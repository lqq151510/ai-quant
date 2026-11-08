"""应用编排层方向 A & C 合流：滚动样本外 (Walk-Forward OOS) 深度时序/动态双时点 KG-GAT 训练器 + 真实份额账本 RL 5-Seed 严格消融引擎.

彻底解决 PROTOTYPE_REVIEW.md 揭示的四大回测失真：
1. 消除收盘同刻成交 (Look-Ahead Bias)：特征严格截至 t 日收盘 (15:00 CST)，标签统一使用 timeline.open_labels (t+1 开盘至 t+h+1 开盘含分红除权真实持有收益)，训练切分强制执行 purge_training (label_end < cutoff)。
2. 消除隐式免费日度再平衡与首日回撤遗漏：所有策略信号通过 timeline.execution_schedule 推迟至下一交易日开盘，统一进入 ledger.simulate 真实份额账本 (100股整数手、T+1锁仓、持仓份额自然漂移、除权除息/折算、涨跌停拒单、DuckDB 净值恒等式核验)。
3. 消除 RL 样本内预测泄漏：通过 Walk-Forward 滚动时间窗生成 100% 样本外 (OOS) Alpha 预测流、CQR 共形分位数区间与 Sticky-HMM/Wasserstein 状态敞口，RL 仅在 OOS 序列上训练与评估。
4. 公平同口径消融对照：在完全相同的调仓周期 (h=5 或 20)、相同费率 (正常/双倍摩擦) 下，同台对比：
   - Equal-Weight (等权基准)
   - Momentum (动量基准)
   - WalkForward-Stage2-TopK (Löwdin GBDT + 双时点动态 KG-GAT 样本外直接轮动)
   - WalkForward-HRP (层次风险平价 + CQR 不确定性排雷)
   - Rule-Prior-Only (仅手写风险先验/未训练网络对照，剥离先验伪增益)
   - Offline-IQL (离线隐式 Q 学习 + HMM/Wasserstein 门控)
   - PPO-5Seed (5 个独立随机种子 PPO 智能体，报告均值 ± 标准差与区块 Bootstrap 风险检验)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 将 research_v1 加入模块搜索路径，复用唯一权威的真实份额账本、时点隔离与区块重采样底座
_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESEARCH_V1 = _REPO_ROOT / "research_v1"
if str(_RESEARCH_V1) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_V1))

from quant_research.ledger import Costs, normalize_actions, simulate  # noqa: E402
from quant_research.risk import block_resample, summarize  # noqa: E402
from quant_research.timeline import execution_schedule, open_labels, purge_training, sessions  # noqa: E402

from src.domain.ensemble.gbdt_stacking import OrthogonalTreeGNNStacker  # noqa: E402
from src.domain.features.pipeline import (  # noqa: E402
    FEATURE_COLUMNS,
    MICRO_FEATURE_COLUMNS,
    compute_raw_feature_columns,
    cross_sectional_mad_zscore,
    lowdin_symmetric_orthogonalize,
)
from src.domain.kg_gnn.bitemporal_kg_bridge import (  # noqa: E402
    BitemporalKGBundle,
    build_bitemporal_bundle_from_news_and_edges,
    compute_bitemporal_3hop_features_and_adjacency,
)
from src.domain.kg_gnn.graph_model import SpatioTemporalKGAlphaModel, build_hybrid_kg_adjacency  # noqa: E402
from src.domain.portfolio.hrp import HierarchicalRiskParityOptimizer  # noqa: E402
from src.domain.regime.market_hmm import StickyMarketRegimeHMM  # noqa: E402
from src.domain.rl_portfolio.a_share_env import ASharePortfolioEnv  # noqa: E402
from src.domain.rl_portfolio.offline_iql import IQLNetworks, train_offline_iql_agent  # noqa: E402
from src.domain.rl_portfolio.ppo_agent import (  # noqa: E402
    PPOPortfolioActorCritic,
    build_alpha_risk_prior,
    train_ppo_portfolio_agent,
)
from src.domain.timeseries_dl.losses import CrossSectionalICLoss  # noqa: E402
from src.domain.uncertainty.quantile_conformal import fit_and_calibrate_conformal_quantile  # noqa: E402
from src.infrastructure.market_data import build_initial_kg_edges  # noqa: E402

PPO_SEEDS: Tuple[int, ...] = (42, 101, 202, 303, 404)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_audited_snapshot_via_duckdb(snapshot_dir: Path) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    """使用 DuckDB 原生读取并校验不可变 Parquet 快照 (无需 pyarrow 依赖)."""
    snapshot_dir = Path(snapshot_dir).resolve()
    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported snapshot schema_version")
    frames: Dict[str, pd.DataFrame] = {}
    with duckdb.connect(config={"threads": 1}) as con:
        for name, record in manifest["tables"].items():
            pq_path = snapshot_dir / f"{name}.parquet"
            if _file_sha256(pq_path) != record["sha256"]:
                raise ValueError(f"Snapshot SHA-256 mismatch for {name}")
            df = con.execute("SELECT * FROM read_parquet(?)", [str(pq_path)]).df()
            if len(df) != record["rows"]:
                raise ValueError(f"Snapshot row count mismatch for {name}")
            frames[name] = df
    return manifest, frames


@dataclass
class OpenToOpenPanelDataset:
    """对齐 t+1 开盘执行与 open_labels 的严格零未来函数时空面板."""

    dates: List[pd.Timestamp]
    label_end_dates: List[pd.Timestamp]
    symbols: List[str]
    x_seq: np.ndarray            # (T, N, L, F_total)
    y_open_ret_h: np.ndarray     # (T, N) 从 t+1 开盘至 t+h+1 开盘的含分红折算真实持有收益率
    y_alpha_z: np.ndarray        # (T, N) 截面 MAD Z-Score 标准化后的超额收益标签
    y_next_open_ret: np.ndarray  # (T, N) 单步开盘到开盘近似收益 (仅供 RL 内部轻量步进奖励训练，最终评估一律走真实份额账本)
    vol_feats: np.ndarray        # (T, N) 截面波动率特征
    dynamic_adj: np.ndarray      # (T, N, N) 双时点 Bitemporal 显式邻接矩阵序列


def build_open_to_open_panel(
    bars_df: pd.DataFrame,
    calendar_dates: Sequence[pd.Timestamp],
    actions_df: pd.DataFrame,
    kg_bundle: BitemporalKGBundle,
    *,
    horizon: int = 5,
    seq_len: int = 20,
) -> OpenToOpenPanelDataset:
    """构建特征截至 t 日收盘、标签为 t+1 开盘至 t+h+1 开盘 (open_labels) 的零穿越面板."""
    cal_idx = sessions(calendar_dates)
    bars = bars_df.copy()
    bars["date"] = pd.to_datetime(bars["date"], errors="raise").dt.normalize()
    bars = bars[bars["date"].isin(cal_idx)].sort_values(["symbol", "date"]).reset_index(drop=True)
    bars["trade_date"] = bars["date"]

    # 1. 计算截至 t 日收盘的量价与微结构特征
    feat_df = compute_raw_feature_columns(bars)
    symbols = sorted(feat_df["symbol"].unique().tolist())
    n_assets = len(symbols)

    # 2. 调用 research_v1.timeline.open_labels 计算严格含分红与份额折算的 Open(t+1) -> Open(t+h+1) 标签
    labels_h = open_labels(bars, cal_idx, horizon=horizon, actions=actions_df)
    labels_h["date"] = pd.to_datetime(labels_h["date"]).dt.normalize()
    labels_h["label_end"] = pd.to_datetime(labels_h["label_end"]).dt.normalize()

    feat_df = feat_df.merge(
        labels_h[["date", "symbol", "label_end", "target_return"]],
        on=["date", "symbol"],
        how="left",
    )

    # 计算 Open(t+1) -> Open(t+2) 单日开盘真实收益 (含除息调整) 供 RL 训练轨迹奖励塑形
    norm_acts = normalize_actions(actions_df, cal_idx)
    act_map: Dict[Tuple[str, pd.Timestamp], Tuple[float, float]] = {}
    for row in norm_acts.itertuples(index=False):
        act_map[(str(row.symbol), pd.Timestamp(row.ex_date).normalize())] = (
            float(row.cash_per_share),
            float(row.share_multiplier),
        )

    next_open_rets = []
    for sym, grp in feat_df.groupby("symbol", sort=False):
        opens = grp["open"].to_numpy(dtype=np.float64)
        dts = pd.to_datetime(grp["date"]).dt.normalize().tolist()
        r_1d = np.full(len(grp), np.nan, dtype=np.float64)
        for i in range(len(grp) - 2):
            p_in = opens[i + 1]
            p_out = opens[i + 2]
            ex_dt = dts[i + 2]
            cash_div, mult = act_map.get((str(sym), ex_dt), (0.0, 1.0))
            if p_in > 0 and p_out > 0:
                r_1d[i] = (p_out * mult + cash_div) / p_in - 1.0
        next_open_rets.append(pd.Series(r_1d, index=grp.index))
    feat_df["next_open_1d"] = pd.concat(next_open_rets).sort_index()

    all_feat_cols = FEATURE_COLUMNS + MICRO_FEATURE_COLUMNS
    # 仅剔除前 20 日滚动窗口预热期与末尾无 label_end 日期；单只停牌股不丢弃整日截面！
    feat_df = feat_df.dropna(subset=all_feat_cols + ["label_end"]).copy()
    feat_df["target_return"] = feat_df.groupby("date")["target_return"].transform(
        lambda s: s.fillna(s.median() if s.notna().any() else 0.0)
    )
    feat_df["next_open_1d"] = feat_df["next_open_1d"].fillna(0.0)

    # 3. 单日截面 MAD 去极值 + Z-Score 标准化
    for col in all_feat_cols:
        feat_df[col] = feat_df.groupby("date")[col].transform(
            lambda s: pd.Series(cross_sectional_mad_zscore(s.fillna(0.0).to_numpy()), index=s.index)
        )
    feat_df["target_alpha_z"] = feat_df.groupby("date")["target_return"].transform(
        lambda s: pd.Series(cross_sectional_mad_zscore(s.to_numpy()), index=s.index)
    )

    valid_dates = (
        feat_df.groupby("date")["symbol"]
        .count()
        .loc[lambda c: c == n_assets]
        .index.sort_values()
        .tolist()
    )
    feat_df = feat_df[feat_df["date"].isin(valid_dates)].sort_values(["date", "symbol"]).reset_index(drop=True)

    # 4. 计算双时点 Bitemporal 动态邻接矩阵 A_t 与三跳事件传导特征 E_t
    dyn_adj_all, ev_feats_all = compute_bitemporal_3hop_features_and_adjacency(
        valid_dates, symbols, kg_bundle, lookback_days=7
    )

    n_valid = len(valid_dates)
    f_base = len(all_feat_cols)
    base_cube = feat_df[all_feat_cols].to_numpy(dtype=np.float32).reshape(n_valid, n_assets, f_base)
    full_cube = np.concatenate([base_cube, ev_feats_all], axis=-1)  # (T_valid, N, f_base + 4)
    f_total = full_cube.shape[-1]

    y_open_cube = feat_df["target_return"].to_numpy(dtype=np.float32).reshape(n_valid, n_assets)
    y_alpha_cube = feat_df["target_alpha_z"].to_numpy(dtype=np.float32).reshape(n_valid, n_assets)
    y_next1d_cube = feat_df["next_open_1d"].to_numpy(dtype=np.float32).reshape(n_valid, n_assets)
    label_end_series = feat_df.groupby("date")["label_end"].first().loc[valid_dates].tolist()

    # 5. 滑动窗口构造 (T_seq, N, seq_len, F_total)
    x_seq_list, out_dates, out_label_ends = [], [], []
    y_open_list, y_alpha_list, y_next1d_list, vol_list, adj_list = [], [], [], [], []
    for t in range(seq_len - 1, n_valid):
        window = np.transpose(full_cube[t - seq_len + 1 : t + 1], (1, 0, 2))  # (N, L, F_total)
        x_seq_list.append(window)
        out_dates.append(pd.Timestamp(valid_dates[t]))
        out_label_ends.append(pd.Timestamp(label_end_series[t]))
        y_open_list.append(y_open_cube[t])
        y_alpha_list.append(y_alpha_cube[t])
        y_next1d_list.append(y_next1d_cube[t])
        vol_list.append(base_cube[t, :, 7])  # vol_ratio_5_20 截面波动指标
        adj_list.append(dyn_adj_all[t])

    return OpenToOpenPanelDataset(
        dates=out_dates,
        label_end_dates=out_label_ends,
        symbols=symbols,
        x_seq=np.stack(x_seq_list, axis=0),
        y_open_ret_h=np.stack(y_open_list, axis=0),
        y_alpha_z=np.stack(y_alpha_list, axis=0),
        y_next_open_ret=np.stack(y_next1d_list, axis=0),
        vol_feats=np.stack(vol_list, axis=0),
        dynamic_adj=np.stack(adj_list, axis=0),
    )


def _train_fold_kg_gat_and_stacker(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    adj_tr: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    adj_val: np.ndarray,
    x_oos: np.ndarray,
    adj_oos: np.ndarray,
    symbols: List[str],
    base_edges_df: pd.DataFrame,
    *,
    epochs: int = 12,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在单个 Walk-Forward 折叠的已清洗 (Purged) 训练集上训练动态双时点 KG-GAT + Löwdin GBDT Stacker.

    Returns:
        val_preds: (T_val, N) 验证折预测 (用于 CQR 共形分位数校准)
        oos_preds: (T_oos, N) 严格样本外折叠预测
        oos_uncertainty: (T_oos, N) CQR 90% 预测区间宽度
    """
    _set_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    _, n_assets, seq_len, n_feats = x_tr.shape

    # 仅使用当前折叠训练集估计混合先验邻接矩阵 (零未来泄漏)
    hybrid_adj = build_hybrid_kg_adjacency(symbols, base_edges_df, x_tr).to(device)
    model = SpatioTemporalKGAlphaModel(
        adj_prior=hybrid_adj,
        in_features=n_feats,
        seq_len=seq_len,
        d_model=48,
        n_heads=4,
        dropout=0.15,
    ).to(device)

    # 联合训练 iTransformer + PatchTST + 动态 KG 投影头
    for param in model.parameters():
        param.requires_grad = True

    criterion = CrossSectionalICLoss(mse_weight=0.20)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    loader = DataLoader(
        TensorDataset(
            torch.tensor(x_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
            torch.tensor(adj_tr, dtype=torch.float32),
        ),
        batch_size=32,
        shuffle=True,
    )

    best_val_ic = -2.0
    best_state = None

    def _infer(x_arr: np.ndarray, adj_arr: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            xt = torch.tensor(x_arr, dtype=torch.float32, device=device)
            at = torch.tensor(adj_arr, dtype=torch.float32, device=device)
            # 融合训练集隐式先验与当日双时点显式动态图边
            blended_adj = 0.5 * hybrid_adj.unsqueeze(0) + 0.5 * at
            return model(xt, dynamic_adj=blended_adj).cpu().numpy()

    for _ in range(epochs):
        model.train()
        for bx, by, badj in loader:
            bx, by, badj = bx.to(device), by.to(device), badj.to(device)
            optimizer.zero_grad()
            blended = 0.5 * hybrid_adj.unsqueeze(0) + 0.5 * badj
            pred = model(bx, dynamic_adj=blended)
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_p = _infer(x_val, adj_val)
        ics = [
            float(np.corrcoef(val_p[i], y_val[i])[0, 1])
            for i in range(len(val_p))
            if np.std(val_p[i]) > 1e-8 and np.std(y_val[i]) > 1e-8
        ]
        mean_ic = float(np.nanmean(ics)) if ics else -1.0
        if mean_ic > best_val_ic:
            best_val_ic = mean_ic
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    tr_gnn_preds = _infer(x_tr, adj_tr)
    val_gnn_preds = _infer(x_val, adj_val)
    oos_gnn_preds = _infer(x_oos, adj_oos)

    # Löwdin 对称正交化 GBDT + KG-GAT Stacking，并在 Purged 验证集上自适应校准方向与融合权重
    stacker = OrthogonalTreeGNNStacker(random_state=seed)
    blended, oos_tree = stacker.fit_and_blend(
        train_x=x_tr,
        train_y=y_tr,
        val_x=x_val,
        val_y=y_val,
        test_x=x_oos,
        gnn_preds={"train_preds": tr_gnn_preds, "test_preds": oos_gnn_preds},
    )
    oos_ens = blended["test_preds"]

    # 在 Purged 验证集上评估 GNN 与 GBDT 的样本外泛化方向，自适应调节正交专家权重 (零测试集泄漏)
    val_blended, val_tree = stacker.fit_and_blend(
        train_x=x_tr,
        train_y=y_tr,
        val_x=x_val,
        val_y=y_val,
        test_x=x_val,
        gnn_preds={"train_preds": tr_gnn_preds, "test_preds": val_gnn_preds},
    )
    val_ens_p = val_blended["test_preds"]
    ens_val_ics = [
        float(np.corrcoef(val_ens_p[i], y_val[i])[0, 1])
        for i in range(len(y_val))
        if np.std(val_ens_p[i]) > 1e-8 and np.std(y_val[i]) > 1e-8
    ]
    mean_ens_val_ic = float(np.nanmean(ens_val_ics)) if ens_val_ics else 0.0
    # 融合 A 股截面中期动量-资金流锚 (sharpe_20d + cmf_20d - ma_bias_20)
    oos_anchor = (
        0.45 * x_oos[:, :, -1, 5]
        + 0.35 * x_oos[:, :, -1, 8]
        - 0.20 * x_oos[:, :, -1, 6]
    )
    if mean_ens_val_ic < -0.005:
        oos_ENS = -0.55 * oos_ens + 0.45 * oos_anchor
    else:
        oos_ENS = 0.75 * oos_ens + 0.25 * oos_anchor

    # CQR 共形分位数回归校准 (在 purged 验证集校准，在 OOS 折叠推理)
    cqr = fit_and_calibrate_conformal_quantile(
        train_x=x_tr,
        train_y=y_tr,
        val_x=x_val,
        val_y=y_val,
        eval_x=x_oos,
        alpha_miscoverage=0.20,
        epochs=12,
        device=device,
    )
    return val_gnn_preds, oos_ENS, cqr.interval_width


@dataclass
class WalkForwardOOSStream:
    """拼接所有滚动折叠后的纯样本外 (OOS) 预测流."""

    dates: List[pd.Timestamp]
    label_end_dates: List[pd.Timestamp]
    symbols: List[str]
    oos_alpha: np.ndarray          # (T_oos, N)
    oos_uncertainty: np.ndarray    # (T_oos, N)
    oos_hmm_exposure: List[float]  # 长度 T_oos
    y_open_ret_h: np.ndarray       # (T_oos, N)
    y_alpha_z: np.ndarray          # (T_oos, N)
    y_next_open_ret: np.ndarray    # (T_oos, N)
    vol_feats: np.ndarray          # (T_oos, N)
    fold_boundaries: List[Dict[str, str]]


def _build_macro_df_from_panel_returns(dates: Sequence[pd.Timestamp], rets_mat: np.ndarray) -> pd.DataFrame:
    """从截面日收益矩阵 (T, N) 构建供 StickyMarketRegimeHMM 因果滤波的宏观观测序列 (含 1D Wasserstein)."""
    from src.domain.regime.market_hmm import compute_1d_wasserstein_w2

    mkt_ret_1d = np.mean(rets_mat, axis=1)
    cross_disp = np.std(rets_mat, axis=1) + 1e-6
    s_ret = pd.Series(mkt_ret_1d)
    mkt_ret_5d = s_ret.rolling(5, min_periods=1).mean().to_numpy()
    mkt_vol_10d = s_ret.rolling(10, min_periods=2).std().bfill().fillna(0.01).to_numpy()

    w2_list, left_shift_list = [], []
    for t in range(len(dates)):
        if t < 20:
            w2_list.append(0.008)
            left_shift_list.append(0.0)
        else:
            recent = rets_mat[max(0, t - 4) : t + 1].reshape(-1)
            ref = rets_mat[max(0, t - 45) : max(1, t - 4)].reshape(-1)
            w2, l_shift = compute_1d_wasserstein_w2(recent, ref)
            w2_list.append(w2)
            left_shift_list.append(l_shift)

    return pd.DataFrame({
        "trade_date": list(dates),
        "mkt_ret_1d": mkt_ret_1d,
        "mkt_ret_5d": mkt_ret_5d,
        "mkt_vol_10d": mkt_vol_10d,
        "cross_dispersion": cross_disp,
        "wasserstein_w2": w2_list,
        "ot_left_tail_shift": left_shift_list,
    })


def generate_walk_forward_oos_stream(
    panel: OpenToOpenPanelDataset,
    base_edges_df: pd.DataFrame,
    *,
    min_train_sessions: int = 400,
    val_sessions: int = 80,
    oos_step_sessions: int = 160,
    epochs_per_fold: int = 10,
    seed: int = 42,
) -> WalkForwardOOSStream:
    """沿时间轴执行滚动 Walk-Forward 训练与 Purge 隔离，生成零样本内泄漏的 OOS 预测序列."""
    n_total = len(panel.dates)
    if n_total < min_train_sessions + val_sessions + 30:
        # 自动按比例缩放以兼容短周期单测夹具
        min_train_sessions = max(20, int(n_total * 0.45))
        val_sessions = max(10, int(n_total * 0.15))
        oos_step_sessions = max(15, int(n_total * 0.20))

    meta_df = pd.DataFrame({
        "idx": np.arange(n_total),
        "date": panel.dates,
        "label_end": panel.label_end_dates,
    })
    macro_all_df = _build_macro_df_from_panel_returns(panel.dates, panel.y_next_open_ret)

    oos_indices: List[int] = []
    oos_alpha_chunks: List[np.ndarray] = []
    oos_unc_chunks: List[np.ndarray] = []
    oos_hmm_caps: List[float] = []
    fold_Meta: List[Dict[str, str]] = []

    cursor = min_train_sessions + val_sessions
    fold_id = 0
    while cursor < n_total:
        fold_end = min(n_total, cursor + oos_step_sessions)
        oos_cutoff = panel.dates[cursor]
        val_start_idx = cursor - val_sessions
        val_cutoff = panel.dates[val_start_idx]

        # 严格调用 research_v1.timeline.purge_training 切除跨边界标签泄漏 (label_end < cutoff)
        purged_train_df = purge_training(meta_df.iloc[:val_start_idx], val_cutoff)
        purged_val_df = purge_training(meta_df.iloc[val_start_idx:cursor], oos_cutoff)
        oos_slice_df = meta_df.iloc[cursor:fold_end]

        tr_idx = purged_train_df["idx"].to_numpy(dtype=int)
        val_idx = purged_val_df["idx"].to_numpy(dtype=int)
        te_idx = oos_slice_df["idx"].to_numpy(dtype=int)

        if len(tr_idx) < 8 or len(val_idx) < 4 or len(te_idx) == 0:
            break

        _, fold_alpha, fold_unc = _train_fold_kg_gat_and_stacker(
            panel.x_seq[tr_idx],
            panel.y_alpha_z[tr_idx],
            panel.dynamic_adj[tr_idx],
            panel.x_seq[val_idx],
            panel.y_alpha_z[val_idx],
            panel.dynamic_adj[val_idx],
            panel.x_seq[te_idx],
            panel.dynamic_adj[te_idx],
            panel.symbols,
            base_edges_df,
            epochs=epochs_per_fold,
            seed=seed + fold_id,
        )

        # 在当前折叠的历史观测上拟合 Sticky-HMM + 1D Wasserstein，并因果前向滤波 OOS 折叠
        hmm = StickyMarketRegimeHMM(self_transition_prob=0.82, random_state=seed + fold_id)
        hmm.fit(macro_all_df.iloc[:cursor])
        snaps = hmm.filter_sequence(macro_all_df.iloc[:fold_end])
        for snap in snaps[cursor:fold_end]:
            oos_hmm_caps.append(float(snap.recommended_max_exposure))

        oos_indices.extend(te_idx.tolist())
        oos_alpha_chunks.append(fold_alpha)
        oos_unc_chunks.append(fold_unc)
        fold_Meta.append({
            "fold": str(fold_id),
            "train_range": f"{panel.dates[tr_idx[0]].date()}..{panel.dates[tr_idx[-1]].date()} (n={len(tr_idx)})",
            "val_range": f"{panel.dates[val_idx[0]].date()}..{panel.dates[val_idx[-1]].date()} (n={len(val_idx)})",
            "oos_range": f"{panel.dates[te_idx[0]].date()}..{panel.dates[te_idx[-1]].date()} (n={len(te_idx)})",
        })
        cursor = fold_end
        fold_id += 1

    idx_arr = np.array(oos_indices, dtype=int)
    return WalkForwardOOSStream(
        dates=[panel.dates[i] for i in idx_arr],
        label_end_dates=[panel.label_end_dates[i] for i in idx_arr],
        symbols=panel.symbols,
        oos_alpha=np.concatenate(oos_alpha_chunks, axis=0),
        oos_uncertainty=np.concatenate(oos_unc_chunks, axis=0),
        oos_hmm_exposure=oos_hmm_caps,
        y_open_ret_h=panel.y_open_ret_h[idx_arr],
        y_alpha_z=panel.y_alpha_z[idx_arr],
        y_next_open_ret=panel.y_next_open_ret[idx_arr],
        vol_feats=panel.vol_feats[idx_arr],
        fold_boundaries=fold_Meta,
    )


def select_buffered_top_k(
    scores: np.ndarray,
    prev_weights: np.ndarray,
    top_k: int,
    buffer_mult: float = 1.6,
) -> np.ndarray:
    """滞回缓冲带选股算子 (No-Trade Hysteresis Buffer)：已持仓标的只要仍居 Top-M 且高于截面分位阈值即优先保留."""
    n_assets = len(scores)
    k = min(max(1, top_k), n_assets)
    eff_mult = max(buffer_mult, 2.4 if n_assets > 6 else 1.6)
    keep_m = min(n_assets, max(k + 1, int(np.ceil(k * eff_mult))))
    ranked_desc = np.argsort(scores)[::-1]
    top_m_set = set(ranked_desc[:keep_m].tolist())
    # 宽截面下只要已持仓标的仍高于截面 45% 分位数且居 Top-M，即不触发换仓避免印花税损耗
    q_hurdle = float(np.quantile(scores, 0.45 if n_assets > 6 else 0.50))

    retained: List[int] = []
    for idx in ranked_desc:
        if len(retained) >= k:
            break
        if prev_weights[idx] > 1e-3 and idx in top_m_set and scores[idx] >= q_hurdle:
            retained.append(int(idx))

    for idx in ranked_desc:
        if len(retained) >= k:
            break
        if int(idx) not in retained:
            retained.append(int(idx))
    return np.array(sorted(retained), dtype=int)


def apply_turnover_deadband(
    proposed_w: np.ndarray,
    prev_w: np.ndarray,
    *,
    l1_deadband: float = 0.16,
    inertia_blend: float = 0.40,
) -> Tuple[np.ndarray, bool]:
    """L1 换手死区与持仓惯性混合算子：若目标仓位与当前持仓 L1 距离小于阈值则跳过调仓，消除微幅噪音摩擦."""
    if float(np.sum(prev_w)) < 1e-4:
        return proposed_w.copy(), True

    # 当新旧持仓集合完全相同时，允许更大的免交易死区避免每期支付最低佣金与滑点
    same_support = np.array_equal(proposed_w > 1e-3, prev_w > 1e-3)
    effective_blend = inertia_blend if not same_support else 0.65
    blended = (1.0 - effective_blend) * proposed_w + effective_blend * prev_w
    # 清除极小碎股权重并保持总敞口一致
    blended = np.where(blended >= 0.015, blended, 0.0)
    target_sum = float(np.sum(proposed_w))
    curr_sum = float(np.sum(blended))
    if curr_sum > 1e-6 and target_sum > 1e-6:
        blended = blended * (target_sum / curr_sum)

    l1_diff = float(np.sum(np.abs(blended - prev_w)))
    threshold = l1_deadband * (1.35 if same_support else 1.0)
    if l1_diff < threshold:
        return prev_w.copy(), False
    return blended, True


def _weights_from_policy_logits(
    logits: np.ndarray,
    alpha_t: np.ndarray,
    prev_w: np.ndarray,
    current_dd: float,
    hmm_cap: float,
    top_k: int,
) -> np.ndarray:
    """结合策略网络动作 Logits、滞回缓冲带与 HMM/回撤门控输出 (N,) 维目标权重向量."""
    n_assets = len(alpha_t)
    clipped = np.clip(logits, -10.0, 10.0)
    exp_a = np.exp(clipped - np.max(clipped))
    alloc = exp_a / np.sum(exp_a)
    raw_cash = float(alloc[-1])

    k = min(top_k, n_assets)
    # 融合样本外 Alpha 与 RL 策略网络自身输出的资产偏好 Logits 进行滞回选股
    policy_score = 0.55 * alpha_t + 0.45 * np.tanh(clipped[:n_assets])
    top_idx = select_buffered_top_k(policy_score, prev_w, k)
    tilt = 1.0 + 0.35 * np.tanh(clipped[top_idx])
    asset_dist = np.zeros(n_assets, dtype=np.float64)
    asset_dist[top_idx] = tilt / np.sum(tilt)

    target_cash = float(np.clip(0.40 * raw_cash + 0.45 * current_dd, 1.0 - hmm_cap, 0.35))
    equity_exposure = float(np.clip(1.0 - target_cash, 0.0, hmm_cap))
    return (asset_dist * equity_exposure).astype(np.float64)


def _simulate_strategy_on_share_ledger(
    signal_weights_by_date: Mapping[pd.Timestamp, Mapping[str, float]],
    bars_df: pd.DataFrame,
    eval_calendar: pd.DatetimeIndex,
    actions_df: pd.DataFrame,
    *,
    initial_cash: float = 100_000.0,
    cost: Costs = Costs(),
) -> Tuple[Dict[str, Any], pd.Series]:
    """将收盘信号通过 execution_schedule 推迟到 t+1 开盘，并在 research_v1.ledger.simulate 上完成真实份额撮合与对账."""
    scheduled_orders = execution_schedule(eval_calendar, signal_weights_by_date)
    bt = simulate(bars_df, eval_calendar, scheduled_orders, actions_df, initial_cash=initial_cash, cost=cost)
    metrics = summarize(bt.equity, bt.orders, initial_cash)

    # 独立校验净值恒等式 nav == cash + market_value + receivable 与非负现金
    recon_err = float(
        (bt.equity["nav"] - bt.equity["cash"] - bt.equity["market_value"] - bt.equity["receivable"]).abs().max()
    )
    min_cash = float(bt.equity["cash"].min())
    if recon_err > 1e-6 or min_cash < -1e-7:
        raise RuntimeError(f"Share ledger reconciliation failed: err={recon_err}, min_cash={min_cash}")

    metrics["ledger_reconciliation_error"] = recon_err
    metrics["min_cash"] = min_cash
    daily_ret_series = bt.equity.set_index("date")["daily_return"]
    return metrics, daily_ret_series


def run_walk_forward_ablation_experiment(
    bars_df: pd.DataFrame,
    calendar_dates: Sequence[pd.Timestamp],
    actions_df: pd.DataFrame,
    kg_bundle: BitemporalKGBundle,
    base_edges_df: pd.DataFrame,
    *,
    horizon: int = 5,
    top_k: int = 2,
    initial_cash: float = 100_000.0,
    min_train_sessions: int = 400,
    val_sessions: int = 80,
    oos_step_sessions: int = 160,
    epochs_per_fold: int = 8,
    ppo_seeds: Sequence[int] = PPO_SEEDS,
) -> Dict[str, Any]:
    """执行端到端 Walk-Forward 样本外训练 + 真实份额账本同口径消融 (含 5-Seed PPO 与 Prior-Only 对照)."""
    panel = build_open_to_open_panel(
        bars_df, calendar_dates, actions_df, kg_bundle, horizon=horizon, seq_len=20
    )
    oos_stream = generate_walk_forward_oos_stream(
        panel,
        base_edges_df,
        min_train_sessions=min_train_sessions,
        val_sessions=val_sessions,
        oos_step_sessions=oos_step_sessions,
        epochs_per_fold=epochs_per_fold,
    )

    # 将 OOS 序列切分为 RL 样本外训练段 (前 55%) 与最终未见样本外评估段 (后 45%)，严守 purge_training
    n_oos = len(oos_stream.dates)
    rl_split = max(15, int(n_oos * 0.55))
    eval_cutoff = oos_stream.dates[rl_split]
    rl_meta = pd.DataFrame({
        "idx": np.arange(rl_split),
        "date": oos_stream.dates[:rl_split],
        "label_end": oos_stream.label_end_dates[:rl_split],
    })
    rl_train_idx = purge_training(rl_meta, eval_cutoff)["idx"].to_numpy(dtype=int)
    if len(rl_train_idx) < 5:
        rl_train_idx = np.arange(max(1, rl_split - horizon - 1))
    eval_idx = np.arange(rl_split, n_oos)

    eval_dates = [oos_stream.dates[i] for i in eval_idx]
    symbols = oos_stream.symbols
    n_assets = len(symbols)
    k = min(top_k, n_assets)

    # 计算 OOS 评估段 IC / RankIC / ICIR
    eval_alpha = oos_stream.oos_alpha[eval_idx]
    eval_unc = oos_stream.oos_uncertainty[eval_idx]
    eval_target_z = oos_stream.y_alpha_z[eval_idx]
    eval_vol = oos_stream.vol_feats[eval_idx]
    eval_hmm_caps = [oos_stream.oos_hmm_exposure[i] for i in eval_idx]

    daily_ics, daily_rics = [], []
    for t in range(len(eval_idx)):
        p, y = eval_alpha[t], eval_target_z[t]
        if np.std(p) > 1e-8 and np.std(y) > 1e-8:
            ic_v = float(np.corrcoef(p, y)[0, 1])
            ric_v, _ = spearmanr(p, y)
            if np.isfinite(ic_v):
                daily_ics.append(ic_v)
            if np.isfinite(ric_v):
                daily_rics.append(float(ric_v))
    ic_mean = float(np.mean(daily_ics)) if daily_ics else 0.0
    rank_ic_mean = float(np.mean(daily_rics)) if daily_rics else 0.0
    icir = float(ic_mean / (np.std(daily_ics) + 1e-8) * np.sqrt(252.0 / horizon)) if daily_ics else 0.0

    # 准备评估窗口的完整交易日历 (含最后一个信号执行所需顺延交易日)
    full_cal = sessions(calendar_dates)
    eval_cal = full_cal[(full_cal >= eval_dates[0]) & (full_cal <= eval_dates[-1])]
    # 包含 eval_dates[-1] 之后的下一个交易日以便最后一个信号能在 t+1 开盘成交
    after_days = full_cal[full_cal > eval_dates[-1]]
    if len(after_days) > 0:
        eval_cal = eval_cal.union(after_days[:1])

    # 在纯样本外 RL 训练段 (rl_train_idx) 构建训练环境
    rl_train_env = ASharePortfolioEnv(
        alpha_preds=oos_stream.oos_alpha[rl_train_idx],
        vol_features=oos_stream.vol_feats[rl_train_idx],
        real_next_rets=oos_stream.y_next_open_ret[rl_train_idx],
    )

    # 1. 训练 Offline IQL (在纯 OOS 训练段拟合)
    _set_seed(42)
    iql_net = train_offline_iql_agent(rl_train_env, n_behavior_episodes=14, iql_epochs=30)
    iql_net.eval()

    # 2. 对 OOS Alpha 序列执行严格因果指数移动平均 (EMA) 去噪，抑制周度截面排名高频毛刺
    all_next_rets = oos_stream.y_next_open_ret
    smoothed_alpha = np.zeros_like(oos_stream.oos_alpha, dtype=np.float64)
    ema_eta = 0.16 if n_assets > 6 else 0.35
    deadband_val = 0.34 if n_assets > 6 else 0.16
    blend_val = 0.55 if n_assets > 6 else 0.40
    for idx_t in range(len(oos_stream.dates)):
        raw_p = oos_stream.oos_alpha[idx_t] - 0.08 * oos_stream.oos_uncertainty[idx_t]
        smoothed_alpha[idx_t] = raw_p if idx_t == 0 else (1.0 - ema_eta) * smoothed_alpha[idx_t - 1] + ema_eta * raw_p

    # 构造同频率 (每 horizon 个交易日检查一次，结合滞回缓冲带与 L1 换手死区) 的各策略收盘信号表
    eq_signals: Dict[pd.Timestamp, Dict[str, float]] = {}
    stage2_topk_signals: Dict[pd.Timestamp, Dict[str, float]] = {}
    hrp_signals: Dict[pd.Timestamp, Dict[str, float]] = {}
    prior_only_signals: Dict[pd.Timestamp, Dict[str, float]] = {}
    iql_signals: Dict[pd.Timestamp, Dict[str, float]] = {}

    prev_stage2_w = np.zeros(n_assets, dtype=np.float64)
    prev_hrp_w = np.zeros(n_assets, dtype=np.float64)
    prev_prior_w = np.zeros(n_assets, dtype=np.float64)
    prev_iql_w = np.zeros(n_assets, dtype=np.float64)
    prior_nav, prior_peak = 1.0, 1.0
    iql_nav, iql_peak = 1.0, 1.0

    for step_i, global_i in enumerate(eval_idx):
        day_ret = all_next_rets[global_i]
        if step_i % horizon == 0:
            day = oos_stream.dates[global_i]
            penalized_a = smoothed_alpha[global_i]
            v_t = oos_stream.vol_feats[global_i]
            hmm_cap = oos_stream.oos_hmm_exposure[global_i]

            # (a) Equal-Weight (保持原样作为不可变对照基准)
            eq_signals[day] = {sym: 1.0 / n_assets for sym in symbols}

            # (b) WalkForward Stage2 Top-K (Löwdin GBDT + Bitemporal KG-GAT + 滞回缓冲带 + L1 换手死区)
            top_indices = select_buffered_top_k(penalized_a, prev_stage2_w, k)
            prop_stage2_w = np.zeros(n_assets, dtype=np.float64)
            # 按平滑 Alpha 强度做温和倾角分配
            tilts = np.exp(0.35 * (penalized_a[top_indices] - np.max(penalized_a[top_indices])))
            prop_stage2_w[top_indices] = tilts / np.sum(tilts)
            new_stage2_w, trade_stage2 = apply_turnover_deadband(
                prop_stage2_w, prev_stage2_w, l1_deadband=deadband_val, inertia_blend=blend_val
            )
            if trade_stage2:
                stage2_topk_signals[day] = {
                    symbols[idx]: float(new_stage2_w[idx])
                    for idx in range(n_assets)
                    if new_stage2_w[idx] > 1e-4
                }
                prev_stage2_w = new_stage2_w

            # (c) WalkForward HRP + CQR + HMM Cap + 滞回缓冲带 + L1 换手死区
            hrp_candidates = select_buffered_top_k(penalized_a, prev_hrp_w, k)
            hist_start = max(0, global_i - 60)
            hist_rets = all_next_rets[hist_start:global_i]
            if len(hist_rets) >= 10:
                hrp_raw = HierarchicalRiskParityOptimizer.allocate_hrp_weights(
                    hist_returns=hist_rets,
                    candidate_indices=hrp_candidates.tolist(),
                    alpha_tilts=penalized_a,
                )
            else:
                hrp_raw = np.zeros(n_assets, dtype=np.float64)
                hrp_raw[hrp_candidates] = 1.0 / k
            prop_hrp_w = (hrp_raw * hmm_cap).astype(np.float64)
            new_hrp_w, trade_hrp = apply_turnover_deadband(
                prop_hrp_w, prev_hrp_w, l1_deadband=deadband_val, inertia_blend=blend_val
            )
            if trade_hrp:
                hrp_signals[day] = {
                    symbols[idx]: float(new_hrp_w[idx])
                    for idx in range(n_assets)
                    if new_hrp_w[idx] > 1e-4
                }
                prev_hrp_w = new_hrp_w

            # (d) Rule-Prior Only (带真实持仓状态反馈与换手死区)
            prior_dd = (prior_peak - prior_nav) / max(prior_peak, 1e-8)
            prior_cash = max(0.0, 1.0 - float(np.sum(prev_prior_w)))
            obs_prior = np.concatenate([
                penalized_a.astype(np.float32),
                v_t.astype(np.float32),
                prev_prior_w.astype(np.float32),
                np.array([prior_cash, prior_dd], dtype=np.float32),
            ])
            prior_logits = build_alpha_risk_prior(obs_prior, n_assets, top_k=k).squeeze(0).numpy()
            prop_prior_w = _weights_from_policy_logits(
                prior_logits, penalized_a, prev_prior_w, prior_dd, hmm_cap, k
            )
            new_prior_w, trade_prior = apply_turnover_deadband(
                prop_prior_w, prev_prior_w, l1_deadband=deadband_val, inertia_blend=blend_val
            )
            if trade_prior:
                prior_only_signals[day] = {
                    symbols[idx]: float(new_prior_w[idx])
                    for idx in range(n_assets)
                    if new_prior_w[idx] > 1e-4
                }
                prev_prior_w = new_prior_w

            # (e) Offline IQL Policy (带真实持仓状态反馈与策略网络选股/倾角控制)
            iql_dd = (iql_peak - iql_nav) / max(iql_peak, 1e-8)
            iql_cash = max(0.0, 1.0 - float(np.sum(prev_iql_w)))
            obs_iql = np.concatenate([
                penalized_a.astype(np.float32),
                v_t.astype(np.float32),
                prev_iql_w.astype(np.float32),
                np.array([iql_cash, iql_dd], dtype=np.float32),
            ])
            prior_iql_logits = build_alpha_risk_prior(obs_iql, n_assets, top_k=k).squeeze(0).numpy()
            with torch.no_grad():
                obs_t = torch.tensor(obs_iql, dtype=torch.float32).unsqueeze(0)
                iql_delta = iql_net.policy_net(obs_t).squeeze(0).numpy()
            iql_logits = prior_iql_logits + 0.85 * iql_delta
            prop_iql_w = _weights_from_policy_logits(
                iql_logits, penalized_a, prev_iql_w, iql_dd, hmm_cap, k
            )
            new_iql_w, trade_iql = apply_turnover_deadband(
                prop_iql_w, prev_iql_w, l1_deadband=deadband_val, inertia_blend=blend_val
            )
            if trade_iql:
                iql_signals[day] = {
                    symbols[idx]: float(new_iql_w[idx])
                    for idx in range(n_assets)
                    if new_iql_w[idx] > 1e-4
                }
                prev_iql_w = new_iql_w

        # 自然价格漂移更新各策略内部跟踪状态 (零未来函数：仅用于下一交易日状态输入)
        for w_ref in (prev_stage2_w, prev_hrp_w):
            if float(np.sum(w_ref)) > 1e-5:
                cash_part = max(0.0, 1.0 - float(np.sum(w_ref)))
                grown = w_ref * (1.0 + day_ret)
                tot = float(np.sum(grown)) + cash_part
                w_ref[:] = grown / max(tot, 1e-8)

        if float(np.sum(prev_prior_w)) > 1e-5:
            prior_nav *= (1.0 + float(np.dot(prev_prior_w, day_ret)))
            prior_peak = max(prior_peak, prior_nav)
            cash_part = max(0.0, 1.0 - float(np.sum(prev_prior_w)))
            grown = prev_prior_w * (1.0 + day_ret)
            prev_prior_w = grown / max(float(np.sum(grown)) + cash_part, 1e-8)

        if float(np.sum(prev_iql_w)) > 1e-5:
            iql_nav *= (1.0 + float(np.dot(prev_iql_w, day_ret)))
            iql_peak = max(iql_peak, iql_nav)
            cash_part = max(0.0, 1.0 - float(np.sum(prev_iql_w)))
            grown = prev_iql_w * (1.0 + day_ret)
            prev_iql_w = grown / max(float(np.sum(grown)) + cash_part, 1e-8)

    # 3. 5-Seed PPO 训练与真实份额账本评估 (含闭环状态跟踪与 L1 换手死区)
    ppo_seed_schedules: Dict[int, Dict[pd.Timestamp, Dict[str, float]]] = {}
    for seed_val in ppo_seeds:
        _set_seed(seed_val)
        env_seed = ASharePortfolioEnv(
            alpha_preds=oos_stream.oos_alpha[rl_train_idx],
            vol_features=oos_stream.vol_feats[rl_train_idx],
            real_next_rets=oos_stream.y_next_open_ret[rl_train_idx],
        )
        ppo_model = train_ppo_portfolio_agent(env_seed, rollout_episodes=20, ppo_epochs=4, lr=1.5e-3)
        ppo_model.eval()
        seed_sig: Dict[pd.Timestamp, Dict[str, float]] = {}
        prev_ppo_w = np.zeros(n_assets, dtype=np.float64)
        ppo_nav, ppo_peak = 1.0, 1.0
        for step_i, global_i in enumerate(eval_idx):
            day_ret = all_next_rets[global_i]
            if step_i % horizon == 0:
                day = oos_stream.dates[global_i]
                a_t = smoothed_alpha[global_i]
                v_t = oos_stream.vol_feats[global_i]
                hmm_cap = oos_stream.oos_hmm_exposure[global_i]
                ppo_dd = (ppo_peak - ppo_nav) / max(ppo_peak, 1e-8)
                ppo_cash = max(0.0, 1.0 - float(np.sum(prev_ppo_w)))
                obs_vec = np.concatenate([
                    a_t.astype(np.float32),
                    v_t.astype(np.float32),
                    prev_ppo_w.astype(np.float32),
                    np.array([ppo_cash, ppo_dd], dtype=np.float32),
                ])
                prior_logits = build_alpha_risk_prior(obs_vec, n_assets, top_k=k).squeeze(0).numpy()
                with torch.no_grad():
                    obs_t = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
                    mean_delta, _ = ppo_model(obs_t)
                ppo_logits = prior_logits + 0.85 * mean_delta.squeeze(0).numpy()
                prop_ppo_w = _weights_from_policy_logits(
                    ppo_logits, a_t, prev_ppo_w, ppo_dd, hmm_cap, k
                )
                new_ppo_w, trade_ppo = apply_turnover_deadband(
                    prop_ppo_w, prev_ppo_w, l1_deadband=deadband_val, inertia_blend=blend_val
                )
                if trade_ppo:
                    seed_sig[day] = {
                        symbols[idx]: float(new_ppo_w[idx])
                        for idx in range(n_assets)
                        if new_ppo_w[idx] > 1e-4
                    }
                    prev_ppo_w = new_ppo_w

            if float(np.sum(prev_ppo_w)) > 1e-5:
                ppo_nav *= (1.0 + float(np.dot(prev_ppo_w, day_ret)))
                ppo_peak = max(ppo_peak, ppo_nav)
                cash_part = max(0.0, 1.0 - float(np.sum(prev_ppo_w)))
                grown = prev_ppo_w * (1.0 + day_ret)
                prev_ppo_w = grown / max(float(np.sum(grown)) + cash_part, 1e-8)
        ppo_seed_schedules[seed_val] = seed_sig

    # 4. 在真实份额账本 (normal 与 double_friction 两种摩擦情景) 下统一结算全部策略
    base_cost = Costs()
    strategy_Map = {
        "equal_weight": eq_signals,
        "wf_stage2_bitemporal_kg_gat": stage2_topk_signals,
        "wf_hrp_cqr_hmm": hrp_signals,
        "rl_rule_prior_only": prior_only_signals,
        "rl_offline_iql": iql_signals,
    }

    ledger_reports: Dict[str, Dict[str, Any]] = {}
    normal_daily_returns: Dict[str, pd.Series] = {}

    for cost_name, cost_cfg in (("normal", base_cost), ("double_friction", base_cost.stressed())):
        ledger_reports[cost_name] = {}
        for strat_name, sig_schedule in strategy_Map.items():
            m, r_series = _simulate_strategy_on_share_ledger(
                sig_schedule, bars_df, eval_cal, actions_df, initial_cash=initial_cash, cost=cost_cfg
            )
            ledger_reports[cost_name][strat_name] = m
            if cost_name == "normal":
                normal_daily_returns[strat_name] = r_series

        # 汇总 5-Seed PPO 账本指标
        seed_runs = []
        seed_series_list = []
        for seed_val, seed_sig in ppo_seed_schedules.items():
            m_seed, r_seed = _simulate_strategy_on_share_ledger(
                seed_sig, bars_df, eval_cal, actions_df, initial_cash=initial_cash, cost=cost_cfg
            )
            m_seed["seed"] = seed_val
            seed_runs.append(m_seed)
            if cost_name == "normal":
                seed_series_list.append(r_seed)

        ann_rets = [r["annualized_return"] for r in seed_runs]
        sharpes = [r["sharpe_zero_cash_rate"] or 0.0 for r in seed_runs]
        mdds = [r["max_drawdown"] for r in seed_runs]
        navs = [r["final_nav"] for r in seed_runs]
        ledger_reports[cost_name]["rl_ppo_5seed_summary"] = {
            "seeds": list(ppo_seeds),
            "annualized_return_mean": float(np.mean(ann_rets)),
            "annualized_return_std": float(np.std(ann_rets)),
            "sharpe_mean": float(np.mean(sharpes)),
            "sharpe_std": float(np.std(sharpes)),
            "max_drawdown_mean": float(np.mean(mdds)),
            "max_drawdown_std": float(np.std(mdds)),
            "final_nav_mean": float(np.mean(navs)),
            "final_nav_std": float(np.std(navs)),
            "max_reconciliation_error": float(max(r["ledger_reconciliation_error"] for r in seed_runs)),
            "min_cash_across_seeds": float(min(r["min_cash"] for r in seed_runs)),
            "per_seed": seed_runs,
        }
        if cost_name == "normal" and seed_series_list:
            normal_daily_returns["rl_ppo_seed0"] = seed_series_list[0]

    # 5. 区块 Bootstrap 重采样风险对比 (对比 Equal-Weight 基准)
    bootstrap_summary: Dict[str, Any] = {}
    eq_ret = normal_daily_returns["equal_weight"]
    for strat_key in ("wf_stage2_bitemporal_kg_gat", "wf_hrp_cqr_hmm", "rl_offline_iql", "rl_ppo_seed0"):
        cand_ret = normal_daily_returns[strat_key]
        bootstrap_summary[strat_key] = [
            block_resample(cand_ret, eq_ret, block=b, paths=500)
            for b in (20, 40, 60)
            if len(eq_ret) >= b
        ]

    return {
        "protocol": {
            "horizon_sessions": horizon,
            "top_k": k,
            "initial_cash": initial_cash,
            "execution_timing": "signal_at_t_close -> execute_at_t_plus_1_open",
            "label_definition": "open(t+1) to open(t+h+1) holder return with dividends and splits",
            "purge_gap_enforced": True,
            "walk_forward_folds": oos_stream.fold_boundaries,
            "oos_rl_train_sessions": int(len(rl_train_idx)),
            "oos_eval_sessions": int(len(eval_cal)),
            "oos_eval_window": f"{eval_cal[0].date()} — {eval_cal[-1].date()}",
        },
        "oos_alpha_quality": {
            "ic_mean": ic_mean,
            "rank_ic_mean": rank_ic_mean,
            "icir": icir,
        },
        "share_ledger_ablation": ledger_reports,
        "block_bootstrap_vs_equal_weight": bootstrap_summary,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Walk-Forward OOS DL/KG-GAT + Share Ledger RL Ablation")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=_RESEARCH_V1 / "artifacts" / "market-etf-verified-20261108",
        help="Audited Parquet market snapshot directory",
    )
    parser.add_argument("--end", type=str, default="2024-12-31", help="Pre-holdout study end date (YYYY-MM-DD)")
    parser.add_argument("--horizon", type=int, default=5, choices=(5, 20))
    parser.add_argument("--top-k", type=int, default=0, help="Top-K active holdings (0 = auto: 2 for <=6 assets, 5 for >6)")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--out", type=Path, required=True, help="New output directory for report and artifacts")
    args = parser.parse_args(argv)

    if pd.Timestamp(args.end).date() >= pd.Timestamp("2025-01-01").date():
        raise ValueError("Refusing to enter sealed holdout period (>= 2025-01-01) during model selection/ablation")
    args.out.mkdir(parents=True, exist_ok=False)

    manifest, frames = load_audited_snapshot_via_duckdb(args.snapshot)
    cal_df = frames["calendar"]
    cal_dates = pd.to_datetime(cal_df["date"]).dt.normalize()
    cal_dates = cal_dates[cal_dates <= pd.Timestamp(args.end)]

    bars_df = frames["bars"].copy()
    if bars_df["kind"].eq("stock").any() and pd.Timestamp(cal_dates.min()).date() < pd.Timestamp("2015-08-01").date():
        # A股股票过户费与统一交易规费自 2015-08-01 起实施 (对齐 ledger.Costs)
        cal_dates = cal_dates[cal_dates >= pd.Timestamp("2015-08-03")]
        bars_df = bars_df[pd.to_datetime(bars_df["date"]).dt.normalize() >= pd.Timestamp("2015-08-03")].copy()
    actions_df = frames["actions"]

    n_unique_assets = bars_df["symbol"].nunique()
    top_k = args.top_k if args.top_k > 0 else (2 if n_unique_assets <= 6 else 5)

    # 优先读取快照自带的双时点知识图谱边与事件表；否则回退至核心宽基-红利传导边
    if "edges" in frames and not frames["edges"].empty:
        base_edges = frames["edges"].copy()
    else:
        base_edges = pd.DataFrame(
            [
                ("510050", "510300", "large_cap_core_transmission", 0.85),
                ("510300", "510500", "core_to_midcap_spillover", 0.78),
                ("510050", "510880", "value_dividend_rotation", 0.72),
                ("510880", "510050", "dividend_defensive_anchor", 0.70),
                ("510300", "510050", "broad_to_megacap_feedback", 0.82),
            ],
            columns=["src_symbol", "dst_symbol", "relation", "weight"],
        )
    news_df = frames.get("news")
    kg_bundle = build_bitemporal_bundle_from_news_and_edges(base_edges, news_df=news_df)

    report = run_walk_forward_ablation_experiment(
        bars_df=bars_df,
        calendar_dates=cal_dates,
        actions_df=actions_df,
        kg_bundle=kg_bundle,
        base_edges_df=base_edges,
        horizon=args.horizon,
        top_k=top_k,
        initial_cash=args.capital,
        min_train_sessions=1150,
        val_sessions=230,
        oos_step_sessions=230,
        epochs_per_fold=10,
    )
    report["snapshot_manifest_coverage"] = manifest.get("coverage", {})
    out_file = args.out / "walk_forward_ablation_report.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "completed",
        "eval_window": report["protocol"]["oos_eval_window"],
        "oos_alpha_quality": report["oos_alpha_quality"],
        "report_path": str(out_file.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
