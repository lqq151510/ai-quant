"""应用编排层 Stage 4：结合 KG-GAT 实时推理、PPO 仓位决策与 ATR 硬风控的半自动实盘交易计划生成器."""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.application.trainer import get_device
from src.domain.features.pipeline import (
    FEATURE_COLUMNS,
    compute_raw_feature_columns,
    cross_sectional_mad_zscore,
    lowdin_symmetric_orthogonalize,
)
from src.domain.portfolio.hrp import HierarchicalRiskParityOptimizer
from src.domain.regime.market_hmm import MarketRegimeSnapshot
from src.domain.rl_portfolio.ppo_agent import build_alpha_risk_prior
from src.infrastructure.market_data import CORE_UNIVERSE


def compute_latest_atr_and_snapshot(bars_df: pd.DataFrame) -> pd.DataFrame:
    """计算每只标的截至最新交易日的收盘价、14日 ATR、20日均线偏离度及主力资金流指标."""
    df = compute_raw_feature_columns(bars_df)
    g = df.groupby("symbol", group_keys=False)

    prev_close = g["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    df["tr"] = np.maximum(tr1, np.maximum(tr2, tr3))
    df["atr_14"] = g["tr"].transform(lambda s: s.rolling(14, min_periods=5).mean())
    df["ma_20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=10).mean())

    latest_date = df["trade_date"].max()
    snap = df[df["trade_date"] == latest_date].sort_values("symbol").reset_index(drop=True)
    return snap


def infer_latest_live_alpha(
    model: nn.Module,
    bars_df: pd.DataFrame,
    seq_len: int = 20,
) -> tuple[pd.Timestamp, List[str], np.ndarray, np.ndarray]:
    """使用截至最新交易日 (无未来标签) 的最近 20 个交易日真实行情，推理次日截面 Alpha 分数."""
    df = compute_raw_feature_columns(bars_df)
    symbols = sorted(df["symbol"].unique().tolist())

    df = df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    for col in FEATURE_COLUMNS:
        df[col] = df.groupby("trade_date")[col].transform(
            lambda s: pd.Series(
                cross_sectional_mad_zscore(s.fillna(0.0).to_numpy()),
                index=s.index,
            )
        )

    latest_dates = sorted(df["trade_date"].unique().tolist())[-seq_len:]
    sub = df[df["trade_date"].isin(latest_dates)].sort_values(["trade_date", "symbol"])
    n_assets = len(symbols)
    n_feats = len(FEATURE_COLUMNS)

    feat_cube = sub[FEATURE_COLUMNS].to_numpy(dtype=np.float32).reshape(seq_len, n_assets, n_feats)
    x_latest = np.transpose(feat_cube, (1, 0, 2))[None, ...]  # (1, N, L, F)
    latest_vol_z = feat_cube[-1, :, 5]

    device = get_device()
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x_latest, dtype=torch.float32, device=device)
        live_alpha = model(x_t).cpu().numpy().squeeze(0)

    return pd.Timestamp(latest_dates[-1]), symbols, live_alpha, latest_vol_z


def generate_semi_auto_trading_plan(
    bars_df: pd.DataFrame,
    kg_model: nn.Module,
    asset_sentiments: Dict[str, float],
    news_df: pd.DataFrame,
    avg_rl_cash_weight: float,
    latest_regime: Optional[MarketRegimeSnapshot] = None,
    latest_q10: Optional[np.ndarray] = None,
    latest_q90: Optional[np.ndarray] = None,
    hist_raw_rets: Optional[np.ndarray] = None,
    output_path: str = "data/latest_trading_plan.md",
) -> str:
    """生成《A股/ETF 次日半自动实盘交易计划单》并持久化为 Markdown."""
    latest_dt, symbols, live_alpha, latest_vol_z = infer_latest_live_alpha(kg_model, bars_df)
    snap_df = compute_latest_atr_and_snapshot(bars_df)

    # 结合 KG-GAT Alpha 分数与东方财富实时新闻催化情绪合成综合排序分
    n_assets = len(symbols)
    sent_vec = np.array([asset_sentiments.get(s, 0.0) for s in symbols], dtype=np.float32)

    # 提取最新截面真实量价状态，执行三道实盘保命铁闸：
    # 1) 跨市场纳指冲击传染检测 (513100 跳空大跌时压制海外算力映射链)
    # 2) 量价背离反软文证伪 (大跌日剥离正面软文诱多分数)
    # 3) 断头铡刀破位熔断 (跌破 MA20 且大跌或外盘停牌溢价标的强制一票否决)
    nasdaq_row = snap_df[snap_df["symbol"] == "513100"]
    nasdaq_ret = float(nasdaq_row["ret_1d_raw"].iloc[0]) if not nasdaq_row.empty else 0.0
    global_tech_shock = nasdaq_ret < -0.025  # 纳指ETF单日大跌超 2.5% 触发海外科技传染预警

    circuit_breaker_flags: Dict[str, str] = {}
    adjusted_alpha = live_alpha.copy()
    adjusted_sent = sent_vec.copy()

    for i, sym in enumerate(symbols):
        row = snap_df[snap_df["symbol"] == sym].iloc[0]
        close_p = float(row["close"])
        ma20 = float(row["ma_20"]) if pd.notna(row["ma_20"]) else close_p
        atr = float(row["atr_14"]) if pd.notna(row["atr_14"]) else close_p * 0.02
        ret_1d = float(row["ret_1d_raw"])
        below_ma20 = close_p < ma20

        # 铁闸 1: 量价证伪 —— 单日跌幅超 2.5% 时，任何正面新闻视为逆势软文噪音并反转为负向警示
        if ret_1d < -0.025 and adjusted_sent[i] > 0:
            adjusted_sent[i] = -abs(adjusted_sent[i]) - 0.15

        # 铁闸 2: 跨市场海外算力链冲击传染
        if global_tech_shock and sym in ("300308", "515880", "513100", "512480"):
            adjusted_alpha[i] -= 0.35
            circuit_breaker_flags[sym] = "🌍 纳指/海外算力破位传染"

        # 铁闸 3: 断头铡刀破位熔断 (单日跌幅 > 4.0% 或跌破 MA20 且跌幅 > 1.2 * ATR)
        atr_pct = atr / max(close_p, 1e-8)
        if ret_1d <= -0.040 or (below_ma20 and ret_1d <= -1.2 * atr_pct):
            adjusted_alpha[i] = -0.60 + ret_1d
            circuit_breaker_flags[sym] = "🚨 断头铡刀破位熔断"
        elif sym == "513100" and (below_ma20 or adjusted_sent[i] < -0.20):
            adjusted_alpha[i] = -0.50
            circuit_breaker_flags[sym] = "⚠️ 跨境高溢价停牌/外盘破位回避"
        elif not below_ma20 and ret_1d > -0.005:
            # 站稳 MA20 多头趋势且在外盘暴跌日抗跌/逆势红盘的标的获得确定性溢价
            adjusted_alpha[i] += 0.18

    composite_alpha = adjusted_alpha + 0.25 * adjusted_sent
    if latest_q10 is not None and latest_q90 is not None:
        # 融合分位数悲观下界保护 (截面零均值中心化): 惩罚不确定性过宽标的
        width = np.maximum(latest_q90 - latest_q10, 1e-4)
        u_adj = 0.15 * latest_q10 - 0.08 * width
        composite_alpha = composite_alpha + (u_adj - float(np.mean(u_adj)))

    # 筛选未触发熔断且站稳 MA20 的多头候选池，调用 HRP 层次风险平价分配跨板块权重
    eligible_indices = []
    for i, sym in enumerate(symbols):
        row = snap_df[snap_df["symbol"] == sym].iloc[0]
        if sym not in circuit_breaker_flags and float(row["close"]) >= float(row["ma_20"]):
            eligible_indices.append(i)

    # 在合格候选中取 Alpha 前 3 名执行 HRP 层次风险平价
    eligible_sorted = sorted(eligible_indices, key=lambda idx: composite_alpha[idx], reverse=True)[:3]
    hmm_cap = latest_regime.recommended_max_exposure if latest_regime is not None else 0.75
    if global_tech_shock:
        hmm_cap = min(hmm_cap, 0.70)

    if eligible_sorted and hist_raw_rets is not None and len(hist_raw_rets) >= 20:
        hrp_unit_w = HierarchicalRiskParityOptimizer.allocate_hrp_weights(
            hist_returns=hist_raw_rets[-45:],
            candidate_indices=eligible_sorted,
            alpha_tilts=composite_alpha,
        )
        asset_weights = hrp_unit_w * hmm_cap
        cash_reserve = 1.0 - float(np.sum(asset_weights))
    else:
        shock_dd_proxy = 0.045 if global_tech_shock else 0.015
        dummy_obs = np.concatenate([
            composite_alpha,
            latest_vol_z,
            np.zeros(n_assets, dtype=np.float32),
            np.array([avg_rl_cash_weight, shock_dd_proxy], dtype=np.float32),
        ])
        alloc_logits = build_alpha_risk_prior(dummy_obs, n_assets, top_k=3).squeeze(0).numpy()
        for i, sym in enumerate(symbols):
            row = snap_df[snap_df["symbol"] == sym].iloc[0]
            if sym in circuit_breaker_flags or float(row["close"]) < float(row["ma_20"]):
                alloc_logits[i] = -1e4
        exp_a = np.exp(alloc_logits - np.max(alloc_logits))
        alloc_weights = exp_a / np.sum(exp_a)
        asset_weights = alloc_weights[:n_assets]
        cash_reserve = float(alloc_weights[-1])

    # 整理最新新闻摘要映射
    latest_news_map: Dict[str, str] = {}
    if not news_df.empty:
        for sym, grp in news_df.groupby("symbol"):
            latest_news_map[str(sym)] = str(grp.iloc[0]["title"])[:36]

    meta_map = {a.symbol: a for a in CORE_UNIVERSE}
    plan_rows = []
    for i, sym in enumerate(symbols):
        row = snap_df[snap_df["symbol"] == sym].iloc[0]
        close_p = float(row["close"])
        atr = float(row["atr_14"]) if pd.notna(row["atr_14"]) else close_p * 0.02
        ma20 = float(row["ma_20"]) if pd.notna(row["ma_20"]) else close_p
        ret_1d = float(row["ret_1d_raw"]) * 100.0
        w = float(asset_weights[i])

        # 计算触发买入区间 (回踩 0.3*ATR ~ 突破 0.2*ATR)、硬止损位 (-2.0*ATR)、止盈目标 (+3.2*ATR)
        buy_low = close_p - 0.35 * atr
        buy_high = close_p + 0.20 * atr
        stop_loss = close_p - 2.0 * atr
        take_profit = close_p + 3.2 * atr
        trend_tag = "站上MA20(多头)" if close_p >= ma20 else "MA20下方(震荡)"

        if sym in circuit_breaker_flags:
            action = circuit_breaker_flags[sym]
        elif w >= 0.05:
            action = "🔥 主力建仓/持有"
        elif close_p >= ma20:
            action = "👀 观察/回踩低吸"
        else:
            action = "🛡️ 破位/弱势观望"

        smart_flow = float(row["smart_flow_1d"]) if "smart_flow_1d" in row and pd.notna(row["smart_flow_1d"]) else 0.0
        q_interval_str = "--"
        if latest_q10 is not None and latest_q90 is not None:
            q_interval_str = f"[{float(latest_q10[i]):+.2f}, {float(latest_q90[i]):+.2f}]"

        plan_rows.append({
            "symbol": sym,
            "name": meta_map[sym].name,
            "sector": meta_map[sym].sector,
            "close": close_p,
            "ret_1d": ret_1d,
            "smart_flow": smart_flow,
            "alpha_score": float(composite_alpha[i]),
            "q_interval": q_interval_str,
            "sent_score": float(adjusted_sent[i]),
            "target_weight": w,
            "action": action,
            "buy_zone": f"{buy_low:.3f} ~ {buy_high:.3f}",
            "stop_loss": f"{stop_loss:.3f} (-{(close_p - stop_loss) / close_p * 100:.1f}%)",
            "take_profit": f"{take_profit:.3f} (+{(take_profit - close_p) / close_p * 100:.1f}%)",
            "trend": trend_tag,
            "catalyst": latest_news_map.get(sym, "产业链图谱动量协同"),
        })

    plan_df = pd.DataFrame(plan_rows).sort_values("alpha_score", ascending=False).reset_index(drop=True)

    regime_desc = "未启用"
    if latest_regime is not None:
        p_b, p_o, p_c = latest_regime.state_probs
        regime_desc = (
            f"**{latest_regime.state_name}** "
            f"(后验概率: 多头={p_b*100:.1f}%, 震荡={p_o*100:.1f}%, 恐慌={p_c*100:.1f}% | "
            f"1D-Wasserstein距离 W2={latest_regime.wasserstein_w2:.4f}, 左尾偏移={latest_regime.ot_left_tail_shift:+.3f} | "
            f"建议权益上限={hmm_cap*100:.0f}%)"
        )

    lines = [
        f"# 📋 A股/ETF 半自动实盘辅助交易计划单 (基准收盘日: {latest_dt.strftime('%Y-%m-%d')})",
        "",
        f"- **决策引擎**: `Löwdin对称正交化` + `Stage 2B (GBDT + KG-GAT)` + `Conformal 分位数区间校准` + `Sticky-HMM + Wasserstein最优传输` + `HRP + Offline IQL`",
        f"- **Sticky-HMM + Wasserstein 市场状态识别**: {regime_desc}",
        f"- **HRP+HMM 建议账户总仓位**: **{(1.0 - cash_reserve) * 100:.1f}%** | **建议现金防守仓位**: **{cash_reserve * 100:.1f}%**",
        f"- **A股 T+1 执行铁律**: 严禁盘中追高超过买入区间上限；跌破 2×ATR 硬止损红线次日无条件纪律止损。",
        "",
        "## 一、 截面 Alpha 排序与次日操作指令表",
        "",
        "| 排名 | 代码 | 名称 | 板块 | 最新收盘 | 当日涨跌 | 主力流CMF | 综合Alpha | 80%共形区间[q10,q90] | 新闻情绪 | HRP+IQL仓位 | 操作指令 | 建议买入区间 | 2×ATR硬止损位 | 3.2×ATR止盈目标 | 趋势状态 | 最新东方财富资讯催化 |",
        "| :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |",
    ]

    for idx, r in plan_df.iterrows():
        lines.append(
            f"| {idx + 1} | `{r['symbol']}` | **{r['name']}** | {r['sector']} | "
            f"{r['close']:.3f} | {r['ret_1d']:+.2f}% | `{r['smart_flow']:+.2f}` | `{r['alpha_score']:+.3f}` | `{r['q_interval']}` | "
            f"`{r['sent_score']:+.2f}` | **{r['target_weight'] * 100:.1f}%** | {r['action']} | "
            f"`{r['buy_zone']}` | `{r['stop_loss']}` | `{r['take_profit']}` | {r['trend']} | {r['catalyst']} |"
        )

    report_md = "\n".join(lines) + "\n"
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(report_md, encoding="utf-8")
    return report_md
