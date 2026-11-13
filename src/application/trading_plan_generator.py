"""旧版原型的研究报告出口。"""

from pathlib import Path
from tempfile import NamedTemporaryFile
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
)
from src.domain.regime.market_hmm import MarketRegimeSnapshot
from src.infrastructure.market_data import CORE_UNIVERSE, MARKET_DATA_SOURCE_ORIGINS


DEFAULT_REPORT = "data/latest_trading_plan.md"


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
    """从输入行情的最新窗口计算原型截面分数."""
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


def write_prototype_research_report(
    bars_df: pd.DataFrame | None = None,
    output_path: str = DEFAULT_REPORT,
) -> str:
    """Publish coverage and provenance only for the unaccepted legacy engine."""
    lines = [
        "# A 股和 ETF 原型研究状态",
        "",
        "**发布状态：研究专用（未通过验收）**",
        "",
        "旧版引擎仅发布数据来源与日期覆盖。验收依据见仓库内 research_v1/PROTOTYPE_REVIEW.md。",
        "",
        "来源标签：real 为供应商历史行情，synthetic 为合成数据，unknown 为无可核来源。"
        "本链路缺少历史获知时间及修订记录，无法验证 PIT（按当时可得的信息重放）。",
        "旧版新闻没有可核的首次采集时间及归属证据，研究状态页不引用该批新闻。",
        "",
    ]
    if bars_df is None or not {"symbol", "trade_date"}.issubset(bars_df.columns) or bars_df.empty:
        lines.append("当前运行没有可列示的行情覆盖记录。")
    else:
        coverage = bars_df[["symbol", "trade_date"]].copy()
        source = bars_df.get("data_source", pd.Series(index=bars_df.index, dtype="object"))
        origin = bars_df.get("data_origin", pd.Series(index=bars_df.index, dtype="object"))
        verified = source.map(MARKET_DATA_SOURCE_ORIGINS).eq(origin).fillna(False)
        coverage["source"] = source.where(verified, "unknown")
        coverage["origin"] = coverage["source"].map(MARKET_DATA_SOURCE_ORIGINS).fillna("unknown")
        coverage["trade_date"] = pd.to_datetime(coverage["trade_date"], errors="coerce")
        coverage["symbol"] = coverage["symbol"].astype("string")
        valid_symbol = coverage["symbol"].str.fullmatch(r"[0-9]{6}(?:[.](?:SH|SZ))?", na=False)
        invalid_count = int((~valid_symbol).sum())
        coverage = coverage.loc[valid_symbol]
        configured = {asset.symbol for asset in CORE_UNIVERSE}
        present = set(coverage["symbol"].str[:6])
        lines += [
            f"配置标的 {len(configured)} 只，输入含 {len(present)} 只。以下为输入行的起止日期及条数。",
            "",
            "| 标的 | 数据标签 | 来源 | 最早日期 | 最晚日期 | 行数 |",
            "|---|---|---|---|---|---:|",
        ]
        for (symbol, label, provider), group in coverage.groupby(["symbol", "origin", "source"], sort=True):
            dates = group["trade_date"].dropna()
            first = dates.min().strftime("%Y-%m-%d") if not dates.empty else "未知"
            last = dates.max().strftime("%Y-%m-%d") if not dates.empty else "未知"
            lines.append(f"| {symbol} | {label} | {provider} | {first} | {last} | {len(group)} |")
        if missing := configured - present:
            lines += ["", f"配置中缺少行情的标的：{'、'.join(sorted(missing))}。"]
        if invalid_count:
            lines += ["", f"有 {invalid_count} 行的标的代码无法识别，未列入上表。"]

    report = "\n".join(lines) + "\n"
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with NamedTemporaryFile(dir=output.parent, mode="w", encoding="utf-8", delete=False) as temp:
            temp_path = Path(temp.name)
            temp.write(report)
        temp_path.replace(output)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return report


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
    output_path: str = DEFAULT_REPORT,
) -> str:
    """旧接口保留参数兼容，统一经过原型的研究报告出口。"""
    return write_prototype_research_report(bars_df, output_path)
