"""领域层 Stage 1A：零未来函数时序特征工程与截面去极值标准化管道."""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd


FEATURE_COLUMNS: List[str] = [
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "volatility_10d",
    "vol_ratio_5_20",
    "intraday_pos",
]


@dataclass
class SpatioTemporalDataset:
    """按交易日截面组织的时空张量数据集.

    Attributes:
        dates: 有效样本交易日列表, 长度 T
        symbols: 资产代码列表, 长度 N (固定顺序)
        x_seq: 时序特征张量, shape = (T, N, L_window, F_features)
        y_alpha: 未来 H 日截面标准化超额收益标签, shape = (T, N)
        y_raw_ret: 未来 1 日真实单日简单收益率 (用于回测与 RL 结算), shape = (T, N)
    """
    dates: List[pd.Timestamp]
    symbols: List[str]
    x_seq: np.ndarray
    y_alpha: np.ndarray
    y_raw_ret: np.ndarray


def cross_sectional_mad_zscore(values: np.ndarray, clip_n: float = 3.0) -> np.ndarray:
    """在单日截面上执行 MAD 去极值 + Z-Score 标准化 (严格仅使用当日截面数据，零未来泄漏)."""
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    scaled_mad = 1.4826 * max(float(mad), 1e-8)
    clipped = np.clip(values, median - clip_n * scaled_mad, median + clip_n * scaled_mad)
    mean = np.nanmean(clipped)
    std = np.nanstd(clipped)
    if std < 1e-8:
        return np.zeros_like(values)
    return (clipped - mean) / std


def build_spatio_temporal_dataset(
    bars_df: pd.DataFrame,
    seq_len: int = 20,
    pred_horizon: int = 3,
) -> SpatioTemporalDataset:
    """从日线表构建零未来函数泄漏的时空面板张量.

    严格保证：
    1. 任何时刻 t 的特征仅由 [t - 20, t] 历史窗口计算，缺失值仅做历史方向前向填充或置 0。
    2. 截面归一化仅在当日 t 的 N 只标的内部计算，绝不跨日期使用全局均值方差。
    """
    df = bars_df.sort_values(["symbol", "trade_date"]).copy()
    symbols = sorted(df["symbol"].unique().tolist())

    # 1. 按单标的严格沿历史时间轴计算原始量价特征
    g = df.groupby("symbol", group_keys=False)
    df["ret_1d"] = g["close"].pct_change(1)
    df["ret_3d"] = g["close"].pct_change(3)
    df["ret_5d"] = g["close"].pct_change(5)
    df["ret_10d"] = g["close"].pct_change(10)
    df["ret_20d"] = g["close"].pct_change(20)

    df["volatility_10d"] = g["ret_1d"].transform(
        lambda s: s.rolling(10, min_periods=5).std()
    )
    vol_ma5 = g["volume"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    vol_ma20 = g["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    df["vol_ratio_5_20"] = (vol_ma5 / (vol_ma20 + 1e-8)) - 1.0

    # 日内收盘位置：(Close - Low) / (High - Low) - 0.5
    hl_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["intraday_pos"] = ((df["close"] - df["low"]) / (hl_range + 1e-8)) - 0.5

    # 2. 计算预测目标：未来 pred_horizon 日累计收益率 & 次日真实单日收益率 (用于回测结算)
    df["future_ret_h"] = g["close"].shift(-pred_horizon) / df["close"] - 1.0
    df["next_1d_ret"] = g["close"].shift(-1) / df["close"] - 1.0

    # 3. 按每个交易日截面独立执行 MAD 去极值 + Z-Score
    df = df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    for col in FEATURE_COLUMNS:
        df[col] = df.groupby("trade_date")[col].transform(
            lambda s: pd.Series(
                cross_sectional_mad_zscore(s.fillna(0.0).to_numpy()),
                index=s.index,
            )
        )

    # 截面超额收益标签 Z-Score
    df["target_alpha"] = df.groupby("trade_date")["future_ret_h"].transform(
        lambda s: pd.Series(
            cross_sectional_mad_zscore(s.to_numpy()) if s.notna().all() else np.nan,
            index=s.index,
        )
    )

    # 过滤掉前 20 天预热期与末尾无标签日期
    valid_dates = (
        df.dropna(subset=["target_alpha", "next_1d_ret"])
        .groupby("trade_date")["symbol"]
        .count()
        .loc[lambda c: c == len(symbols)]
        .index.sort_values()
        .tolist()
    )
    df_valid = df[df["trade_date"].isin(valid_dates)].sort_values(["trade_date", "symbol"])

    n_dates = len(valid_dates)
    n_assets = len(symbols)
    n_feats = len(FEATURE_COLUMNS)

    feat_cube = df_valid[FEATURE_COLUMNS].to_numpy(dtype=np.float32).reshape(n_dates, n_assets, n_feats)
    alpha_mat = df_valid["target_alpha"].to_numpy(dtype=np.float32).reshape(n_dates, n_assets)
    raw_ret_mat = df_valid["next_1d_ret"].to_numpy(dtype=np.float32).reshape(n_dates, n_assets)

    # 组装长度为 seq_len 的滑动历史窗口: (T_samples, N, L, F)
    x_list, y_alpha_list, y_raw_list, sample_dates = [], [], [], []
    for t in range(seq_len - 1, n_dates):
        # 窗口 [t - seq_len + 1 : t + 1] -> 转置为 (N, L, F)
        win = np.transpose(feat_cube[t - seq_len + 1: t + 1], (1, 0, 2))
        x_list.append(win)
        y_alpha_list.append(alpha_mat[t])
        y_raw_list.append(raw_ret_mat[t])
        sample_dates.append(valid_dates[t])

    return SpatioTemporalDataset(
        dates=sample_dates,
        symbols=symbols,
        x_seq=np.stack(x_list, axis=0),
        y_alpha=np.stack(y_alpha_list, axis=0),
        y_raw_ret=np.stack(y_raw_list, axis=0),
    )


def chronological_split(
    dataset: SpatioTemporalDataset,
    train_ratio: float = 0.65,
    val_ratio: float = 0.15,
    purge_gap: int = 5,
) -> Tuple[SpatioTemporalDataset, SpatioTemporalDataset, SpatioTemporalDataset]:
    """严格按时间顺序切分训练集、验证集、测试集，并在边界加入 Purge Gap 防止预测窗口重叠泄漏."""
    total = len(dataset.dates)
    t_train = int(total * train_ratio)
    t_val = int(total * (train_ratio + val_ratio))

    def _slice(start: int, end: int) -> SpatioTemporalDataset:
        return SpatioTemporalDataset(
            dates=dataset.dates[start:end],
            symbols=dataset.symbols,
            x_seq=dataset.x_seq[start:end],
            y_alpha=dataset.y_alpha[start:end],
            y_raw_ret=dataset.y_raw_ret[start:end],
        )

    train_ds = _slice(0, t_train)
    val_ds = _slice(t_train + purge_gap, t_val)
    test_ds = _slice(t_val + purge_gap, total)
    return train_ds, val_ds, test_ds
