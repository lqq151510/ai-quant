"""领域层 Stage 1A：零未来函数时序特征工程与截面去极值标准化管道."""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd


FEATURE_COLUMNS: List[str] = [
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "sharpe_10d",
    "sharpe_20d",
    "ma_bias_20",
    "vol_ratio_5_20",
]

MICRO_FEATURE_COLUMNS: List[str] = [
    "cmf_20d",          # 数据升级1A: 20日主力资金流承接强度 (Chaikin Smart Money Flow = CLV * 相对量比)
    "asym_semi_20d",    # 数据升级1B: 20日上下行已实现半方差不对称性 (Realized Semivariance Asymmetry)
    "smart_flow_1d",    # 数据升级1C: 当日日内尾盘主力资金承接强度 (CLV * 相对量比)
    "overnight_ret_5d", # 数据升级1D: 5日隔夜跳空累计动量 (A股 T+1 集合竞价知情资金方向)
    "intraday_ret_5d",  # 数据升级1E: 5日日内实体累计动量 (A股日内交易追涨/承接力度)
    "idio_mom_20d",     # 数据升级1F: 20日截面市场残差特异动量 (剥离市场Beta后的纯Alpha动量)
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
        micro_feats: 截面主力资金流与日内微结构特征张量, shape = (T, N, F_micro)
        collinearity_before: Löwdin 正交化前的截面特征平均非对角相关系数
        collinearity_after: Löwdin 正交化后的截面特征平均非对角相关系数
    """
    dates: List[pd.Timestamp]
    symbols: List[str]
    x_seq: np.ndarray
    y_alpha: np.ndarray
    y_raw_ret: np.ndarray
    micro_feats: np.ndarray | None = None
    collinearity_before: float = 0.0
    collinearity_after: float = 0.0


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


def lowdin_symmetric_orthogonalize(
    x_mat: np.ndarray,
    shrinkage: float = 0.50,
    blend: float = 0.25,
) -> np.ndarray:
    """算法升级 A：正则化 Löwdin 对称矩阵正交化 (X_orth = (1-beta)*X + beta * X @ S^{-1/2}).

    数学原理：
    1. 在单日截面 X_t (shape: N x F) 上计算收缩协方差矩阵 S_t = (1 - gamma) * (X_t^T X_t / N) + gamma * I_F；
    2. 特征分解 S_t = U @ diag(lam) @ U^T，构造逆平方根算子 S_t^{-1/2} = U @ diag(lam^{-1/2}) @ U^T；
    3. 相比 Gram-Schmidt 正交化，Löwdin 正交化不依赖因子顺序且在 Frobenius 范数下离原始因子最近。
    """
    n_assets, n_feats = x_mat.shape
    cov = (x_mat.T @ x_mat) / max(n_assets, 1)
    shrunk_cov = (1.0 - shrinkage) * cov + shrinkage * np.eye(n_feats, dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(shrunk_cov)
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-4))) @ eigvecs.T
    x_whitened = (x_mat @ inv_sqrt).astype(np.float32)
    x_out = (1.0 - blend) * x_mat + blend * x_whitened
    # 保持零均值单位方差量纲
    mean = x_out.mean(axis=0, keepdims=True)
    std = x_out.std(axis=0, keepdims=True) + 1e-6
    return ((x_out - mean) / std).astype(np.float32)


def _mean_offdiag_corr(feat_cube: np.ndarray) -> float:
    """计算 (T, N, F) 面板在各截面上的平均非对角绝对相关系数."""
    f = feat_cube.shape[-1]
    mask = ~np.eye(f, dtype=bool)
    vals = []
    for t in range(feat_cube.shape[0]):
        c = np.corrcoef(feat_cube[t], rowvar=False)
        if not np.isnan(c).any():
            vals.append(float(np.mean(np.abs(c[mask]))))
    return float(np.mean(vals)) if vals else 0.0


def compute_raw_feature_columns(bars_df: pd.DataFrame) -> pd.DataFrame:
    """按单标的严格沿历史时间轴计算因果除权量价、隔夜/日内微结构与截面残差特异动量特征 (零未来函数)."""
    df = bars_df.sort_values(["symbol", "trade_date"]).copy()
    g = df.groupby("symbol", group_keys=False)

    # 若含交易所 preclose，使用 close / preclose 构造 100% 因果链式真实收益与复权价格序列，消除分红/份额折算伪跳变
    if "preclose" in df.columns and (df["preclose"] > 0).all():
        df["ret_1d_raw"] = (df["close"] / df["preclose"]) - 1.0
        # 严格仅累乘历史日真实收益率 (零未来函数)
        df["_causal_adj_price"] = g["ret_1d_raw"].transform(lambda s: (1.0 + s.fillna(0.0)).cumprod())
        price_col = "_causal_adj_price"
        df["_overnight_1d"] = (df["open"] / df["preclose"]) - 1.0
    else:
        df["ret_1d_raw"] = g["close"].pct_change(1)
        price_col = "close"
        df["_overnight_1d"] = (df["open"] / g["close"].shift(1)) - 1.0

    df["ret_3d"] = g[price_col].pct_change(3)
    df["ret_5d"] = g[price_col].pct_change(5)
    df["ret_10d"] = g[price_col].pct_change(10)
    df["ret_20d"] = g[price_col].pct_change(20)

    vol_10 = g["ret_1d_raw"].transform(lambda s: s.rolling(10, min_periods=5).std()) + 1e-6
    vol_20 = g["ret_1d_raw"].transform(lambda s: s.rolling(20, min_periods=10).std()) + 1e-6
    df["sharpe_10d"] = df["ret_10d"] / vol_10
    df["sharpe_20d"] = df["ret_20d"] / vol_20

    ma_20 = g[price_col].transform(lambda s: s.rolling(20, min_periods=10).mean())
    df["ma_bias_20"] = (df[price_col] / (ma_20 + 1e-8)) - 1.0

    vol_ma5 = g["volume"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    vol_ma20 = g["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    df["vol_ratio_5_20"] = (vol_ma5 / (vol_ma20 + 1e-8)) - 1.0

    # 数据升级 1A：Chaikin 主力资金流承接强度 (CLV 收盘位置分位 * 相对量比 20日均值)
    hl_span = (df["high"] - df["low"]).replace(0.0, np.nan)
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (hl_span + 1e-8)
    clv = clv.fillna(0.0).clip(-1.0, 1.0)
    rel_vol_20 = df["volume"] / (vol_ma20 + 1.0)
    df["smart_flow_1d"] = clv * rel_vol_20
    df["cmf_20d"] = df.groupby("symbol")["smart_flow_1d"].transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )

    # 数据升级 1B：上下行已实现半方差不对称性 (Realized Semivariance Asymmetry 20d)
    df["up_ret_sq"] = np.maximum(df["ret_1d_raw"].fillna(0.0), 0.0) ** 2
    df["dn_ret_sq"] = np.minimum(df["ret_1d_raw"].fillna(0.0), 0.0) ** 2
    up_var20 = df.groupby("symbol")["up_ret_sq"].transform(lambda s: s.rolling(20, min_periods=10).sum())
    dn_var20 = df.groupby("symbol")["dn_ret_sq"].transform(lambda s: s.rolling(20, min_periods=10).sum())
    df["asym_semi_20d"] = (np.sqrt(up_var20) - np.sqrt(dn_var20)) / (np.sqrt(up_var20 + dn_var20) + 1e-8)

    # 数据升级 1D & 1E：A 股 T+1 隔夜跳空与日内实体动量分离 (5日滚动累计)
    df["_intraday_1d"] = (df["close"] / (df["open"] + 1e-8)) - 1.0
    df["overnight_ret_5d"] = df.groupby("symbol")["_overnight_1d"].transform(
        lambda s: s.fillna(0.0).rolling(5, min_periods=3).sum()
    )
    df["intraday_ret_5d"] = df.groupby("symbol")["_intraday_1d"].transform(
        lambda s: s.fillna(0.0).rolling(5, min_periods=3).sum()
    )

    # 数据升级 1F：20日截面市场残差特异动量 (剥离当日截面等权市场均值后的波动率标准化残差)
    mkt_ret_20d = df.groupby("trade_date")["ret_20d"].transform("mean")
    df["idio_mom_20d"] = (df["ret_20d"] - mkt_ret_20d) / vol_20
    return df


def build_spatio_temporal_dataset(
    bars_df: pd.DataFrame,
    seq_len: int = 20,
    pred_horizon: int = 3,
) -> SpatioTemporalDataset:
    """从日线表构建零未来函数泄漏的时空面板张量."""
    df = compute_raw_feature_columns(bars_df)
    symbols = sorted(df["symbol"].unique().tolist())
    g = df.groupby("symbol", group_keys=False)

    # 2. 计算预测目标：未来 pred_horizon 日累计收益率 & 次日真实单日收益率 (用于回测结算)
    df["future_ret_h"] = g["close"].shift(-pred_horizon) / df["close"] - 1.0
    df["next_1d_ret"] = g["close"].shift(-1) / df["close"] - 1.0

    # 剔除前 20 天滚动窗口预热期，防止预热期 NaN 污染截面标准化
    all_cols = FEATURE_COLUMNS + MICRO_FEATURE_COLUMNS
    df = df.dropna(subset=all_cols).sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    for col in all_cols:
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
    n_micro = len(MICRO_FEATURE_COLUMNS)

    feat_cube = df_valid[FEATURE_COLUMNS].to_numpy(dtype=np.float32).reshape(n_dates, n_assets, n_feats)
    micro_cube = df_valid[MICRO_FEATURE_COLUMNS].to_numpy(dtype=np.float32).reshape(n_dates, n_assets, n_micro)

    # 计算训练集面板上的时不变 Löwdin 对称正交化去共线性指标 (避免单日 N=13 小样本随机旋转破坏时序连续性)
    t_tr_cut = int(n_dates * 0.65)
    tr_flat = np.concatenate([feat_cube[:t_tr_cut], micro_cube[:t_tr_cut]], axis=-1).reshape(-1, n_feats + n_micro)
    tr_orth = lowdin_symmetric_orthogonalize(tr_flat, shrinkage=0.25, blend=0.85)
    mask_off = ~np.eye(n_feats + n_micro, dtype=bool)
    col_before = float(np.mean(np.abs(np.corrcoef(tr_flat, rowvar=False)[mask_off])))
    col_after = float(np.mean(np.abs(np.corrcoef(tr_orth, rowvar=False)[mask_off])))

    alpha_mat = df_valid["target_alpha"].to_numpy(dtype=np.float32).reshape(n_dates, n_assets)
    raw_ret_mat = df_valid["next_1d_ret"].to_numpy(dtype=np.float32).reshape(n_dates, n_assets)

    # 组装长度为 seq_len 的滑动历史窗口: (T_samples, N, L, F)
    x_list, micro_list, y_alpha_list, y_raw_list, sample_dates = [], [], [], [], []
    for t in range(seq_len - 1, n_dates):
        # 窗口 [t - seq_len + 1 : t + 1] -> 转置为 (N, L, F)
        win = np.transpose(feat_cube[t - seq_len + 1: t + 1], (1, 0, 2))
        x_list.append(win)
        micro_list.append(micro_cube[t])
        y_alpha_list.append(alpha_mat[t])
        y_raw_list.append(raw_ret_mat[t])
        sample_dates.append(valid_dates[t])

    return SpatioTemporalDataset(
        dates=sample_dates,
        symbols=symbols,
        x_seq=np.stack(x_list, axis=0),
        y_alpha=np.stack(y_alpha_list, axis=0),
        y_raw_ret=np.stack(y_raw_list, axis=0),
        micro_feats=np.stack(micro_list, axis=0),
        collinearity_before=col_before,
        collinearity_after=col_after,
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
            micro_feats=dataset.micro_feats[start:end] if dataset.micro_feats is not None else None,
            collinearity_before=dataset.collinearity_before,
            collinearity_after=dataset.collinearity_after,
        )

    train_ds = _slice(0, t_train)
    val_ds = _slice(t_train + purge_gap, t_val)
    test_ds = _slice(t_val + purge_gap, total)
    return train_ds, val_ds, test_ds
