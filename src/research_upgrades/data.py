"""Optional AkShare inputs and explicit, point-in-time-aware feature construction."""

from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


BASE_FEATURES = [
    "ret_3d", "ret_5d", "ret_10d", "ret_20d", "sharpe_10d", "sharpe_20d",
    "ma_bias_20", "vol_ratio_5_20",
]
EXTRA_FEATURES = [
    "clv", "clv_available", "flow_super_ratio", "flow_large_ratio", "flow_available",
    "intraday_log_up_down_variance", "intraday_available",
]
ALL_FEATURES = BASE_FEATURES + EXTRA_FEATURES
_FIVE_MINUTE_ENDS = tuple(
    pd.date_range("2000-01-01 09:35", "2000-01-01 11:30", freq="5min").strftime("%H:%M")
) + tuple(pd.date_range("2000-01-01 13:05", "2000-01-01 15:00", freq="5min").strftime("%H:%M"))


def _market_for_symbol(symbol: str) -> str:
    if symbol.startswith(("4", "8", "92")):
        return "bj"
    return "sh" if symbol.startswith(("5", "6", "9")) else "sz"


def _normalize_flow(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    cols = {"日期": "trade_date", "超大单净流入-净占比": "flow_super_ratio", "大单净流入-净占比": "flow_large_ratio"}
    if not set(cols).issubset(raw.columns):
        raise ValueError("AkShare flow response is missing required fields")
    result = raw[list(cols)].rename(columns=cols).copy()
    result["symbol"] = symbol
    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce").dt.normalize()
    for col in ("flow_super_ratio", "flow_large_ratio"):
        result[col] = pd.to_numeric(result[col], errors="coerce") / 100.0
    return result.dropna(subset=["trade_date"])


def _normalize_minutes(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    cols = {"时间": "timestamp", "开盘": "open", "收盘": "close"}
    if not set(cols).issubset(raw.columns):
        raise ValueError("AkShare minute response is missing required fields")
    result = raw[list(cols)].rename(columns=cols).copy()
    result["symbol"] = symbol
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    for col in ("open", "close"):
        result[col] = pd.to_numeric(result[col], errors="coerce")
    return result.dropna(subset=["timestamp"])


def fetch_akshare_inputs(symbols: list[str], timeout: float = 8.0):
    """Fetch snapshots without retries; stop on provider rate limit.

    Returned historical data may have been revised by the provider. The fetch
    timestamp alone does not establish what was known on an earlier trade date.
    """
    import akshare as ak
    from akshare.exceptions import RateLimitError
    from akshare.utils.context import config

    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    symbols = list(dict.fromkeys(str(x) for x in symbols))
    if any(len(s) != 6 or not s.isdigit() for s in symbols):
        raise ValueError("Symbols must be six-digit strings")
    collected_at = datetime.now(ZoneInfo("Asia/Shanghai"))
    completed_cutoff = pd.Timestamp(collected_at.date())
    if collected_at.time() < time(15, 30):
        completed_cutoff -= pd.Timedelta(days=1)
    frames_flow, frames_minute, status = [], [], []
    previous_timeout = config.timeout
    config.timeout = timeout
    rate_limited = False
    try:
        for symbol in symbols:
            market = _market_for_symbol(symbol)
            is_etf = symbol.startswith(("15", "16", "51", "56", "58"))
            for kind in ("flow", "minute"):
                if rate_limited:
                    status.append({"symbol": symbol, "kind": kind, "status": "skipped_rate_limit"})
                    continue
                try:
                    if kind == "flow":
                        raw = ak.stock_individual_fund_flow(stock=symbol, market=market)
                        data = _normalize_flow(raw, symbol)
                        data = data[data["trade_date"] <= completed_cutoff]
                        frames_flow.append(data)
                    else:
                        fn = ak.fund_etf_hist_min_em if is_etf else ak.stock_zh_a_hist_min_em
                        raw = fn(symbol=symbol, period="5", adjust="")
                        data = _normalize_minutes(raw, symbol)
                        data = data[data["timestamp"].dt.normalize() <= completed_cutoff]
                        frames_minute.append(data)
                    status.append({"symbol": symbol, "kind": kind, "status": "ok" if len(data) else "empty", "rows": len(data)})
                except RateLimitError as exc:
                    rate_limited = True
                    status.append({"symbol": symbol, "kind": kind, "status": "rate_limited", "error_type": type(exc).__name__})
                except Exception as exc:
                    # Do not persist exception messages, which may include proxy credentials.
                    status.append({"symbol": symbol, "kind": kind, "status": "failed", "error_type": type(exc).__name__})
    finally:
        config.timeout = previous_timeout
    empty_flow = pd.DataFrame(columns=["symbol", "trade_date", "flow_super_ratio", "flow_large_ratio"])
    empty_minute = pd.DataFrame(columns=["symbol", "timestamp", "open", "close"])
    return (
        pd.concat(frames_flow, ignore_index=True) if frames_flow else empty_flow,
        pd.concat(frames_minute, ignore_index=True) if frames_minute else empty_minute,
        {"source": "AkShare/东方财富", "collected_at": collected_at.isoformat(),
         "after_close_cutoff": str(completed_cutoff.date()), "results": status},
    )


def intraday_variance(minutes: pd.DataFrame) -> pd.DataFrame:
    """Five-minute close-to-close semivariances, excluding the overnight gap.

    Only a complete normal A-share session with 48 distinct ending timestamps is
    accepted. Incomplete sessions, duplicates and invalid prices remain missing.
    """
    columns = ["symbol", "trade_date", "intraday_log_up_down_variance"]
    if minutes.empty:
        return pd.DataFrame(columns=columns)
    df = minutes.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["trade_date"] = df["timestamp"].dt.normalize()
    rows = []
    for (symbol, date), group in df.groupby(["symbol", "trade_date"], sort=True):
        group = group.sort_values("timestamp")
        times = tuple(group["timestamp"].dt.strftime("%H:%M"))
        prices = pd.to_numeric(group["close"], errors="coerce").to_numpy(dtype=float)
        first_open = pd.to_numeric(group["open"], errors="coerce").iloc[0]
        if (times != _FIVE_MINUTE_ENDS or not group["timestamp"].eq(group["timestamp"].dt.floor("min")).all()
                or not np.isfinite(prices).all() or (prices <= 0).any() or not np.isfinite(first_open) or first_open <= 0):
            continue
        returns = np.diff(np.log(np.r_[first_open, prices]))
        up = float(np.square(returns[returns > 0]).sum())
        down = float(np.square(returns[returns < 0]).sum())
        rows.append({"symbol": symbol, "trade_date": date,
                     "intraday_log_up_down_variance": float(np.log((up + 1e-12) / (down + 1e-12)))})
    return pd.DataFrame(rows, columns=columns)


def _standardize_available(frame: pd.DataFrame, column: str, mask: pd.Series) -> pd.Series:
    """Daily cross-section, using only observed values; missing remains zero."""
    output = pd.Series(0.0, index=frame.index)
    for _, indexes in frame.groupby("trade_date").groups.items():
        idx = pd.Index(indexes)
        idx = idx[mask.loc[idx].to_numpy()]
        if len(idx) < 2:
            continue
        values = frame.loc[idx, column].to_numpy(dtype=float)
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        if mad > 1e-12:
            values = np.clip(values, median - 3 * 1.4826 * mad, median + 3 * 1.4826 * mad)
        std = values.std()
        if std > 1e-12:
            output.loc[idx] = (values - values.mean()) / std
    return output


def prepare_feature_panel(bars: pd.DataFrame, flow: pd.DataFrame | None = None,
                          minutes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build daily features without forward filling missing flow or intraday data."""
    required = {"symbol", "trade_date", "open", "high", "low", "close", "volume"}
    if not required.issubset(bars.columns):
        raise ValueError(f"Missing daily bars columns: {sorted(required - set(bars.columns))}")
    df = bars.copy()
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="raise").dt.normalize()
    if df.duplicated(["symbol", "trade_date"]).any():
        raise ValueError("Daily bars contain duplicate symbol/date keys")
    universe = frozenset(df["symbol"])
    if len(universe) < 3 or not df.groupby("trade_date")["symbol"].agg(lambda s: frozenset(s) == universe).all():
        raise ValueError("The research panel needs at least three symbols and a balanced daily universe")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if not np.isfinite(df[["open", "high", "low", "close", "volume"]].to_numpy()).all():
        raise ValueError("Daily bars contain missing or non-finite prices/volume")
    if ((df[["open", "high", "low", "close"]] <= 0).any(axis=1) | (df["volume"] < 0) |
        (df["high"] < df[["open", "close", "low"]].max(axis=1)) |
        (df["low"] > df[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Daily bars contain invalid OHLC or volume values")
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    group = df.groupby("symbol", group_keys=False)
    df["ret_1d"] = group["close"].pct_change(fill_method=None)
    for horizon in (3, 5, 10, 20):
        df[f"ret_{horizon}d"] = group["close"].pct_change(horizon, fill_method=None)
    for horizon in (10, 20):
        vol = group["ret_1d"].transform(lambda s: s.rolling(horizon, min_periods=horizon // 2).std())
        df[f"sharpe_{horizon}d"] = df[f"ret_{horizon}d"] / (vol + 1e-6)
    ma = group["close"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    df["ma_bias_20"] = df["close"] / ma - 1.0
    v5 = group["volume"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    v20 = group["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    df["vol_ratio_5_20"] = v5 / (v20 + 1e-8) - 1.0
    span = df["high"] - df["low"]
    df["clv_available"] = (span > 0).astype(float)
    df["clv"] = np.where(span > 0, (2 * df["close"] - df["high"] - df["low"]) / span.replace(0, np.nan), np.nan)

    if flow is None or flow.empty:
        df["flow_super_ratio"] = np.nan
        df["flow_large_ratio"] = np.nan
    else:
        flow = flow.copy()
        flow["symbol"] = flow["symbol"].astype(str).str.zfill(6)
        flow["trade_date"] = pd.to_datetime(flow["trade_date"]).dt.normalize()
        if flow.duplicated(["symbol", "trade_date"]).any():
            raise ValueError("Flow data has duplicate symbol/date keys")
        for col in ("flow_super_ratio", "flow_large_ratio"):
            flow[col] = pd.to_numeric(flow[col], errors="coerce")
        df = df.merge(flow[["symbol", "trade_date", "flow_super_ratio", "flow_large_ratio"]],
                      on=["symbol", "trade_date"], how="left", validate="one_to_one")
    valid_flow = np.isfinite(df[["flow_super_ratio", "flow_large_ratio"]]).all(axis=1)
    df["flow_available"] = valid_flow.astype(float)
    micro = intraday_variance(minutes) if minutes is not None else intraday_variance(pd.DataFrame())
    if micro.empty:
        df["intraday_log_up_down_variance"] = np.nan
    else:
        micro["symbol"] = micro["symbol"].astype(str).str.zfill(6)
        df = df.merge(micro, on=["symbol", "trade_date"], how="left", validate="one_to_one")
    df["intraday_available"] = np.isfinite(df["intraday_log_up_down_variance"]).astype(float)

    for name in BASE_FEATURES:
        df[name] = _standardize_available(df, name, np.isfinite(df[name]))
    for name, availability in (("clv", "clv_available"), ("flow_super_ratio", "flow_available"),
                               ("flow_large_ratio", "flow_available"),
                               ("intraday_log_up_down_variance", "intraday_available")):
        df[name] = _standardize_available(df, name, df[availability].astype(bool))
    return df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
