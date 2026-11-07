"""Deterministic engineering fixture, deliberately isolated from market ingestion."""

import numpy as np
import pandas as pd

from .protocol import ETF_SYMBOLS


def fixture() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(47)
    dates = pd.bdate_range("2024-01-02", periods=185).difference(pd.DatetimeIndex(["2024-04-04", "2024-04-05"]))
    stock_symbols = tuple(str(s) for s in range(600000, 600012))
    rows = []
    for i, symbol in enumerate(ETF_SYMBOLS + stock_symbols):
        close = (2.5 + i / 2) * np.cumprod(1 + rng.normal(0.00035, 0.009, size=len(dates)))
        if symbol == "510300":
            close[90:] -= 0.1
        previous = np.r_[close[0], close[:-1]]
        if symbol == "510300":
            previous[90] -= 0.1
        open_ = previous * (1 + rng.normal(0, 0.0025, len(dates)))
        volume = rng.integers(100_000, 2_000_000, size=len(dates))
        rows.append(pd.DataFrame({
            "date": dates, "symbol": symbol, "kind": "etf" if symbol in ETF_SYMBOLS else "stock",
            "open": open_, "high": np.maximum(open_, close) * 1.002,
            "low": np.minimum(open_, close) * 0.998, "close": close,
            "volume": volume, "amount": close * volume, "tradable": True,
            "upper_limit": previous * 1.1, "lower_limit": previous * 0.9,
        }))
    return {
        "calendar": pd.DataFrame({"date": dates}), "bars": pd.concat(rows, ignore_index=True),
        "actions": pd.DataFrame([{
            "symbol": "510300", "record_date": dates[89], "ex_date": dates[90],
            "pay_date": dates[94], "cash_per_share": 0.1, "share_multiplier": 1.0,
        }]),
        "membership": pd.DataFrame({"symbol": stock_symbols, "effective_date": dates[0],
                                    "known_at": pd.Timestamp("2023-12-29T15:00:00+08:00")}),
    }
