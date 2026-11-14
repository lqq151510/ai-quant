"""Session-indexed labels, execution timing and point-in-time cohort selection."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from .protocol import HORIZONS


def sessions(values: Iterable[object]) -> pd.DatetimeIndex:
    days = pd.DatetimeIndex(pd.to_datetime(list(values), errors="raise")).normalize()
    if days.empty or days.hasnans or days.tz is not None or not days.is_unique or not days.is_monotonic_increasing:
        raise ValueError("expected a nonempty, unique, ascending, timezone-naive exchange calendar")
    return days


def label_windows(calendar: Iterable[object], horizon: int) -> pd.DataFrame:
    if horizon not in HORIZONS:
        raise ValueError("horizon must be 5 or 20 exchange sessions")
    days = pd.Series(sessions(calendar), name="date")
    return pd.DataFrame({"date": days, "entry": days.shift(-1), "label_end": days.shift(-(horizon + 1))})


def purge_training(samples: pd.DataFrame, cutoff: object) -> pd.DataFrame:
    """Fit only observations whose outcome is fully known before the next partition."""
    boundary = pd.Timestamp(cutoff)
    return samples.loc[(samples["date"] < boundary) & (samples["label_end"] < boundary)].copy()


def execution_schedule(
    calendar: Iterable[object], signals: Mapping[object, Mapping[str, float]]
) -> dict[pd.Timestamp, Mapping[str, float]]:
    days = sessions(calendar)
    offsets = {day: i for i, day in enumerate(days)}
    scheduled = {}
    for signal, weights in signals.items():
        day = pd.Timestamp(signal).normalize()
        if day not in offsets:
            raise ValueError(f"signal is not on an exchange session: {signal}")
        i = offsets[day]
        if i + 1 < len(days):
            if days[i + 1] in scheduled:
                raise ValueError(f"duplicate execution date: {days[i + 1]}")
            scheduled[days[i + 1]] = weights
    return scheduled


def _main_board(symbol: str) -> bool:
    return symbol.startswith(("600", "601", "603", "605", "000", "001", "002", "003"))


def select_stock_cohort(bars: pd.DataFrame, membership: pd.DataFrame, calendar: Iterable[object]) -> dict:
    """Rank dated CSI 300 members by strictly prior 60-session turnover, once a month.

    Membership rows describe complete snapshots. ``known_at`` may differ by
    snapshot version; the latest known complete version is used as of 15:00 CST.
    Missing snapshots or amount observations never fall back to today's members.
    """
    days = sessions(calendar)
    members = membership.copy()
    members["effective_date"] = pd.to_datetime(members["effective_date"], errors="raise").dt.normalize()
    members["known_at"] = pd.to_datetime(members["known_at"], errors="raise", utc=True)
    stocks = bars.loc[bars["kind"].eq("stock"), ["date", "symbol", "amount"]].copy()
    stocks["date"] = pd.to_datetime(stocks["date"], errors="raise").dt.normalize()
    amounts = stocks.pivot(index="date", columns="symbol", values="amount").reindex(days)
    prior = amounts.shift(1).rolling(60, min_periods=60).mean()
    out: dict[pd.Timestamp, tuple[str, ...]] = {}
    selected: tuple[str, ...] = ()
    month = None
    for day in days:
        key = (day.year, day.month)
        if key != month:
            asof = (day + pd.Timedelta(hours=15)).tz_localize("Asia/Shanghai").tz_convert("UTC")
            available = members[(members["effective_date"] <= day) & (members["known_at"] <= asof)]
            selected = ()
            if not available.empty:
                latest_effective = available["effective_date"].max()
                current = available[available["effective_date"].eq(latest_effective)]
                version = current[current["known_at"].eq(current["known_at"].max())]
                eligible = [s for s in set(version["symbol"].astype(str)) if _main_board(s) and s in prior]
                rankings = prior.loc[day, eligible].dropna()
                selected = tuple(sorted(rankings.index, key=lambda s: (-rankings[s], s))[:50])
            month = key
        out[day] = selected
    return out


def open_labels(bars: pd.DataFrame, calendar: Iterable[object], horizon: int, actions: pd.DataFrame) -> pd.DataFrame:
    """Open-to-open hypothetical holder return, accounting for cash and unit changes.

    Marked labels include accrued dividends, without assuming their reinvestment.
    Corporate actions on the entry session already precede entry and are excluded.
    Rows with unpriced/disabled entry or exit are left unknown, not filled with zero.
    """
    windows = label_windows(calendar, horizon)
    days = pd.DatetimeIndex(windows["date"])
    from .ledger import normalize_actions

    actions = normalize_actions(actions, days)
    prices = bars.assign(date=pd.to_datetime(bars["date"], errors="raise").dt.normalize())
    if prices.duplicated(["date", "symbol"]).any():
        raise ValueError("duplicate bars for label calculation")
    out = []
    for symbol, asset in prices.groupby("symbol", sort=True):
        aligned = asset.set_index("date").reindex(days)
        px = pd.to_numeric(aligned["open"]).to_numpy(float)
        tradable = aligned["tradable"].eq(True).to_numpy(bool)
        priced = tradable & np.isfinite(px) & (px > 0)
        values = np.full(len(days), np.nan)
        events = []
        if not actions.empty:
            for row in actions.loc[actions["symbol"].eq(symbol)].itertuples(index=False):
                ex = pd.Timestamp(row.ex_date).normalize()
                if ex in days:
                    j = days.get_loc(ex)
                    record_date = getattr(row, "record_date", None)
                    if pd.isna(record_date):
                        record_date = days[j - 1] if j > 0 else ex
                    events.append((j, pd.Timestamp(record_date).normalize(), row.cash_per_share, row.share_multiplier))
        for i in range(len(days) - horizon - 1):
            start, stop = i + 1, i + horizon + 1
            if not (priced[start] and priced[stop]):
                continue
            shares, cash = 1.0, 0.0
            for j, record, dividend, multiplier in sorted(events):
                if start < j <= stop and days[start] <= record:
                    cash += shares * float(dividend)
                    shares *= float(multiplier)
            values[i] = (shares * px[stop] + cash) / px[start] - 1
        out.append(windows.assign(symbol=symbol, target_return=values))
    return pd.concat(out, ignore_index=True) if out else windows.iloc[:0].assign(symbol="", target_return=np.nan)
