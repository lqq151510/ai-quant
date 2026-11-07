"""Pre-registered baseline schedules; all execution goes through one cash ledger."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from .ledger import Costs, normalize_actions, simulate
from .protocol import CAPITALS, ETF_SYMBOLS, HORIZONS
from .risk import block_resample, summarize
from .timeline import execution_schedule, select_stock_cohort, sessions


def market_close_returns(bars: pd.DataFrame, actions: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    prices = bars.pivot(index="date", columns="symbol", values="close").reindex(days)
    adjusted_current = prices.copy()
    for action in normalize_actions(actions, days).itertuples(index=False):
        day = pd.Timestamp(action.ex_date).normalize()
        if day in days and action.symbol in prices:
            adjusted_current.loc[day, action.symbol] = (
                adjusted_current.loc[day, action.symbol] * float(action.share_multiplier)
                + float(action.cash_per_share)
            )
    return adjusted_current / prices.shift(1) - 1


def momentum_weights(scores: pd.Series, volatility: pd.Series, kind: str) -> dict[str, float]:
    positive = scores[scores.gt(0) & np.isfinite(scores)]
    names = sorted(positive.index, key=lambda symbol: (-positive.loc[symbol], symbol))
    if kind == "stock":
        return {s: 0.1 for s in names[:10]}
    valid = volatility.reindex(names).dropna()
    valid = valid[valid.gt(0) & np.isfinite(valid)]
    inverse = 1 / valid
    return {s: min(0.35, float(v)) for s, v in (inverse / inverse.sum()).items()}


def build_targets(
    days: pd.DatetimeIndex, cohort: Mapping, returns: pd.DataFrame, kind: str, horizon: int
) -> dict[str, dict]:
    momentum = np.expm1(np.log1p(returns.where(returns > -1)).rolling(20, min_periods=20).sum())
    volatility = returns.rolling(20, min_periods=20).std()
    eq, mom = {}, {}
    for day in days[::horizon]:
        names = list(cohort.get(day, ()))
        eq[day] = {s: 1 / len(names) for s in names} if names else {}
        mom[day] = momentum_weights(momentum.loc[day].reindex(names), volatility.loc[day], kind)
    return {
        "cash": {}, "equal_weight": execution_schedule(days, eq), "momentum": execution_schedule(days, mom),
        "csi300_etf_hold": execution_schedule(days, {days[0]: {"510300": 1.0}}),
    }


def run_baselines(
    frames: Mapping[str, pd.DataFrame], output: Path, *, start: object, end: object, origin: str,
    kinds: tuple[str, ...] = ("etf", "stock"),
) -> list[dict]:
    """Run every line/initial capital and persist independent, inspectable ledgers."""
    if origin not in {"synthetic_fixture", "audited_market"}:
        raise ValueError("baseline data must be an explicit fixture or audited market input")
    if not kinds or set(kinds) - {"etf", "stock"}:
        raise ValueError("invalid market lines")
    calendar = sessions(frames["calendar"]["date"])
    days = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    if days.empty:
        raise ValueError("no sessions in baseline window")
    bars = frames["bars"].copy()
    bars["date"] = pd.to_datetime(bars["date"], errors="raise").dt.normalize()
    actions = frames["actions"]
    adjusted = market_close_returns(bars, actions, calendar)
    stock = select_stock_cohort(bars, frames["membership"], calendar) if "stock" in kinds else {}
    etf = {d: ETF_SYMBOLS for d in days}
    result = []
    cost = Costs()
    normal_series: dict[tuple[str, int, int], dict[str, pd.Series]] = {}
    for kind, cohort in (("etf", etf), ("stock", stock)):
        if kind not in kinds:
            continue
        universe_bars = bars[bars["kind"].eq(kind) | bars["symbol"].eq("510300")]
        for h in HORIZONS:
            schedules = build_targets(days, cohort, adjusted, kind, h)
            for initial_cash in CAPITALS:
                for case, settings in (("normal", cost), ("double_friction", cost.stressed())):
                    for name, orders in schedules.items():
                        item = simulate(universe_bars, days, orders, actions, initial_cash=initial_cash, cost=settings)
                        dest = output / "ledgers" / f"{kind}-{h}d-{initial_cash}-{case}-{name}"
                        dest.mkdir(parents=True, exist_ok=False)
                        for table in ("equity", "orders", "positions", "cash_events"):
                            getattr(item, table).to_parquet(dest / f"{table}.parquet", index=False)
                        metrics = summarize(item.equity, item.orders, initial_cash)
                        import duckdb

                        with duckdb.connect(config={"threads": 1}) as db:
                            err, cash_floor = db.execute(
                                "SELECT max(abs(nav - cash - market_value - receivable)), min(cash) FROM read_parquet(?)",
                                [str(dest / "equity.parquet")],
                            ).fetchone()
                        if err is None or err > 1e-6 or cash_floor < -1e-7:
                            raise RuntimeError(f"ledger does not reconcile: {dest.name}")
                        metrics.update({"ledger_reconciliation_error": err, "min_cash": cash_floor})
                        row = {"origin": origin, "line": f"{kind}-{h}d", "capital": initial_cash,
                               "cost_case": case, "strategy": name, **metrics}
                        result.append(row)
                        if case == "normal":
                            normal_series.setdefault((kind, h, initial_cash), {})[name] = item.equity.set_index("date")["daily_return"]
    for row in result:
        if row["strategy"] != "momentum" or row["cost_case"] != "normal":
            continue
        kind, horizon = row["line"].split("-")
        pair = normal_series[(kind, int(horizon[:-1]), row["capital"])]
        row["resampling"] = [block_resample(pair["momentum"], pair["equal_weight"], block=b)
                             for b in (20, 40, 60) if len(days) >= b]
    return result
