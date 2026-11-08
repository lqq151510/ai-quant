"""Pre-registered baseline schedules; all execution goes through one cash ledger."""

from __future__ import annotations

import json
from collections.abc import Mapping
from math import fsum
from pathlib import Path

import numpy as np
import pandas as pd

from .ledger import BacktestResult, CloseState, Costs, normalize_actions, simulate
from .protocol import CAPITALS, ETF_SYMBOLS, HORIZONS
from .risk import block_resample, summarize
from .timeline import execution_schedule, select_stock_cohort, sessions


def market_close_returns(bars: pd.DataFrame, actions: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    if "valuation_close" in bars:
        marks = bars.pivot(index="date", columns="symbol", values="valuation_close").reindex(days)
    else:
        marks = None
    prices = bars.pivot(index="date", columns="symbol", values="close").reindex(days)
    if marks is not None:
        prices = prices.where(marks.isna(), marks)
    previous = prices.shift(1)
    adjusted_current = prices.copy()
    for action in normalize_actions(actions, days).itertuples(index=False):
        day = pd.Timestamp(action.ex_date).normalize()
        if day in days and action.symbol in prices:
            if marks is not None and pd.notna(marks.loc[day, action.symbol]) and action.share_multiplier != 1:
                # A halted split's carry mark represents no price movement. Its tick rounding
                # and the later publication of the exact ratio must not create a return signal.
                price = previous.loc[day, action.symbol]
            else:
                price = prices.loc[day, action.symbol] * float(action.share_multiplier)
            adjusted_current.loc[day, action.symbol] = price + float(action.cash_per_share)
    return adjusted_current / previous - 1


def momentum_weights(scores: pd.Series, volatility: pd.Series, kind: str) -> dict[str, float]:
    positive = scores[scores.gt(0) & np.isfinite(scores)]
    names = sorted(positive.index, key=lambda symbol: (-positive.loc[symbol], symbol))
    if kind == "stock":
        return {s: 0.1 for s in names[:10]}
    valid = volatility.reindex(names).dropna()
    valid = valid[valid.gt(0) & np.isfinite(valid)]
    inverse = 1 / valid
    return {s: min(0.35, float(v)) for s, v in (inverse / inverse.sum()).items()}


def _momentum_scores(returns: pd.DataFrame) -> pd.DataFrame:
    return np.expm1(np.log1p(returns.where(returns > -1)).rolling(20, min_periods=20).sum())


def build_targets(
    days: pd.DatetimeIndex, cohort: Mapping, returns: pd.DataFrame, kind: str, horizon: int
) -> dict[str, dict]:
    momentum = _momentum_scores(returns)
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


def build_allocation_decisions(
    days: pd.DatetimeIndex, cohort: Mapping, returns: pd.DataFrame, kind: str, horizon: int
) -> dict[pd.Timestamp, dict[str, dict[str, float]]]:
    from .allocation import RETURN_LOOKBACK, allocation_targets

    momentum = _momentum_scores(returns.loc[:days[-1]])
    return {
        day: allocation_targets(
            momentum.loc[day].reindex(list(cohort.get(day, ()))),
            returns.loc[:day].tail(RETURN_LOOKBACK),
            kind,
        )
        for day in days[::horizon]
    }


def _allocation_target_metrics(schedule: Mapping) -> dict:
    gross = {day: fsum(weights.values()) for day, weights in schedule.items()}
    invested_hhi = [
        fsum((weight / gross[day]) ** 2 for weight in weights.values())
        for day, weights in schedule.items() if gross[day] > 0
    ]
    return {
        "execution_sessions": len(gross),
        "invested_execution_sessions": len(invested_hhi),
        "mean_gross_weight": fsum(gross.values()) / len(gross) if gross else None,
        "max_single_name_weight": max((max(weights.values(), default=0.0) for weights in schedule.values()), default=0.0),
        "mean_invested_hhi": fsum(invested_hhi) / len(invested_hhi) if invested_hhi else None,
    }


def _latest_allocation_decision(
    days: pd.DatetimeIndex, calendar: pd.DatetimeIndex, horizon: int, decisions: Mapping
) -> dict:
    decision_day = max(decisions)
    next_index = calendar.get_loc(decision_day) + 1
    next_day = calendar[next_index] if next_index < len(calendar) else None
    if decision_day < days[-1]:
        status = "historical_decision"
    else:
        status = "next_session_unknown" if next_day is None else "pending_outside_window"
    return {
        "as_of_session": days[-1].date().isoformat(),
        "decision_session": decision_day.date().isoformat(),
        "next_exchange_session": next_day.date().isoformat() if next_day is not None else None,
        "rebalance_sessions": horizon,
        "status": status,
        "strategies": {
            strategy: {"targets": weights, "cash_weight": max(0.0, 1.0 - fsum(weights.values()))}
            for strategy, weights in decisions[decision_day].items()
        },
    }


def _simulate_optimized(
    bars: pd.DataFrame, days: pd.DatetimeIndex, actions: pd.DataFrame,
    hrp_decisions: Mapping, exposure_by_day: pd.Series, kind: str,
    initial_cash: float, cost: Costs, *, use_band: bool, use_breadth: bool,
) -> tuple[BacktestResult, dict[str, int]]:
    from .strategy_optimization import rebalance_target

    diagnostics = dict.fromkeys(("scheduled_count", "skipped_count", "defensive_count", "unknown_breadth_count"), 0)

    def on_close(state: CloseState) -> Mapping[str, float] | None:
        day = pd.Timestamp(state.day)
        if day not in hrp_decisions:
            return None
        diagnostics["scheduled_count"] += 1
        hrp_target = hrp_decisions[day]["allocation_hrp"]
        scale = 1.0
        if use_breadth:
            observed_scale = float(exposure_by_day.loc[day])
            if np.isfinite(observed_scale):
                scale = observed_scale
                diagnostics["defensive_count"] += int(scale < 1.0 and bool(hrp_target))
            else:
                diagnostics["unknown_breadth_count"] += 1
        target = rebalance_target(
            hrp_target, state.weights, kind,
            use_band=use_band, exposure_scale=scale,
        )
        diagnostics["skipped_count"] += int(target is None and bool(state.weights))
        return target

    result = simulate(bars, days, {}, actions, initial_cash=initial_cash, cost=cost, on_close=on_close)
    return result, diagnostics


def run_baselines(
    frames: Mapping[str, pd.DataFrame], output: Path, *, start: object, end: object, origin: str,
    kinds: tuple[str, ...] = ("etf", "stock"), allocation_study: bool = False,
    strategy_optimization: bool = False,
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
    plan = None
    if allocation_study or strategy_optimization:
        from .allocation import GROSS_BUDGET_FRACTION, RETURN_LOOKBACK, require_allocation_dependencies

        require_allocation_dependencies()
        plan = {
            "origin": origin, "purpose": "paper_allocation_study",
            "target_gross_rule": "gross_budget_fraction * min(1, eligible_count * single_name_cap)",
            "gross_budget_fraction": GROSS_BUDGET_FRACTION,
            "return_lookback_sessions": RETURN_LOOKBACK,
            "lines": {},
        }
    bars = frames["bars"].copy()
    bars["date"] = pd.to_datetime(bars["date"], errors="raise").dt.normalize()
    actions = frames["actions"]
    observed_calendar = calendar[calendar <= days[-1]]
    bars = bars[bars["date"] <= days[-1]]
    adjusted = market_close_returns(bars, actions, observed_calendar)
    optimization_variants = {}
    exposure_by_day = pd.Series(dtype=float)
    if strategy_optimization:
        from .strategy_optimization import OPTIMIZED_STRATEGIES, market_exposure_scale

        optimization_variants = OPTIMIZED_STRATEGIES
        exposure_by_day = market_exposure_scale(_momentum_scores(adjusted))
    stock = select_stock_cohort(bars, frames["membership"], observed_calendar) if "stock" in kinds else {}
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
            allocation_metrics = {}
            decisions = {}
            if plan is not None:
                decisions = build_allocation_decisions(days, cohort, adjusted, kind, h)
                plan["lines"][f"{kind}-{h}d"] = _latest_allocation_decision(days, calendar, h, decisions)
                for strategy in decisions[days[0]]:
                    schedule = execution_schedule(days, {day: targets[strategy] for day, targets in decisions.items()})
                    schedules[strategy] = schedule
                    allocation_metrics[strategy] = _allocation_target_metrics(schedule)

            for initial_cash in CAPITALS:
                for case, settings in (("normal", cost), ("double_friction", cost.stressed())):
                    paired_rows = {}
                    for name in (*schedules, *optimization_variants):
                        diagnostics = None
                        if name in optimization_variants:
                            use_band, use_breadth = optimization_variants[name]
                            item, diagnostics = _simulate_optimized(
                                universe_bars, days, actions, decisions, exposure_by_day, kind,
                                initial_cash, settings, use_band=use_band, use_breadth=use_breadth,
                            )
                        else:
                            item = simulate(
                                universe_bars, days, schedules[name], actions, initial_cash=initial_cash, cost=settings,
                            )
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
                        if name in allocation_metrics:
                            row["allocation_target_metrics"] = allocation_metrics[name]
                        if diagnostics is not None:
                            row["optimization_diagnostics"] = diagnostics
                        if strategy_optimization and (name == "allocation_hrp" or diagnostics is not None):
                            row["mean_realized_gross_weight"] = float(
                                (item.equity["market_value"] / item.equity["nav"]).mean()
                            )
                        paired_rows[name] = row
                        result.append(row)
                        if case == "normal":
                            normal_series.setdefault((kind, h, initial_cash), {})[name] = item.equity.set_index("date")["daily_return"]
                    for name in optimization_variants:
                        candidate = paired_rows[name]
                        control = paired_rows["allocation_hrp"]
                        candidate["vs_hrp"] = {
                            metric: candidate[metric] - control[metric]
                            for metric in ("net_return", "max_drawdown", "traded_notional_over_average_nav",
                                           "mean_realized_gross_weight")
                        }
    for row in result:
        if row["strategy"] != "momentum" or row["cost_case"] != "normal":
            continue
        kind, horizon = row["line"].split("-")
        pair = normal_series[(kind, int(horizon[:-1]), row["capital"])]
        row["resampling"] = [block_resample(pair["momentum"], pair["equal_weight"], block=b)
                             for b in (20, 40, 60) if len(days) >= b]
    if plan is not None:
        (output / "allocation_plan.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    return result
