"""Run an exploratory volume and moving-average comparison on the verified ETF snapshot."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import json

import numpy as np
import pandas as pd

from quant_research.baseline import market_close_returns
from quant_research.ledger import Costs, simulate
from quant_research.momentum_study import END, load_market
from quant_research.protocol import ETF_SYMBOLS, HOLDOUT_START, VALIDATION_START
from quant_research.risk import block_resample, summarize
from quant_research.snapshot import file_sha256
from quant_research.timeline import sessions


DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[2]
SNAPSHOT = ROOT / "research_v1/artifacts/market-etf-verified-20261108"
ARMS = {"baseline": (), "volume": ("volume",), "ma": ("ma",), "volume_ma": ("volume", "ma")}
CASH = 100_000
SLOT_WEIGHT = 0.05
MIN_TURNOVER = 30_000_000
VOLUME_RATIO = 1.5
MA_WINDOWS = (5, 10, 20)


def build_features(frames: dict[str, pd.DataFrame], days: pd.DatetimeIndex) -> pd.DataFrame:
    bars, actions = frames["bars"], frames["actions"]
    returns = market_close_returns(bars, actions, days).reindex(columns=ETF_SYMBOLS)
    if returns.iloc[1:].isna().any().any() or returns.le(-1).any().any():
        raise ValueError("the audited ETF panel must have a continuous total-return history")
    adjusted = (1 + returns.fillna(0)).cumprod()
    momentum = (1 + returns).rolling(5).apply(np.prod, raw=True) - 1

    amount = bars.pivot(index="date", columns="symbol", values="amount").reindex(days)
    tradable = bars.pivot(index="date", columns="symbol", values="tradable").reindex(days)
    volume = bars.pivot(index="date", columns="symbol", values="volume").reindex(days)
    splits = pd.DataFrame(1.0, index=days, columns=ETF_SYMBOLS)
    for action in actions.itertuples(index=False):
        splits.loc[action.ex_date, action.symbol] = action.share_multiplier
    # Before taking a rolling average, express volume in a common split-adjusted unit.
    # Dividend events only affect the total-return price, not the volume denominator.
    common_unit_volume = volume / splits.cumprod()
    reference_volume = common_unit_volume.shift(1).rolling(5).mean().where(lambda v: v > 0)
    relative_volume = common_unit_volume / reference_volume
    moving_averages = {n: adjusted.rolling(n).mean() for n in MA_WINDOWS}

    features = []
    for symbol in ETF_SYMBOLS:
        average_turnover = amount[symbol].rolling(5).mean()
        ma_eligible = moving_averages[max(MA_WINDOWS)][symbol].notna()
        below_ma = pd.concat(
            [adjusted[symbol].lt(moving_averages[n][symbol]) for n in MA_WINDOWS], axis=1
        ).any(axis=1)
        table = pd.DataFrame({
            "symbol": symbol,
            "momentum_5": momentum[symbol],
            "turnover_mean_5": average_turnover,
            "return_1": returns[symbol],
            "volume_ratio": relative_volume[symbol],
            "volume": (returns[symbol].lt(0) & relative_volume[symbol].ge(VOLUME_RATIO))
            .where(relative_volume[symbol].notna()),
            "ma": below_ma.where(ma_eligible),
            "signal_tradable": tradable[symbol],
        }, index=days)
        features.append(table)
    return pd.concat(features).rename_axis("signal_date").reset_index()


def schedules(
    days: pd.DatetimeIndex, features: pd.DataFrame, phase: int,
) -> tuple[dict[str, dict[pd.Timestamp, dict]], pd.DataFrame]:
    paired_days = days[phase:-2:2]
    by_signal = features.loc[features["signal_date"].isin(paired_days)]
    eligible = by_signal.loc[
        by_signal["signal_tradable"] & by_signal["momentum_5"].gt(0)
        & by_signal["turnover_mean_5"].ge(MIN_TURNOVER)
    ].sort_values(["signal_date", "momentum_5", "symbol"], ascending=[True, False, True])
    if eligible[["volume", "ma"]].isna().any().any():
        raise ValueError("a shared candidate has a missing volume or MA signal")

    per_day = {date: frame for date, frame in eligible.groupby("signal_date")}
    targets = {arm: {} for arm in ARMS}
    decisions = []
    for day in paired_days:
        offset = days.get_loc(day)
        entry, exit_day = days[offset + 1:offset + 3]
        candidates = per_day.get(day, eligible.iloc[:0])
        for arm, factors in ARMS.items():
            kept = {}
            for row in candidates.itertuples(index=False):
                vetoes = [factor for factor in factors if getattr(row, factor)]
                if not vetoes:
                    kept[row.symbol] = SLOT_WEIGHT
                decisions.append({
                    "arm": arm, "signal_date": day.date(), "entry_date": entry.date(),
                    "exit_date": exit_day.date(), "symbol": row.symbol, "momentum_5": row.momentum_5,
                    "turnover_mean_5": row.turnover_mean_5, "return_1": row.return_1,
                    "volume_ratio": row.volume_ratio, "volume_veto": row.volume, "ma_veto": row.ma,
                    "buy_signal": not vetoes, "vetoes": ",".join(vetoes),
                })
            targets[arm][entry] = kept
            targets[arm][exit_day] = {}
    return targets, pd.DataFrame(decisions)


def evaluate(
    days: pd.DatetimeIndex, frames: dict[str, pd.DataFrame], features: pd.DataFrame,
    phase: int, cost: Costs,
) -> tuple[dict, dict[str, pd.DataFrame]]:
    targets, decisions = schedules(days, features, phase)
    data = {"orders": [], "equity": []}
    stats, daily_returns = {}, {}
    baseline_fills = set()
    exits = set(days[phase + 2::2].date)
    entries = set(days[phase + 1:-1:2].date)
    for arm, targets_for_arm in targets.items():
        result = simulate(frames["bars"], days, targets_for_arm, frames["actions"],
                          initial_cash=CASH, cost=cost)
        equity, orders = result.equity, result.orders
        discrepancy = (equity["nav"] - equity["cash"] - equity["market_value"] - equity["receivable"]).abs()
        if discrepancy.max() > 1e-6 or equity["cash"].min() < 0 or equity["date"].tolist() != days.date.tolist():
            raise RuntimeError(f"incomplete or unbalanced ledger for {arm}")
        delayed = result.positions.loc[result.positions["date"].isin(exits)]
        retries = set()
        for position in delayed.itertuples(index=False):
            offset = days.get_loc(pd.Timestamp(position.date))
            if offset + 1 < len(days):
                next_day = days[offset + 1]
                if position.symbol in targets_for_arm.get(next_day, {}):
                    raise RuntimeError(f"a delayed {position.symbol} exit overlaps a new entry for {arm}")
                retries.add((next_day.date(), position.symbol))
        if not set(orders.loc[orders["side"].eq("buy"), "date"]) <= entries:
            raise RuntimeError(f"buy order outside entry dates for {arm}")
        off_schedule = orders.loc[orders["side"].eq("sell") & ~orders["date"].isin(exits)]
        if not set(zip(off_schedule["date"], off_schedule["symbol"])) <= retries:
            raise RuntimeError(f"sell order outside exit dates for {arm}")
        stats[arm] = summarize(equity, orders, CASH)
        selected = decisions.loc[decisions["arm"].eq(arm)]
        fills = orders.loc[orders["filled_qty"].gt(0)]
        buy_fills = fills.loc[fills["side"].eq("buy")]
        entry_fills = set(zip(buy_fills["date"], buy_fills["symbol"]))
        if arm == "baseline":
            baseline_fills = entry_fills
        elif not entry_fills <= baseline_fills:
            raise RuntimeError(f"a filtered entry for {arm} has no paired baseline fill")
        vetoed = selected.loc[~selected["buy_signal"]]
        effective_vetoes = baseline_fills & set(zip(vetoed["entry_date"], vetoed["symbol"]))
        stats[arm].update({
            "common_candidates": len(selected),
            "vetoed_candidates": len(vetoed),
            "vetoes_of_filled_baseline_orders": len(effective_vetoes),
            "unique_symbols_in_effective_vetoes": len({symbol for _, symbol in effective_vetoes}),
            "buy_fills": len(buy_fills),
            "sell_fills": int(fills["side"].eq("sell").sum()),
            "failed_sell_attempts": int((orders["side"].eq("sell") & orders["status"].ne("filled")).sum()),
            "delayed_exit_session_positions": len(delayed),
            "closing_positions": int(result.positions["date"].eq(days[-1].date()).sum()),
            "order_issues": dict(Counter(orders.loc[orders["reason"].ne(""), "reason"])),
            "average_close_exposure": float((equity["market_value"] / equity["nav"]).mean()),
            "ledger_max_discrepancy": float(discrepancy.max()),
        })
        if stats[arm]["buy_fills"] != stats[arm]["sell_fills"] or stats[arm]["closing_positions"]:
            raise RuntimeError(f"one-day position round trips do not reconcile for {arm}")
        daily_returns[arm] = equity.set_index("date")["daily_return"]
        data["orders"].append(orders.assign(arm=arm))
        data["equity"].append(equity.assign(arm=arm))

    baseline = stats["baseline"]
    comparisons = {}
    for arm in tuple(ARMS)[1:]:
        paired = block_resample(daily_returns[arm], daily_returns["baseline"], block=20, days=len(days))
        comparisons[arm] = {
            "net_return_delta": stats[arm]["net_return"] - baseline["net_return"],
            "max_drawdown_delta": stats[arm]["max_drawdown"] - baseline["max_drawdown"],
            "paired_resampled_net_return_delta_interval_95": paired["paired_terminal_excess_interval"],
        }
    return {
        "start": str(days[0].date()), "end": str(days[-1].date()), "phase": phase,
        "entry_opportunities": len(targets["baseline"]) // 2,
        "metrics": stats, "comparison_to_baseline": comparisons,
    }, {**{name: pd.concat(rows, ignore_index=True) for name, rows in data.items()}, "decisions": decisions}


def run(snapshot: Path, output: Path) -> dict:
    audit, frames = load_market(snapshot, end=END)
    days = sessions(frames["calendar"]["date"])
    features = build_features(frames, days)
    warmed = days[max(MA_WINDOWS) - 1:]
    cost = Costs()
    report = {
        "evidence_class": "reviewed historical exploration; not independent out-of-sample evidence",
        "symbol_universe": ETF_SYMBOLS,
        "market_audit": audit,
        "snapshot_manifest_sha256": file_sha256(snapshot / "manifest.json"),
        "source_sha256": file_sha256(Path(__file__)),
        "last_observed_date": str(days[-1].date()), "sealed_holdout_start": str(HOLDOUT_START),
        "initial_virtual_cash": CASH,
        "signals": {
            "common": "positive five-session total return; last five sessions' mean RMB turnover >= 30 million; tradable at signal close",
            "volume": "veto a negative total-return day with share volume >= 1.5x mean volume of the preceding five sessions, with split normalization",
            "ma": "veto if the action-adjusted close is below any of its 5/10/20-session moving averages, inclusive of the signal close",
            "volume_ma": "veto if either volume or MA vetoes; blocked candidates stay in cash",
        },
        "execution": {
            "timing": "signal at close, buy next session open, attempt liquidation the following open and retry a failed exit on the next available open; alternate sessions in two separate overlapping phases",
            "sizing": "target 5% of opening NAV per candidate, with round-lot rounding and no reallocation; cash yield 0",
            "fees": {"commission_rate": cost.commission_rate, "minimum_commission": cost.minimum_commission,
                     "slippage_bps_per_side": cost.slippage_bps, "stamp_and_transfer_for_etf": 0},
            "limitations": "no 09:45 book, depth, 1% last-close cap or absolute CNY 5000 cap; ledger models open prices and applicable exchange limits",
        },
        "limitations": [
            "Four high-liquidity domestic ETFs; no individual stocks, cross-border ETFs or news inputs.",
            "The verified snapshot is retrospective, with limited proof of contemporaneous publication and revision history.",
            "The two entry phases share calendar observations; intervals are descriptive 20-session block resamples, not independent forecasts.",
        ],
        "periods": {},
    }
    evidence = {"orders": [], "equity": [], "decisions": []}
    development = warmed[warmed.date < VALIDATION_START]
    reviewed = warmed[warmed.date >= VALIDATION_START]
    for window, calendar in (("development_2015_2022", development), ("reviewed_2023_2024", reviewed)):
        report["periods"][window] = []
        for phase in (0, 1):
            result, tables = evaluate(calendar, frames, features, phase, cost)
            report["periods"][window].append(result)
            for name, table in tables.items():
                evidence[name].append(table.assign(window=window, phase=phase))

    output.mkdir(parents=True, exist_ok=True)
    for name, tables in evidence.items():
        pd.concat(tables, ignore_index=True).to_csv(output / f"{name}.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DIRECTORY / "historical-results")
    args = parser.parse_args()
    report = run(args.snapshot, args.output)
    for window, phases in report["periods"].items():
        for result in phases:
            print(window, result["phase"])
            for arm, metrics in result["metrics"].items():
                print(f"  {arm:<10} {metrics['net_return']:8.4%} {metrics['max_drawdown']:8.4%} "
                      f"fees={metrics['transaction_fees']:.2f} buys={metrics['buy_fills']} "
                      f"vetoes={metrics['vetoed_candidates']}/{metrics['common_candidates']}")


if __name__ == "__main__":
    main()
