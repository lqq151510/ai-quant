"""Replay and audit the 60-session ETF reversal on two already explored years."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .baseline import build_targets, market_close_returns
from .capital_scale import CAPITALS, reversal_exposure_targets
from .ledger import Costs, simulate
from .momentum_study import END, load_market
from .protocol import ETF_SYMBOLS, HOLDOUT_START, MAX_DRAWDOWN, VALIDATION_START
from .risk import summarize
from .snapshot import file_sha256
from .timeline import sessions

HORIZON = 60
YEARS = (2023, 2024)
SIGNALS = ("reverse", "equal_weight_match_reverse", "equal_weight")


def subwindow_metrics(equity: pd.DataFrame, orders: pd.DataFrame, year: int, initial_cash: float) -> dict:
    """Measure local returns/drawdown, including the immediately preceding NAV."""
    dates = pd.to_datetime(equity["date"])
    year_nav = equity.loc[dates.dt.year.eq(year)]
    prior = equity.loc[dates.dt.year.lt(year), "nav"]
    opening_nav = initial_cash if prior.empty else float(prior.iloc[-1])
    year_orders = orders.loc[pd.to_datetime(orders["date"]).dt.year.eq(year)]
    metrics = summarize(year_nav, year_orders, opening_nav)
    return {
        **metrics,
        "starting_nav": opening_nav,
        "calmar": metrics["annualized_return"] / metrics["max_drawdown"] if metrics["max_drawdown"] else None,
        "mean_exposure": float((year_nav["market_value"] / year_nav["nav"]).mean()),
        "board_lot_rejections": int(year_orders["reason"].eq("board_lot").sum()),
    }


def schedules_for(days: pd.DatetimeIndex, returns: pd.DataFrame) -> dict:
    targets = build_targets(
        days, dict.fromkeys(days, ETF_SYMBOLS), returns, "etf", HORIZON, reverse_momentum=True,
    )
    reverse = targets["momentum"]
    matched = reversal_exposure_targets(reverse)["equal_weight_match_reverse"]
    return {"reverse": reverse, "equal_weight_match_reverse": matched, "equal_weight": targets["equal_weight"]}


def run_study(snapshot: Path, output: Path) -> dict:
    audit, frames = load_market(snapshot, end=END)
    calendar = sessions(frames["calendar"]["date"])
    days = calendar[calendar >= pd.Timestamp(VALIDATION_START)]
    if tuple(days.year.unique()) != YEARS or days[-1].date() != END:
        raise ValueError("missing sessions in the declared pre-holdout study window")
    bars, actions = frames["bars"], frames["actions"]
    returns = market_close_returns(bars, actions, calendar)
    year_days = {year: days[days.year == year] for year in YEARS}
    windows = [("continuation", days, YEARS)]
    windows.extend(("fresh_cash", year_days[year], (year,)) for year in YEARS)
    metrics, schedules, ledgers, orders = [], [], [], []
    for mode, run_days, result_years in windows:
        for signal, targets in schedules_for(run_days, returns).items():
            schedules.extend({
                "mode": mode, "signal": signal,
                "signal_date": run_days[run_days.get_loc(day) - 1].date().isoformat(),
                "execution_date": day.date().isoformat(), "target_weights": weights,
            } for day, weights in targets.items())
            for capital in CAPITALS:
                for cost_name, cost in (("normal", Costs()), ("double_friction", Costs().stressed())):
                    result = simulate(bars, run_days, targets, actions, initial_cash=capital, cost=cost)
                    nav, log = result.equity, result.orders
                    discrepancy = (nav["nav"] - nav["cash"] - nav["market_value"] - nav["receivable"]).abs().max()
                    if (discrepancy > 1e-6 or nav["cash"].min() < 0 or nav["date"].tolist() != run_days.date.tolist()
                            or not set(log["date"]) <= set(pd.DatetimeIndex(targets).date)):
                        raise RuntimeError(f"failed ledger reconciliation for {mode}/{signal}/{capital}/{cost_name}")
                    labels = dict(mode=mode, signal=signal, initial_cash=capital, cost_case=cost_name)
                    ledgers.append(nav.assign(**labels))
                    orders.append(log.assign(**labels))
                    for year in result_years:
                        metrics.append({
                            **labels, "year": year,
                            "scheduled_reviews": sum(day.year == year for day in targets),
                            **subwindow_metrics(nav, log, year, capital),
                        })

    report = {
        "evidence": "already_explored_2023_2024_subwindows",
        "snapshot_manifest_sha256": file_sha256(snapshot / "manifest.json"),
        "audit": audit,
        "latest_input_date": END.isoformat(),
        "sealed_holdout_start": HOLDOUT_START.isoformat(),
        "review_interval_sessions": HORIZON,
        "max_drawdown_screen": MAX_DRAWDOWN,
        "metrics": metrics, "schedules": schedules,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(render(report), encoding="utf-8")
    pd.concat(ledgers, ignore_index=True).to_csv(output / "equity.csv", index=False)
    pd.concat(orders, ignore_index=True).to_csv(output / "orders.csv", index=False)
    return report


def render(report: dict) -> str:
    lines = [
        "# 60 日反向策略的年度子窗", "",
        "续跑切片沿用完整账本和原定复核日。年度重启以现金开始，并在年内重新按 60 个交易日排定复核。",
        "两种口径均以该年起始权益为回撤基点。年化收益按 252 个交易日折算，Calmar 为年化收益除以最大回撤。",
        "等目标仓位组在相同复核日，将反向组当次目标总仓位均分至四只 ETF。实际仓位会随价格、整手与成交漂移。",
        "亏损、无成交或 Calmar 无法计算的场景均保留。该窗口此前已被查看，2025 年起的封存行情未进入计算。", "",
        "| 口径 | 年度 | 本金 | 成本 | 策略 | 复核 | 净收益 | 年化收益 | 回撤 | Calmar | 平均持仓 |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(report["metrics"], key=lambda r: (r["mode"], r["year"], r["initial_cash"], r["cost_case"], SIGNALS.index(r["signal"]))):
        calmar = f"{row['calmar']:.2f}" if row["calmar"] is not None else "未定义"
        lines.append(
            f"| {row['mode']} | {row['year']} | {row['initial_cash']:,} | {row['cost_case']} | {row['signal']} | "
            f"{row['scheduled_reviews']} | {row['net_return']:+.2%} | {row['annualized_return']:+.2%} | "
            f"{row['max_drawdown']:.2%} | {calmar} | {row['mean_exposure']:.2%} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run_study(args.snapshot, args.out)


if __name__ == "__main__":
    main()
