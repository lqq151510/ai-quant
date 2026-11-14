"""Compare a fixed 12% volatility target with same-cadence ETF allocations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import market_close_returns
from .capital_scale import CAPITALS, attribute_actual_fills
from .ledger import Costs, simulate
from .momentum_study import END, load_market
from .protocol import ETF_SYMBOLS, HOLDOUT_START, MAX_DRAWDOWN, VALIDATION_START
from .reversal_subwindows import subwindow_metrics
from .risk import summarize
from .snapshot import file_sha256
from .timeline import execution_schedule, sessions

TARGET_VOL = 0.12
LOOKBACK = 60
REVIEW_INTERVAL = 60
STATIC_EXPOSURE = 0.60
RETURN_GAP = 0.02
NAMES = {"vol_target": "12% 波动率目标", "equal_full": "满仓等权", "equal_60": "固定 60% 等权"}


def build_schedules(returns: pd.DataFrame, days: pd.DatetimeIndex) -> tuple[dict, list[dict]]:
    """Estimate basket volatility at the close and trade at the following open."""
    basket = returns.reindex(columns=ETF_SYMBOLS).replace([np.inf, -np.inf], np.nan)
    # Sum requires all four observations. Partial cross sections cannot change basket composition.
    basket_returns = basket.sum(axis=1, min_count=len(ETF_SYMBOLS)) / len(ETF_SYMBOLS)
    vol = basket_returns.rolling(LOOKBACK, min_periods=LOOKBACK).std(ddof=1) * np.sqrt(252)
    signal_days = days[:-1:REVIEW_INTERVAL]
    signals: dict[str, dict] = {name: {} for name in NAMES}
    evidence = []
    for day in signal_days:
        estimate = float(vol.get(day, np.nan))
        if not np.isfinite(estimate) or estimate <= 0:
            raise ValueError(f"cannot estimate basket volatility at {day.date()}")
        target_exposure = min(1.0, TARGET_VOL / estimate)
        for name, exposure in (("vol_target", target_exposure), ("equal_full", 1.0), ("equal_60", STATIC_EXPOSURE)):
            signals[name][day] = dict.fromkeys(ETF_SYMBOLS, exposure / len(ETF_SYMBOLS))
        evidence.append({
            "signal_date": day.date().isoformat(),
            "execution_date": days[days.get_loc(day) + 1].date().isoformat(),
            "estimated_annual_volatility": estimate,
            "target_exposure": target_exposure,
            "projected_annual_volatility": target_exposure * estimate,
        })
    return {name: execution_schedule(days, schedule) for name, schedule in signals.items()}, evidence


def run_study(snapshot: Path, output: Path) -> dict:
    audit, frames = load_market(snapshot, end=END)
    calendar = sessions(frames["calendar"]["date"])
    days = calendar[calendar >= pd.Timestamp(VALIDATION_START)]
    if days.empty or days[-1].date() != END:
        raise ValueError("incomplete historical study window")
    bars, actions = frames["bars"], frames["actions"]
    schedules, signal_evidence = build_schedules(market_close_returns(bars, actions, calendar), days)
    openings = bars.assign(date=bars["date"].dt.date).set_index(["date", "symbol"])["open"]
    metrics, annual, equity, orders = [], [], [], []
    costs = Costs()
    for case, cost in (("normal", costs), ("double_friction", costs.stressed())):
        for capital in CAPITALS:
            for name, schedule in schedules.items():
                result = simulate(bars, days, schedule, actions, initial_cash=capital, cost=cost)
                nav, log = result.equity, result.orders
                residual = (nav["nav"] - nav["cash"] - nav["market_value"] - nav["receivable"]).abs().max()
                if (residual > 1e-6 or nav["cash"].min() < 0 or nav["date"].tolist() != days.date.tolist()
                        or not set(log["date"]) <= set(pd.DatetimeIndex(schedule).date)):
                    raise RuntimeError(f"ledger failed reconciliation for {name}/{capital}/{case}")
                info = dict(initial_cash=capital, cost_case=case, strategy=name)
                filled = log.loc[log["filled_qty"].gt(0)]
                summary = summarize(nav, log, capital)
                metrics.append({
                    **info, **summary,
                    "mean_exposure": float((nav["market_value"] / nav["nav"]).mean()),
                    "board_lot_rejections": int(log["reason"].eq("board_lot").sum()),
                    "ledger_max_discrepancy": float(residual),
                    "attribution": attribute_actual_fills(filled, openings, final_nav=summary["final_nav"], capital=capital),
                })
                annual.extend({**info, "year": year, **subwindow_metrics(nav, log, year, capital)} for year in (2023, 2024))
                equity.append(nav.assign(**info))
                orders.append(log.assign(**info))
    lookup = {(row["strategy"], row["initial_cash"], row["cost_case"]): row for row in metrics}
    screen = []
    for capital in CAPITALS:
        for case in ("normal", "double_friction"):
            row = lookup["vol_target", capital, case]
            full = lookup["equal_full", capital, case]
            fixed = lookup["equal_60", capital, case]
            gap = row["net_return"] - full["net_return"]
            checks = {
                "positive": row["net_return"] > 0,
                "drawdown_at_most_15pct": row["max_drawdown"] <= MAX_DRAWDOWN,
                "within_2pp_of_full": gap >= -RETURN_GAP,
            }
            screen.append({
                "initial_cash": capital, "cost_case": case, **checks,
                "passes_primary_screen": all(checks.values()),
                "return_gap_vs_full": gap,
                "return_gap_vs_fixed_60": row["net_return"] - fixed["net_return"],
                "drawdown_gap_vs_fixed_60": row["max_drawdown"] - fixed["max_drawdown"],
            })
    report = {
        "evidence": "already_explored_2023_2024_historical_comparison",
        "window": [days[0].date().isoformat(), END.isoformat()],
        "sealed_holdout_start": HOLDOUT_START.isoformat(),
        "snapshot_manifest_sha256": file_sha256(snapshot / "manifest.json"),
        "source_sha256": file_sha256(Path(__file__)),
        "audit": audit, "target_volatility": TARGET_VOL, "lookback_sessions": LOOKBACK,
        "review_interval_sessions": REVIEW_INTERVAL, "full_return_gap_screen": RETURN_GAP,
        "signals": signal_evidence, "metrics": metrics, "annual_continuation": annual, "screen": screen,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(render(report), encoding="utf-8")
    pd.concat(equity, ignore_index=True).to_csv(output / "equity.csv", index=False)
    pd.concat(orders, ignore_index=True).to_csv(output / "orders.csv", index=False)
    return report


def render(report: dict) -> str:
    lines = [
        "# 四只 ETF 等权加 12% 波动率目标", "",
        "三组每 60 个交易日复核，同一收盘算信号并在下一交易日开盘模拟执行。"
        "目标组用最近 60 日四只 ETF 等权复权收益的样本标准差乘 √252 估计年化波动，"
        "总目标仓位为 12% 除以估计波动，最高 100%，剩余为现金。",
        "现金收益为零，成交保留整手、公司行动、正常与双倍摩擦。"
        "全窗筛选要求净收益为正、最大回撤 ≤15%、收益较满仓等权最多低 2 个百分点。"
        "固定 60% 等权用于比较动态仓位本身。波动率目标不保证实际波动率或回撤。",
        "本报告只使用已查看的 2023–2024 年，年度净收益按完整账本跨年续跑。", "",
        "| 本金 | 成本 | 策略 | 净收益 | 最大回撤 | 实现年化波动 | 平均持仓 | 全期换手 | 2023 净收益 | 2024 净收益 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    annual = {(r["strategy"], r["initial_cash"], r["cost_case"], r["year"]): r for r in report["annual_continuation"]}
    for row in report["metrics"]:
        name, capital, case = row["strategy"], row["initial_cash"], row["cost_case"]
        years = [annual[name, capital, case, y]["net_return"] for y in (2023, 2024)]
        lines.append(
            f"| {capital:,} | {case} | {NAMES[name]} | {row['net_return']:+.2%} | {row['max_drawdown']:.2%} | "
            f"{row['annualized_volatility']:.2%} | {row['mean_exposure']:.2%} | "
            f"{row['traded_notional_over_average_nav']:.2f} 倍 | {years[0]:+.2%} | {years[1]:+.2%} |"
        )
    lines += ["", "| 本金 | 成本 | 达到全窗筛选 | 与满仓等权差（百分点） | 与固定 60% 等权差（百分点） |",
              "|---:|---|---|---:|---:|"]
    for row in report["screen"]:
        lines.append(
            f"| {row['initial_cash']:,} | {row['cost_case']} | {'通过' if row['passes_primary_screen'] else '未通过'} | "
            f"{100 * row['return_gap_vs_full']:+.2f} | {100 * row['return_gap_vs_fixed_60']:+.2f} |"
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
