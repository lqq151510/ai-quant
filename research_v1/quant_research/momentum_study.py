"""Frozen, offline dual-momentum experiment on an audited ETF snapshot.

Previously inspected dates remain historical exploration, not new OOS evidence.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Mapping

import duckdb
import numpy as np
import pandas as pd

from .baseline import market_close_returns
from .ledger import BacktestResult, CloseState, Costs, simulate
from .protocol import CAPITALS, ETF_SYMBOLS, HOLDOUT_START, MAX_DRAWDOWN, RESEARCH_START, ResearchPeriod
from .risk import block_resample, summarize
from .snapshot import audit_snapshot, file_sha256
from .timeline import sessions


LOOKBACK = 126
REVIEW_INTERVAL = 20
SLOT_WEIGHT = 0.35
END = date(2024, 12, 31)
STRATEGIES = ("equal70_monthly", "dual_momentum126_monthly", "band70_weekly")
TARGET = dict.fromkeys(ETF_SYMBOLS, 0.175)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRIOR = ROOT / "artifacts/etf-deadband-study-20261110/results/report.json"


def require_unsealed(end: object) -> date:
    end = pd.Timestamp(end).date()
    if end >= HOLDOUT_START:
        raise ValueError("sealed holdout starts 2025-01-01; this study cannot open it")
    return end


def load_market(snapshot: Path, *, end: object = END) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Hash full immutable inputs, but only materialize rows before the holdout."""
    end = require_unsealed(end)
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or set(manifest.get("tables", {})) != {"bars", "calendar", "actions"}:
        raise ValueError("unexpected ETF snapshot schema")
    frames = {}
    with duckdb.connect(config={"threads": 1}) as db:
        for name, metadata in manifest["tables"].items():
            path = snapshot / f"{name}.parquet"
            if file_sha256(path) != metadata["sha256"]:
                raise ValueError(f"snapshot hash mismatch: {name}")
            count = db.execute("SELECT count(*) FROM read_parquet(?)", [str(path)]).fetchone()[0]
            if count != metadata["rows"]:
                raise ValueError(f"snapshot row count mismatch: {name}")
            column = "ex_date" if name == "actions" else "date"
            frame = db.execute(
                f"SELECT * FROM read_parquet(?) WHERE {column} <= ? ORDER BY {column}", [str(path), end],
            ).df()
            frame[column] = pd.to_datetime(frame[column]).dt.normalize()
            if not frame[column].lt(pd.Timestamp(HOLDOUT_START)).all():
                raise ValueError("sealed rows entered the research frames")
            if name in {"bars", "actions"} and not set(frame["symbol"]) <= set(ETF_SYMBOLS):
                raise ValueError("snapshot contains assets outside the frozen ETF universe")
            frames[name] = frame
    audit = audit_snapshot(manifest, frames, ResearchPeriod(RESEARCH_START, end), "etf")
    if audit["status"] != "ready":
        raise ValueError(f"snapshot failed local audit: {audit}")
    return audit, frames


def momentum_scores(
    bars: pd.DataFrame, actions: pd.DataFrame, days: pd.DatetimeIndex,
) -> pd.DataFrame:
    """126 past total-return periods need 127 closes; never forward-fill a gap."""
    returns = market_close_returns(bars, actions, days).reindex(columns=ETF_SYMBOLS)
    valid = returns.where(np.isfinite(returns) & returns.gt(-1))
    return np.expm1(np.log1p(valid).rolling(LOOKBACK, min_periods=LOOKBACK).sum())


def momentum_target(scores: pd.Series) -> dict[str, float]:
    """Each positive winner owns one 35% slot; unused slots remain cash."""
    eligible = [s for s in ETF_SYMBOLS if s in scores and np.isfinite(scores[s]) and scores[s] > 0]
    ordered = sorted(eligible, key=lambda symbol: (-float(scores[symbol]), symbol))
    return {symbol: SLOT_WEIGHT for symbol in ordered[:2]}


def review_days(days: pd.DatetimeIndex, interval: int = REVIEW_INTERVAL) -> pd.DatetimeIndex:
    if interval < 1:
        raise ValueError("review interval must be positive")
    return days[:-1:interval]


def run_policy(
    frames: Mapping[str, pd.DataFrame], days: pd.DatetimeIndex, reviews: pd.DatetimeIndex,
    initial_cash: int, cost: Costs, strategy: str, scores: pd.DataFrame,
) -> tuple[dict, BacktestResult, pd.DataFrame]:
    if strategy not in STRATEGIES or days.empty or reviews.empty:
        raise ValueError("unknown strategy or empty policy calendar")
    if not reviews.is_unique or not reviews.is_monotonic_increasing or not reviews.isin(days[:-1]).all():
        raise ValueError("reviews must be unique ordered sessions with a following open")
    next_open = {day.date(): days[days.get_loc(day) + 1].date() for day in reviews}
    decisions = []

    def policy(state: CloseState) -> dict[str, float] | None:
        if state.date not in next_open:
            return None
        target = momentum_target(scores.loc[pd.Timestamp(state.date)]) if strategy == STRATEGIES[1] else TARGET
        gross = sum(state.weights.values())
        largest_gap = max(abs(state.weights.get(symbol, 0) - target.get(symbol, 0)) for symbol in ETF_SYMBOLS)
        if not decisions:
            reason = "initial"
        elif strategy != "band70_weekly":
            reason = "scheduled"
        elif largest_gap >= 0.025 and abs(gross - 0.70) >= 0.05:
            reason = "individual_and_total"
        elif largest_gap >= 0.025:
            reason = "individual"
        elif abs(gross - 0.70) >= 0.05:
            reason = "total"
        else:
            reason = "inside_band"
        trigger = reason != "inside_band"
        decisions.append({
            "signal_date": state.date, "eligible_open_date": next_open[state.date],
            "trigger": trigger, "reason": reason, "gross_weight": gross,
            "largest_individual_deviation": largest_gap, "target_gross": sum(target.values()),
            **{f"weight_{s}": state.weights.get(s, 0.0) for s in ETF_SYMBOLS},
            **{f"target_{s}": target.get(s, 0.0) for s in ETF_SYMBOLS},
            **{f"momentum_{s}": float(scores.loc[pd.Timestamp(state.date), s]) for s in ETF_SYMBOLS},
        })
        return dict(target) if trigger else None

    result = simulate(frames["bars"], days, {}, frames["actions"], initial_cash=initial_cash,
                      cost=cost, close_policy=policy)
    decisions = pd.DataFrame(decisions)
    if list(decisions["signal_date"]) != list(reviews.date):
        raise RuntimeError("policy did not observe every frozen review session")
    allowed = set(decisions.loc[decisions["trigger"], "eligible_open_date"])
    if not set(result.orders["date"]) <= allowed:
        raise RuntimeError("an order executed outside its next-open schedule")
    equity = result.equity
    post_entry = equity.loc[equity["date"].ge(next_open[reviews[0].date()])]
    gross = post_entry["market_value"] / post_entry["nav"]
    stats = summarize(equity, result.orders, initial_cash)
    stats.update({
        "review_count": len(decisions), "trigger_count": int(decisions["trigger"].sum()),
        "days_with_fills": int(result.orders.loc[result.orders["filled_qty"].gt(0), "date"].nunique()),
        "gross_weight_after_first_entry_attempt": {
            "min": float(gross.min()), "mean": float(gross.mean()), "max": float(gross.max()),
        },
        "ledger_reconciliation_error": float((equity["nav"] - equity["cash"] - equity["market_value"] - equity["receivable"]).abs().max()),
        "min_cash": float(equity["cash"].min()),
    })
    if stats["ledger_reconciliation_error"] > 1e-6 or stats["min_cash"] < -1e-7:
        raise RuntimeError("share ledger reconciliation failed")
    return stats, result, decisions


def verify_prior(actual: dict, expected: dict) -> None:
    """Fail closed if a previously reported baseline cannot be reproduced."""
    for key, value in expected.items():
        if key not in actual:
            raise ValueError(f"prior baseline missing metric: {key}")
        observed = actual[key]
        if isinstance(value, dict):
            verify_prior(observed, value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isclose(observed, value, rel_tol=1e-12, abs_tol=1e-10):
                raise ValueError(f"prior baseline mismatch {key}: {observed} != {value}")
        elif observed != value:
            raise ValueError(f"prior baseline mismatch {key}")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _markdown(report: dict) -> str:
    lines = ["# 固定双动量候选历史探索", "", "状态：historical_exploration。2023—2024已被查看，本次结果不是新的独立样本外证据。", "",
             "|窗口|模拟本金|成本|策略|净收益|年化收益|最大回撤|费用|成交笔数|", "|---|---:|---|---|---:|---:|---:|---:|---:|"]
    for row in report["metrics"]:
        lines.append(f"|{row['window']}|{row['initial_cash']}|{row['cost_case']}|{row['strategy']}|{row['net_return']:.2%}|{row['annualized_return']:.2%}|{row['max_drawdown']:.2%}|{row['transaction_fees']:.2f}|{row['filled_orders']}|")
    lines.extend(["", f"主窗历史筛选通过：{report['candidate_passed']}；回顾窗历史筛选通过：{report['retrospective_screen_passed']}；独立前向验证：pending。", "",
                  "参数：126期含分红收益、正动量前2、每只35%、20交易日复核。只有1只合格则35%，无合格则现金。",
                  "复核日收盘决策，下一交易日开盘模拟；含整手约束、费用、滑点、限价及公司行动账本。",
                  "回顾窗共同暖启动需127个收盘观察，因此不覆盖2015最早阶段。现金收益假设为0。",
                  "历史供应商快照是事后采集，缺完整历史known_at证据；因果公司行动计算不保证供应商历史数据从未修订。",
                  "配对重采样是历史情景，不是未来获利概率。2025起封存行情未进入研究计算。",
                  "冻结协议见protocol.json；逐日账本、订单、决策见daily.csv、orders.csv、decisions.csv。", ""])
    lines.extend(["|窗口|模拟本金|成本|回撤≤15%|超过月度等权|超过周度容差|全部通过|",
                  "|---|---:|---|---|---|---|---|"])
    for row in report["acceptance"]:
        check = row["checks"]
        lines.append(f"|{row['window']}|{row['initial_cash']}|{row['cost_case']}|{check['drawdown_at_most_15_percent']}|{check['net_return_above_equal70_monthly']}|{check['net_return_above_band70_weekly']}|{row['all_passed']}|")
    lines.append("")
    return "\n".join(lines)


def run_study(snapshot: Path, out: Path, prior_path: Path = DEFAULT_PRIOR) -> dict:
    """Freeze the protocol before simulations; preserve a failed run for inspection."""
    out.mkdir(parents=True, exist_ok=False)
    try:
        audit, frames = load_market(snapshot)
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        if file_sha256(snapshot / "manifest.json") != prior["protocol"]["snapshot_manifest_sha256"]:
            raise ValueError("prior baseline uses a different snapshot")
        calendar = sessions(frames["calendar"]["date"])
        scores = momentum_scores(frames["bars"], frames["actions"], calendar)
        complete = scores.index[np.isfinite(scores).all(axis=1)]
        if complete.empty:
            raise ValueError("not enough common history for 126-period momentum")
        common_start = complete[0]
        primary_spec = prior["protocol"]["windows"]["primary_2023_2024"]
        primary = calendar[(calendar >= pd.Timestamp(primary_spec["start"])) & (calendar <= pd.Timestamp(primary_spec["end"]))]
        prior_reviews = pd.DatetimeIndex(pd.to_datetime(primary_spec["review_dates"]))
        if len(primary) != primary_spec["sessions"] or primary.empty or primary[0] < common_start:
            raise ValueError("prior evaluation calendar or common warm-up is inconsistent")
        retrospective = calendar[(calendar >= common_start) & (calendar.year <= 2022)]
        windows = {"primary_2023_2024": (primary, prior_reviews),
                   "retrospective_warmed_2015_2022": (retrospective, review_days(retrospective, 5))}
        files = [Path(__file__), *(Path(__file__).parent / name for name in (
            "baseline.py", "ledger.py", "protocol.py", "risk.py", "snapshot.py", "timeline.py"))]
        source_hashes = {}
        for path in files:
            relative = path.relative_to(ROOT)
            content = path.read_bytes()
            source_hashes[str(relative)] = hashlib.sha256(content).hexdigest()
            destination = out / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        config = {
            "lookback_return_periods": LOOKBACK, "required_closes": LOOKBACK + 1,
            "review_interval_sessions": REVIEW_INTERVAL, "top_k": 2, "slot_weight": SLOT_WEIGHT,
            "positive_only": True, "tie_break": "symbol_ascending", "missing_history": "exclude_without_forward_fill",
            "band70_individual_tolerance": 0.025, "band70_gross_tolerance": 0.05,
            "capitals_simulated": list(CAPITALS), "thresholds_searched": False,
            "risk_screen_max_drawdown": MAX_DRAWDOWN, "cash_yield": 0,
        }
        protocol = {
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "config": config,
            "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            "source_sha256": source_hashes,
            "environment": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                            "duckdb": duckdb.__version__},
            "snapshot_manifest_sha256": file_sha256(snapshot / "manifest.json"),
            "prior_report_sha256": file_sha256(prior_path), "audit": audit,
            "latest_input_date": END.isoformat(), "sealed_holdout_start": HOLDOUT_START.isoformat(),
            "execution": "signal at close, actual share ledger at next exchange open",
            "evidence_class": "previously inspected historical exploration; not new independent OOS",
            "supplier_limitation": "retrospectively captured snapshot; full historical known_at and revision evidence unavailable",
            "common_warmup_first_signal": str(common_start.date()),
            "windows": {name: {"start": str(days[0].date()), "end": str(days[-1].date()), "sessions": len(days),
                               "weekly_review_dates": [str(d.date()) for d in weekly],
                               "monthly_review_dates": [str(d.date()) for d in review_days(days)]}
                        for name, (days, weekly) in windows.items()},
            "promotion_rule": "all four primary capital/cost cases beat both baselines net of costs and max drawdown <= 15%; forward validation still required",
            "independent_forward_validation": "pending; no future outcomes claimed",
        }
        _write_json(out / "protocol.json", protocol)
        metrics, comparisons, paired, checks = [], [], [], []
        evidence = {"daily": [], "orders": [], "decisions": []}
        for window, (days, weekly) in windows.items():
            for capital in CAPITALS:
                for case, cost in (("normal", Costs()), ("double_friction", Costs().stressed())):
                    context = {"window": window, "initial_cash": capital, "cost_case": case}
                    stats_by_strategy, returns = {}, {}
                    for strategy in STRATEGIES:
                        reviews = weekly if strategy == "band70_weekly" else review_days(days)
                        stats, result, decisions = run_policy(frames, days, reviews, capital, cost, strategy, scores)
                        if window == "primary_2023_2024" and strategy == "band70_weekly":
                            expected = [row for row in prior["metrics"] if row["window"] == window and row["initial_cash"] == capital
                                        and row["cost_case"] == case and row["strategy"] == "band_70"]
                            if len(expected) != 1:
                                raise ValueError("prior report does not have exactly one matching baseline")
                            verify_prior(stats, {k: v for k, v in expected[0].items() if k not in {"window", "initial_cash", "cost_case", "strategy"}})
                        labels = {**context, "strategy": strategy}
                        metrics.append({**labels, **stats})
                        stats_by_strategy[strategy] = stats
                        returns[strategy] = result.equity.set_index("date")["daily_return"]
                        for name, table in (("daily", result.equity), ("orders", result.orders), ("decisions", decisions)):
                            evidence[name].append(table.assign(**labels))
                    candidate = stats_by_strategy[STRATEGIES[1]]
                    local_checks = {"drawdown_at_most_15_percent": candidate["max_drawdown"] <= MAX_DRAWDOWN}
                    for baseline in (STRATEGIES[0], STRATEGIES[2]):
                        base = stats_by_strategy[baseline]
                        local_checks[f"net_return_above_{baseline}"] = candidate["net_return"] > base["net_return"]
                        comparisons.append({**context, "baseline": baseline,
                                            "net_return_gain": candidate["net_return"] - base["net_return"],
                                            "drawdown_change": candidate["max_drawdown"] - base["max_drawdown"],
                                            "fee_change": candidate["transaction_fees"] - base["transaction_fees"]})
                        paired.extend({**context, "baseline": baseline,
                                       **block_resample(returns[STRATEGIES[1]], returns[baseline], block=b, paths=2000)}
                                      for b in (20, 40, 60))
                    checks.append({**context, "checks": local_checks, "all_passed": all(local_checks.values())})
                    print(window, capital, case, f"candidate net={candidate['net_return']:.4%} DD={candidate['max_drawdown']:.4%}", flush=True)
        for relative, expected in source_hashes.items():
            if file_sha256(ROOT / relative) != expected:
                raise RuntimeError(f"research source changed during the frozen run: {relative}")
        report = {"status": "historical_exploration", "protocol": protocol,
                  "matched_prior_primary_band_baseline": True, "metrics": metrics,
                  "comparisons": comparisons, "acceptance": checks, "paired_resampling": paired,
                  "candidate_passed": all(row["all_passed"] for row in checks if row["window"] == "primary_2023_2024"),
                  "retrospective_screen_passed": all(row["all_passed"] for row in checks if row["window"] == "retrospective_warmed_2015_2022"),
                  "independent_forward_validation": "pending", "new_independent_oos_improvement_proven": False}
        for name, tables in evidence.items():
            pd.concat(tables, ignore_index=True).to_csv(out / f"{name}.csv", index=False)
        _write_json(out / "report.json", report)
        (out / "report.md").write_text(_markdown(report), encoding="utf-8")
        return report
    except Exception as exc:
        _write_json(out / "failure.json", {"status": "failed", "type": type(exc).__name__, "message": str(exc),
                                          "failed_at_utc": datetime.now(timezone.utc).isoformat()})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prior-report", type=Path, default=DEFAULT_PRIOR)
    args = parser.parse_args()
    run_study(args.snapshot.resolve(), args.out.resolve(), args.prior_report.resolve())


if __name__ == "__main__":
    main()
