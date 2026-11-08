"""Compare one predictive feature or fitting change at a time on common rows."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import market_close_returns
from .feature_upgrades import LowdinTransform, daily_bar_features, time_decay_weights, wasserstein_drift
from .fund_flow import SOURCE_FIELDS, capture_features_asof
from .protocol import (
    ETF_SYMBOLS, HOLDOUT_START, HORIZONS, LAST_EXPLORED_SESSION, RESEARCH_START, VALIDATION_START, ResearchPeriod,
)
from .timeline import open_labels, purge_training, select_stock_cohort, sessions


_PRICE = ("return_1d", "momentum_5d", "momentum_20d", "volatility_20d")
_OHLC = ("clv", "up_down_variance_ratio")
_DRIFT = "wasserstein_drift"
_REQUIRED = [*_PRICE, *_OHLC, _DRIFT, "target_return"]
_HALF_LIFE = 120
_MODEL = {"max_depth": 4, "min_samples_leaf": 20, "random_state": 47}
_CAPTURE_CUTOFF_HOUR = 20
_CAPTURE_MINIMUMS = {"training_rows": 40, "evaluation_rows": 20, "sessions_per_split": 5}


def _feature_rows(
    bars: pd.DataFrame, returns: pd.DataFrame, cohort: Mapping,
    days: pd.DatetimeIndex, drift: pd.Series,
) -> pd.DataFrame:
    keys = pd.DataFrame(((day, symbol) for day in days for symbol in cohort[day]), columns=["date", "symbol"])
    keys["date"] = pd.to_datetime(keys["date"])
    log_returns = np.log1p(returns.where(returns.gt(-1) & np.isfinite(returns)))
    values = (
        returns,
        np.expm1(log_returns.rolling(5).sum()),
        np.expm1(log_returns.rolling(20).sum()),
        log_returns.rolling(20).std(),
    )
    row = days.get_indexer(keys["date"])
    column = returns.columns.get_indexer(keys["symbol"])
    present = column >= 0
    for name, wide in zip(_PRICE, values, strict=True):
        keys[name] = np.nan
        keys.loc[present, name] = wide.to_numpy()[row[present], column[present]]
    keys[_DRIFT] = keys["date"].map(drift)
    return keys.merge(daily_bar_features(bars, sessions=days), on=["date", "symbol"], how="left",
                      validate="one_to_one")


def _complete_rows(samples: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    finite = np.isfinite(samples[_REQUIRED])
    clean = samples.loc[finite.all(axis=1)].sort_values(["date", "symbol"])
    summary = {
        "candidate_rows": len(samples),
        "shared_rows": len(clean),
        "missing_or_nonfinite_by_field": {key: int(n) for key, n in (~finite).sum().items()},
    }
    if not clean.empty:
        summary.update(first_signal=clean["date"].min().date().isoformat(),
                       last_signal=clean["date"].max().date().isoformat(),
                       last_label_end=clean["label_end"].max().date().isoformat())
    return clean, summary


def _predictive_metrics(evaluation: pd.DataFrame, prediction: np.ndarray) -> dict:
    actual = evaluation["target_return"].to_numpy()
    daily_rank_ic = []
    pairs = evaluation[["date", "target_return"]].assign(prediction=prediction)
    for _, group in pairs.groupby("date", sort=False):
        ranks = group[["target_return", "prediction"]].rank()
        if ranks.nunique().gt(1).all():
            daily_rank_ic.append(ranks["target_return"].corr(ranks["prediction"]))
    return {
        "mae": float(np.mean(np.abs(prediction - actual))),
        "mean_daily_spearman": float(np.mean(daily_rank_ic)) if daily_rank_ic else None,
        "rank_ic_sessions": len(daily_rank_ic),
    }


def _compare_line(
    samples: pd.DataFrame, days: pd.DatetimeIndex, cutoff: pd.Timestamp,
    kind: str, flow_observations: pd.DataFrame | None, minute_observations: pd.DataFrame | None,
) -> dict:
    try:
        from sklearn.tree import DecisionTreeRegressor
    except ModuleNotFoundError as exc:
        raise ValueError("Feature study requires `uv sync --extra allocation` from research_v1.") from exc

    eligible_train = purge_training(samples, cutoff)
    train, train_summary = _complete_rows(eligible_train)
    evaluation, eval_summary = _complete_rows(samples.loc[samples["date"].ge(cutoff)])
    train_summary["purged_before_feature_filter"] = int(samples["date"].lt(cutoff).sum() - len(eligible_train))
    result = {"training": train_summary, "evaluation": eval_summary, "variants": {}}

    def evaluate(
        train_x: np.ndarray, eval_x: np.ndarray, *, sample_weight: np.ndarray | None = None,
        training_rows: pd.DataFrame = train, evaluation_rows: pd.DataFrame = evaluation,
    ) -> dict:
        model = DecisionTreeRegressor(**_MODEL)
        model.fit(train_x, training_rows["target_return"].to_numpy(), sample_weight=sample_weight)
        return _predictive_metrics(evaluation_rows, model.predict(eval_x))

    for name, label, observations, columns in (
        ("fund_flow", "flow", flow_observations, list(SOURCE_FIELDS.values())),
        ("intraday", "minute", minute_observations, ["minute_up_down_variance_ratio"]),
    ):
        if observations is None:
            continue

        def available_capture(common: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
            as_of = (common["date"] + pd.Timedelta(hours=_CAPTURE_CUTOFF_HOUR)).dt.tz_localize("Asia/Shanghai")
            joined = capture_features_asof(observations, common.assign(kind=kind, as_of=as_of), columns)
            finite = np.isfinite(joined[columns])
            matched = joined.loc[finite.all(axis=1)]
            return matched, {
                "shared_rows": len(common), f"{label}_complete_rows": len(matched),
                f"{label}_complete_sessions": int(matched["date"].nunique()),
                "coverage_fraction": len(matched) / len(common) if len(common) else None,
                "missing_or_nonfinite_by_field": {key: int(n) for key, n in (~finite).sum().items()},
            }

        covered_train, covered_train_summary = available_capture(train)
        covered_eval, covered_eval_summary = available_capture(evaluation)
        paired = {"status": "data_insufficient", "training": covered_train_summary,
                  "evaluation": covered_eval_summary}
        if (len(covered_train) >= _CAPTURE_MINIMUMS["training_rows"]
                and len(covered_eval) >= _CAPTURE_MINIMUMS["evaluation_rows"]
                and min(covered_train["date"].nunique(), covered_eval["date"].nunique())
                >= _CAPTURE_MINIMUMS["sessions_per_split"]):
            paired.update(status="evaluated", variants={
                method: evaluate(covered_train[fields].to_numpy(), covered_eval[fields].to_numpy(),
                                 training_rows=covered_train, evaluation_rows=covered_eval)
                for method, fields in (("price_baseline", list(_PRICE)), (f"plus_{name}", [*_PRICE, *columns]))
            })
        result[name] = paired
    if len(train) < 2 or evaluation.empty:
        return {**result, "status": "insufficient_shared_samples"}

    train_prices = train[list(_PRICE)].to_numpy()
    eval_prices = evaluation[list(_PRICE)].to_numpy()
    whitener = LowdinTransform().fit(train_prices)
    train_positions = days.get_indexer(train["date"])
    weights = time_decay_weights(train_positions.max() - train_positions, half_life=_HALF_LIFE)

    result["variants"] = {
        "price_baseline": evaluate(train_prices, eval_prices),
        "plus_daily_ohlc": evaluate(train[[*_PRICE, *_OHLC]].to_numpy(), evaluation[[*_PRICE, *_OHLC]].to_numpy()),
        "lowdin": {
            **evaluate(whitener.transform(train_prices), whitener.transform(eval_prices)),
            "effective_rank": whitener.effective_rank_, "discarded_rank": whitener.discarded_rank_,
        },
        "time_decay": evaluate(train_prices, eval_prices, sample_weight=weights),
        "plus_wasserstein": evaluate(train[[*_PRICE, _DRIFT]].to_numpy(), evaluation[[*_PRICE, _DRIFT]].to_numpy()),
    }
    return {**result, "status": "evaluated"}


def run_feature_study(
    frames: Mapping[str, pd.DataFrame], *, days: pd.DatetimeIndex, cutoff: pd.Timestamp,
    kinds: tuple[str, ...], origin: str, flow_observations: pd.DataFrame | None = None,
    minute_observations: pd.DataFrame | None = None,
) -> dict:
    if origin not in {"synthetic_fixture", "audited_market"}:
        raise ValueError("feature study requires an explicit fixture or audited market input")
    bars = frames["bars"].assign(date=pd.to_datetime(frames["bars"]["date"], errors="raise").dt.normalize())
    bars = bars.loc[bars["date"].isin(days)]
    actions = frames["actions"]
    returns = market_close_returns(bars, actions, days)
    drift = wasserstein_drift(returns.reindex(columns=ETF_SYMBOLS), reference_sessions=60, recent_sessions=10)
    lines = {}
    for kind in kinds:
        kind_bars = bars.loc[bars["kind"].eq(kind)]
        if kind == "stock":
            cohort = select_stock_cohort(bars, frames["membership"], days)
        else:
            kind_bars = kind_bars.loc[kind_bars["symbol"].isin(ETF_SYMBOLS)]
            cohort = dict.fromkeys(days, ETF_SYMBOLS)
        features = _feature_rows(kind_bars, returns, cohort, days, drift)
        for horizon in HORIZONS:
            labels = open_labels(kind_bars, days, horizon, actions)
            samples = features.merge(labels, on=["date", "symbol"], how="left", validate="one_to_one")
            lines[f"{kind}-{horizon}d"] = _compare_line(
                samples, days, cutoff, kind, flow_observations, minute_observations,
            )
    return {
        "origin": origin,
        "status": "engineering_only" if origin == "synthetic_fixture" else "validation_only",
        "scope": "synthetic engineering check" if origin == "synthetic_fixture" else "pre-holdout validation",
        "evaluation_start": cutoff.date().isoformat(),
        "calendar_start": days[0].date().isoformat(), "calendar_end": days[-1].date().isoformat(),
        "model": {"name": "DecisionTreeRegressor", **_MODEL},
        "sample_policy": "Same complete rows for all five variants within each line and split.",
        "label": "Corporate-action-aware return from next session open to the open one horizon later.",
        "purge_rule": "Training label_end strictly precedes evaluation_start.",
        "feature_metadata": {
            "price_baseline": {"columns": list(_PRICE), "volatility_basis": "20-session adjusted log returns"},
            "plus_daily_ohlc": {"added_columns": list(_OHLC), "variance_ratio_sessions": 20},
            "lowdin": {"transformed_columns": list(_PRICE), "fit_partition": "training only", "rcond": 1e-12},
            "time_decay": {"half_life_exchange_sessions": _HALF_LIFE, "fit_partition": "training only"},
            "plus_wasserstein": {"added_columns": [_DRIFT], "return_symbols": list(ETF_SYMBOLS),
                                 "reference_sessions": 60, "recent_sessions": 10, "minimum_finite_fraction": 0.8},
        },
        "lines": lines,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="run all four lines on the synthetic fixture")
    audited = commands.add_parser("audited", help="evaluate only audit-ready market snapshots before 2025")
    forward = commands.add_parser("forward", help=f"run a separate market study after {LAST_EXPLORED_SESSION}")
    forward.add_argument("--start", type=date.fromisoformat, required=True)
    forward.add_argument("--evaluation-start", type=date.fromisoformat, required=True)
    for command in (audited, forward):
        command.add_argument("snapshot", type=Path)
        command.add_argument("--end", type=date.fromisoformat, required=True)
        command.add_argument("--kind", choices=("all", "etf", "stock"), default="all")
        command.add_argument("--fund-flow-capture", type=Path, action="append", default=[])
        command.add_argument("--intraday-capture", type=Path, action="append", default=[])
    for command in (demo, audited, forward):
        command.add_argument("--out", type=Path, required=True, help="new JSON file for the results")
    args = parser.parse_args(argv)

    try:
        kinds = ("etf", "stock")
        flow_observations = minute_observations = None
        if args.command == "demo":
            from .demo import fixture

            origin, frames = "synthetic_fixture", fixture()
            days = sessions(frames["calendar"]["date"])
            cutoff = days[len(days) * 7 // 10]
        else:
            from .snapshot import audit_snapshot, file_sha256, read_snapshot

            is_forward = args.command == "forward"
            if is_forward:
                if not LAST_EXPLORED_SESSION < args.start < args.evaluation_start < args.end:
                    raise ValueError(f"forward study requires {LAST_EXPLORED_SESSION} < start < evaluation-start < end")
                period = ResearchPeriod(args.start, args.end)
                cutoff = pd.Timestamp(args.evaluation_start)
            else:
                if not VALIDATION_START <= args.end < HOLDOUT_START:
                    raise ValueError("audited feature study requires a validation end in 2023 or 2024")
                period = ResearchPeriod(RESEARCH_START, args.end)
                cutoff = pd.Timestamp(VALIDATION_START)
            manifest, frames = read_snapshot(args.snapshot)
            kinds = kinds if args.kind == "all" else (args.kind,)
            # The Wasserstein regime feature uses frozen ETF returns on every line.
            gates = [audit_snapshot(manifest, frames, period, kind, allow_forward_period=is_forward)
                     for kind in sorted({"etf", *kinds})]
            if any(gate["status"] != "ready" for gate in gates):
                print(json.dumps(gates, ensure_ascii=False, indent=2), file=sys.stderr)
                return 2
            origin = "audited_market"
            dates = pd.to_datetime(frames["calendar"]["date"], errors="raise")
            start, end = pd.Timestamp(period.start), pd.Timestamp(period.end)
            days = sessions(dates.loc[dates.between(start, end)])
            if args.fund_flow_capture:
                from .fund_flow import read_fund_flow

                flow_observations = pd.concat([read_fund_flow(path) for path in args.fund_flow_capture]).drop_duplicates()
                flow_observations = flow_observations.loc[flow_observations["date"].between(start, end)]
            if args.intraday_capture:
                from .intraday import read_intraday_capture

                minute_observations = pd.concat([read_intraday_capture(path) for path in args.intraday_capture]).drop_duplicates()
                minute_observations = minute_observations.loc[minute_observations["date"].between(start, end)]
        report = run_feature_study(frames, days=days, cutoff=cutoff, kinds=kinds, origin=origin,
                                   flow_observations=flow_observations, minute_observations=minute_observations)
        if args.command == "forward":
            report.update(status="forward_research_only", scope="forward-only market research",
                          forward_period={"start": period.start.isoformat(), "end": period.end.isoformat(),
                                          "last_explored_session": LAST_EXPLORED_SESSION.isoformat()})
        if args.command != "demo":
            report["snapshot_manifest_sha256"] = file_sha256(args.snapshot / "manifest.json")
            if args.fund_flow_capture:
                report["fund_flow_metadata"] = {
                    "cutoff": f"{_CAPTURE_CUTOFF_HOUR:02d}:00 Asia/Shanghai on the signal session",
                    "comparison": "Separate paired price-only and price-plus-flow fits on the same covered rows.",
                    "minimum_samples": _CAPTURE_MINIMUMS,
                    "capture_sha256": [file_sha256(path / "capture.json") for path in args.fund_flow_capture],
                }
            if args.intraday_capture:
                report["intraday_metadata"] = {
                    "cutoff": f"{_CAPTURE_CUTOFF_HOUR:02d}:00 Asia/Shanghai on the signal session",
                    "comparison": "Separate paired price-only and price-plus-minute fits on the same covered rows.",
                    "minimum_samples": _CAPTURE_MINIMUMS,
                    "capture_sha256": [file_sha256(path / "intraday.json") for path in args.intraday_capture],
                }
        content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as stream:
            stream.write(content)
        print(json.dumps({"origin": origin, "output": str(args.out.resolve())}, ensure_ascii=False))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
