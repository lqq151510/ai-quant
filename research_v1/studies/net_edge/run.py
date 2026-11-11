"""Compare a forecast-based one-lot entry filter with the same unfiltered limit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from dataclasses import asdict, replace
from datetime import date

import duckdb
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
RESEARCH_ROOT = HERE.parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))

from quant_research.baseline import market_close_returns
from quant_research.ledger import Costs, _fill_value, _money, opening_execution_price, simulate
from quant_research.protocol import ETF_SYMBOLS, HOLDOUT_START, MAX_DRAWDOWN, RESEARCH_START, ResearchPeriod
from quant_research.risk import summarize
from quant_research.snapshot import audit_snapshot, file_sha256
from quant_research.timeline import purge_training, sessions


def require_unsealed(end: object) -> date:
    end = pd.Timestamp(end).date()
    if end >= HOLDOUT_START:
        raise ValueError("sealed holdout starts 2025-01-01; this study cannot open it")
    return end


def load_market(snapshot: Path, *, end: object) -> tuple[dict, dict[str, pd.DataFrame]]:
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


def features(asset, actions, days, symbol):
    returns = market_close_returns(asset, actions, days)[symbol]
    logs = np.log1p(returns.where(returns.gt(-1)))
    data = {f"return_{n}": np.expm1(logs.rolling(n, min_periods=n).sum()) for n in (5, 20, 60)}
    data["volatility_20"] = returns.rolling(20, min_periods=20).std()
    return pd.DataFrame(data, index=days)


def forecasts(asset, actions, days, protocol):
    symbol = protocol["symbol"]
    start = pd.Timestamp(protocol["evaluation_start"])
    columns = protocol["features"]
    prices = asset.set_index("date").reindex(days)
    panel = features(asset, actions, days, symbol)
    prefix = days[days < start]
    prefix_features = features(
        asset.loc[asset["date"] <= prefix[-1]],
        actions.loc[actions["ex_date"] <= prefix[-1]],
        prefix, symbol,
    )
    pd.testing.assert_frame_equal(panel.loc[prefix], prefix_features, check_exact=True)
    panel["date"] = days
    panel["entry_date"] = panel["date"].shift(-1)
    panel["label_end"] = panel["date"].shift(-2)
    panel["limit"] = prices["close"]
    panel["observed_exit_open"] = prices["open"].shift(-2)
    panel["target_return"] = panel["observed_exit_open"] / panel["limit"] - 1
    usable = panel.dropna(subset=[*columns, "target_return"])
    decision_days = days[(days >= start) & (days <= days[-3])]
    validation = usable.loc[usable.index >= start]
    if not validation.index.equals(decision_days):
        raise ValueError("Incomplete decision window")
    predictions = []
    folds = []
    for quarter, fold in validation.groupby(validation.index.to_period("Q"), sort=True):
        cutoff = fold.index[0]
        train = purge_training(usable, cutoff)
        if train.empty:
            raise ValueError(f"No mature outcomes before {cutoff}")
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(train[columns], train["target_return"])
        prediction = fold[["date", "entry_date", "label_end", "limit", "observed_exit_open"]].copy()
        prediction["forecast_exit"] = fold["limit"] * (1 + model.predict(fold[columns]))
        if not np.isfinite(prediction["forecast_exit"]).all() or not prediction["forecast_exit"].gt(0).all():
            raise ValueError("Invalid forecast price")
        predictions.append(prediction)
        folds.append({
            "quarter": str(quarter), "evaluation_start": str(cutoff.date()),
            "evaluation_end": str(fold.index[-1].date()), "train_rows": len(train),
            "last_training_outcome": str(train["label_end"].max().date()),
        })
    return pd.concat(predictions, ignore_index=True), folds


def expected_net_at_cap(row, cost, quantity):
    exit_proxy = opening_execution_price(row.forecast_exit, cost, "sell")
    _, _, debit = _fill_value(row.entry_date.date(), "etf", "buy", quantity, row.limit, cost)
    _, _, credit = _fill_value(row.label_end.date(), "etf", "sell", quantity, exit_proxy, cost)
    return _money(debit + credit)


def measure(result, planned, cash):
    equity = result.equity
    error = (equity["nav"] - equity["cash"] - equity["market_value"] - equity["receivable"]).abs().max()
    if error > 1e-6 or equity["cash"].min() < 0:
        raise RuntimeError("Invalid cash or NAV reconciliation")
    orders = result.orders
    buys = orders.loc[orders["side"].eq("buy")]
    if len(buys) != planned:
        raise RuntimeError("Scheduled buys and order audit disagree")
    sells = orders.loc[orders["side"].eq("sell")]
    metrics = summarize(equity, orders, cash)
    metrics.update({
        "net_pnl": _money(metrics["final_nav"] - cash),
        "entry_attempts": planned,
        "filled_buys": int(buys["filled_qty"].gt(0).sum()),
        "filled_sells": int(sells["filled_qty"].gt(0).sum()),
        "buy_rejections": buys.loc[buys["filled_qty"].eq(0), "reason"].value_counts().to_dict(),
        "position_unresolved": bool(equity["market_value"].iloc[-1] > 0),
        "dividend_cash_paid": _money(result.cash_events["cash_delta"].sum()),
    })
    return metrics


def source_hashes():
    modules = ("baseline", "ledger", "protocol", "risk", "snapshot", "timeline")
    sources = [Path(__file__).resolve(), *(RESEARCH_ROOT / "quant_research" / f"{name}.py" for name in modules)]
    return {str(path.relative_to(RESEARCH_ROOT)): file_sha256(path) for path in sources}


def run(snapshot: Path, out: Path):
    if out.exists():
        raise FileExistsError(out)
    protocol_raw = (HERE / "protocol.json").read_bytes()
    protocol = json.loads(protocol_raw)
    sources = source_hashes()
    audit, data = load_market(snapshot, end=protocol["data_end"])
    symbol = protocol["symbol"]
    asset = data["bars"].loc[data["bars"]["symbol"].eq(symbol)].copy()
    actions = data["actions"].loc[data["actions"]["symbol"].eq(symbol)].copy()
    days = sessions(data["calendar"]["date"].sort_values())
    predictions, folds = forecasts(asset, actions, days, protocol)
    eval_days = days[days >= pd.Timestamp(protocol["evaluation_start"])]
    cash = protocol["initial_cash_virtual"]
    quantity = protocol["quantity"]
    normal = Costs()
    costs = {"normal": normal, "double": normal.stressed(), "proportional": replace(normal, minimum_commission=0)}
    report = {
        "protocol": protocol, "protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(),
        "snapshot_manifest_sha256": file_sha256(snapshot / "manifest.json"),
        "source_sha256": sources, "audit": audit, "folds": folds,
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "decision_count": len(predictions),
        "forecast_mean_abs_error_cny": float((predictions["forecast_exit"] - predictions["observed_exit_open"]).abs().mean()),
        "previous_close_mean_abs_error_cny": float((predictions["limit"] - predictions["observed_exit_open"]).abs().mean()),
        "costs": {}, "screen": {},
    }
    daily, orders, events, decision_audit = [], [], [], []
    original = dict(zip(predictions["entry_date"], predictions["limit"], strict=True))
    for name, cost in costs.items():
        decision = predictions.copy()
        decision_cost_name = "normal" if name == "double" else name
        decision_cost = costs[decision_cost_name]
        decision["expected_net_at_cap"] = [
            expected_net_at_cap(row, decision_cost, quantity) for row in decision.itertuples(index=False)
        ]
        decision["eligible"] = decision["expected_net_at_cap"].gt(protocol["minimum_expected_net_cny_exclusive"])
        allowed = {row.entry_date: row.limit for row in decision.loc[decision["eligible"]].itertuples(index=False)}
        group = {
            "assumptions": asdict(cost), "decision_cost_scenario": decision_cost_name,
            "candidate_signals": len(allowed), "signals_filtered": len(original) - len(allowed),
            "expected_net_min": float(decision["expected_net_at_cap"].min()),
            "expected_net_max": float(decision["expected_net_at_cap"].max()),
            "strategies": {},
        }
        for strategy, entries in (("prior_close_limit", original), ("forecast_filter", allowed), ("cash", {})):
            result = simulate(asset, eval_days, {}, actions, initial_cash=cash, cost=cost, entry_limits=entries)
            group["strategies"][strategy] = measure(result, len(entries), cash)
            key = {"cost_scenario": name, "strategy": strategy}
            daily.append(result.equity.assign(**key))
            if not result.orders.empty:
                orders.append(result.orders.assign(**key))
            if not result.cash_events.empty:
                events.append(result.cash_events.assign(**key))
            if strategy != "cash":
                buys = result.orders.loc[result.orders["side"].eq("buy")].set_index("date")
                buy_status = buys["status"].where(buys["filled_qty"].gt(0), buys["reason"])
                mapped_status = decision["entry_date"].dt.date.map(buy_status).astype("string")
                decision[f"{strategy}_buy_status"] = mapped_status.fillna("filtered")
        baseline = group["strategies"]["prior_close_limit"]
        candidate = group["strategies"]["forecast_filter"]
        group["candidate_incremental_pnl"] = _money(candidate["net_pnl"] - baseline["net_pnl"])
        group["baseline_fills_on_filtered_days"] = int(
            (decision["prior_close_limit_buy_status"].eq("filled") & ~decision["eligible"]).sum()
        )
        report["costs"][name] = group
        if name in {"normal", "double"}:
            report["screen"][name] = {
                "positive_net": candidate["net_pnl"] > 0,
                "beats_prior_close_limit": candidate["net_pnl"] > baseline["net_pnl"],
                "drawdown_within_limit": candidate["max_drawdown"] <= MAX_DRAWDOWN,
                "all_positions_resolved": not candidate["position_unresolved"],
            }
        decision_audit.append(decision.assign(cost_scenario=name))
    report["passed_historical_screen"] = all(all(checks.values()) for checks in report["screen"].values())
    if sources != source_hashes() or (HERE / "protocol.json").read_bytes() != protocol_raw:
        raise RuntimeError("Study source or protocol changed during replay")
    out.mkdir(parents=True, exist_ok=False)
    for name, tables in (("daily", daily), ("orders", orders), ("cash_events", events), ("decisions", decision_audit)):
        if tables:
            pd.concat(tables, ignore_index=True).to_csv(out / f"{name}.csv", index=False)
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"historical_screen_passed": report["passed_historical_screen"], "decision_count": len(predictions)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True, help="Verified local ETF snapshot directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.snapshot, args.out)
