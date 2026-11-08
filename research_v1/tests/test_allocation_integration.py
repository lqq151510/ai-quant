import json
from itertools import product

import pandas as pd
import pytest

from quant_research import allocation, baseline
from quant_research.cli import main
from quant_research.demo import fixture
from quant_research.protocol import CAPITALS, RESEARCH_LINES
from quant_research.snapshot import write_snapshot
from quant_research.timeline import select_stock_cohort, sessions

STRATEGIES = {"allocation_equal", "allocation_inverse_vol", "allocation_hrp"}


def test_allocation_decisions_use_pre_window_history_and_monthly_cohort(monkeypatch):
    pytest.importorskip("scipy")
    pytest.importorskip("sklearn")
    frames = fixture()
    calendar = sessions(frames["calendar"]["date"])
    days = calendar[80:86]
    membership = pd.concat([
        frames["membership"],
        pd.DataFrame({
            "symbol": ["600000", "600001"],
            "effective_date": pd.Timestamp("2024-05-01"),
            "known_at": pd.to_datetime(["2024-04-30T14:00:00+08:00", "2024-05-01T16:00:00+08:00"]),
        }),
    ], ignore_index=True)
    bars = frames["bars"]
    cohort = select_stock_cohort(bars, membership, calendar)
    returns = baseline.market_close_returns(bars, frames["actions"], calendar)

    inputs = {}
    allocate = allocation.allocation_targets

    def capture(scores, history, kind):
        inputs[history.index[-1]] = (scores, history)
        return allocate(scores, history, kind)

    monkeypatch.setattr(allocation, "allocation_targets", capture)
    decisions = baseline.build_allocation_decisions(days, cohort, returns, "stock", 5)
    assert list(decisions) == [days[0], days[-1]]
    assert set(decisions[days[0]]["allocation_equal"])
    assert len(inputs[days[0]][0]) == 12
    assert list(inputs[days[-1]][0].index) == ["600000"]
    for day, (_, history) in inputs.items():
        assert len(history) == 60
        assert history.index[0] < days[0]
        pd.testing.assert_frame_equal(history, returns.loc[:day].tail(60))

    later = returns.copy()
    later.loc[later.index > days[-1]] = -0.95
    assert baseline.build_allocation_decisions(days, cohort, later, "stock", 5) == decisions


def test_allocation_study_compares_four_lines_through_the_same_ledger(tmp_path, monkeypatch):
    pytest.importorskip("scipy")
    pytest.importorskip("sklearn")
    frames = fixture()
    calendar = sessions(frames["calendar"]["date"])
    days = calendar[71:77]
    schedules = []
    simulate = baseline.simulate

    def capture(bars, calendar, targets, *args, **kwargs):
        schedules.append(targets)
        return simulate(bars, calendar, targets, *args, **kwargs)

    monkeypatch.setattr(baseline, "simulate", capture)
    results = baseline.run_baselines(
        frames, tmp_path, start=days[0], end=days[-1], origin="synthetic_fixture", allocation_study=True,
    )
    plan = json.loads((tmp_path / "allocation_plan.json").read_text())
    assert plan["origin"] == "synthetic_fixture"
    assert plan["gross_budget_fraction"] == 0.80
    assert set(plan["lines"]) == set(RESEARCH_LINES)
    assert len(results) == len(schedules) == 112

    actual_runs = set()
    for row, schedule in zip(results, schedules, strict=True):
        strategy = row["strategy"]
        if strategy not in STRATEGIES:
            assert "allocation_target_metrics" not in row
            continue
        line = row["line"]
        actual_runs.add((line, row["capital"], row["cost_case"], strategy))
        assert row["ledger_reconciliation_error"] < 1e-6
        assert row["min_cash"] >= 0
        assert row["traded_notional_over_average_nav"] > 0

        # The first decision is Friday at the close, so the first open is Monday.
        assert list(schedule) == [days[1]]
        metrics = row["allocation_target_metrics"]
        assert metrics["execution_sessions"] == metrics["invested_execution_sessions"] == 1
        assert 0 < metrics["mean_invested_hhi"] <= 1
        assert metrics["max_single_name_weight"] == max(schedule[days[1]].values())
        assert metrics["mean_gross_weight"] == pytest.approx(sum(schedule[days[1]].values()))

        latest = plan["lines"][line]
        assert latest["as_of_session"] == days[-1].date().isoformat()
        assert set(latest["strategies"]) == STRATEGIES
        target = latest["strategies"][strategy]
        assert sum(target["targets"].values()) + target["cash_weight"] == pytest.approx(1)
        if line.endswith("-5d"):
            assert latest["status"] == "pending_outside_window"
            assert latest["rebalance_sessions"] == 5
            assert latest["decision_session"] == days[-1].date().isoformat()
            assert latest["next_exchange_session"] == calendar[77].date().isoformat()
        else:
            assert latest["status"] == "historical_decision"
            assert latest["rebalance_sessions"] == 20
            assert latest["decision_session"] == days[0].date().isoformat()
            assert latest["next_exchange_session"] == days[1].date().isoformat()
            assert target["targets"] == schedule[days[1]]

    assert actual_runs == set(product(RESEARCH_LINES, CAPITALS, ("normal", "double_friction"), STRATEGIES))
    order_dir = tmp_path / "ledgers" / "etf-5d-100000-normal-allocation_equal"
    orders = pd.read_parquet(order_dir / "orders.parquet")
    assert set(pd.to_datetime(orders["date"])) == {days[1]}
    filled = orders.loc[orders["filled_qty"] > 0].iloc[0]
    bar = frames["bars"].loc[frames["bars"]["date"].eq(days[1]) & frames["bars"]["symbol"].eq(filled.symbol)].iloc[0]
    assert filled.price == pytest.approx(bar.open * (1 + baseline.Costs().slippage_bps / 10_000))


def test_allocation_flag_keeps_market_and_holdout_gates(tmp_path, capsys):
    snapshot = tmp_path / "fixture"
    write_snapshot(snapshot, origin="synthetic_fixture", frames=fixture(), sources=[])
    output = tmp_path / "results"
    args = ["baseline", str(snapshot), "--kind", "etf", "--out", str(output), "--allocation-study"]

    assert main([*args, "--end", "2025-01-01"]) == 2
    assert "sealed test interval" in capsys.readouterr().err
    assert not output.exists()

    assert main([*args, "--end", "2024-12-31"]) == 2
    result, = json.loads(capsys.readouterr().out)
    assert result["status"] == "data_insufficient"
    assert "synthetic fixtures are excluded from market evidence" in result["issues"]
    assert not output.exists()


def test_missing_allocation_extra_leaves_output_available_for_retry(tmp_path, monkeypatch, capsys):
    def missing_dependencies():
        raise ValueError("Run uv sync --extra allocation")

    monkeypatch.setattr(allocation, "require_allocation_dependencies", missing_dependencies)
    output = tmp_path / "results"

    assert main(["demo", "--out", str(output), "--allocation-study"]) == 2
    assert "uv sync --extra allocation" in capsys.readouterr().err
    assert not output.exists()
