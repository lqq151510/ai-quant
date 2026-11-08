import json

import numpy as np
import pandas as pd
import pytest

from quant_research.demo import fixture
from quant_research.feature_study import main, run_feature_study
from quant_research.protocol import ETF_SYMBOLS
from quant_research.snapshot import write_snapshot
from quant_research.timeline import sessions


def test_synthetic_study_scores_five_variants_for_each_purged_line(tmp_path):
    pytest.importorskip("scipy")
    pytest.importorskip("sklearn")
    output = tmp_path / "study.json"
    assert main(["demo", "--out", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["origin"] == "synthetic_fixture"
    assert set(report["lines"]) == {"etf-5d", "etf-20d", "stock-5d", "stock-20d"}
    assert report["feature_metadata"]["time_decay"]["half_life_exchange_sessions"] == 120
    for line in report["lines"].values():
        assert line["status"] == "evaluated"
        assert line["training"]["last_label_end"] < report["evaluation_start"]
        assert line["evaluation"]["first_signal"] >= report["evaluation_start"]
        assert line["training"]["purged_before_feature_filter"] > 0
        assert line["training"]["shared_rows"] > 0 and line["evaluation"]["shared_rows"] > 0
        assert set(line["variants"]) == set(report["feature_metadata"])
        assert all(np.isfinite(metrics["mae"]) for metrics in line["variants"].values())


def test_audited_cli_rejects_synthetic_snapshot_and_sealed_interval(tmp_path, capsys):
    snap = tmp_path / "snapshot"
    out = tmp_path / "study.json"
    write_snapshot(snap, origin="synthetic_fixture", frames=fixture(), sources=[])
    assert main(["audited", str(snap), "--end", "2024-12-31", "--out", str(out)]) == 2
    assert "synthetic fixtures are excluded" in capsys.readouterr().err
    assert main(["audited", str(snap), "--end", "2025-01-01", "--out", str(out)]) == 2
    assert "2023 or 2024" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize("start, evaluation_start, end", [
    ("2025-01-01", "2026-12-01", "2027-01-01"),
    ("2026-11-06", "2026-12-01", "2027-01-01"),
    ("2026-11-09", "2026-11-09", "2027-01-01"),
    ("2026-11-09", "2027-01-01", "2027-01-01"),
])
def test_forward_cli_rejects_invalid_boundaries_before_reading_snapshot(tmp_path, capsys, start, evaluation_start, end):
    out = tmp_path / "study.json"
    assert main(["forward", str(tmp_path / "missing"), "--start", start, "--evaluation-start", evaluation_start,
                 "--end", end, "--out", str(out)]) == 2
    assert "2026-11-06 < start < evaluation-start < end" in capsys.readouterr().err
    assert not out.exists()


def test_forward_cli_rejects_synthetic_snapshot(tmp_path, capsys):
    snap = tmp_path / "snapshot"
    out = tmp_path / "study.json"
    write_snapshot(snap, origin="synthetic_fixture", frames=fixture(), sources=[])
    assert main(["forward", str(snap), "--start", "2026-11-09", "--evaluation-start", "2027-02-01",
                 "--end", "2027-05-01", "--kind", "etf", "--out", str(out)]) == 2
    assert "synthetic fixtures are excluded" in capsys.readouterr().err
    assert not out.exists()


def test_forward_cli_keeps_forward_dates_and_enables_paired_captures(tmp_path, monkeypatch):
    pytest.importorskip("scipy")
    pytest.importorskip("sklearn")
    from quant_research.fund_flow import SOURCE_FIELDS

    frames = fixture()
    offset = pd.Timedelta(weeks=157)
    for name in ("calendar", "bars"):
        frames[name]["date"] += offset
    for column in ("record_date", "ex_date", "pay_date"):
        frames["actions"][column] += offset
    old = frames["bars"].iloc[[0, 0]].assign(date=pd.Timestamp("2025-01-02"))
    frames["bars"] = pd.concat([old, frames["bars"]], ignore_index=True)
    days = sessions(frames["calendar"]["date"])
    frames["calendar"] = pd.concat([pd.DataFrame({"date": [old["date"].iloc[0]]}), frames["calendar"]])
    start, cutoff, end = (days[i].date().isoformat() for i in (1, 128, -2))
    snap = tmp_path / "snapshot"
    write_snapshot(snap, origin="market", frames=frames, sources=[{"test_fixture": True}], coverage=dict.fromkeys(
        ("exchange_calendar", "etf_raw_bars", "etf_actions", "etf_limits"), "verified",
    ))
    flow = pd.DataFrame(((day, symbol) for day in days for symbol in ETF_SYMBOLS), columns=["date", "symbol"])
    flow["kind"] = "etf"
    flow["known_at"] = (flow["date"] + pd.Timedelta(hours=19)).dt.tz_localize("Asia/Shanghai")
    for field in SOURCE_FIELDS.values():
        flow[field] = np.sin(np.arange(len(flow)))
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "capture.json").write_text("{}")
    monkeypatch.setattr("quant_research.fund_flow.read_fund_flow", lambda _: flow)
    minutes = flow[["date", "symbol", "kind", "known_at"]].assign(
        minute_up_down_variance_ratio=1.0 + np.square(np.sin(np.arange(len(flow)))),
    )
    (capture / "intraday.json").write_text("{}")
    monkeypatch.setattr("quant_research.intraday.read_intraday_capture", lambda _: minutes)

    output = tmp_path / "forward.json"
    assert main(["forward", str(snap), "--start", start, "--evaluation-start", cutoff, "--end", end,
                 "--kind", "etf", "--fund-flow-capture", str(capture), "--intraday-capture", str(capture),
                 "--out", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["status"] == "forward_research_only"
    assert (report["calendar_start"], report["evaluation_start"], report["calendar_end"]) == (start, cutoff, end)
    for line in report["lines"].values():
        assert line["training"]["first_signal"] >= start
        assert line["training"]["last_label_end"] < cutoff
        assert line["evaluation"]["first_signal"] >= cutoff
        assert line["evaluation"]["last_label_end"] <= end
        assert line["fund_flow"]["status"] == "evaluated"
        assert set(line["intraday"]["variants"]) == {"price_baseline", "plus_intraday"}


def test_fund_flow_is_paired_separately_and_late_versions_stay_unavailable():
    pytest.importorskip("scipy")
    pytest.importorskip("sklearn")
    from quant_research.fund_flow import SOURCE_FIELDS

    frames = fixture()
    days = sessions(frames["calendar"]["date"])
    options = {"days": days, "cutoff": days[len(days) * 7 // 10], "kinds": ("etf",), "origin": "synthetic_fixture"}
    decisions = pd.DataFrame(((day, symbol) for day in days for symbol in ETF_SYMBOLS), columns=["date", "symbol"])
    decisions["kind"] = "etf"
    for offset, field in enumerate(SOURCE_FIELDS.values()):
        decisions[field] = np.sin(np.arange(len(decisions)) / 17 + offset)
    decisions["known_at"] = (decisions["date"] + pd.Timedelta(hours=21)).dt.tz_localize("Asia/Shanghai")
    late = run_feature_study(frames, **options, flow_observations=decisions)["lines"]
    decisions["known_at"] -= pd.Timedelta(hours=2)
    early = run_feature_study(frames, **options, flow_observations=decisions)["lines"]
    for line, original in late.items():
        assert original["fund_flow"]["status"] == "data_insufficient"
        assert original["fund_flow"]["training"]["flow_complete_rows"] == 0
        assert early[line]["fund_flow"]["status"] == "evaluated"
        assert early[line]["variants"] == original["variants"]
        assert early[line]["training"] == original["training"]
        assert set(early[line]["fund_flow"]["variants"]) == {"price_baseline", "plus_fund_flow"}
