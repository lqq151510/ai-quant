import json
from datetime import date

import pandas as pd
import pytest

from quant_research.demo import fixture
from quant_research.protocol import RESEARCH_START, ResearchPeriod
from quant_research.snapshot import audit_snapshot, read_snapshot, write_snapshot


def test_immutable_snapshot_and_tamper_detection(tmp_path):
    path = tmp_path / "snapshot"
    write_snapshot(path, origin="synthetic_fixture", frames=fixture(), sources=[])
    manifest, frames = read_snapshot(path)
    assert manifest["origin"] == "synthetic_fixture"
    assert len(frames["bars"]) > 0
    with pytest.raises(FileExistsError):
        write_snapshot(path, origin="synthetic_fixture", frames=fixture(), sources=[])
    file = path / "bars.parquet"
    with file.open("ab") as output:
        output.write(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_snapshot(path)


def test_bounded_read_filters_before_materialization_and_checks_full_counts(tmp_path, monkeypatch):
    path = tmp_path / "snapshot"
    dates = pd.to_datetime(["2024-12-31", "2025-01-01"])
    timestamps = ["2024-12-31T23:59:59+08:00", "2024-12-31T16:00:00Z"]
    data = {
        "bars": pd.DataFrame({"date": dates}),
        "calendar": pd.DataFrame({"date": dates}),
        "actions": pd.DataFrame({"ex_date": dates}),
        "membership": pd.DataFrame({"effective_date": [dates[0]] * 2, "known_at": timestamps}),
        "securities": pd.DataFrame({"listed_at": dates}),
        "news": pd.DataFrame({"ready_at": timestamps}),
        "edges": pd.DataFrame({"known_at": timestamps, "valid_from": timestamps}),
    }
    manifest = write_snapshot(path, origin="synthetic_fixture", frames=data, sources=[])

    def forbid_full_read(*args, **kwargs):
        raise AssertionError("bounded path must not use the unfiltered reader")

    monkeypatch.setattr(pd, "read_parquet", forbid_full_read)
    _, frames = read_snapshot(path, end=date(2024, 12, 31))
    assert all(len(frame) == 1 for frame in frames.values())
    assert frames["news"]["ready_at"].tolist() == [timestamps[0]]
    manifest["tables"]["bars"]["rows"] = 1
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="row count mismatch: bars"):
        read_snapshot(path, end=date(2024, 12, 31))


def test_fixture_and_unverified_market_are_not_market_evidence(tmp_path):
    path = tmp_path / "snapshot"
    manifest = write_snapshot(path, origin="synthetic_fixture", frames=fixture(), sources=[])
    period = ResearchPeriod(RESEARCH_START, pd.Timestamp("2026-11-06").date())
    for kind in ("stock", "etf"):
        audit = audit_snapshot(manifest, fixture(), period, kind)
        assert audit["status"] == "data_insufficient"
        assert "synthetic fixtures are excluded from market evidence" in audit["issues"]
    manifest["origin"] = "market"
    assert audit_snapshot(manifest, fixture(), period, "etf")["status"] == "data_insufficient"


def test_market_requires_provenance(tmp_path):
    with pytest.raises(ValueError, match="source evidence"):
        write_snapshot(tmp_path / "market", origin="market", frames=fixture(), sources=[])


def test_verified_labels_cannot_hide_future_membership_or_missing_daily_bars():
    from datetime import date

    frames = fixture()
    frames["calendar"] = pd.DataFrame({"date": pd.bdate_range("2015-01-01", "2015-04-30")})
    frames["bars"] = frames["bars"].iloc[:1].assign(kind="stock", symbol="600000", date=pd.Timestamp("2015-01-05"))
    frames["membership"] = pd.DataFrame({"symbol": ["600000"], "effective_date": ["2020-01-01"],
                                          "known_at": ["2020-01-01T00:00:00Z"]})
    frames["securities"] = pd.DataFrame({"symbol": ["600000"], "listed_at": ["1999-01-01"],
                                         "delisted_at": [None], "board": ["main"]})
    manifest = {"origin": "market", "sources": [{"provider": "test"}],
                "coverage": {k: "verified" for k in ("exchange_calendar", "stock_raw_bars", "stock_actions",
                                                        "stock_limits", "stock_membership")}}
    audit = audit_snapshot(manifest, frames, ResearchPeriod(RESEARCH_START, date(2015, 4, 30)), "stock")
    assert audit["status"] == "data_insufficient"
    assert "no point-in-time membership for one or more study sessions" in audit["issues"]


def test_security_metadata_schema_cannot_be_replaced_by_coverage_flag():
    from datetime import date

    frames = fixture()
    frames["securities"] = pd.DataFrame({"unrelated": [1]})
    audit = audit_snapshot({"origin": "market", "sources": [{}], "coverage": {}}, frames,
                           ResearchPeriod(RESEARCH_START, date(2024, 12, 31)), "stock")
    assert "listing/delisting metadata missing or invalid schema" in audit["issues"]
