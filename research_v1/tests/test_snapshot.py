import pandas as pd
import pytest
from datetime import timedelta

from quant_research.demo import fixture
from quant_research.protocol import LAST_EXPLORED_SESSION, RESEARCH_START, ResearchPeriod
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


def test_provisional_etf_limits_cannot_be_promoted_by_manifest_label():
    frames = fixture()
    frames["bars"]["limit_source"] = "rule_derived_candidate"
    manifest = {"origin": "market", "sources": [{}], "coverage": {"etf_limits": "verified"}}
    audit = audit_snapshot(manifest, frames, ResearchPeriod(RESEARCH_START, pd.Timestamp("2024-12-31").date()), "etf")
    assert audit["status"] == "data_insufficient"
    assert "provisional price limits cannot satisfy verified coverage" in audit["issues"]


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


def test_forward_audit_requires_opt_in_and_preserves_source_gate():
    manifest = {"origin": "synthetic_fixture", "sources": []}
    start = LAST_EXPLORED_SESSION + timedelta(days=1)
    period = ResearchPeriod(start, start + timedelta(days=30))
    assert "start differs from pre-registered study window" in audit_snapshot(manifest, fixture(), period, "etf")["issues"]
    forward = audit_snapshot(manifest, fixture(), period, "etf", allow_forward_period=True)
    assert "start differs from pre-registered study window" not in forward["issues"]
    assert "synthetic fixtures are excluded from market evidence" in forward["issues"]
    past = ResearchPeriod(LAST_EXPLORED_SESSION, start)
    assert "forward period must begin after the last explored session" in audit_snapshot(
        manifest, fixture(), past, "etf", allow_forward_period=True,
    )["issues"]
