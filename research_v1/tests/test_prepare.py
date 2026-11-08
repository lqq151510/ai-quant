import json
from datetime import date

import pandas as pd
import pytest

from quant_research.prepare import _resolve_split_ex_date, _round_mill, dividend_per_share, prepare_etf_snapshot
from quant_research.protocol import ETF_SYMBOLS, RESEARCH_START, ResearchPeriod
from quant_research.snapshot import audit_snapshot, file_sha256, read_snapshot


def captured_csvs(tmp_path):
    probe, calendar = tmp_path / "probe", tmp_path / "calendar"
    (probe / "raw").mkdir(parents=True)
    (calendar / "raw").mkdir(parents=True)
    dates = pd.bdate_range("2015-01-01", periods=5)
    sessions = calendar / "raw" / "calendar.csv"
    pd.DataFrame({"trade_date": dates}).to_csv(sessions, index=False)
    (calendar / "audit.json").write_text(json.dumps({"sha256": file_sha256(sessions)}))
    sources = []
    for symbol in ETF_SYMBOLS:
        path = probe / "raw" / f"{symbol}.csv"
        pd.DataFrame(dict(date=dates, open=10., high=11., low=9., close=10., volume=10000,
                          amount=100000)).to_csv(path, index=False)
        sources.append({"request": {"key": f"etf_bars_{symbol}"},
                        "files": [{"path": f"raw/{symbol}.csv", "sha256": file_sha256(path)}]})
    (probe / "audit.json").write_text(json.dumps({"origin": "market_provider_probe", "sources": sources}))
    return probe, sessions, ResearchPeriod(RESEARCH_START, date(2015, 1, 7))


def test_preparation_does_not_promote_unverified_actions(tmp_path):
    probe, calendar, period = captured_csvs(tmp_path)
    out = tmp_path / "snapshot"
    prepare_etf_snapshot(probe, calendar, out, period)
    manifest, frames = read_snapshot(out)
    assert manifest["origin"] == "market"
    assert len(frames["bars"]) == 20
    audit = audit_snapshot(manifest, frames, period, "etf")
    assert audit["status"] == "data_insufficient"
    assert "etf_limits: not independently verified" in audit["issues"]


def test_preparation_checks_captured_csv_hashes(tmp_path):
    probe, calendar, period = captured_csvs(tmp_path)
    with (probe / "raw" / "510300.csv").open("a") as f:
        f.write("tampering\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare_etf_snapshot(probe, calendar, tmp_path / "out", period)


@pytest.mark.parametrize("raw", ["每10份派现金0.8000", "0.8000元", 0.8])
def test_real_vendor_per_ten_units_distribution_format(raw):
    assert dividend_per_share(raw) == pytest.approx(0.08)


def test_unknown_distribution_units_never_turn_into_cash():
    for raw in ("每100份派现金0.8000", "待公布", float("nan")):
        with pytest.raises(ValueError, match="unrecognized"):
            dividend_per_share(raw)


def test_split_record_date_maps_to_next_exchange_session_and_mill_rounding():
    sessions = pd.DatetimeIndex(pd.to_datetime(["2015-04-13", "2015-04-14", "2015-04-15", "2022-08-26", "2022-08-29"]))
    assert _resolve_split_ex_date(pd.Timestamp("2015-04-14"), sessions) == pd.Timestamp("2015-04-15")
    assert _resolve_split_ex_date(pd.Timestamp("2022-08-26"), sessions) == pd.Timestamp("2022-08-29")
    assert _round_mill(8.001 * 1.1) == pytest.approx(8.801)
    assert _round_mill(8.001 * 0.9) == pytest.approx(7.201)
