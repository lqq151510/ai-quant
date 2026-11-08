from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from quant_research import intraday
from quant_research.snapshot import file_sha256

OBSERVED = "2024-04-02T08:00:00+00:00"


def _session(day="2024-04-02"):
    morning = pd.date_range(f"{day} 09:31", f"{day} 11:30", freq="min")
    afternoon = pd.date_range(f"{day} 13:01", f"{day} 15:00", freq="min")
    morning_price = 10 * np.exp(np.r_[0, 0.02, np.full(118, 0.01)])
    return pd.DataFrame({
        "时间": morning.append(afternoon).strftime("%Y-%m-%d %H:%M:%S"),
        "收盘": np.r_[morning_price, np.full(120, 1000.0)],
    })


def _summary(frame, observed_at_utc=OBSERVED, started_at_utc=None):
    return intraday.summarize_sessions(
        frame, kind="stock", symbol="600000", started_at_utc=started_at_utc or observed_at_utc,
        observed_at_utc=observed_at_utc,
    )


def test_regular_segments_exclude_opening_anchor_and_lunch_and_overnight_jumps():
    first, second = _session("2024-04-01"), _session()
    second["收盘"] *= 2000
    anchor = pd.DataFrame({"时间": ["2024-04-02 09:30:00", "2024-04-02 13:00:00"], "收盘": [0, 0]})
    rows = _summary(pd.concat([second, anchor, first], ignore_index=True))
    assert [row["session_date"] for row in rows] == ["2024-04-01", "2024-04-02"]
    assert rows[-1]["ignored_anchor_rows"] == 2
    assert rows[-1]["first_regular_bar_at"] == "2024-04-02T09:31:00+08:00"
    assert rows[-1]["last_regular_bar_at"] == "2024-04-02T15:00:00+08:00"
    for row in rows:
        assert row["status"] == "complete"
        assert row["capture_observed_at_utc"] == OBSERVED
        assert row["return_count"] == 238
        assert row["minute_coverage"] == 1
        assert row["realized_up_variance"] == pytest.approx(0.0004)
        assert row["realized_down_variance"] == pytest.approx(0.0001)
        assert row["realized_up_down_variance_ratio"] == pytest.approx(4)


@pytest.mark.parametrize("case", ["gap", "duplicate", "bad_price", "not_yet_closed", "straddles_close"])
def test_incomplete_minutes_keep_coverage_but_omit_variance(case):
    frame = _session()
    observed = OBSERVED
    started = None
    if case == "gap":
        frame = frame.drop(index=55)
    elif case == "duplicate":
        frame = pd.concat([frame, frame.iloc[[55]]], ignore_index=True)
    elif case == "bad_price":
        frame.loc[55, "收盘"] = 0
    elif case == "not_yet_closed":
        observed = "2024-04-02T06:59:59+00:00"
    else:
        started = "2024-04-02T06:59:59+00:00"
    row, = _summary(frame, observed, started)
    assert row["observed_minutes"] == (239 if case == "gap" else 240)
    assert row["missing_minutes"] == (1 if case == "gap" else 0)
    assert row["duplicate_regular_rows"] == (1 if case == "duplicate" else 0)
    assert row["invalid_regular_prices"] == (1 if case == "bad_price" else 0)
    assert row["status"] == "incomplete_session"
    assert row["return_count"] == 0
    assert row["realized_up_variance"] is row["realized_down_variance"] is None
    assert row["realized_up_down_variance_ratio"] is None


def test_no_downside_preserves_complete_coverage_and_returns_unknown_ratio():
    frame = _session()
    frame["收盘"] = np.arange(240) + 10
    row, = _summary(frame)
    assert row["complete_grid"] is True
    assert row["status"] == "zero_downside"
    assert row["realized_up_variance"] > 0
    assert row["realized_down_variance"] == 0
    assert row["realized_up_down_variance_ratio"] is None


def test_probe_writes_verified_capture_only(tmp_path, monkeypatch):
    def collect(request, raw):
        assert request.provider == "akshare_eastmoney"
        assert request.arguments == {"symbol": request.key.split("_")[1], "period": "1", "adjust": ""}
        assert request.endpoint in ("stock_zh_a_hist_min_em", "fund_etf_hist_min_em")
        file = raw / f"{request.key}.csv"
        _session().to_csv(file, index=False)
        return {
            "request": asdict(request), "status": "captured", "started_at_utc": OBSERVED, "observed_at_utc": OBSERVED,
            "files": [{"path": f"raw/{file.name}", "rows": 240,
                       "sha256": file_sha256(file) if request.key.startswith("stock") else "wrong-hash"}],
        }

    monkeypatch.setattr(intraday, "_collect", collect)
    output = tmp_path / "output"
    result = intraday.run_intraday(output, stock="600000", etf="510300")

    assert result["status"] == "sample_available"
    assert result["acquisition_status"] == "partial"
    stock, etf = result["sources"]
    assert stock["validation"]["status"] == "verified_capture"
    assert etf["validation"] == {"status": "rejected", "reason": "captured minute file hash mismatch"}
    assert len(result["sessions"]) == 1
    parquet = output / result["parquet"]["path"]
    assert result["parquet"]["sha256"] == file_sha256(parquet)
    assert pd.read_parquet(parquet).iloc[0]["capture_observed_at_utc"] == OBSERVED
    feature, = intraday.read_intraday_capture(output).itertuples(index=False)
    assert feature.date == pd.Timestamp("2024-04-02")
    assert (feature.symbol, feature.kind) == ("600000", "stock")
    assert feature.known_at == pd.Timestamp(OBSERVED)
    assert feature.minute_up_down_variance_ratio == pytest.approx(4)

    parquet.write_bytes(parquet.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        intraday.read_intraday_capture(output)


def test_probe_skips_same_provider_after_rate_limit(tmp_path, monkeypatch):
    calls = []

    def rate_limit(request, raw):
        calls.append(request.key)
        return {
            "request": asdict(request), "status": "failed", "observed_at_utc": OBSERVED,
            "error": {"type": "Push2RateLimitError"}, "files": [],
        }

    monkeypatch.setattr(intraday, "_collect", rate_limit)
    output = tmp_path / "output"
    result = intraday.run_intraday(output, stock="600000", etf="510300")

    assert calls == ["stock_600000_minutes"]
    assert result["status"] == "data_insufficient"
    assert result["acquisition_status"] == "failed"
    assert result["sources"][1]["status"] == "skipped_after_rate_limit"
    assert pd.read_parquet(output / "sessions.parquet").empty
    features = intraday.read_intraday_capture(output)
    assert features.empty
    assert features.columns.tolist() == ["date", "symbol", "kind", "known_at", "minute_up_down_variance_ratio"]
    assert str(features["known_at"].dt.tz) == "UTC"


@pytest.mark.parametrize(("stock", "etf"), [("510300", "159919"), ("600000", "600519")])
def test_cross_kind_symbol_is_rejected_before_capture(tmp_path, stock, etf):
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="six-digit code"):
        intraday.run_intraday(output, stock=stock, etf=etf)
    assert not output.exists()
