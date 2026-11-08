"""Capture a bounded one-minute provider sample and measure session semivariance."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from .providers import _Request, _collect
from .snapshot import file_sha256

_TZ = "Asia/Shanghai"
_SEGMENTS = (("09:31", "11:30"), ("13:01", "15:00"))
_ANCHORS = ("09:30", "13:00")
_EXPECTED_MINUTES = 240
_COLUMNS = (
    "kind", "symbol", "session_date", "capture_started_at_utc", "capture_observed_at_utc", "session_close",
    "first_regular_bar_at", "last_regular_bar_at", "status", "complete_grid",
    "expected_minutes", "observed_minutes", "minute_coverage", "missing_minutes",
    "duplicate_regular_rows", "ignored_anchor_rows", "unexpected_rows",
    "invalid_regular_prices", "return_count", "realized_up_variance",
    "realized_down_variance", "realized_up_down_variance_ratio",
)


def summarize_sessions(
    frame: pd.DataFrame, *, kind: str, symbol: str, started_at_utc: str, observed_at_utc: str,
) -> list[dict]:
    """Assess minute completeness using the provider's local bar-end timestamps."""
    if not {"时间", "收盘"} <= set(frame.columns):
        raise ValueError("minute sample lacks time or close")
    started, captured = pd.Timestamp(started_at_utc), pd.Timestamp(observed_at_utc)
    if any(pd.isna(value) or value.tzinfo is None for value in (started, captured)):
        raise ValueError("capture times must have explicit time zones")
    if started > captured:
        raise ValueError("capture starts after observation")
    started, captured = started.tz_convert("UTC"), captured.tz_convert("UTC")
    timestamps = pd.to_datetime(frame["时间"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    if timestamps.isna().any():
        raise ValueError("minute sample has missing timestamps")
    local = timestamps.dt.tz_localize(_TZ)
    price = pd.to_numeric(frame["收盘"], errors="coerce")
    bars = pd.Series(price.to_numpy(dtype=float, na_value=np.nan), index=pd.DatetimeIndex(local))

    rows = []
    for day, closes in bars.groupby(bars.index.normalize()):
        day_label = day.date().isoformat()
        segments = [
            pd.date_range(f"{day_label} {start}", f"{day_label} {end}", freq="min", tz=_TZ)
            for start, end in _SEGMENTS
        ]
        expected = segments[0].append(segments[1])
        anchors = pd.DatetimeIndex([f"{day_label} {time}" for time in _ANCHORS], tz=_TZ)
        regular = closes[closes.index.isin(expected)]
        present = regular.index.unique()
        observed = len(present)
        duplicates = len(regular) - observed
        ignored = int(closes.index.isin(anchors).sum())
        unexpected = len(closes) - len(regular) - ignored
        invalid = int((~np.isfinite(regular) | regular.le(0)).sum())
        complete = (
            observed == _EXPECTED_MINUTES and not (duplicates or unexpected or invalid)
            and day.weekday() < 5 and expected[-1] <= started
        )
        row = {
            "kind": kind, "symbol": symbol, "session_date": day_label,
            "capture_started_at_utc": started.isoformat(), "capture_observed_at_utc": captured.isoformat(),
            "session_close": expected[-1].isoformat(),
            "first_regular_bar_at": present.min().isoformat() if observed else None,
            "last_regular_bar_at": present.max().isoformat() if observed else None,
            "status": "incomplete_session", "complete_grid": complete,
            "expected_minutes": _EXPECTED_MINUTES, "observed_minutes": observed,
            "minute_coverage": observed / _EXPECTED_MINUTES, "missing_minutes": _EXPECTED_MINUTES - observed,
            "duplicate_regular_rows": duplicates, "ignored_anchor_rows": ignored,
            "unexpected_rows": unexpected, "invalid_regular_prices": invalid,
            "return_count": 0, "realized_up_variance": None, "realized_down_variance": None,
            "realized_up_down_variance_ratio": None,
        }
        if complete:
            returns = np.concatenate([np.diff(np.log(regular.loc[segment].to_numpy())) for segment in segments])
            up = float(np.square(returns[returns > 0]).sum())
            down = float(np.square(returns[returns < 0]).sum())
            row.update(
                status="complete" if down > 0 else "zero_downside",
                return_count=len(returns), realized_up_variance=up, realized_down_variance=down,
                realized_up_down_variance_ratio=up / down if down > 0 else None,
            )
        rows.append(row)
    return rows


def run_intraday(output: Path, *, stock: str, etf: str) -> dict:
    requests = []
    for kind, symbol, endpoint in (
        ("stock", stock, "stock_zh_a_hist_min_em"), ("etf", etf, "fund_etf_hist_min_em"),
    ):
        prefixes = "036" if kind == "stock" else "15"
        if not re.fullmatch(f"[{prefixes}][0-9]{{5}}", symbol):
            raise ValueError(f"{kind} symbol must be a six-digit code starting with one of {prefixes}")
        requests.append((kind, symbol, _Request(
            f"{kind}_{symbol}_minutes", "akshare_eastmoney", endpoint,
            {"symbol": symbol, "period": "1", "adjust": ""}, "时间", ("时间", "收盘"),
        )))

    output = Path(output)
    raw = output / "raw"
    output.mkdir(parents=True, exist_ok=False)
    raw.mkdir()
    sources, rows = [], []
    rate_limited = False
    for kind, symbol, request in requests:
        if rate_limited:
            sources.append({"request": asdict(request), "status": "skipped_after_rate_limit", "files": []})
            continue
        record = _collect(request, raw)
        sources.append(record)
        rate_limited = "RateLimit" in record.get("error", {}).get("type", "")
        if record["status"] != "captured":
            continue
        try:
            expected_path = f"raw/{request.key}.csv"
            item, = record["files"]
            if item["path"] != expected_path:
                raise ValueError("unexpected captured minute file path")
            content = (output / expected_path).read_bytes()
            if hashlib.sha256(content).hexdigest() != item["sha256"]:
                raise ValueError("captured minute file hash mismatch")
            frame = pd.read_csv(BytesIO(content), dtype={"时间": str})
            if len(frame) != item["rows"]:
                raise ValueError("captured minute file row count mismatch")
            session_rows = summarize_sessions(
                frame, kind=kind, symbol=symbol, started_at_utc=record["started_at_utc"],
                observed_at_utc=record["observed_at_utc"],
            )
        except (OSError, ValueError) as exc:
            record["validation"] = {"status": "rejected", "reason": str(exc)}
            continue
        record["validation"] = {
            "status": "verified_capture",
            "observed_dates": len(session_rows),
            "complete_sessions": sum(row["complete_grid"] for row in session_rows),
            "usable_sessions": sum(row["status"] == "complete" for row in session_rows),
        }
        rows.extend(session_rows)

    verified = sum(record.get("validation", {}).get("status") == "verified_capture" for record in sources)
    usable = sum(row["status"] == "complete" for row in rows)
    path = output / "sessions.parquet"
    pd.DataFrame.from_records(rows, columns=_COLUMNS).to_parquet(path, index=False)
    report = {
        "schema_version": 1, "origin": "market_provider_probe",
        "status": "sample_available" if usable else "data_insufficient",
        "acquisition_status": "complete" if verified == len(requests) else "partial" if verified else "failed",
        "reported_at_utc": datetime.now(timezone.utc).isoformat(),
        "session_timezone": _TZ,
        "expected_bar_end_segments": _SEGMENTS,
        "coverage_scope": "provider-returned dates only",
        "method": "positive squared log returns divided by negative squared log returns within each session segment",
        "sources": sources, "sessions": rows,
        "parquet": {"path": path.name, "rows": len(rows), "sha256": file_sha256(path)},
    }
    (output / "intraday.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return report


def read_intraday_capture(capture: Path) -> pd.DataFrame:
    """Load point-in-time minute features from a verified provider capture."""
    capture = Path(capture)
    report = json.loads((capture / "intraday.json").read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("origin") != "market_provider_probe":
        raise ValueError("unsupported intraday capture")
    evidence = report.get("parquet", {})
    path = capture / "sessions.parquet"
    if evidence.get("path") != path.name:
        raise ValueError("unexpected intraday Parquet path")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != evidence.get("sha256"):
        raise ValueError("intraday Parquet hash mismatch")
    sessions = pd.read_parquet(BytesIO(content))
    if len(sessions) != evidence.get("rows"):
        raise ValueError("intraday Parquet row count mismatch")
    ratio_column = "realized_up_down_variance_ratio"
    required = {"session_date", "symbol", "kind", "capture_observed_at_utc", "status", "complete_grid", ratio_column}
    if not required <= set(sessions.columns):
        raise ValueError("intraday Parquet lacks feature columns")
    ratios = pd.to_numeric(sessions[ratio_column], errors="coerce")
    accepted = sessions.loc[sessions["status"].eq("complete") & sessions["complete_grid"].eq(True) & np.isfinite(ratios)]
    observations = accepted["capture_observed_at_utc"].map(pd.Timestamp)
    if any(pd.isna(value) or value.tzinfo is None for value in observations):
        raise ValueError("capture observation must have an explicit time zone")
    dates = pd.to_datetime(accepted["session_date"], format="%Y-%m-%d", errors="raise")
    if dates.isna().any():
        raise ValueError("minute features require a session date")
    return pd.DataFrame({
        "date": dates.dt.normalize(),
        "symbol": accepted["symbol"].astype("string"),
        "kind": accepted["kind"].astype("string"),
        "known_at": pd.to_datetime(observations, utc=True),
        "minute_up_down_variance_ratio": ratios.loc[accepted.index].astype(float),
    }).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", required=True)
    parser.add_argument("--etf", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run_intraday(args.out, stock=args.stock, etf=args.etf)
    except (OSError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": report["status"], "acquisition_status": report["acquisition_status"],
        "report": str((args.out / "intraday.json").resolve()),
    }, ensure_ascii=False))
    return 0 if report["status"] == "sample_available" else 2


if __name__ == "__main__":
    raise SystemExit(main())
