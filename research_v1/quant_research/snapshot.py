"""Immutable, verified Parquet snapshots and a fail-closed research gate."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import numpy as np

from .protocol import ETF_SYMBOLS, RESEARCH_START, ResearchPeriod

TABLES = {"bars", "calendar", "actions", "membership", "securities", "news", "edges"}
ORIGINS = {"market", "synthetic_fixture"}
BAR_COLUMNS = {
    "date", "symbol", "kind", "open", "high", "low", "close", "volume", "amount",
    "tradable", "upper_limit", "lower_limit",
}


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_snapshot(
    path: Path,
    *,
    origin: str,
    frames: Mapping[str, pd.DataFrame],
    sources: list[dict[str, Any]],
    coverage: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if origin not in ORIGINS or not frames or frames.keys() - TABLES:
        raise ValueError("invalid snapshot origin or tables")
    if origin == "market" and not sources:
        raise ValueError("market snapshots require source evidence")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=".snapshot-") as work:
        staging = Path(work) / "snapshot"
        staging.mkdir()
        records = {}
        for name, frame in sorted(frames.items()):
            file = staging / f"{name}.parquet"
            frame.to_parquet(file, index=False)
            records[name] = {"rows": len(frame), "sha256": file_sha256(file)}
        manifest = {
            "schema_version": 1, "origin": origin, "tables": records, "sources": sources,
            "coverage": dict(coverage or {}), "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.rename(staging, path)
    return manifest


def read_snapshot(path: Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["tables"]
    if manifest.get("schema_version") != 1 or manifest.get("origin") not in ORIGINS:
        raise ValueError("unsupported snapshot manifest")
    if not records or records.keys() - TABLES:
        raise ValueError("unexpected snapshot table")
    frames = {}
    for name, record in records.items():
        file = path / f"{name}.parquet"
        if file_sha256(file) != record["sha256"]:
            raise ValueError(f"snapshot hash mismatch: {name}")
        frames[name] = pd.read_parquet(file)
        if len(frames[name]) != record["rows"]:
            raise ValueError(f"snapshot row count mismatch: {name}")
    return manifest, frames


def audit_snapshot(
    manifest: Mapping[str, Any], frames: Mapping[str, pd.DataFrame], period: ResearchPeriod, kind: str
) -> dict[str, Any]:
    if kind not in {"stock", "etf"}:
        raise ValueError("unknown market line")
    problems: list[str] = []
    if manifest.get("origin") != "market":
        problems.append("synthetic fixtures are excluded from market evidence")
    if not manifest.get("sources"):
        problems.append("source evidence missing")
    if period.start != RESEARCH_START:
        problems.append("start differs from pre-registered study window")
    coverage = manifest.get("coverage", {})
    for key in ("exchange_calendar", f"{kind}_raw_bars", f"{kind}_actions", f"{kind}_limits"):
        if coverage.get(key) != "verified":
            problems.append(f"{key}: not independently verified")
    cal = frames.get("calendar", pd.DataFrame())
    days = pd.DatetimeIndex([])
    if "date" not in cal or cal.empty:
        problems.append("exchange calendar missing")
    else:
        all_days = pd.DatetimeIndex(pd.to_datetime(cal["date"], errors="raise")).normalize()
        days = all_days[(all_days.date >= period.start) & (all_days.date <= period.end)]
        if not len(days) or not days.is_unique or not days.is_monotonic_increasing:
            problems.append("empty, duplicate or unsorted exchange sessions")
        # The requested dates may be weekends/holidays, but not a truncated multi-day feed.
        elif (days[0].date() - period.start).days > 10 or (period.end - days[-1].date()).days > 10:
            problems.append("exchange calendar does not span requested period")

    counts: dict[str, int] = {}
    bars = frames.get("bars", pd.DataFrame())
    if not BAR_COLUMNS <= set(bars):
        problems.append(f"bars missing columns: {sorted(BAR_COLUMNS - set(bars))}")
    else:
        bars = bars[bars["kind"] == kind].copy()
        bars["date"] = pd.to_datetime(bars["date"], errors="raise").dt.normalize()
        bars = bars[bars["date"].isin(days)]
        counts = {str(k): int(v) for k, v in bars.groupby("symbol").size().items()}
        if bars.empty or bars.duplicated(["symbol", "date"]).any():
            problems.append("missing or duplicate bars")
        if kind == "etf" and any(counts.get(s, 0) != len(days) for s in ETF_SYMBOLS):
            problems.append("frozen ETF universe has incomplete sessions")
        if not np.isfinite(bars[["open", "high", "low", "close", "volume", "amount"]].to_numpy(float)).all() or (bars[["open", "close"]] <= 0).any().any():
            problems.append("missing or non-positive raw prices")
        if bars["tradable"].isna().any() or not pd.api.types.is_bool_dtype(bars["tradable"]):
            problems.append("unknown tradability")
        trades = bars[bars["tradable"].eq(True)]
        if trades[["upper_limit", "lower_limit"]].isna().any().any():
            problems.append("tradeable bars have unknown price limits")
    actions = frames.get("actions")
    if actions is None or not {"symbol", "ex_date", "pay_date", "cash_per_share", "share_multiplier"} <= set(actions):
        problems.append("corporate action records missing")
    if kind == "stock":
        if period.start < date(2015, 8, 1):
            problems.append("default execution fees lack verified pre-August-2015 stock rules")
        if coverage.get("stock_membership") != "verified":
            problems.append("point-in-time stock membership not independently verified")
        members = frames.get("membership", pd.DataFrame())
        if not {"symbol", "effective_date", "known_at"} <= set(members) or members.empty:
            problems.append("dated membership records missing")
        elif members[["effective_date", "known_at"]].isna().any().any():
            problems.append("membership effective or known timestamp missing")
        securities = frames.get("securities", pd.DataFrame())
        meta_cols = {"symbol", "listed_at", "delisted_at", "board"}
        if not meta_cols <= set(securities) or securities.empty:
            problems.append("listing/delisting metadata missing or invalid schema")
        elif securities["symbol"].duplicated().any() or securities[["symbol", "listed_at", "board"]].isna().any().any():
            problems.append("duplicate or incomplete security metadata")
        elif {"symbol", "effective_date", "known_at"} <= set(members) and len(days) and BAR_COLUMNS <= set(bars):
            # A coverage assertion cannot replace validation of actual as-of records.
            metadata = securities.set_index("symbol").copy()
            metadata["listed_at"] = pd.to_datetime(metadata["listed_at"], errors="raise").dt.normalize()
            metadata["delisted_at"] = pd.to_datetime(metadata["delisted_at"], errors="raise").dt.normalize()
            if (metadata["delisted_at"].notna() & (metadata["delisted_at"] < metadata["listed_at"])).any():
                problems.append("delisting precedes listing")
            members = members.copy()
            members["effective_date"] = pd.to_datetime(members["effective_date"], errors="raise").dt.normalize()
            members["known_at"] = pd.to_datetime(members["known_at"], errors="raise", utc=True)
            keys = set(zip(bars["date"], bars["symbol"].astype(str)))
            complete_versions = members.groupby(["effective_date", "known_at"])["symbol"].agg(lambda s: tuple(s.astype(str)))
            snapshots = [(effective, known, symbols) for (effective, known), symbols in complete_versions.items()]
            if any(len(symbols) != 300 or len(set(symbols)) != 300 for _, _, symbols in snapshots):
                problems.append("CSI300 membership snapshots must contain 300 unique members")
            for day in days:
                cutoff = (day + pd.Timedelta(hours=15)).tz_localize("Asia/Shanghai").tz_convert("UTC")
                available = [(effective, known, symbols) for effective, known, symbols in snapshots
                             if effective <= day and known <= cutoff]
                if not available:
                    problems.append("no point-in-time membership for one or more study sessions")
                    break
                _, _, symbols = max(available, key=lambda item: (item[0], item[1]))
                if not set(symbols) <= set(metadata.index):
                    problems.append("historical members missing listing metadata")
                    break
                required = metadata.loc[list(symbols)]
                active = required[(required["board"] == "main") & (required["listed_at"] <= day)
                                  & (required["delisted_at"].isna() | (required["delisted_at"] >= day))]
                if active.empty or any((day, str(symbol)) not in keys for symbol in active.index):
                    problems.append("missing daily bars for historically active main-board members")
                    break

    return {
        "origin": manifest.get("origin"), "kind": kind, "status": "ready" if not problems else "data_insufficient",
        "start": period.start.isoformat(), "end": period.end.isoformat(), "sessions": len(days),
        "rows_by_symbol": counts, "issues": problems,
    }
