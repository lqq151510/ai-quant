"""Normalize captured ETF tables without promoting unverified action evidence."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .protocol import ETF_SYMBOLS, ResearchPeriod
from .snapshot import file_sha256, write_snapshot


def dividend_per_share(value: object) -> float:
    """Parse the vendor's explicitly per-ten-units cash text, never its first digit."""
    match = re.fullmatch(r"(?:每10份派现金)?([0-9]+(?:\.[0-9]+)?)(?:元)?", str(value).strip())
    if match is None:
        raise ValueError(f"unrecognized per-ten-units distribution format: {value}")
    return float(match.group(1)) / 10


def prepare_etf_snapshot(probe: Path, calendar_csv: Path, output: Path, period: ResearchPeriod) -> dict:
    audit = json.loads((probe / "audit.json").read_text(encoding="utf-8"))
    if audit.get("origin") != "market_provider_probe":
        raise ValueError("only captured market-provider probes can be normalized")
    calendar_csv = calendar_csv.resolve()
    calendar_audit_path = calendar_csv.parent.parent / "audit.json"
    calendar_audit = json.loads(calendar_audit_path.read_text(encoding="utf-8"))

    # Accept only a calendar whose hash is present in its acquisition evidence.
    def contains_hash(value: object, digest: str) -> bool:
        if isinstance(value, dict):
            return value.get("sha256") == digest or any(contains_hash(v, digest) for v in value.values())
        if isinstance(value, list):
            return any(contains_hash(v, digest) for v in value)
        return False

    if not contains_hash(calendar_audit, file_sha256(calendar_csv)):
        raise ValueError("calendar hash absent from captured acquisition evidence")
    raw_calendar = pd.read_csv(calendar_csv)
    date_col = "trade_date" if "trade_date" in raw_calendar else "date"
    dates = pd.to_datetime(raw_calendar[date_col], errors="raise").dt.normalize()
    dates = dates[(dates.dt.date >= period.start) & (dates.dt.date <= period.end)]
    if dates.empty or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("empty, duplicate or unsorted calendar")
    frames, actions = [], []
    sources = []
    for record in audit["sources"]:
        key = record["request"]["key"]
        if not (key.startswith("etf_bars_") or key.startswith("etf_actions_")):
            continue
        symbol = key.rsplit("_", 1)[1]
        if symbol not in ETF_SYMBOLS:
            continue
        sources.append(record)
        for evidence in record.get("files", []):
            path = (probe / evidence["path"]).resolve()
            if not path.is_relative_to(probe.resolve()) or file_sha256(path) != evidence["sha256"]:
                raise ValueError("source CSV hash mismatch or path outside probe")
            raw = pd.read_csv(path)
            if key.startswith("etf_bars_"):
                raw["date"] = pd.to_datetime(raw["date"], errors="raise").dt.normalize()
                bars = raw.loc[raw["date"].isin(dates), ["date", "open", "high", "low", "close", "volume", "amount"]].copy()
                bars["symbol"], bars["kind"] = symbol, "etf"
                bars["tradable"] = bars["volume"].gt(0)
                bars["upper_limit"], bars["lower_limit"] = np.nan, np.nan
                frames.append(bars)
            elif "权益登记日" in raw:
                for row in raw.to_dict("records"):
                    cash = dividend_per_share(row["每10份分红"])
                    actions.append({"symbol": symbol, "record_date": row["权益登记日"], "ex_date": row["除息日"],
                                    "pay_date": row["分红发放日"], "cash_per_share": cash, "share_multiplier": 1.})
            elif "拆分折算日" in raw:
                for row in raw.to_dict("records"):
                    if str(row["拆分折算日"]) in {"暂无拆分信息", "暂无拆分记录", "nan"}:
                        continue
                    numerator, denominator = str(row["拆分折算比例"]).split(":")
                    actions.append({"symbol": symbol, "record_date": None, "ex_date": row["拆分折算日"],
                                    "pay_date": None, "cash_per_share": 0.,
                                    "share_multiplier": float(denominator) / float(numerator)})
    if len(frames) != len(ETF_SYMBOLS):
        raise ValueError("some frozen ETF sources were not captured")
    bars = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    full = all(set(group["date"]) == set(dates) and len(group) == len(dates) for _, group in bars.groupby("symbol"))
    action_data = pd.DataFrame(actions, columns=["symbol", "record_date", "ex_date", "pay_date", "cash_per_share", "share_multiplier"])
    for name in ("record_date", "ex_date", "pay_date"):
        action_data[name] = pd.to_datetime(action_data[name], errors="raise")
    action_data = action_data[(action_data["ex_date"].dt.date >= period.start) & (action_data["ex_date"].dt.date <= period.end)]
    sources.append({"provider": "captured_third_party_calendar", "sha256": file_sha256(calendar_csv),
                    "path": str(calendar_csv), "audit": calendar_audit})
    return write_snapshot(output, origin="market", frames={"bars": bars, "calendar": pd.DataFrame({"date": dates}),
                          "actions": action_data}, sources=sources,
                          coverage={"exchange_calendar": "cross_checked_third_party", "etf_raw_bars": "verified" if full else "incomplete",
                                    "etf_actions": "unverified_third_party", "etf_limits": "missing"})
