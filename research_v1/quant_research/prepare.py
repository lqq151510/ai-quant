"""Normalize captured ETF tables without promoting unverified action evidence."""

from __future__ import annotations

import json
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd

from .protocol import ETF_SYMBOLS, ResearchPeriod
from .snapshot import file_sha256, write_snapshot

_MILL = Decimal("0.001")


def _round_mill(value: float) -> float:
    return float((Decimal(str(value))).quantize(_MILL, rounding=ROUND_HALF_UP))


def dividend_per_share(value: object) -> float:
    """Parse the vendor's explicitly per-ten-units cash text, never its first digit."""
    match = re.fullmatch(r"(?:每10份派现金)?([0-9]+(?:\.[0-9]+)?)(?:元)?", str(value).strip())
    if match is None:
        raise ValueError(f"unrecognized per-ten-units distribution format: {value}")
    return float(match.group(1)) / 10


def _resolve_split_ex_date(record_day: pd.Timestamp, session_index: pd.DatetimeIndex) -> pd.Timestamp:
    """Map an SSE ETF share conversion record date (拆分折算日) to its effective ex-split trading session."""
    pos = session_index.searchsorted(record_day)
    if pos < len(session_index) and session_index[pos] == record_day:
        if pos + 1 < len(session_index):
            return session_index[pos + 1]
    return record_day


def prepare_etf_snapshot(
    probe: Path,
    calendar_csv: Path,
    output: Path,
    period: ResearchPeriod,
    *,
    bulk_dir: Path | None = None,
) -> dict:
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
    all_dates = pd.DatetimeIndex(pd.to_datetime(raw_calendar[date_col], errors="raise").dt.normalize()).sort_values()
    dates = pd.Series(all_dates[(all_dates.date >= period.start) & (all_dates.date <= period.end)], name="date")
    if dates.empty or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("empty, duplicate or unsorted calendar")
    session_idx = pd.DatetimeIndex(dates)
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
                    rec_dt = pd.Timestamp(row["拆分折算日"]).normalize()
                    ex_dt = _resolve_split_ex_date(rec_dt, all_dates) if bulk_dir is not None else rec_dt
                    actions.append({"symbol": symbol, "record_date": rec_dt if bulk_dir is not None else None,
                                    "ex_date": ex_dt, "pay_date": None, "cash_per_share": 0.,
                                    "share_multiplier": float(denominator) / float(numerator)})
    if len(frames) != len(ETF_SYMBOLS):
        raise ValueError("some frozen ETF sources were not captured")
    bars = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    full = all(set(group["date"]) == set(dates) and len(group) == len(dates) for _, group in bars.groupby("symbol"))
    action_data = pd.DataFrame(actions, columns=["symbol", "record_date", "ex_date", "pay_date", "cash_per_share", "share_multiplier"])
    for name in ("record_date", "ex_date", "pay_date"):
        action_data[name] = pd.to_datetime(action_data[name], errors="raise")
    action_data = action_data[(action_data["ex_date"].dt.date >= period.start) & (action_data["ex_date"].dt.date <= period.end)].reset_index(drop=True)
    sources.append({"provider": "captured_third_party_calendar", "sha256": file_sha256(calendar_csv),
                    "path": str(calendar_csv), "audit": calendar_audit})
    coverage = {
        "exchange_calendar": "cross_checked_third_party",
        "etf_raw_bars": "verified" if full else "incomplete",
        "etf_actions": "unverified_third_party",
        "etf_limits": "missing",
    }

    if bulk_dir is not None:
        bulk_dir = bulk_dir.resolve()
        gap_report = json.loads((bulk_dir / "gap_report.json").read_text(encoding="utf-8"))
        # 1. Verify 12-year BaoStock calendar hashes and exact session equality
        bs_cal_frames = []
        for yr_info in gap_report["calendar"]["years"]:
            yr = int(yr_info["year"])
            yr_path = (bulk_dir / "raw" / f"trading_calendar_baostock_{yr}.csv").resolve()
            if not yr_path.is_relative_to(bulk_dir) or file_sha256(yr_path) != yr_info["sha256"]:
                raise ValueError(f"baostock calendar hash mismatch for {yr}")
            bs_cal_frames.append(pd.read_csv(yr_path))
        bs_cal = pd.concat(bs_cal_frames, ignore_index=True)
        bs_trading = pd.to_datetime(
            bs_cal.loc[bs_cal["is_trading_day"].astype(str).eq("1"), "calendar_date"], errors="raise"
        ).dt.normalize()
        bs_days = pd.DatetimeIndex(bs_trading[(bs_trading.dt.date >= period.start) & (bs_trading.dt.date <= period.end)]).sort_values()
        if not bs_days.equals(session_idx):
            raise ValueError("cross-provider exchange calendar session mismatch")

        # 2. Verify BaoStock & Tencent ETF bar hashes, suspension rows, price limits, and preclose-action bijection
        bs_bar_map = gap_report["alternate_bar_sources"]["baostock_query_history_k_data_plus"]
        tx_bar_map = gap_report["alternate_bar_sources"]["tencent_stock_zh_a_hist_tx"]
        verified_frames = []
        action_lookup = {
            (row.symbol, pd.Timestamp(row.ex_date).normalize()): row
            for row in action_data.itertuples(index=False)
        }
        for symbol in ETF_SYMBOLS:
            bs_meta = bs_bar_map[symbol]
            tx_meta = tx_bar_map[symbol]
            bs_path = (bulk_dir / bs_meta["path"]).resolve()
            tx_path = (bulk_dir / tx_meta["path"]).resolve()
            if file_sha256(bs_path) != bs_meta["sha256"] or file_sha256(tx_path) != tx_meta["sha256"]:
                raise ValueError(f"multi-source ETF bar hash mismatch for {symbol}")
            bs_raw = pd.read_csv(bs_path)
            bs_raw["date"] = pd.to_datetime(bs_raw["date"], errors="raise").dt.normalize()
            sub = bs_raw.loc[bs_raw["date"].isin(session_idx)].sort_values("date").reset_index(drop=True)
            if len(sub) != len(session_idx) or not pd.DatetimeIndex(sub["date"]).equals(session_idx):
                raise ValueError(f"incomplete sessions in multi-source bars for {symbol}")
            sub["symbol"] = symbol
            sub["kind"] = "etf"
            sub["tradable"] = sub["volume"].gt(0) & sub["tradestatus"].astype(int).eq(1)
            sub["upper_limit"] = sub["preclose"].map(lambda p: _round_mill(float(p) * 1.1))
            sub["lower_limit"] = sub["preclose"].map(lambda p: _round_mill(float(p) * 0.9))

            # Verify 100% of tradeable bars stay within 10% SSE price limits
            active = sub[sub["tradable"]]
            if (
                (active["low"] < active["lower_limit"] - 1e-6).any()
                or (active["high"] > active["upper_limit"] + 1e-6).any()
                or (active["open"] < active["lower_limit"] - 1e-6).any()
                or (active["open"] > active["upper_limit"] + 1e-6).any()
                or (active["close"] < active["lower_limit"] - 1e-6).any()
                or (active["close"] > active["upper_limit"] + 1e-6).any()
            ):
                raise ValueError(f"OHLC breaches derived price limits for {symbol}")

            # Verify 100% bijection between preclose transitions and corporate actions
            prev_closes = sub["close"].shift(1)
            for idx in range(1, len(sub)):
                day_ts = pd.Timestamp(sub.loc[idx, "date"]).normalize()
                prev_c = float(prev_closes.iloc[idx])
                pre_c = float(sub.loc[idx, "preclose"])
                act = action_lookup.get((symbol, day_ts))
                if act is None:
                    if abs(pre_c - prev_c) > 0.001 + 1e-6:
                        raise ValueError(f"unexplained preclose jump for {symbol} on {day_ts.date()}")
                else:
                    expected_ref = (prev_c - float(act.cash_per_share)) / float(act.share_multiplier)
                    if act.share_multiplier == 1.0:
                        if abs(pre_c - expected_ref) > 0.001 + 1e-6:
                            raise ValueError(f"dividend ex-reference mismatch for {symbol} on {day_ts.date()}")
                    elif abs(pre_c - expected_ref) / max(pre_c, 1e-6) > 0.002:
                        raise ValueError(f"split ex-reference mismatch for {symbol} on {day_ts.date()}")

            verified_frames.append(
                sub[["date", "symbol", "kind", "open", "high", "low", "close", "preclose",
                     "volume", "amount", "tradable", "upper_limit", "lower_limit"]]
            )

        bars = pd.concat(verified_frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
        sources.append({
            "provider": "cross_checked_bulk_evidence",
            "path": str(bulk_dir),
            "gap_report_sha256": file_sha256(bulk_dir / "gap_report.json"),
        })
        coverage = {
            "exchange_calendar": "verified",
            "etf_raw_bars": "verified",
            "etf_actions": "verified",
            "etf_limits": "verified",
        }

    return write_snapshot(
        output,
        origin="market",
        frames={"bars": bars, "calendar": pd.DataFrame({"date": dates}), "actions": action_data},
        sources=sources,
        coverage=coverage,
    )
