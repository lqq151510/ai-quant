"""Reconcile captured ETF sessions and retain derived limits as candidates."""

from __future__ import annotations

from datetime import date
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path

import pandas as pd

from .prepare import dividend_per_share
from .protocol import ETF_SYMBOLS, ResearchPeriod
from .snapshot import file_sha256, write_snapshot


_TICK = Decimal("0.001")
_PRICES = ("open", "high", "low", "close")
_HALTS = {date(2015, 4, 13), date(2015, 4, 14)}
_SPLIT_2015 = "https://www.sse.com.cn/disclosure/fund/announcement/c/2015-04-15/510500_20150415_2.pdf"
_HALT_2015 = "https://www.sse.com.cn/disclosure/fund/announcement/c/2015-04-07/510500_20150408_1.pdf"
_SPLIT_2022 = "https://www.sse.com.cn/disclosure/fund/announcement/c/new/2022-08-23/510500_20220823_1_aMMa3ZNB.pdf"
_RULE_SOURCES = (
    "https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20150912_3988782.shtml",
    "https://www.sse.com.cn/lawandrules/sselawsrules2025/repeal/rules/c/c_20200313_10785127.shtml",
    "https://www.sse.com.cn/lawandrules/sselawsrules2025/repeal/rules/c/c_20250612_10824490.shtml",
    "https://www.sse.com.cn/lawandrules/sselawsrules2025/fund/trading/c/c_20260424_10817739.shtml",
)


def _price(value: Decimal) -> Decimal:
    return value.quantize(_TICK, rounding=ROUND_HALF_UP)


def _limits(reference: Decimal) -> tuple[Decimal, Decimal]:
    return _price(reference * Decimal("0.9")), _price(reference * Decimal("1.1"))


def _csv(bulk: Path, records: dict, name: str, used: list[dict]) -> pd.DataFrame:
    record = records.get(name)
    if record is None or record.get("status") != "success":
        raise ValueError(f"missing successful captured file: {name}")
    path = (bulk / name).resolve()
    if not path.is_relative_to(bulk) or file_sha256(path) != record.get("sha256"):
        raise ValueError(f"source CSV hash mismatch or path outside capture: {name}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if len(frame) != record.get("rows"):
        raise ValueError(f"source CSV row count mismatch: {name}")
    used.append({k: record[k] for k in ("path", "sha256", "rows", "source_function", "params", "observed_at_utc")})
    return frame


def _date(value: str) -> date:
    return date.fromisoformat(value)


def _calendar(bulk: Path, records: dict, used: list[dict], period: ResearchPeriod) -> tuple[set[date], dict]:
    observed = {}
    for year in range(period.start.year, period.end.year + 1):
        rows = _csv(bulk, records, f"raw/trading_calendar_baostock_{year}.csv", used)
        for day, status in rows[["calendar_date", "is_trading_day"]].itertuples(index=False, name=None):
            session = _date(day)
            if session.year != year or session in observed or status not in {"0", "1"}:
                raise ValueError(f"invalid captured calendar date or flag: {day}")
            observed[session] = status == "1"
    window = set(pd.date_range(period.start, period.end).date)
    if not window <= observed.keys():
        raise ValueError("captured calendar does not cover the requested dates")
    sessions = {day for day in window if observed[day]}
    if not sessions:
        raise ValueError("captured calendar has no exchange sessions in the requested period")
    evidence = {"status": "single_vendor", "baostock_sessions": len(sessions)}

    # The prior calendar comparison records the exact Sina bytes. Resolve only
    # the expected sibling capture, so relocated copies can still be reconciled.
    comparison = records.get("calendar_diff.csv", {}).get("params", {})
    old_path = Path(comparison.get("sina_csv", ""))
    if old_path.name != "trading_calendar_sina.csv" or old_path.parent.name != "raw":
        return sessions, evidence
    sina_path = bulk.parent / old_path.parent.parent.name / "raw" / old_path.name
    audit_path = sina_path.parent.parent / "audit.json"
    if not sina_path.is_file() or not audit_path.is_file():
        return sessions, evidence
    if not sina_path.resolve().is_relative_to(bulk.parent) or not audit_path.resolve().is_relative_to(bulk.parent):
        raise ValueError("Sina calendar evidence is outside the sibling capture")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    digest = file_sha256(sina_path)
    if audit.get("request", {}).get("key") != "trading_calendar_sina" or not any(
        item.get("sha256") == digest and item.get("path") == f"raw/{sina_path.name}" for item in audit.get("files", [])
    ):
        raise ValueError("Sina calendar hash absent from captured acquisition evidence")
    sina_days = [_date(day) for day in pd.read_csv(sina_path, dtype=str)["trade_date"]]
    in_window = [day for day in sina_days if day in window]
    if len(set(in_window)) != len(in_window):
        raise ValueError("duplicate dates in captured Sina calendar")
    sina = set(in_window)
    evidence.update(
        status="cross_checked_third_party" if sessions == sina else "vendor_mismatch",
        sina_sha256=digest, sina_audit_sha256=file_sha256(audit_path),
        sina_sessions=len(sina), baostock_only=sorted(day.isoformat() for day in sessions - sina),
        sina_only=sorted(day.isoformat() for day in sina - sessions),
    )
    return sessions, evidence


def _official_splits() -> list[dict]:
    return [
        {"symbol": "510500", "record_date": date(2015, 4, 13), "ex_date": date(2015, 4, 14),
         "pay_date": None, "cash_per_share": 0.0, "share_multiplier": 0.28032483,
         "split_rounding": "floor", "share_available_date": date(2015, 4, 15),
         "action_source": "issuer_announcement", "source_url": _SPLIT_2015},
        {"symbol": "510500", "record_date": date(2022, 8, 26), "ex_date": date(2022, 8, 29),
         "pay_date": None, "cash_per_share": 0.0, "share_multiplier": 1.14539,
         "split_rounding": "ceil", "share_available_date": date(2022, 8, 29),
         "action_source": "issuer_announcement", "source_url": _SPLIT_2022},
    ]


def _actions(bulk: Path, records: dict, used: list[dict], symbol: str, period: ResearchPeriod) -> list[dict]:
    actions = []
    vendor_dates = set()
    for row in _csv(bulk, records, f"raw/etf_actions_f10_{symbol}_splits.csv", used).to_dict("records"):
        value = row["拆分折算日"]
        if value not in {"", "暂无拆分信息", "暂无拆分记录"}:
            vendor_dates.add(_date(value))
    official = [event for event in _official_splits() if event["symbol"] == symbol]
    # The vendor reports 2022's registration date, while the position changes at
    # the next opening. The 2015 vendor date already denotes the ex-date.
    known_dates = {date(2015, 4, 14), date(2022, 8, 26)} if official else set()
    for day in vendor_dates:
        if period.start <= day <= period.end and day not in known_dates:
            raise ValueError(f"unresolved captured ETF split: {symbol} {day}")
    actions.extend(official)
    dividends = _csv(bulk, records, f"raw/etf_actions_f10_{symbol}_dividends.csv", used)
    for row in dividends.to_dict("records"):
        ex_date = _date(row["除息日"])
        if ex_date > period.end:
            continue
        record_date, pay_date = _date(row["权益登记日"]), _date(row["分红发放日"])
        if not record_date < ex_date <= pay_date:
            raise ValueError(f"invalid captured ETF dividend dates: {symbol} {ex_date}")
        actions.append({
            "symbol": symbol, "record_date": record_date, "ex_date": ex_date,
            "pay_date": pay_date, "cash_per_share": dividend_per_share(row["每10份分红"]),
            "share_multiplier": 1.0, "split_rounding": "exact", "share_available_date": None,
            "action_source": "eastmoney_f10", "source_url": f"https://fundf10.eastmoney.com/fhsp_{symbol}.html",
        })
    by_day = {event["ex_date"]: event for event in actions}
    if len(by_day) != len(actions):
        raise ValueError(f"duplicate captured ETF actions: {symbol}")
    return actions


def _bars(
    bulk: Path, records: dict, used: list[dict], symbol: str,
    sessions: set[date], actions: list[dict], period: ResearchPeriod,
) -> tuple[list[dict], dict]:
    baostock = _csv(bulk, records, f"raw/etf_bars_baostock_{symbol}.csv", used)
    if not baostock["code"].eq(f"sh.{symbol}").all() or not baostock["adjustflag"].eq("3").all():
        raise ValueError(f"BaoStock prices are not unadjusted records for {symbol}")
    tencent = _csv(bulk, records, f"raw/etf_bars_tx_{symbol}.csv", used)
    if baostock["date"].duplicated().any() or tencent["date"].duplicated().any():
        raise ValueError(f"duplicate captured ETF bar date for {symbol}")
    tencent_by_day = tencent.set_index("date").to_dict("index")
    action_by_day = {event["ex_date"]: event for event in actions}
    out = []
    counts = dict.fromkeys(("ohlc_matched", "dividend_references_matched", "split_references_matched",
                           "references_unchecked", "verified_halts", "derived_candidates"), 0)
    issues = []
    for raw_day in tencent_by_day:
        day = _date(raw_day)
        if period.start <= day <= period.end and day not in sessions:
            issues.append({"date": raw_day, "check": "tencent_bar_outside_calendar"})
    previous = None
    for row in baostock.sort_values("date").to_dict("records"):
        day = _date(row["date"])
        if day > period.end:
            break
        in_scope = day in sessions
        if period.start <= day and not in_scope:
            issues.append({"date": row["date"], "check": "baostock_bar_outside_calendar"})
            continue
        status = row["tradestatus"]
        prices = {name: Decimal(row[name]) for name in (*_PRICES, "preclose", "volume", "amount")}
        if (status not in {"0", "1"} or any(not value.is_finite() for value in prices.values())
                or any(prices[name] <= 0 for name in (*_PRICES, "preclose"))
                or min(prices["volume"], prices["amount"]) < 0
                or prices["low"] > min(prices["open"], prices["close"])
                or prices["high"] < max(prices["open"], prices["close"])
                or (status == "1") != (prices["volume"] > 0)):
            raise ValueError(f"invalid captured ETF price or tradability: {symbol} {day}")
        event = action_by_day.get(day)
        expected = previous
        if event is not None and previous is not None:
            cash = Decimal(str(event["cash_per_share"]))
            multiplier = Decimal(str(event["share_multiplier"]))
            expected = _price((previous - cash) / multiplier)
        tradable = status == "1"
        official_halt = symbol == "510500" and day in _HALTS
        if official_halt and (tradable or prices["amount"] != 0):
            raise ValueError(f"BaoStock conflicts with official ETF halt: {symbol} {day}")
        valuation = expected if official_halt and event is not None else None
        if official_halt and event is not None and (valuation is None or valuation <= 0):
            raise ValueError(f"missing pre-fold carrying price for {symbol} {day}")

        if in_scope:
            counterpart = tencent_by_day.get(row["date"])
            matched = counterpart is not None and all(prices[name] == Decimal(counterpart[name]) for name in _PRICES)
            if tradable and matched:
                counts["ohlc_matched"] += 1
            elif tradable or counterpart is not None:
                issues.append({"date": row["date"], "check": "tencent_ohlc_or_tradability_mismatch"})
            elif not official_halt:
                issues.append({"date": row["date"], "check": "unverified_halt"})
            reference_ok = expected is None or prices["preclose"] == expected
            reference_check = "vendor_halt_reference"
            if tradable:
                if expected is None:
                    counts["references_unchecked"] += 1
                    reference_check = "first_observed_reference"
                elif not reference_ok:
                    reference_check = "reference_mismatch"
                    issues.append({"date": row["date"], "check": reference_check,
                                   "expected": str(expected), "observed": str(prices["preclose"])})
                elif event is not None and event["cash_per_share"]:
                    counts["dividend_references_matched"] += 1
                    reference_check = "cash_dividend_reconciled"
                elif event is not None or (symbol == "510500" and day == date(2015, 4, 15)):
                    counts["split_references_matched"] += 1
                    reference_check = "issuer_split_reconciled"
                else:
                    reference_check = "previous_valuation_reconciled"

            lower = upper = None
            limit_source = "not_applicable_halt"
            if tradable:
                lower, upper = _limits(prices["preclose"])
                if prices["low"] < lower or prices["high"] > upper:
                    issues.append({"date": row["date"], "check": "candidate_limit_range_conflict",
                                   "candidate_lower": str(lower), "candidate_upper": str(upper)})
                    lower = upper = None
                elif not reference_ok:
                    lower = upper = None
                limit_source = "rule_derived_candidate" if lower is not None else "unresolved"
                counts["derived_candidates"] += int(lower is not None)
            counts["verified_halts"] += int(official_halt)
            out.append({
                "date": pd.Timestamp(day), "symbol": symbol, "kind": "etf", "tradable": tradable,
                **{name: float(value) for name, value in prices.items()},
                "upper_limit": float(upper) if upper is not None else None,
                "lower_limit": float(lower) if lower is not None else None,
                "preclose_source": "baostock_unadjusted", "preclose_check": reference_check,
                "limit_source": limit_source, "tradability_source": "issuer_announcement" if official_halt else "baostock",
                "valuation_close": float(valuation) if valuation is not None else None,
                "valuation_source": "issuer_formula_derived" if valuation is not None else None,
            })
        previous = valuation if valuation is not None else prices["close"]

    observed = {row["date"].date() for row in out}
    if observed != sessions:
        issues.extend({"date": day.isoformat(), "check": "missing_baostock_bar"} for day in sorted(sessions - observed))
    for event_day in action_by_day:
        if period.start <= event_day <= period.end and event_day not in observed:
            issues.append({"date": event_day.isoformat(), "check": "action_date_has_no_session_bar"})
    return out, {"rows": len(out), **counts, "issues": issues}


def prepare_bulk_etf_snapshot(bulk: Path, output: Path, period: ResearchPeriod) -> dict:
    """Build an immutable research snapshot while leaving market evidence fail-closed."""
    bulk = Path(bulk).resolve()
    source_manifest = bulk / "manifest.json"
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    if source.get("complete") is not True:
        raise ValueError("bulk capture manifest is incomplete")
    named = [item for item in source["items"] if item.get("path")]
    records = {item["path"]: item for item in named}
    if len(records) != len(named):
        raise ValueError("duplicate source paths in bulk capture manifest")
    used = []
    sessions, calendar = _calendar(bulk, records, used, period)
    bars, actions, audits = [], [], {}
    for symbol in ETF_SYMBOLS:
        events = _actions(bulk, records, used, symbol, period)
        rows, audit = _bars(bulk, records, used, symbol, sessions, events, period)
        bars.extend(rows)
        actions.extend(event for event in events if event["ex_date"] in sessions)
        audits[symbol] = audit
    if not bars:
        raise ValueError("no BaoStock ETF rows overlap the captured calendar")

    bar_frame = pd.DataFrame(bars).sort_values(["date", "symbol"], ignore_index=True)
    for column in ("upper_limit", "lower_limit", "valuation_close"):
        bar_frame[column] = bar_frame[column].astype(float)
    action_frame = pd.DataFrame(actions, columns=list(_official_splits()[0]))
    for column in ("record_date", "ex_date", "pay_date", "share_available_date"):
        action_frame[column] = pd.to_datetime(action_frame[column])
    action_frame = action_frame.sort_values(["ex_date", "symbol"], ignore_index=True)
    clean = all(not audit["issues"] for audit in audits.values())
    evidence = {
        "provider": "etf_bulk_reconciliation", "capture_manifest_sha256": file_sha256(source_manifest),
        "input_files": used, "calendar": calendar, "by_symbol": audits,
        "corporate_action_sources": [_HALT_2015, _SPLIT_2015, _SPLIT_2022],
        "limit_rule_sources": list(_RULE_SOURCES),
        "limit_derivation": {"reference": "BaoStock unadjusted preclose", "rate": "0.10", "tick": "0.001",
                             "rounding": "ROUND_HALF_UP", "status": "ordinary_rule_candidate"},
    }
    return write_snapshot(
        output, origin="market", sources=[evidence],
        frames={"bars": bar_frame, "actions": action_frame,
                "calendar": pd.DataFrame({"date": pd.to_datetime(sorted(sessions))})},
        coverage={"exchange_calendar": calendar["status"], "etf_raw_bars": "cross_checked_ohlc" if clean else "mismatch",
                  "etf_actions": "partially_verified", "etf_limits": "rule_derived_candidate"},
    )
