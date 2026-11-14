"""Collect exact-session unadjusted close evidence for the study holdings."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from scheduled_open import has_saved_capture


BASE = Path(__file__).resolve().parent
CST = ZoneInfo("Asia/Shanghai")
DAY = date(2026, 11, 16)
PRIOR_DAY = date(2026, 11, 13)
SESSIONS = (PRIOR_DAY, *(date(2026, 11, day) for day in range(16, 21)))
CLOSE = time(15, 0)
GROUPS = ("baseline", "filtered")
PROTOCOL_SHA256 = "477e3dcfcf51b4b21ddb3fd51ebee38cdfb126c4fcb5c9fc8b715da4dcf692fe"
SHORTLIST_SHA256 = "137ed04d32777fd23db94bad04be9562d17a8262531f34f71f9272bb24439d7c"


def positive_decimal(value):
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("invalid_numeric_market_field") from exc
    if not number.is_finite() or number <= 0:
        raise ValueError("nonpositive_or_nonfinite_market_field")
    return number


def download(url):
    request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"})
    with urlopen(request, timeout=8) as response:
        payload = response.read(2_000_001)
    if len(payload) > 2_000_000:
        raise ValueError("market_payload_too_large")
    return payload, datetime.now(CST)


def save_raw(out, symbol, vendor, payload):
    digest = hashlib.sha256(payload).hexdigest()
    path = Path("close-quotes") / symbol / f"{vendor}-{digest}.raw"
    target = out / path
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        if target.read_bytes() != payload:
            raise ValueError("raw_evidence_checksum_mismatch")
    return {"raw_file": path.as_posix(), "raw_sha256": digest}


def tencent_rows(payload, code, day, prior_day):
    response = json.loads(payload)
    if response.get("code") != 0:
        raise ValueError("tencent_unsuccessful_response")
    security = response["data"][code]
    rows = security.get("day") or []
    if [row[0] for row in rows][-2:] != [prior_day.isoformat(), day.isoformat()]:
        raise ValueError("tencent_exact_session_close_missing")
    return security, rows


def parse_tencent(payload, code, previous_close, observed_at, *, day=DAY, prior_day=PRIOR_DAY):
    security, rows = tencent_rows(payload, code, day, prior_day)
    previous, today = rows[-2:]
    raw_close = positive_decimal(today[2])
    positive_decimal(today[5])
    quote = security["qt"][code]
    quote_at = datetime.strptime(quote[30], "%Y%m%d%H%M%S").replace(tzinfo=CST)
    if (
        quote[2] != code[2:] or quote_at.date() != day
        or quote_at.time() < CLOSE or quote_at > observed_at
    ):
        raise ValueError("tencent_same_day_post_close_quote_missing")
    if raw_close != positive_decimal(quote[3]):
        raise ValueError("tencent_daily_and_quote_close_mismatch")
    if previous_close != positive_decimal(previous[2]) or previous_close != positive_decimal(quote[4]):
        raise ValueError("previous_close_mismatch_corporate_action_unresolved")
    return raw_close, quote_at


def parse_sina(payload, *, day=DAY, require_latest=True):
    body = payload.decode("utf-8-sig")
    start, end = body.find("[{"), body.rfind("]")
    if start == -1 or end < start:
        raise ValueError("sina_invalid_daily_payload")
    rows = json.loads(body[start:end + 1])
    considered = rows[-1:] if require_latest else rows
    matches = [row for row in considered if row.get("day") == day.isoformat()]
    if len(matches) != 1:
        raise ValueError("sina_exact_session_close_missing")
    positive_decimal(matches[0]["volume"])
    return positive_decimal(matches[0]["close"])


def prior_session(day):
    if day not in SESSIONS[1:]:
        raise ValueError("date_outside_frozen_experiment")
    return SESSIONS[SESSIONS.index(day) - 1]


def collect(out, symbol, previous_close, *, day=DAY):
    code = symbol[-2:].lower() + symbol[:6]
    previous_day = prior_session(day)
    tencent_url = ("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
                   f"?param={code},day,{previous_day},{day},12,")
    sina_url = ("https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_a=/"
                f"CN_MarketDataService.getKLineData?symbol={code}&scale=240&ma=no&datalen=8")
    record = {}
    recover_previous = previous_close is None
    try:
        payload, observed = download(tencent_url)
        record["tencent"] = {"source_url": tencent_url, "observed_at": observed.isoformat(),
                             **save_raw(out, symbol, "tencent", payload)}
        if recover_previous:
            _, rows = tencent_rows(payload, code, day, previous_day)
            previous_close = positive_decimal(rows[-2][2])
        close, quote_at = parse_tencent(payload, code, previous_close, observed, day=day, prior_day=previous_day)
        try:
            payload, observed = download(sina_url)
            record["sina"] = {"source_url": sina_url, "observed_at": observed.isoformat(),
                              **save_raw(out, symbol, "sina", payload)}
            sina_close = parse_sina(payload, day=day)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            record["sina"] = {**record.get("sina", {}), "status": "unavailable", "error": str(exc)}
            if recover_previous:
                raise ValueError("prior_close_evidence_missing") from exc
        else:
            if sina_close != close:
                raise ValueError("tencent_sina_close_mismatch")
            record["sina"]["status"] = "agreed"
            if recover_previous:
                try:
                    sina_previous = parse_sina(payload, day=previous_day, require_latest=False)
                except (ValueError, KeyError, TypeError, IndexError) as exc:
                    raise ValueError("prior_close_evidence_missing") from exc
                if sina_previous != previous_close:
                    raise ValueError("previous_close_mismatch_corporate_action_unresolved")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        return symbol, None, str(exc), record
    return symbol, {
        "date": day.isoformat(), "price_type": "unadjusted_close", "close": str(close),
        "quote_at": quote_at.isoformat(), "previous_close": str(previous_close),
        "previous_close_basis": "dual_source_recovery" if recover_previous else "opening_reference",
        **record["tencent"], "secondary_check": record["sina"],
    }, None, record


def held_symbols(opening, candidates=None, *, day=DAY):
    if opening.get("date") != day.isoformat() or opening.get("mode") != "forward":
        raise ValueError("valid same-day forward opening evidence required")
    if day == DAY:
        reference = {row["symbol"]: row["previous_close"] for row in candidates["candidates"]}
    else:
        reference = opening.get("prior_closes", {})
        if not isinstance(reference, dict):
            raise ValueError("opening prior_closes must be an object")
    held = {}
    for group in GROUPS:
        for symbol, units in opening[group]["positions"].items():
            if not re.fullmatch(r"\d{6}\.(?:SH|SZ)", symbol) or (day == DAY and symbol not in reference):
                raise ValueError("opening position outside supported securities")
            if type(units) is not int or units <= 0:
                raise ValueError("opening position has invalid quantity")
            value = reference.get(symbol)
            held[symbol] = positive_decimal(value) if value is not None else None
    return held


def load_frozen_inputs(day):
    protocol = (BASE / "protocol.md").read_bytes()
    if hashlib.sha256(protocol).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("frozen protocol hash mismatch")
    if day != DAY:
        return None
    shortlist = (BASE / "monday-preparation/candidates.json").read_bytes()
    if hashlib.sha256(shortlist).hexdigest() != SHORTLIST_SHA256:
        raise ValueError("Monday shortlist hash mismatch")
    return json.loads(shortlist)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, default=DAY)
    day = parser.parse_args().date
    prior_session(day)
    now = datetime.now(CST)
    if now.date() != day or now.time() < CLOSE:
        raise SystemExit(f"收盘采集只在北京时间 {day} 当天 15:00 后运行")
    candidates = load_frozen_inputs(day)
    out = BASE / day.isoformat()
    destination = out / "close_marks.json"
    if (out / "daily.json").is_file():
        if not destination.is_file():
            raise SystemExit("已有 daily.json，但缺少原收盘采集证据")
        print(destination.read_text(encoding="utf-8"))
        return
    opening_path = out / "opening.json"
    if day == DAY and not opening_path.exists() and not opening_path.is_symlink():
        if not has_saved_capture(out, legacy_retry=True):
            parser.exit(1, "周一开盘账本缺失，未找到可用于恢复的已保存盘口。\n")
        project = BASE.parents[1]
        research = project / "research_v1"
        try:
            subprocess.run(
                [str(research / ".venv/bin/python"), str(BASE / "scheduled_open.py"), "--date", str(day)],
                cwd=project, env={**os.environ, "PYTHONPATH": str(research)},
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as exc:
            lines = exc.stderr.strip().splitlines() if exc.stderr else []
            reason = lines[-1] if lines else f"开盘程序退出码 {exc.returncode}"
            parser.exit(1, f"周一开盘账本恢复失败：{reason}\n")
        except OSError as exc:
            parser.exit(1, f"周一开盘账本恢复失败：{exc}\n")
        if not opening_path.is_file():
            parser.exit(1, "周一开盘恢复未生成账本。\n")
    opening = json.loads(opening_path.read_bytes())
    held = held_symbols(opening, candidates, day=day)
    result = {"date": day.isoformat(), "marks": {}, "errors": {}, "attempts": {}}
    if destination.is_file():
        previous = json.loads(destination.read_bytes())
        if previous.get("date") != day.isoformat() or not set(previous.get("marks", {})) <= set(held):
            raise ValueError("existing close_marks does not match opening positions")
        result = previous
    to_collect = sorted(set(held) - set(result["marks"]))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for symbol, mark, error, evidence in pool.map(lambda s: collect(out, s, held[s], day=day), to_collect):
            result["attempts"][symbol] = evidence
            if mark is None:
                result["errors"][symbol] = error
            else:
                result["marks"][symbol] = mark
                result["errors"].pop(symbol, None)
    with tempfile.NamedTemporaryFile("w", dir=out, encoding="utf-8", delete=False) as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    try:
        if (out / "daily.json").is_file():
            if not destination.is_file():
                raise ValueError("daily ledger was published without the close marks")
            print(destination.read_text(encoding="utf-8"))
            return
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
