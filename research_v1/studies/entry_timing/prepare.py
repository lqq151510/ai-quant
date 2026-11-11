"""Capture public price references and construct common historical opportunities."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from decimal import Decimal
import csv
import hashlib
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


BASE = Path(__file__).resolve().parent
TZ = ZoneInfo("Asia/Shanghai")
PRICE_FIELDS = ("open", "high", "low", "close")
FIVE_MINUTE_ENDS = tuple(
    f"{minute // 60:02d}:{minute % 60:02d}"
    for start, stop in ((9 * 60 + 35, 11 * 60 + 30), (13 * 60 + 5, 15 * 60))
    for minute in range(start, stop + 1, 5)
)
FIELDS = (
    "symbol", "signal_date", "entry_date", "exit_date", "previous_close",
    "overnight_open", "delayed_open", "exit_open",
)


def download(url, referrer):
    request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": referrer})
    for attempt in range(2):
        try:
            with urlopen(request, timeout=10) as response:
                payload = response.read(2_000_001)
            if len(payload) > 2_000_000:
                raise ValueError("Unexpectedly large quote response")
            return payload
        except (URLError, TimeoutError):
            if attempt:
                raise


def parse_minutes(payload):
    body = payload.decode("utf-8-sig")
    rows = json.loads(body[body.index("["):body.rindex("]") + 1])
    days = defaultdict(dict)
    for row in rows:
        stamp = datetime.strptime(row["day"], "%Y-%m-%d %H:%M:%S")
        label = stamp.strftime("%H:%M")
        if stamp.second or label in days[stamp.date()]:
            raise ValueError(f"Duplicate or off-grid bar: {stamp}")
        prices = {key: Decimal(row[key]) for key in PRICE_FIELDS}
        if any(not p.is_finite() or p <= 0 for p in prices.values()):
            raise ValueError(f"Invalid bar prices: {stamp}")
        if not all(prices["low"] <= prices[key] <= prices["high"] for key in ("open", "close")):
            raise ValueError(f"Inconsistent OHLC: {stamp}")
        days[stamp.date()][label] = {**prices, "volume": int(row["volume"])}
    return days


def parse_daily(payload, code, field):
    rows = json.loads(payload)["data"][code][field]
    result = {date.fromisoformat(row[0]): tuple(map(Decimal, row[1:5])) for row in rows}
    if len(rows) != len(result):
        raise ValueError(f"Duplicate daily date for {code}")
    return result


def capture_symbol(symbol, protocol, sessions):
    code = f"sh{symbol}"
    start, end = sessions[0], sessions[-1]
    sina = (
        f"https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_{code}_5=/"
        f"CN_MarketDataService.getKLineData?symbol={code}&scale=5&ma=no&datalen=1023"
    )
    daily_url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,{start},{end},40,"
    quote_url = f"https://qt.gtimg.cn/q={code}"
    urls = {"sina_5m": sina, "tencent_daily": daily_url, "tencent_adjusted": daily_url + "qfq", "tencent_quote": quote_url}
    raw = {
        label: download(url, "https://finance.sina.com.cn/" if label == "sina_5m" else "https://gu.qq.com/")
        for label, url in urls.items()
    }
    bars = parse_minutes(raw["sina_5m"])
    daily = parse_daily(raw["tencent_daily"], code, "day")
    adjusted = parse_daily(raw["tencent_adjusted"], code, "qfqday")
    quote = raw["tencent_quote"].decode("gb18030").split('"')[1].split("~")
    if (
        set(daily) != set(sessions) or quote[2] != symbol or quote[30][:8] != end.strftime("%Y%m%d")
        or quote[30][8:] < "150000" or Decimal(quote[4]) != daily[sessions[-2]][1]
        or Decimal(quote[3]) != daily[end][1]
    ):
        raise ValueError(f"Unverified session calendar or final quote for {symbol}")
    if not set(sessions[:-1]) <= set(adjusted) or any(daily[d] != adjusted[d] for d in sessions[:-1]):
        raise ValueError(f"Price adjustment evidence needs review for {symbol}")
    for day in sessions:
        intraday = bars[day]
        if set(intraday) != set(FIVE_MINUTE_ENDS) or any(v["volume"] <= 0 for v in intraday.values()):
            raise ValueError(f"Incomplete traded bars for {symbol} on {day}")
        open_, close, high, low = daily[day]
        if (
            intraday["09:35"]["open"] != open_ or intraday["15:00"]["close"] != close
            or max(v["high"] for v in intraday.values()) != high
            or min(v["low"] for v in intraday.values()) != low
        ):
            raise ValueError(f"Daily and five-minute OHLC differ for {symbol} on {day}")
    entries = protocol["entry_labels"]
    opportunities = [
        dict(zip(FIELDS, (
            symbol, before, day, after, daily[before][1],
            bars[day][entries["overnight_limit"]]["open"],
            bars[day][entries["wait_0935"]]["open"], bars[after][protocol["exit_label"]]["open"],
        ), strict=True))
        for before, day, after in zip(sessions[:-2], sessions[1:-1], sessions[2:], strict=True)
    ]
    return opportunities, urls, raw, len(bars), {
        "opening_reference": str(bars[end][entries["overnight_limit"]]["open"]),
        "delayed_reference": str(bars[end][entries["wait_0935"]]["open"]),
        "close": str(daily[end][1]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=BASE / "protocol.json")
    parser.add_argument("--out", type=Path, required=True, help="New local input directory")
    args = parser.parse_args()
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    start = date.fromisoformat(protocol["first_complete_session"])
    end = date.fromisoformat(protocol["last_complete_session"])
    # This frozen span has no weekday exchange closures. Both feeds verify the complete calendar.
    calendar_days = [start + timedelta(days=delta) for delta in range((end - start).days + 1)]
    sessions = [day for day in calendar_days if day.weekday() < 5]
    output = args.out
    if output.exists():
        raise FileExistsError(output)
    symbols = protocol["symbols"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        captures = list(pool.map(lambda symbol: capture_symbol(symbol, protocol, sessions), symbols))
    output.mkdir(exist_ok=False)
    manifest = {
        "captured_at": datetime.now(TZ).isoformat(),
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "sessions": [day.isoformat() for day in sessions],
        "complete_sessions_per_symbol": len(sessions),
        "sources": {},
    }
    with (output / "opportunities.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for symbol, (opportunities, urls, raw, raw_dates, pending) in zip(symbols, captures, strict=True):
            writer.writerows(opportunities)
            sources = {}
            for label, payload in raw.items():
                filename = f"{symbol}-{label}.txt"
                (output / filename).write_bytes(payload)
                sources[label] = {"url": urls[label], "file": filename, "sha256": hashlib.sha256(payload).hexdigest()}
            manifest["sources"][symbol] = {"series": sources, "sina_reported_dates": raw_dates, "last_day_reference": pending}
    manifest["opportunities_sha256"] = hashlib.sha256((output / "opportunities.csv").read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(sessions)} complete sessions; {(len(sessions) - 2) * len(symbols)} paired opportunities")


if __name__ == "__main__":
    main()
