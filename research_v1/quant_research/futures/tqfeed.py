"""Optional TqSdk quote-only adapter; it cannot submit orders to a broker."""

from __future__ import annotations

import fcntl
import getpass
import json
import math
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zoneinfo import ZoneInfo

from .engine import Contract, FuturesError, number, parse_day
from .paper import timestamp

SHANGHAI = ZoneInfo("Asia/Shanghai")


def quote_event(quote: object, contract: Contract, day: date, observed: datetime, max_age: float) -> dict | None:
    """Translate only recent, complete market data; time means local observation time."""
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise FuturesError("observation time must have a timezone")
    try:
        raw_time = str(quote.datetime)
        exchange_time = datetime.fromisoformat(raw_time)
        if exchange_time.tzinfo is None:
            exchange_time = exchange_time.replace(tzinfo=SHANGHAI)
        age = (observed - exchange_time).total_seconds()
        values = {"last": float(quote.last_price), "bid": float(quote.bid_price1), "ask": float(quote.ask_price1),
                  "bid_size": int(quote.bid_volume1), "ask_size": int(quote.ask_volume1),
                  "upper_limit": float(quote.upper_limit), "lower_limit": float(quote.lower_limit)}
        tick = Decimal(str(quote.price_tick))
        multiplier = Decimal(str(quote.volume_multiple))
    except (ValueError, TypeError, OverflowError, AttributeError, InvalidOperation):
        return None
    if not -2 <= age <= max_age or not all(math.isfinite(float(value)) for value in values.values()):
        return None
    if not tick.is_finite() or not multiplier.is_finite() or tick <= 0 or multiplier <= 0:
        return None
    if tick != contract.tick_size or multiplier != contract.multiplier:
        raise FuturesError("feed contract tick_size or multiplier does not match config")
    if min(values["last"], values["bid"], values["ask"], values["lower_limit"]) <= 0:
        return None
    if not values["lower_limit"] <= values["bid"] <= values["ask"] <= values["upper_limit"]:
        return None
    if not values["lower_limit"] <= values["last"] <= values["upper_limit"] or min(values["bid_size"], values["ask_size"]) < 0:
        return None
    return {"type": "quote", "time": observed.isoformat(), "exchange_time": exchange_time.isoformat(),
            "trading_day": day.isoformat(), "symbol": contract.symbol, "source": "tqsdk", **values}


def completed_bar(row: object, contract: Contract, day: date, observed: datetime, period: int,
                  last_bar_end: datetime | None, max_age: float) -> dict | None:
    try:
        start_ns = int(row["datetime"])
        close = float(row["close"])
        end = datetime.fromtimestamp(start_ns / 1_000_000_000, tz=timezone.utc) + timedelta(seconds=period)
    except (ValueError, TypeError, OverflowError, KeyError, OSError):
        return None
    if not math.isfinite(close) or close <= 0 or end > observed:
        return None
    if (observed - end).total_seconds() > max_age or (last_bar_end is not None and end <= last_bar_end):
        return None
    return {"type": "bar", "time": observed.isoformat(), "bar_end": end.isoformat(),
            "trading_day": day.isoformat(), "symbol": contract.symbol, "close": close, "source": "tqsdk"}


def _tail(events: Path, symbol: str) -> tuple[dict | None, datetime | None]:
    last, last_bar = None, None
    if not events.exists():
        return last, last_bar
    raw = events.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise FuturesError("event file has an incomplete last line")
    try:
        for line in raw.splitlines():
            item = json.loads(line)
            if not isinstance(item, dict) or item.get("symbol") != symbol:
                raise FuturesError("event file contains a different symbol or invalid event")
            if item.get("type") == "bar":
                last_bar = timestamp(item.get("bar_end"), "bar_end")
            last = item
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FuturesError("event file contains invalid JSON") from exc
    return last, last_bar


def append_settlement(events: Path, contract: Contract, *, day: str, price: object,
                      observed: datetime | None = None) -> None:
    trading_day = parse_day(day, "trading_day")
    settlement = number(price, "settlement price", minimum=Decimal("0"))
    if settlement <= 0 or not events.is_file():
        raise FuturesError("settlement must be positive and event file must exist")
    now = observed or datetime.now(SHANGHAI)
    if now.tzinfo is None or now.utcoffset() is None:
        raise FuturesError("settlement observation time requires a timezone")
    lock_path = events.with_name(events.name + ".feed.lock")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FuturesError("stop the feed before appending settlement") from exc
        last, _ = _tail(events, contract.symbol)
        if not last or last.get("trading_day") != trading_day.isoformat() or last.get("type") == "settlement":
            raise FuturesError("settlement day must match the last unsettled event")
        if now < timestamp(last.get("time"), "last event time"):
            raise FuturesError("settlement observation time precedes the last event")
        item = {"type": "settlement", "time": now.isoformat(), "trading_day": trading_day.isoformat(),
                "symbol": contract.symbol, "price": str(settlement), "source": "manual_unverified"}
        with events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def run_feed(events: Path, contract: Contract, *, trading_day: str, seconds: int,
             period: int = 60, max_age: float = 30, sdk: object | None = None) -> None:
    day = parse_day(trading_day, "trading_day")
    if day > contract.exit_by or seconds < 1 or seconds > 86400 or period not in (60, 300, 900):
        raise FuturesError("check trading day, duration (1–86400 seconds), and bar period (60/300/900)")
    if not 1 <= max_age <= 60:
        raise FuturesError("max_age must be 1–60 seconds")
    username = os.environ.get("TQ_ACCOUNT") or (input("快期行情账号: ").strip() if sys.stdin.isatty() else "")
    password = os.environ.get("TQ_PASSWORD") or (getpass.getpass("快期行情密码: ") if sys.stdin.isatty() else "")
    if not username or not password:
        raise FuturesError("行情需要快期账号；请在终端输入或设置 TQ_ACCOUNT/TQ_PASSWORD")
    if sdk is None:
        try:
            import tqsdk as sdk
        except ImportError as exc:
            raise FuturesError("未安装可选依赖 tqsdk，参见 FUTURES_PAPER.md") from exc
    if not events.parent.is_dir():
        raise FuturesError("event directory must already exist")
    lock_path = events.with_name(events.name + ".feed.lock")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FuturesError("another feed is writing this file") from exc
        last, last_bar = _tail(events, contract.symbol)
        if last:
            prior_day = str(last.get("trading_day", ""))
            if prior_day > day.isoformat() or (prior_day == day.isoformat() and last.get("type") == "settlement"):
                raise FuturesError("event file has already advanced beyond this trading day")
            if prior_day < day.isoformat() and last.get("type") != "settlement":
                raise FuturesError("record prior day's official settlement before starting the next trading day")
        last_event_time = timestamp(last["time"], "last event time") if last else None
        last_exchange = None
        written = 0
        # Explicit local TqSim is used only to create a market-data client. No order API is called.
        try:
            sim = sdk.TqSim(init_balance=1)
            with sdk.TqApi(sim, auth=sdk.TqAuth(username, password)) as api:
                quote = api.get_quote(contract.symbol)
                bars = api.get_kline_serial(contract.symbol, duration_seconds=period, data_length=4)
                deadline = time.monotonic() + seconds
                with events.open("a", encoding="utf-8") as handle:
                    while time.monotonic() < deadline:
                        if not api.wait_update(deadline=time.time() + min(5, max(0.1, deadline-time.monotonic()))):
                            continue
                        observed = datetime.now(SHANGHAI)
                        if last_event_time and observed <= last_event_time:
                            continue
                        current = quote_event(quote, contract, day, observed, max_age)
                        if current is None or current["exchange_time"] == last_exchange:
                            continue
                        if last_exchange and timestamp(current["exchange_time"], "exchange_time") <= timestamp(last_exchange, "exchange_time"):
                            continue
                        handle.write(json.dumps(current, ensure_ascii=False, allow_nan=False) + "\n")
                        last_exchange, last_event_time = current["exchange_time"], observed
                        written += 1
                        # The last K line is in progress; only inspect the penultimate one.
                        if len(bars) >= 2:
                            bar = completed_bar(bars.iloc[-2], contract, day, observed, period, last_bar, max_age)
                            if bar is not None:
                                handle.write(json.dumps(bar, ensure_ascii=False, allow_nan=False) + "\n")
                                last_bar = timestamp(bar["bar_end"], "bar_end")
                        handle.flush()
                        os.fsync(handle.fileno())
        except FuturesError:
            raise
        except Exception as exc:
            # SDK exceptions may contain credentials or service URLs: don't put them in logs.
            raise FuturesError(f"行情连接或读取失败（{type(exc).__name__}）；未记录账号或密码") from None
    print(json.dumps({"source": "tqsdk_quote_only", "quote_events": written,
                      "events": str(events.resolve())}, ensure_ascii=False), flush=True)
