"""Append-only market events and a restartable local futures paper account.

This module has no broker, network or real-account API. The feed is separate
from the matcher so that the same event format works with recorded data.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import tempfile
import time
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Mapping

from .engine import Contract, FuturesError, MovingAverage, ZERO, money, number, parse_day, whole
from .output import report_html, sha256


def timestamp(value: object, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise FuturesError(f"{name} must be an ISO timestamp including a timezone") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise FuturesError(f"{name} requires a timezone")
    return parsed


def initial_state(contract: Contract, strategy: MovingAverage, capital: object, config_hash: str,
                  events_path: Path, signal_mode: str = "moving_average") -> dict:
    amount = money(number(capital, "capital", minimum=ZERO))
    if amount <= ZERO or strategy.lots > contract.max_lots or signal_mode not in ("moving_average", "external"):
        raise FuturesError("capital must be positive and lots must not exceed max_lots")
    return {"schema": "futures-paper-v1", "symbol": contract.symbol, "capital": str(amount),
            "cash": str(amount), "position": 0, "basis": None, "mark": None,
            "config_sha256": config_hash, "events_path": str(events_path.resolve()),
            "account_code_sha256": {name: sha256(Path(__file__).with_name(name)) for name in ("paper.py", "engine.py")},
            "strategy": {"mode": signal_mode, "fast": strategy.fast, "slow": strategy.slow, "lots": strategy.lots},
            "offset": 0, "prefix_sha256": hashlib.sha256(b"").hexdigest(),
            "last_event_time": None, "last_quote_time": None, "last_bar_end": None,
            "current_trading_day": None, "last_settled_day": None,
            "closes": [], "pending": None, "orders": [], "equity": [], "warnings": []}


def _metrics(state: dict, contract: Contract) -> tuple[Decimal, Decimal, Decimal]:
    cash = Decimal(state["cash"])
    qty = int(state["position"])
    mark = Decimal(state["mark"]) if state["mark"] is not None else ZERO
    basis = Decimal(state["basis"]) if state["basis"] is not None else ZERO
    equity = money(cash + Decimal(qty) * (mark - basis) * contract.multiplier)
    margin = money(Decimal(abs(qty)) * mark * contract.multiplier * contract.margin_rate)
    return equity, margin, money(equity - margin)


def _quote(event: Mapping[str, object], contract: Contract) -> dict:
    required = {"type", "time", "symbol", "trading_day", "last", "bid", "ask", "bid_size", "ask_size", "upper_limit", "lower_limit"}
    missing = required - event.keys()
    if missing:
        raise FuturesError(f"quote missing: {sorted(missing)}")
    vals = {key: number(event[key], key, minimum=ZERO) for key in ("last", "bid", "ask", "upper_limit", "lower_limit")}
    vals.update({key: whole(event[key], key) for key in ("bid_size", "ask_size")})
    if min(vals[k] for k in ("last", "bid", "ask", "lower_limit")) <= ZERO:
        raise FuturesError("quote prices must be positive")
    if not vals["lower_limit"] <= vals["bid"] <= vals["ask"] <= vals["upper_limit"] or not vals["lower_limit"] <= vals["last"] <= vals["upper_limit"]:
        raise FuturesError("crossed quote or quote outside daily limits")
    return vals


def _execute(state: dict, contract: Contract, quote: dict, event_time: datetime) -> None:
    pending = state["pending"]
    if pending is None or event_time <= timestamp(pending["time"], "pending time"):
        return
    target = int(pending["target"])
    qty = int(state["position"])
    delta = target - qty
    if not delta:
        state["pending"] = None
        return
    if (event_time - timestamp(pending["time"], "pending time")).total_seconds() > 60:
        state["orders"].append({"trading_day": event_time.isoformat(), "requested_target": target,
                                "filled_delta": 0, "fill_price": "", "status": "cancelled", "reason": "signal_expired_60s"})
        state["pending"] = None
        return
    side = 1 if delta > 0 else -1
    price = quote["ask"] + contract.tick_size * contract.slippage_ticks if side > 0 else quote["bid"] - contract.tick_size * contract.slippage_ticks
    available_lots = quote["ask_size"] if side > 0 else quote["bid_size"]
    reject = ""
    filled = 0
    fees = ZERO
    if price <= ZERO or price > quote["upper_limit"] or price < quote["lower_limit"]:
        reject = "price_limit_or_slippage"
    elif available_lots == 0:
        reject = "no_quoted_liquidity"
    else:
        wanted = min(abs(delta), available_lots)
        close_count = min(wanted, abs(qty)) if qty * delta < 0 else 0
        if close_count:
            basis = Decimal(state["basis"])
            pnl = money(Decimal(close_count * (1 if qty > 0 else -1)) * (price - basis) * contract.multiplier)
            close_fee = money(contract.fee_close * close_count)
            state["cash"] = str(money(Decimal(state["cash"]) + pnl - close_fee))
            qty += side * close_count
            filled += side * close_count
            fees += close_fee
            state["position"] = qty
            if qty == 0:
                state["basis"] = None
        open_count = wanted - close_count
        if open_count:
            candidate = qty + side * open_count
            old_basis = Decimal(state["basis"]) if state["basis"] is not None else ZERO
            new_basis = ((old_basis * abs(qty) + price * open_count) / abs(candidate))
            open_fee = money(contract.fee_open * open_count)
            cash = money(Decimal(state["cash"]) - open_fee)
            equity_after = money(cash + Decimal(candidate) * (quote["last"] - new_basis) * contract.multiplier)
            margin_after = money(Decimal(abs(candidate)) * max(quote["last"], price) * contract.multiplier * contract.margin_rate)
            if abs(candidate) > contract.max_lots or equity_after <= ZERO or margin_after > equity_after * contract.max_margin_fraction:
                reject = "insufficient_margin_or_risk_limit"
            else:
                qty = candidate
                filled += side * open_count
                fees += open_fee
                state["cash"], state["position"], state["basis"] = str(cash), qty, str(new_basis)
        if wanted < abs(delta) and not reject:
            reject = "limited_quoted_liquidity"
    state["orders"].append({"trading_day": event_time.isoformat(), "requested_target": target,
                            "filled_delta": filled, "fill_price": str(price) if filled else "",
                            "fee": str(fees), "status": "filled" if not reject else "partial" if filled else "rejected",
                            "reason": reject})
    # Never repeatedly submit a stale or rejected idea. The next completed bar can create another.
    state["pending"] = None


def apply_event(state: dict, event: Mapping[str, object], contract: Contract, strategy: MovingAverage) -> None:
    if event.get("symbol") != contract.symbol:
        raise FuturesError("event symbol does not match this session")
    now = timestamp(event.get("time"), "event time")
    previous = timestamp(state["last_event_time"], "last event time") if state["last_event_time"] else None
    if previous is not None and now < previous:
        raise FuturesError("events must be in nondecreasing observation-time order")
    kind = event.get("type")
    day = parse_day(event.get("trading_day"), "trading_day")
    if day > contract.exit_by:
        raise FuturesError("event trading_day is after exit_by")
    if kind == "quote":
        values = _quote(event, contract)
        if state["last_quote_time"] and now <= timestamp(state["last_quote_time"], "last quote time"):
            raise FuturesError("quote timestamps must be strictly increasing")
        previous_day = state["current_trading_day"]
        if previous_day and day.isoformat() < previous_day:
            raise FuturesError("trading_day moved backwards")
        if previous_day and day.isoformat() != previous_day and state["position"] and state["last_settled_day"] != previous_day:
            raise FuturesError("a held position requires the prior trading day's settlement event before new-day quotes")
        if state["last_settled_day"] == day.isoformat():
            raise FuturesError("quote received after settlement for the same trading day")
        state["current_trading_day"] = day.isoformat()
        state["mark"] = str(values["last"])
        if day == contract.exit_by:
            if state["position"]:
                # Set an exit intent at this quote; it can fill only on a later quote.
                if not state["pending"] or state["pending"]["target"] != 0:
                    state["pending"] = {"target": 0, "time": now.isoformat(), "reason": "exit_by"}
            else:
                state["pending"] = None  # Never open on the last permitted holding day.
        _execute(state, contract, values, now)
        state["last_quote_time"] = now.isoformat()
        equity, margin, available = _metrics(state, contract)
        snapshot = {"trading_day": now.isoformat(), "equity": str(equity), "cash": state["cash"],
                    "position": state["position"], "margin": str(margin), "available": str(available)}
        bucket = now.replace(second=0, microsecond=0)
        if state["equity"] and timestamp(state["equity"][-1]["trading_day"], "equity time").replace(second=0, microsecond=0) == bucket:
            state["equity"][-1] = snapshot
        else:
            state["equity"].append(snapshot)
        if state["position"] and available < ZERO and "margin_call" not in state["warnings"]:
            state["warnings"].append("margin_call")
    elif kind == "bar":
        if state["current_trading_day"] != day.isoformat() or state["last_settled_day"] == day.isoformat():
            raise FuturesError("bar requires an active quote from the same unsettled trading day")
        end = timestamp(event.get("bar_end"), "bar_end")
        if end > now or (state["last_bar_end"] and end <= timestamp(state["last_bar_end"], "last bar end")):
            raise FuturesError("bar must be completed and strictly later than the prior bar")
        close = number(event.get("close"), "close", minimum=ZERO)
        if close <= ZERO:
            raise FuturesError("bar close must be positive")
        state["last_bar_end"] = end.isoformat()
        state["closes"] = (state["closes"] + [str(close)])[-strategy.slow:]
        if state["strategy"]["mode"] == "moving_average" and len(state["closes"]) >= strategy.slow and day < contract.exit_by:
            closes = [Decimal(value) for value in state["closes"]]
            fast = sum(closes[-strategy.fast:], ZERO) / strategy.fast
            slow = sum(closes[-strategy.slow:], ZERO) / strategy.slow
            target = strategy.lots if fast > slow else -strategy.lots if fast < slow else 0
            state["pending"] = {"target": target, "time": now.isoformat(), "reason": "completed_bar"}
    elif kind == "signal":
        if state["strategy"]["mode"] != "external":
            raise FuturesError("external signal requires external strategy mode")
        if state["current_trading_day"] != day.isoformat() or state["last_settled_day"] == day.isoformat() or day >= contract.exit_by:
            raise FuturesError("external signal requires an active, unexpired trading day")
        cutoff = timestamp(event.get("feature_cutoff"), "feature_cutoff")
        if cutoff > now:
            raise FuturesError("feature_cutoff cannot be after signal availability")
        model = event.get("model_id")
        target = event.get("target")
        if not isinstance(model, str) or not model.strip():
            raise FuturesError("external signal requires model_id")
        if isinstance(target, bool) or not isinstance(target, int) or abs(target) > contract.max_lots:
            raise FuturesError("external target must be an integer within max_lots")
        state["pending"] = {"target": target, "time": now.isoformat(), "reason": "external_signal",
                            "feature_cutoff": cutoff.isoformat(), "model_id": model}
    elif kind == "settlement":
        if state["current_trading_day"] != day.isoformat() or state["last_settled_day"] == day.isoformat():
            raise FuturesError("settlement must be unique and match the current trading day")
        price = number(event.get("price"), "settlement price", minimum=ZERO)
        if price <= ZERO:
            raise FuturesError("settlement price must be positive")
        if state["position"]:
            pnl = money(Decimal(state["position"]) * (price - Decimal(state["basis"])) * contract.multiplier)
            state["cash"] = str(money(Decimal(state["cash"]) + pnl))
            state["basis"] = str(price)
        state["last_settled_day"] = day.isoformat()
        state["pending"] = None
    else:
        raise FuturesError("event type must be quote, bar, signal, or settlement")
    state["last_event_time"] = now.isoformat()


def _atomic(path: Path, value: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _advance(state: dict, raw: bytes, contract: Contract, strategy: MovingAverage) -> dict:
    offset = state["offset"]
    if len(raw) < offset or hashlib.sha256(raw[:offset]).hexdigest() != state["prefix_sha256"]:
        raise FuturesError("event file was truncated or previously processed data was changed")
    # Ignore an incomplete final line while an external producer is writing it.
    end = raw.rfind(b"\n") + 1
    if end <= offset:
        return state
    next_state = copy.deepcopy(state)
    for line in raw[offset:end].splitlines():
        if not line.strip():
            raise FuturesError("blank event lines are not permitted")
        try:
            event = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FuturesError("invalid UTF-8 JSON event") from exc
        if not isinstance(event, dict):
            raise FuturesError("each event must be a JSON object")
        apply_event(next_state, event, contract, strategy)
    next_state["offset"] = end
    next_state["prefix_sha256"] = hashlib.sha256(raw[:end]).hexdigest()
    return next_state


def run_session(session: Path, events: Path, config_path: Path, contract: Contract, strategy: MovingAverage,
                *, capital: object, watch: bool = False, poll_seconds: float = 2,
                signal_mode: str = "moving_average") -> None:
    if not 0.2 <= poll_seconds <= 60:
        raise FuturesError("poll_seconds must be between 0.2 and 60")
    if not events.is_file():
        raise FuturesError("events file must already exist")
    if not session.exists():
        session.mkdir(parents=True, exist_ok=False)
        created = True
    else:
        created = False
        if not (session / "state.json").is_file():
            raise FuturesError("existing session directory is not a futures paper session")
    with (session / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FuturesError("paper session is already running") from exc
        expected = initial_state(contract, strategy, capital, sha256(config_path), events, signal_mode)
        if created:
            state = expected
            _atomic(session / "state.json", json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        else:
            state = json.loads((session / "state.json").read_text(encoding="utf-8"))
            for key in ("schema", "symbol", "capital", "config_sha256", "events_path", "strategy", "account_code_sha256"):
                if state.get(key) != expected[key]:
                    raise FuturesError(f"session {key} differs; use the original inputs or a new session")
        while True:
            raw = events.read_bytes()
            updated = _advance(state, raw, contract, strategy)
            if updated is not state or not (session / "report.html").exists():
                state = updated
                _atomic(session / "state.json", json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
                explanation = ("本地虚拟资金；行情事件和结算价由输入方提供，尚未经独立核验。"
                               "成交按下一笔报价和显示手数近似，未模拟排队或市场冲击。")
                report = report_html(title="期货本地模拟账户", explanation=explanation, equity=state["equity"],
                                     orders=state["orders"], status="paper_only", capital=state["capital"],
                                     symbol=state["symbol"], pending=state["pending"])
                _atomic(session / "report.html", report)
                print(json.dumps({"processed_bytes": state["offset"], "position": state["position"],
                                  "report": str((session / "report.html").resolve())}, ensure_ascii=False), flush=True)
            if not watch:
                return
            time.sleep(poll_seconds)
