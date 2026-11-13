"""Evaluate one paper buy observation against the original signal and limit."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation


def _timestamp(value: object, name: str) -> datetime:
    if isinstance(value, str):
        try:
            result = datetime.fromisoformat(value)
            if result.utcoffset() is not None:
                return result
        except ValueError:
            pass
    raise ValueError(f"{name} must be an ISO 8601 timestamp with a timezone")


def _price(value: object, name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be a positive price") from exc
    if not result.is_finite() or result <= 0:
        raise ValueError(f"{name} must be a positive price")
    return result


def _units(value: object, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def evaluate_paper_buy(observation: dict) -> dict:
    """Return a decision and its input evidence for later order reconciliation."""
    if not isinstance(observation, dict):
        raise ValueError("observation must be a JSON object")
    symbol = observation.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol is required")
    as_of = _timestamp(observation.get("as_of"), "as_of")
    signal = observation.get("signal")
    if signal is not None and type(signal) is not bool:
        raise ValueError("signal must be true, false or null")
    reason = "signal_unavailable" if signal is None else "signal_not_triggered"
    if signal is True:
        reason = _entry_reason(observation, as_of)
    return {
        "schema_version": 1,
        "account_type": "paper",
        "eligible": reason == "within_limit",
        "reason": reason,
        "observation": observation,
    }


def _entry_reason(observation: dict, as_of: datetime) -> str:
    signal_at = _timestamp(observation.get("signal_at"), "signal_at")
    expires_at = _timestamp(observation.get("expires_at"), "expires_at")
    if expires_at <= signal_at:
        raise ValueError("expires_at must be after signal_at")
    limit = _price(observation.get("limit_price"), "limit_price")
    units = _units(observation.get("ordered_units"), "ordered_units", 1)
    max_age = _units(observation.get("max_quote_age_seconds"), "max_quote_age_seconds", 1)
    if as_of < signal_at:
        return "signal_not_available_yet"
    if as_of >= expires_at:
        return "signal_expired"

    quote = observation.get("quote")
    if quote is None:
        return "quote_unavailable"
    if not isinstance(quote, dict) or quote.get("symbol") != observation["symbol"]:
        raise ValueError("quote must belong to the observed symbol")
    quote_at = _timestamp(quote.get("at"), "quote.at")
    age = (as_of - quote_at).total_seconds()
    if age < 0:
        return "quote_from_future"
    if quote_at < signal_at or age > max_age:
        return "quote_stale"
    if quote.get("ask_price") is None or quote.get("ask_units") is None:
        return "ask_data_unavailable"
    ask = _price(quote["ask_price"], "quote.ask_price")
    ask_units = _units(quote["ask_units"], "quote.ask_units", 0)
    if ask_units == 0:
        return "no_ask_liquidity"
    if ask > limit:
        return "ask_above_limit"
    if ask_units < units:
        return "insufficient_ask_units"
    return "within_limit"
