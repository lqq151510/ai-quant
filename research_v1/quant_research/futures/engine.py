"""Single-contract futures simulation with explicit assumptions and daily settlement.

Only completed daily bars reach the strategy. Its target is executable at the
following supplied session's open. This is a research approximation, not an
exchange matching engine.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Mapping, Protocol, Sequence

CENT = Decimal("0.01")
ZERO = Decimal("0")
ONE = Decimal("1")
SYMBOL = re.compile(r"^[A-Z]{2,6}\.[A-Za-z]{1,5}(\d{3,4})$")


class FuturesError(ValueError):
    """Invalid or insufficient inputs for the paper futures account."""


def number(value: object, name: str, *, minimum: Decimal | None = None) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise FuturesError(f"{name} must be a finite number") from exc
    if not result.is_finite() or (minimum is not None and result < minimum):
        raise FuturesError(f"{name} must be finite and >= {minimum}")
    return result


def money(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def whole(value: object, name: str, *, minimum: int = 0) -> int:
    parsed = number(value, name, minimum=Decimal(minimum))
    if parsed != parsed.to_integral_value():
        raise FuturesError(f"{name} must be an integer")
    return int(parsed)


def parse_day(value: object, name: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise FuturesError(f"{name} must be an ISO date (YYYY-MM-DD)") from exc


def validate_symbol(symbol: str) -> str:
    match = SYMBOL.fullmatch(symbol)
    if not match or match.group(1) in {"000", "888", "999", "0000", "8888", "9999"}:
        raise FuturesError("use a dated contract such as SHFE.ag2701, not a continuous/index symbol")
    return symbol


@dataclass(frozen=True)
class Contract:
    symbol: str
    multiplier: Decimal
    tick_size: Decimal
    margin_rate: Decimal
    max_margin_fraction: Decimal
    fee_open: Decimal
    fee_close: Decimal
    slippage_ticks: int
    max_lots: int
    exit_by: date

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Contract:
        fields = {"symbol", "multiplier", "tick_size", "margin_rate", "max_margin_fraction",
                  "fee_open", "fee_close", "slippage_ticks", "max_lots", "exit_by"}
        missing = fields - raw.keys()
        extra = raw.keys() - fields
        if missing or extra:
            raise FuturesError(f"contract configuration: missing={sorted(missing)}, extra={sorted(extra)}")
        spec = cls(
            symbol=validate_symbol(str(raw["symbol"])),
            multiplier=number(raw["multiplier"], "multiplier", minimum=ZERO),
            tick_size=number(raw["tick_size"], "tick_size", minimum=ZERO),
            margin_rate=number(raw["margin_rate"], "margin_rate", minimum=ZERO),
            max_margin_fraction=number(raw["max_margin_fraction"], "max_margin_fraction", minimum=ZERO),
            fee_open=number(raw["fee_open"], "fee_open", minimum=ZERO),
            fee_close=number(raw["fee_close"], "fee_close", minimum=ZERO),
            slippage_ticks=whole(raw["slippage_ticks"], "slippage_ticks"),
            max_lots=whole(raw["max_lots"], "max_lots", minimum=1),
            exit_by=parse_day(raw["exit_by"], "exit_by"),
        )
        if spec.multiplier <= ZERO or spec.tick_size <= ZERO:
            raise FuturesError("multiplier and tick_size must be positive")
        if not ZERO < spec.margin_rate <= ONE or not ZERO < spec.max_margin_fraction <= ONE:
            raise FuturesError("margin_rate and max_margin_fraction must be in (0, 1]")
        return spec


@dataclass(frozen=True)
class Bar:
    trading_day: date
    symbol: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    settlement: Decimal
    upper_limit: Decimal
    lower_limit: Decimal

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object], contract: Contract) -> Bar:
        names = {"trading_day", "symbol", "open", "high", "low", "close", "settlement", "upper_limit", "lower_limit"}
        missing = names - raw.keys()
        if missing:
            raise FuturesError(f"bar missing columns: {', '.join(sorted(missing))}")
        day = parse_day(raw["trading_day"], "trading_day")
        symbol = validate_symbol(str(raw["symbol"]))
        if symbol != contract.symbol or day > contract.exit_by:
            raise FuturesError("bar symbol differs from contract or trading_day is after exit_by")
        values = {name: number(raw[name], name, minimum=ZERO) for name in names - {"trading_day", "symbol"}}
        if any(value <= ZERO for value in values.values()):
            raise FuturesError("bar prices, settlement and limits must all be positive")
        if not values["lower_limit"] < values["upper_limit"]:
            raise FuturesError("lower_limit must be below upper_limit")
        if not values["low"] <= min(values["open"], values["close"]) <= max(values["open"], values["close"]) <= values["high"]:
            raise FuturesError("inconsistent OHLC prices")
        if not values["lower_limit"] <= values["low"] <= values["high"] <= values["upper_limit"]:
            raise FuturesError("OHLC prices are outside supplied daily limits")
        if not values["lower_limit"] <= values["settlement"] <= values["upper_limit"]:
            raise FuturesError("settlement is outside supplied daily limits")
        return cls(trading_day=day, symbol=symbol, **values)


def read_bars(path: Path, contract: Contract) -> list[Bar]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise FuturesError("CSV header is missing or contains duplicate columns")
        result = [Bar.from_mapping(row, contract) for row in reader]
    if not result:
        raise FuturesError("bars are empty")
    if any(a.trading_day >= b.trading_day for a, b in zip(result, result[1:])):
        raise FuturesError("bars must have unique and strictly increasing trading days")
    return result


class CloseStrategy(Protocol):
    def target(self, history: Sequence[Bar], position: int) -> int | None:
        """Return a signed target from completed bars, or None to keep the position."""


@dataclass(frozen=True)
class MovingAverage:
    fast: int
    slow: int
    lots: int = 1

    def __post_init__(self) -> None:
        if not 1 <= self.fast < self.slow or self.lots < 1:
            raise FuturesError("strategy requires 1 <= fast < slow and lots >= 1")

    def target(self, history: Sequence[Bar], position: int) -> int | None:
        if len(history) < self.slow:
            return None
        prices = [bar.close for bar in history]
        fast = sum(prices[-self.fast:], ZERO) / self.fast
        slow = sum(prices[-self.slow:], ZERO) / self.slow
        return self.lots if fast > slow else -self.lots if fast < slow else 0


@dataclass(frozen=True)
class RidgeForecast:
    """Expanding-time ridge baseline predicting next session's open-to-settle return."""

    lookback: int = 3
    min_train: int = 30
    train_window: int = 120
    alpha: float = 1.0
    threshold: float = 0.0005
    lots: int = 1

    def __post_init__(self) -> None:
        import math

        if (self.lookback < 1 or self.min_train < 3 or self.train_window < self.min_train or self.lots < 1
                or not math.isfinite(self.alpha) or self.alpha <= 0
                or not math.isfinite(self.threshold) or self.threshold < 0):
            raise FuturesError("invalid ridge strategy parameters")

    def target(self, history: Sequence[Bar], position: int) -> int | None:
        # A training label for decision i needs session i+1 to be complete.
        if len(history) < self.lookback + self.min_train + 1:
            return None
        import numpy as np

        def features(index: int) -> list[float]:
            return [float(history[index-j].close / history[index-j-1].close - ONE)
                    for j in range(self.lookback)]

        indices = list(range(self.lookback, len(history) - 1))[-self.train_window:]
        x = np.array([features(index) for index in indices], dtype=float)
        y = np.array([float((history[index+1].settlement - history[index+1].open) / history[index+1].open)
                      for index in indices], dtype=float)
        current = np.array(features(len(history) - 1), dtype=float)
        center, scale = x.mean(axis=0), x.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        normalized = (x - center) / scale
        coefficient = np.linalg.solve(normalized.T @ normalized + self.alpha * np.eye(self.lookback),
                                      normalized.T @ (y - y.mean()))
        prediction = float(y.mean() + ((current - center) / scale) @ coefficient)
        if not np.isfinite(prediction):
            raise FuturesError("ridge prediction is not finite")
        return self.lots if prediction > self.threshold else -self.lots if prediction < -self.threshold else 0


@dataclass(frozen=True)
class Result:
    equity: list[dict[str, object]]
    orders: list[dict[str, object]]
    decisions: list[dict[str, object]]
    status: str
    warnings: tuple[str, ...]


def backtest(bars: Sequence[Bar], contract: Contract, strategy: CloseStrategy, *, capital: object) -> Result:
    initial = money(number(capital, "capital", minimum=ZERO))
    if initial <= ZERO:
        raise FuturesError("capital must be positive")
    if not bars:
        raise FuturesError("bars are empty")
    if any(a.trading_day >= b.trading_day for a, b in zip(bars, bars[1:])):
        raise FuturesError("bars must have unique and strictly increasing trading days")
    if any(bar.symbol != contract.symbol or bar.trading_day > contract.exit_by for bar in bars):
        raise FuturesError("bar symbol differs from contract or trading_day is after exit_by")

    balance, peak = initial, initial
    position = 0
    previous_settle: Decimal | None = None
    pending: int | None = None
    pending_date: date | None = None
    history: list[Bar] = []
    equity: list[dict[str, object]] = []
    orders: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    warnings: list[str] = []
    status = "completed"

    for bar in bars:
        day = bar.trading_day
        gap = money(Decimal(position) * (bar.open - previous_settle) * contract.multiplier) if position and previous_settle is not None else ZERO
        opening_equity = balance + gap
        old_position = position
        fee_total, slip_total = ZERO, ZERO
        requested = pending
        decision_date = pending_date
        reason = "strategy"
        # exit_by is a user-provided last permitted holding day, not inferred from a code.
        if day == contract.exit_by:
            requested = 0
            reason = "exit_by"
        if requested is not None and requested != position:
            delta = requested - position
            side = 1 if delta > 0 else -1
            fill_price = bar.open + Decimal(side * contract.slippage_ticks) * contract.tick_size
            filled = 0
            closed, opened = 0, 0
            reject = ""
            if (side > 0 and (bar.open >= bar.upper_limit or fill_price > bar.upper_limit)) or (
                side < 0 and (bar.open <= bar.lower_limit or fill_price < bar.lower_limit)
            ) or fill_price <= ZERO:
                reject = "price_limit_or_slippage"
            else:
                close_count = min(abs(delta), abs(position)) if position * delta < 0 else 0
                if close_count:
                    close_fee = money(contract.fee_close * close_count)
                    close_slip = money(abs(fill_price - bar.open) * contract.multiplier * close_count)
                    opening_equity -= close_fee + close_slip
                    fee_total += close_fee
                    slip_total += close_slip
                    position += side * close_count
                    filled += side * close_count
                    closed = close_count
                open_count = abs(delta) - close_count
                if open_count:
                    candidate = position + side * open_count
                    open_fee = money(contract.fee_open * open_count)
                    open_slip = money(abs(fill_price - bar.open) * contract.multiplier * open_count)
                    after_cost = opening_equity - open_fee - open_slip
                    margin = money(Decimal(abs(candidate)) * max(bar.open, fill_price) * contract.multiplier * contract.margin_rate)
                    if abs(candidate) > contract.max_lots:
                        reject = "max_lots"
                    elif after_cost <= ZERO or margin > after_cost * contract.max_margin_fraction:
                        reject = "insufficient_margin_or_risk_limit"
                    else:
                        position = candidate
                        opening_equity = after_cost
                        fee_total += open_fee
                        slip_total += open_slip
                        filled += side * open_count
                        opened = open_count
            orders.append({"decision_day": decision_date.isoformat() if decision_date else "",
                           "trading_day": day.isoformat(), "trigger": reason, "requested_target": requested,
                           "position_before": old_position, "filled_delta": filled, "position_after": position,
                           "open_price": str(bar.open), "fill_price": str(fill_price) if filled else "",
                           "closed_lots": closed, "opened_lots": opened, "fee": str(fee_total),
                           "slippage": str(slip_total), "status": "filled" if not reject else "partial" if filled else "rejected",
                           "reason": reject})
        intraday = money(Decimal(position) * (bar.settlement - bar.open) * contract.multiplier)
        balance = money(opening_equity + intraday)
        margin = money(Decimal(abs(position)) * bar.settlement * contract.multiplier * contract.margin_rate)
        available = money(balance - margin)
        peak = max(peak, balance)
        drawdown = (peak - balance) / peak if peak > ZERO else ZERO
        margin_call = position != 0 and available < ZERO
        if margin_call and "margin_call" not in warnings:
            warnings.append("margin_call")
        equity.append({"trading_day": day.isoformat(), "position": position, "settlement": str(bar.settlement),
                       "gap_pnl": str(gap), "intraday_pnl": str(intraday), "fees": str(fee_total),
                       "slippage": str(slip_total), "equity": str(balance), "margin": str(margin),
                       "available": str(available), "drawdown": str(drawdown), "margin_call": margin_call})
        previous_settle = bar.settlement
        history.append(bar)
        pending = strategy.target(tuple(history), position)
        if pending is not None:
            if isinstance(pending, bool) or not isinstance(pending, int) or abs(pending) > contract.max_lots:
                raise FuturesError("strategy target must be an integer within max_lots")
            pending_date = day
            decisions.append({"trading_day": day.isoformat(), "position": position, "target": pending,
                              "execution": "next_supplied_session_open"})
        else:
            pending_date = None
        if day == contract.exit_by and position:
            status = "exit_unresolved"
            warnings.append("position_remains_on_exit_by")
    if pending is not None and bars[-1].trading_day < contract.exit_by:
        warnings.append("last_decision_not_executed")
    if bars[-1].trading_day < contract.exit_by and position:
        warnings.append("open_position_at_end")
    return Result(equity, orders, decisions, status, tuple(warnings))
