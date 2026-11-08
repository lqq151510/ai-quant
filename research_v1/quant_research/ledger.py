"""Deterministic opening auction approximation for A shares and stock ETFs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
from math import floor, isfinite
from types import MappingProxyType

import pandas as pd


class LedgerError(ValueError):
    """The supplied inputs cannot support an auditable account."""


DateLike = date | pd.Timestamp | str
Rates = tuple[tuple[str, float], ...]
LOT_SIZE = 100
_CENT = Decimal("0.01")
_EPSILON = 1e-9
_BARS = ("date", "symbol", "kind", "open", "high", "low", "close",
         "volume", "amount", "tradable", "upper_limit", "lower_limit")
_ACTIONS = ("symbol", "ex_date", "pay_date", "cash_per_share", "share_multiplier")
_EQUITY = ("date", "cash", "market_value", "receivable", "nav", "daily_return", "drawdown")
_ORDERS = ("date", "symbol", "side", "requested_qty", "filled_qty", "price",
           "commission", "stamp_tax", "transfer_fee", "status", "reason", "notional",
           "cash_delta")
_POSITIONS = ("date", "symbol", "quantity", "sellable_quantity", "close", "market_value")
_EVENTS = ("date", "symbol", "event", "cash_delta", "receivable_delta", "quantity_delta")


def _date(value: DateLike) -> date:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise LedgerError("missing date")
    if ts.tzinfo is not None:
        ts = ts.tz_convert("Asia/Shanghai")
    return ts.date()


def _money(value: float) -> float:
    return float(Decimal(str(value)).quantize(_CENT, rounding=ROUND_HALF_UP))


def _positive(value: float) -> bool:
    return isfinite(value) and value > 0


@dataclass(frozen=True)
class Costs:
    """Commission and slippage are assumptions. Stock taxes are dated rates."""

    commission_rate: float = 0.0003
    minimum_commission: float = 5.0
    slippage_bps: float = 5.0
    # MOF single-sided stamp duty and 2023 halving announcements.
    stamp_rates: Rates = (("2008-09-19", 0.001), ("2023-08-28", 0.0005))
    # ChinaClear unified turnover-based A-share fee from August 2015.
    transfer_rates: Rates = (("2015-08-01", 0.00002), ("2022-04-29", 0.00001))

    def __post_init__(self) -> None:
        for value in (self.commission_rate, self.minimum_commission, self.slippage_bps):
            if not isfinite(value) or value < 0:
                raise LedgerError("cost assumptions must be finite and non-negative")
        if self.slippage_bps >= 10_000:
            raise LedgerError("slippage_bps must be below 10000")
        for schedule in (self.stamp_rates, self.transfer_rates):
            dates = [date.fromisoformat(effective) for effective, _ in schedule]
            if dates != sorted(set(dates)):
                raise LedgerError("fee dates must be unique and increasing")
            if any(not isfinite(rate) or rate < 0 for _, rate in schedule):
                raise LedgerError("fee rates must be finite and non-negative")

    def stressed(self) -> Costs:
        """Double assumed commission, its floor, and slippage."""
        return replace(self, commission_rate=2 * self.commission_rate,
                       minimum_commission=2 * self.minimum_commission,
                       slippage_bps=2 * self.slippage_bps)

    @staticmethod
    def _rate(day: date, schedule: Rates) -> float:
        iso = day.isoformat()
        for effective, rate in reversed(schedule):
            if effective <= iso:
                return rate
        raise LedgerError(f"missing historical stock fee schedule for {iso}")

    def charges(self, day: date, kind: str, side: str, notional: float) -> tuple[float, float, float]:
        commission = _money(max(self.minimum_commission, notional * self.commission_rate))
        if kind == "etf":
            return commission, 0.0, 0.0
        transfer = _money(notional * self._rate(day, self.transfer_rates))
        stamp = _money(notional * self._rate(day, self.stamp_rates)) if side == "sell" else 0.0
        return commission, stamp, transfer


@dataclass
class _Lot:
    quantity: int
    available: date


@dataclass
class _Position:
    lots: list[_Lot] = field(default_factory=list)

    @property
    def quantity(self) -> int:
        return sum(lot.quantity for lot in self.lots)

    def sellable(self, day: date) -> int:
        return sum(lot.quantity for lot in self.lots if lot.available <= day)

    def buy(self, day: date, quantity: int) -> None:
        self.lots.append(_Lot(quantity, day + timedelta(days=1)))

    def sell(self, day: date, quantity: int) -> None:
        if quantity > self.sellable(day):
            raise LedgerError("sell would use units that are still locked")
        for lot in sorted(self.lots, key=lambda entry: entry.available):
            if lot.available <= day:
                consumed = min(lot.quantity, quantity)
                lot.quantity -= consumed
                quantity -= consumed
        self.lots = [lot for lot in self.lots if lot.quantity]

    def apply_split(self, multiplier: float, available: date, rounding: str) -> None:
        if rounding != "exact":
            self._apply_rounded_split(multiplier, available, rounding)
            return
        # Round only exact integer quantities. Fractional entitlements need
        # their own verified cash-in-lieu inputs.
        quantities = [lot.quantity * multiplier for lot in self.lots]
        if any(abs(qty - round(qty)) > 1e-8 for qty in quantities):
            raise LedgerError("split creates fractional units in a held lot")
        if multiplier >= 1:
            extra = sum(round(qty) for qty in quantities) - self.quantity
            if extra:
                self.lots.append(_Lot(extra, available))
        else:
            for lot, quantity in zip(self.lots, quantities, strict=True):
                lot.quantity = round(quantity)
            self.lots = [lot for lot in self.lots if lot.quantity]

    def _apply_rounded_split(self, multiplier: float, available: date, rounding: str) -> None:
        factor = Decimal(str(multiplier))
        mode = ROUND_FLOOR if rounding == "floor" else ROUND_CEILING
        old_total = new_total = 0
        updated = []
        # Round the account total once. Assign the integer units in release
        # order so acquiring shares in multiple orders does not change entitlement.
        for lot in sorted(self.lots, key=lambda entry: entry.available):
            old_total += lot.quantity
            converted_total = int((old_total * factor).to_integral_value(rounding=mode))
            converted = converted_total - new_total
            new_total = converted_total
            if multiplier >= 1:
                updated.append(lot)
                converted -= lot.quantity
            if converted:
                updated.append(_Lot(converted, max(lot.available, available)))
        self.lots = updated


@dataclass(frozen=True)
class _Action:
    symbol: str
    ex_date: date
    record_date: date
    pay_date: date | None
    cash_per_share: float
    share_multiplier: float
    share_available_date: date
    split_rounding: str


@dataclass(frozen=True)
class CloseState:
    """Settled portfolio at a session close, before the next opening auction."""

    day: date
    weights: Mapping[str, float]
    cash_weight: float
    drawdown: float
    nav: float


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.DataFrame
    orders: pd.DataFrame
    positions: pd.DataFrame
    cash_events: pd.DataFrame
    close_state: CloseState | None = None


def _require_columns(frame: pd.DataFrame, names: Sequence[str], label: str) -> None:
    missing = set(names).difference(frame.columns)
    if missing:
        raise LedgerError(f"{label} missing columns: {', '.join(sorted(missing))}")


def _prepare_bars(bars: pd.DataFrame) -> pd.DataFrame:
    _require_columns(bars, _BARS, "bars")
    columns = list(_BARS)
    columns.extend(column for column in ("preclose", "valuation_close") if column in bars)
    data = bars.loc[:, columns].copy()
    data["date"] = data["date"].map(_date)
    if data["symbol"].isna().any() or data["symbol"].eq("").any():
        raise LedgerError("bars have missing symbols")
    if not data["kind"].isin(("stock", "etf")).all():
        raise LedgerError("ledger supports only stock and stock ETF bars")
    if data.duplicated(["date", "symbol"]).any():
        raise LedgerError("duplicate symbol and date in bars")
    if data.groupby("symbol")["kind"].nunique().gt(1).any():
        raise LedgerError("security kind changed within the snapshot")
    if data["tradable"].isna().any() or not data["tradable"].isin((True, False)).all():
        raise LedgerError("bars require an explicit tradable flag")
    for column in set(columns).difference(("date", "symbol", "kind", "tradable")):
        data[column] = pd.to_numeric(data[column], errors="raise").astype(float)
    if "valuation_close" in data:
        marks = data.loc[data["valuation_close"].notna()]
        if (
            not marks["valuation_close"].map(_positive).all()
            or marks["tradable"].any() or marks["kind"].ne("etf").any()
        ):
            raise LedgerError("valuation_close requires a positive mark for a nontradable ETF")
    return data


def _prepare_targets(
    targets: Mapping[DateLike, Mapping[str, float]], calendar: set[date]
) -> dict[date, dict[str, float]]:
    result = {}
    for raw_day, weights in targets.items():
        day = _date(raw_day)
        if day not in calendar or day in result:
            raise LedgerError(f"unknown or duplicate execution day: {day}")
        values = {symbol: float(weight) for symbol, weight in weights.items()}
        if any(not isfinite(weight) or weight < 0 for weight in values.values()):
            raise LedgerError("target weights must be finite and non-negative")
        if sum(values.values()) > 1.0 + _EPSILON:
            raise LedgerError("target weights cannot use leverage")
        result[day] = values
    return result


def _prepare_actions(actions: pd.DataFrame | None, days: list[date]) -> list[_Action]:
    if actions is None or actions.empty:
        return []
    _require_columns(actions, _ACTIONS, "actions")
    preceding_day = {day: days[i - 1] for i, day in enumerate(days) if i}
    result = []
    keys = set()
    for row in actions.itertuples(index=False):
        ex_date = _date(row.ex_date)
        if not days[0] <= ex_date <= days[-1]:
            continue
        if ex_date not in preceding_day and ex_date != days[0]:
            raise LedgerError(f"ex-date is not a supplied session: {ex_date}")
        key = row.symbol, ex_date
        if key in keys:
            raise LedgerError(f"duplicate corporate action: {key}")
        keys.add(key)
        default_record = preceding_day.get(ex_date, ex_date - timedelta(days=1))
        record = getattr(row, "record_date", None)
        record = default_record if pd.isna(record) else _date(record)
        pay = None if pd.isna(row.pay_date) else _date(row.pay_date)
        available = getattr(row, "share_available_date", None)
        available = ex_date if pd.isna(available) else _date(available)
        cash = float(row.cash_per_share)
        multiplier = float(row.share_multiplier)
        rounding = getattr(row, "split_rounding", None)
        rounding = "exact" if pd.isna(rounding) else rounding
        if rounding not in ("exact", "floor", "ceil") or (rounding != "exact" and multiplier == 1):
            raise LedgerError(f"invalid split rounding policy: {key}")
        if (
            record >= ex_date or (record >= days[0] and record not in preceding_day and record != days[0])
            or (pay is not None and pay < ex_date) or available < ex_date
            or not isfinite(cash) or cash < 0 or not _positive(multiplier)
            or (cash > 0 and pay is None)
        ):
            raise LedgerError(f"invalid corporate action dates or quantities: {key}")
        if multiplier != 1 and record != default_record and record >= days[0]:
            raise LedgerError(f"share change requires previous session record date: {key}")
        if multiplier < 1 and available != ex_date and rounding == "exact":
            raise LedgerError(f"reverse split needs immediate delivery: {key}")
        result.append(_Action(row.symbol, ex_date, record, pay, cash, multiplier, available, rounding))
    return result


def _open_price(row: object | None, side: str, cost: Costs) -> tuple[float | None, str]:
    if row is None:
        return None, "missing_bar"
    if not row.tradable or row.volume == 0:
        return None, "suspended"
    if not _positive(row.open) or not _positive(row.volume):
        return None, "invalid_open_or_volume"
    if not _positive(row.lower_limit) or not _positive(row.upper_limit):
        return None, "missing_price_limits"
    if not row.lower_limit <= row.open <= row.upper_limit or row.lower_limit == row.upper_limit:
        return None, "invalid_price_limits"
    if side == "buy" and row.open >= row.upper_limit - _EPSILON:
        return None, "limit_up"
    if side == "sell" and row.open <= row.lower_limit + _EPSILON:
        return None, "limit_down"
    direction = 1 if side == "buy" else -1
    price = row.open * (1 + direction * cost.slippage_bps / 10_000)
    if not row.lower_limit - _EPSILON <= price <= row.upper_limit + _EPSILON:
        return None, "slippage_outside_limit"
    return price, ""


def normalize_actions(actions: pd.DataFrame | None, calendar: Sequence[DateLike]) -> pd.DataFrame:
    """Share the ledger's corporate-action validation with labels and features."""
    days = [_date(day) for day in calendar]
    if not days or days != sorted(set(days)):
        raise LedgerError("calendar must be non-empty, unique and increasing")
    events = _prepare_actions(actions, days)
    return pd.DataFrame([event.__dict__ for event in events], columns=[
        "symbol", "ex_date", "record_date", "pay_date", "cash_per_share",
        "share_multiplier", "share_available_date", "split_rounding",
    ])


def _fill_value(
    day: date, kind: str, side: str, quantity: int, price: float, cost: Costs
) -> tuple[float, tuple[float, float, float], float]:
    notional = _money(quantity * price)
    charges = cost.charges(day, kind, side, notional)
    cash_delta = _money((-notional if side == "buy" else notional) - sum(charges))
    return notional, charges, cash_delta


def _affordable(day: date, kind: str, quantity: int, price: float, cash: float, cost: Costs) -> int:
    low, high = 0, quantity // LOT_SIZE
    while low < high:
        midpoint = (low + high + 1) // 2
        _, _, delta = _fill_value(day, kind, "buy", midpoint * LOT_SIZE, price, cost)
        if cash + delta >= -_EPSILON:
            low = midpoint
        else:
            high = midpoint - 1
    return low * LOT_SIZE


def _liquidation_quantity(position: _Position, day: date, requested: int) -> int:
    sellable = position.sellable(day)
    quantity = min(sellable, requested)
    if quantity == position.quantity:
        return quantity
    return quantity // LOT_SIZE * LOT_SIZE


def _valuation_close(row: object) -> float:
    mark = getattr(row, "valuation_close", float("nan"))
    return mark if _positive(mark) else row.close


def _rebalance(
    day: date, weights: dict[str, float], rows: dict[str, object],
    positions: dict[str, _Position], prior_close: dict[str, float],
    cash: float, receivable: float, cost: Costs, orders: list[tuple],
) -> float:
    opening_nav = cash + receivable
    for symbol, position in positions.items():
        if not position.quantity:
            continue
        row = rows[symbol]
        mark = getattr(row, "valuation_close", float("nan"))
        if row.tradable and row.volume > 0 and _positive(row.open):
            mark = row.open
        elif not _positive(mark):
            mark = prior_close[symbol]
        opening_nav += position.quantity * mark
    desired = []
    for symbol in sorted(set(positions) | set(weights)):
        position = positions.get(symbol, _Position())
        row = rows.get(symbol)
        weight = weights.get(symbol, 0.0)
        if weight and (row is None or not _positive(row.open)):
            orders.append((day, symbol, "target", None, 0, None, 0.0, 0.0, 0.0,
                           "rejected", "missing_bar" if row is None else "invalid_open_or_volume", 0.0, 0.0))
            continue
        raw_target = floor(opening_nav * weight / row.open + _EPSILON) if weight else 0
        target_qty = raw_target // LOT_SIZE * LOT_SIZE
        if weight and target_qty == 0 and position.quantity == 0:
            orders.append((day, symbol, "buy", raw_target, 0, None, 0.0, 0.0, 0.0,
                           "rejected", "board_lot", 0.0, 0.0))
            continue
        delta = target_qty - position.quantity
        if delta:
            desired.append((delta > 0, -weight, symbol, abs(delta)))
    for buying, _, symbol, requested in sorted(desired):
        row = rows.get(symbol)
        position = positions.setdefault(symbol, _Position())
        side = "buy" if buying else "sell"
        price, reason = _open_price(row, side, cost)
        quantity = 0
        notional, charges, cash_delta = 0.0, (0.0, 0.0, 0.0), 0.0
        if price is not None:
            if buying:
                quantity = _affordable(day, row.kind, requested, price, cash, cost)
                reason = "board_lot" if requested < LOT_SIZE else "insufficient_cash"
            else:
                quantity = _liquidation_quantity(position, day, requested)
                reason = "t_plus_one" if requested > position.sellable(day) else "board_lot"
            if quantity:
                notional, charges, cash_delta = _fill_value(day, row.kind, side, quantity, price, cost)
                if cash + cash_delta < -_EPSILON:
                    quantity = 0
                    notional, charges, cash_delta = 0.0, (0.0, 0.0, 0.0), 0.0
                    reason = "insufficient_cash_for_fees"
                else:
                    cash = _money(cash + cash_delta)
                    if buying:
                        position.buy(day, quantity)
                    else:
                        position.sell(day, quantity)
        status = "filled" if quantity == requested else "partial" if quantity else "rejected"
        orders.append((day, symbol, side, requested, quantity, price if quantity else None,
                       *charges, status, "" if status == "filled" else reason, notional, cash_delta))
    return cash


def simulate(
    bars: pd.DataFrame,
    calendar: Sequence[DateLike],
    targets: Mapping[DateLike, Mapping[str, float]],
    actions: pd.DataFrame | None = None,
    *,
    initial_cash: float,
    cost: Costs = Costs(),
    on_close: Callable[[CloseState], Mapping[str, float] | None] | None = None,
) -> BacktestResult:
    """Simulate opening target weights on a supplied exchange calendar.

    Static targets already refer to execution dates. With empty static targets,
    on_close can read the settled account and schedule weights for the next
    session. None retains current positions and an empty mapping requests
    liquidation. The final close is returned without invoking the callback.

    The caller verifies corporate action coverage. Open prices approximate
    executable liquidity. Limits, explicit halts, and absent prices can prevent
    fills. Stock ETF trading is the secondary market case only. Dividends are
    credited at the supplied cash per share on the first session on or after
    payment.
    """
    if not _positive(initial_cash) or _money(initial_cash) == 0:
        raise LedgerError("initial cash must be finite and positive to the cent")
    days = [_date(day) for day in calendar]
    if not days or days != sorted(set(days)):
        raise LedgerError("calendar must be non-empty, unique and increasing")
    if targets and on_close is not None:
        raise LedgerError("static targets and a close callback cannot be combined")
    execution = _prepare_targets(targets, set(days))
    bar_data = _prepare_bars(bars)
    groups = bar_data.groupby("date", sort=False)
    security_kinds = bar_data.drop_duplicates("symbol").set_index("symbol")["kind"].to_dict()
    event_data = _prepare_actions(actions, days)
    ex_events, record_events = defaultdict(list), defaultdict(list)
    for event in event_data:
        if event.split_rounding != "exact" and security_kinds.get(event.symbol) == "stock":
            raise LedgerError(f"fractional split rounding requires an ETF: {event.symbol}")
        ex_events[event.ex_date].append(event)
        record_events[event.record_date].append(event)
    positions: dict[str, _Position] = {}
    previous_close: dict[str, float] = {}
    entitled: dict[_Action, int] = {}
    pending: list[tuple[date, str, float]] = []
    equity_rows, order_rows, position_rows, event_rows = [], [], [], []
    cash = peak = last_nav = _money(initial_cash)
    close_state = None

    for index, day in enumerate(days):
        rows = {}
        if day in groups.indices:
            rows = {row.symbol: row for row in groups.get_group(day).itertuples(index=False)}
        for symbol, position in positions.items():
            if position.quantity and (symbol not in rows or not _positive(_valuation_close(rows[symbol]))):
                raise LedgerError(f"missing valuation for held {symbol} on {day}")
        for event in ex_events[day]:
            quantity = entitled.get(event, 0)
            if not quantity:
                continue
            amount = _money(quantity * event.cash_per_share)
            if amount and event.pay_date is not None:
                pending.append((event.pay_date, event.symbol, amount))
                event_rows.append((day, event.symbol, "dividend_receivable", 0.0, amount, 0))
            if event.share_multiplier != 1:
                position = positions[event.symbol]
                if position.quantity != quantity:
                    raise LedgerError(f"share count changed after record date: {event.symbol} {day}")
                position.apply_split(event.share_multiplier, event.share_available_date, event.split_rounding)
                event_rows.append((day, event.symbol, "share_change", 0.0, 0.0,
                                   position.quantity - quantity))
        remaining = []
        for pay_date, symbol, amount in pending:
            if pay_date <= day:
                cash = _money(cash + amount)
                event_rows.append((day, symbol, "dividend_paid", amount, -amount, 0))
            else:
                remaining.append((pay_date, symbol, amount))
        pending = remaining
        receivable = _money(sum(amount for _, _, amount in pending))
        action_symbols = {event.symbol for event in ex_events[day]}
        for symbol, position in positions.items():
            if not position.quantity:
                continue
            row = rows[symbol]
            reference = getattr(row, "preclose", float("nan"))
            tolerance = 0.001 if row.kind == "etf" else 0.01
            if (
                symbol in previous_close and _positive(reference) and symbol not in action_symbols
                and abs(reference - previous_close[symbol]) > tolerance + _EPSILON
            ):
                raise LedgerError(f"unexplained ex-rights reference for {symbol} on {day}")
        if day in execution:
            opening_reference = previous_close.copy()
            for event in ex_events[day]:
                if event.symbol in opening_reference:
                    opening_reference[event.symbol] = (
                        (opening_reference[event.symbol] - event.cash_per_share) / event.share_multiplier
                    )
            cash = _rebalance(day, execution[day], rows, positions, opening_reference,
                              cash, receivable, cost, order_rows)
        market_values = {}
        for symbol, position in sorted(positions.items()):
            if position.quantity:
                close = _valuation_close(rows[symbol])
                if not _positive(close):
                    raise LedgerError(f"missing closing valuation for filled {symbol} on {day}")
                previous_close[symbol] = close
                value = _money(position.quantity * close)
                market_values[symbol] = value
                position_rows.append((day, symbol, position.quantity, position.sellable(day), close, value))
        for event in record_events[day]:
            if event.symbol in positions:
                entitled[event] = positions[event.symbol].quantity
        market_value = _money(sum(market_values.values()))
        nav = _money(cash + market_value + receivable)
        if nav <= 0:
            raise LedgerError(f"non-positive account value on {day}")
        peak = max(peak, nav)
        drawdown = 1.0 - nav / peak
        equity_rows.append((day, cash, market_value, receivable, nav,
                            nav / last_nav - 1.0, drawdown))
        last_nav = nav
        close_state = CloseState(
            day=day,
            weights=MappingProxyType({symbol: value / nav for symbol, value in market_values.items()}),
            cash_weight=cash / nav,
            drawdown=drawdown,
            nav=nav,
        )
        if on_close is not None and index + 1 < len(days):
            weights = on_close(close_state)
            if weights is not None:
                next_day = days[index + 1]
                execution.update(_prepare_targets({next_day: weights}, {next_day}))

    return BacktestResult(
        pd.DataFrame(equity_rows, columns=_EQUITY),
        pd.DataFrame(order_rows, columns=_ORDERS),
        pd.DataFrame(position_rows, columns=_POSITIONS),
        pd.DataFrame(event_rows, columns=_EVENTS),
        close_state,
    )
