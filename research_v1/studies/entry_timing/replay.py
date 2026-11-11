"""Replay paired opening-entry proxies with the repository's fee model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from dataclasses import asdict, dataclass, fields, replace
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from quant_research.ledger import LOT_SIZE, Costs, _fill_value


OPENING = "overnight_limit"
DELAYED = "wait_0935"
ARMS = (OPENING, DELAYED)
NORMAL = "repo_normal"
DOUBLE = "repo_double"
ZERO = Decimal("0.00")


@dataclass(frozen=True)
class Opportunity:
    symbol: str
    signal_date: date
    entry_date: date
    exit_date: date
    previous_close: Decimal
    overnight_open: Decimal
    delayed_open: Decimal
    exit_open: Decimal


@dataclass(frozen=True)
class Trade:
    scenario: str
    arm: str
    symbol: str
    signal_date: date
    entry_date: date
    exit_date: date
    requested_quantity: int
    filled_quantity: int
    limit_price: Decimal
    entry_reference: Decimal
    projected_entry_price: Decimal
    entry_price: Decimal | None
    exit_reference: Decimal
    exit_price: Decimal | None
    buy_fees: Decimal
    sell_fees: Decimal
    fees: Decimal
    net_pnl: Decimal
    status: str
    reason: str


def load_opportunities(
    path: Path, symbols: set[str], sessions: list[date], expected_sha256: str,
) -> list[Opportunity]:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("Opportunity CSV SHA256 differs from the audit manifest")
    adjacent = {
        entry: (signal, exit_day)
        for signal, entry, exit_day in zip(sessions, sessions[1:], sessions[2:])
    }
    columns = [field.name for field in fields(Opportunity)]
    opportunities = []
    seen = set()
    with io.StringIO(data.decode("utf-8"), newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != columns:
            raise ValueError(f"Expected opportunity CSV columns {', '.join(columns)}")
        for line, row in enumerate(reader, 2):
            try:
                symbol = row["symbol"]
                if symbol not in symbols:
                    raise ValueError(f"unsupported symbol {symbol!r}")
                signal, entry, exit_day = (date.fromisoformat(row[name]) for name in columns[1:4])
                if adjacent.get(entry) != (signal, exit_day):
                    raise ValueError("signal, entry and T+1 exit must be consecutive audited sessions")
                prices = [Decimal(row[name]) for name in columns[4:]]
                if any(not price.is_finite() or price <= 0 for price in prices):
                    raise ValueError("all prices must be positive and finite")
                key = (symbol, entry)
                if key in seen:
                    raise ValueError(f"duplicate opportunity for {symbol} on {entry}")
                seen.add(key)
                opportunities.append(Opportunity(symbol, signal, entry, exit_day, *prices))
            except (ValueError, TypeError, InvalidOperation) as exc:
                raise ValueError(f"{path.name}, line {line}: {exc}") from exc
    expected = {(symbol, entry) for symbol in symbols for entry in adjacent}
    if seen != expected:
        missing = ", ".join(f"{symbol}/{entry}" for symbol, entry in sorted(expected - seen))
        raise ValueError(f"Incomplete symbol and session opportunities: {missing}")
    return sorted(opportunities, key=lambda opportunity: (opportunity.entry_date, opportunity.symbol))


def execution_price(reference: Decimal, tick: Decimal, cost: Costs, side: str) -> Decimal:
    slippage = Decimal(str(cost.slippage_bps)) / 10_000
    factor = 1 + slippage if side == "buy" else 1 - slippage
    rounding = ROUND_CEILING if side == "buy" else ROUND_FLOOR
    price = (reference * factor / tick).to_integral_value(rounding=rounding) * tick
    if price <= 0:
        raise ValueError("Execution price is nonpositive after slippage and tick rounding")
    return price


def replay_arm(
    opportunity: Opportunity, arm: str, scenario: str, quantity: int,
    kind: str, tick: Decimal, cost: Costs,
) -> Trade:
    reference = opportunity.overnight_open if arm == OPENING else opportunity.delayed_open
    projected = execution_price(reference, tick, cost, "buy")
    common = {
        "scenario": scenario, "arm": arm, "symbol": opportunity.symbol,
        "signal_date": opportunity.signal_date, "entry_date": opportunity.entry_date,
        "exit_date": opportunity.exit_date, "requested_quantity": quantity,
        "limit_price": opportunity.previous_close, "entry_reference": reference,
        "projected_entry_price": projected, "exit_reference": opportunity.exit_open,
    }
    if projected > opportunity.previous_close:
        return Trade(**common, filled_quantity=0, entry_price=None, exit_price=None,
                     buy_fees=ZERO, sell_fees=ZERO, fees=ZERO, net_pnl=ZERO,
                     status="unfilled", reason="projected_price_above_limit")

    exit_price = execution_price(opportunity.exit_open, tick, cost, "sell")
    _, buy_costs, buy_cash = _fill_value(
        opportunity.entry_date, kind, "buy", quantity, float(projected), cost,
    )
    _, sell_costs, sell_cash = _fill_value(
        opportunity.exit_date, kind, "sell", quantity, float(exit_price), cost,
    )
    buy_fees = sum(Decimal(str(amount)) for amount in buy_costs)
    sell_fees = sum(Decimal(str(amount)) for amount in sell_costs)
    return Trade(
        **common, filled_quantity=quantity, entry_price=projected, exit_price=exit_price,
        buy_fees=buy_fees, sell_fees=sell_fees, fees=buy_fees + sell_fees,
        net_pnl=Decimal(str(buy_cash)) + Decimal(str(sell_cash)),
        status="filled_proxy", reason="",
    )


def group_summary(trades: list[Trade]) -> dict:
    groups = {arm: [trade for trade in trades if trade.arm == arm] for arm in ARMS}
    summary = {}
    for arm, group in groups.items():
        count = len(group)
        fills = sum(trade.filled_quantity > 0 for trade in group)
        net = sum((trade.net_pnl for trade in group), ZERO)
        summary[arm] = {
            "opportunities": count,
            "fills": fills,
            "unfilled": count - fills,
            "profitable": sum(trade.net_pnl > 0 for trade in group),
            "net_pnl": net,
            "fees": sum((trade.fees for trade in group), ZERO),
            "net_per_opportunity": net / count if count else None,
        }

    pair_cases = {
        label: {"count": 0, "wait_minus_overnight_net": ZERO}
        for label in ("both_filled", "overnight_only", "wait_only", "neither_filled")
    }
    case_names = {
        (False, False): "neither_filled", (False, True): "wait_only",
        (True, False): "overnight_only", (True, True): "both_filled",
    }
    better = worse = 0
    for first, second in zip(groups[OPENING], groups[DELAYED], strict=True):
        if (first.symbol, first.entry_date) != (second.symbol, second.entry_date):
            raise ValueError("Unpaired opportunity in summary")
        case = case_names[bool(first.filled_quantity), bool(second.filled_quantity)]
        delta = second.net_pnl - first.net_pnl
        pair_cases[case]["count"] += 1
        pair_cases[case]["wait_minus_overnight_net"] += delta
        better += delta > 0
        worse += delta < 0
    summary["paired"] = {
        "wait_minus_overnight_net": summary[DELAYED]["net_pnl"] - summary[OPENING]["net_pnl"],
        "wait_better": better,
        "wait_worse": worse,
        "tied": len(groups[OPENING]) - better - worse,
        "by_fill_case": pair_cases,
    }
    return summary


def run(protocol_path: Path, opportunities_path: Path, out: Path) -> None:
    manifest = json.loads(opportunities_path.with_name("manifest.json").read_text(encoding="utf-8"))
    protocol_data = protocol_path.read_bytes()
    protocol_hash = hashlib.sha256(protocol_data).hexdigest()
    if protocol_hash != manifest["protocol_sha256"]:
        raise ValueError("Protocol SHA256 differs from the audit manifest")
    protocol = json.loads(protocol_data)
    sessions = [date.fromisoformat(day) for day in manifest["sessions"]]
    if (
        len(sessions) < 3 or sessions != sorted(set(sessions))
        or sessions[0] != date.fromisoformat(protocol["first_complete_session"])
        or sessions[-1] != date.fromisoformat(protocol["last_complete_session"])
    ):
        raise ValueError("Manifest sessions must be unique, ordered and match protocol boundaries")
    symbols = protocol["symbols"]
    quantity = protocol["quantity"]
    if quantity != LOT_SIZE:
        raise ValueError(f"This experiment requires one lot of {LOT_SIZE} units")
    instrument_specs = {}
    for symbol, metadata in symbols.items():
        kind = metadata["kind"]
        tick = Decimal(metadata["tick"])
        if kind not in ("etf", "stock") or not tick.is_finite() or tick <= 0:
            raise ValueError(f"Invalid instrument specification for {symbol}")
        instrument_specs[symbol] = kind, tick
    opportunities = load_opportunities(
        opportunities_path, set(symbols), sessions, manifest["opportunities_sha256"],
    )
    normal = Costs()
    scenarios = {
        NORMAL: normal, DOUBLE: normal.stressed(),
        "proportional_commission": replace(normal, minimum_commission=0),
    }
    report = {
        "protocol_sha256": protocol_hash,
        "opportunities_sha256": manifest["opportunities_sha256"],
        "opportunities": len(opportunities),
        "scenarios": {},
    }
    all_trades = []
    for name, cost in scenarios.items():
        trades = []
        for opportunity in opportunities:
            kind, tick = instrument_specs[opportunity.symbol]
            for arm in ARMS:
                trades.append(replay_arm(opportunity, arm, name, quantity, kind, tick, cost))
        all_trades.extend(trades)
        report["scenarios"][name] = {
            "costs": asdict(cost),
            "overall": group_summary(trades),
            "by_symbol": {
                symbol: group_summary([trade for trade in trades if trade.symbol == symbol])
                for symbol in symbols
            },
        }
    checks = {}
    for scenario in (NORMAL, DOUBLE):
        overall = report["scenarios"][scenario]["overall"]
        checks[scenario] = {
            "positive_wait_net": overall[DELAYED]["net_pnl"] > 0,
            "positive_wait_increment": overall["paired"]["wait_minus_overnight_net"] > 0,
        }
    report["screen"] = {
        "checks": checks,
        "passed": all(all(scenario_checks.values()) for scenario_checks in checks.values()),
    }

    out.mkdir(exist_ok=False)
    with (out / "trades.csv").open("x", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=[field.name for field in fields(Trade)])
        writer.writeheader()
        writer.writerows(asdict(trade) for trade in all_trades)
    (out / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--opportunities", type=Path, required=True,
                        help="Audited opportunity CSV with a manifest.json beside it")
    parser.add_argument("--out", type=Path, required=True, help="A new results directory")
    args = parser.parse_args()
    run(args.protocol, args.opportunities, args.out)


if __name__ == "__main__":
    main()
