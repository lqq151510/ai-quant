"""Compare opening-confirmation selections at the same fee-inclusive entry spend."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import combinations
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'paper-open-confirmation-20261120'))

from confirm import INITIAL, compare_archive, compare_pilot


SIGNALS = ('confirmed', 'below_previous_close', 'unknown')
BLOCKED = 'below_previous_close'
ZERO = Fraction()
INITIAL_CASH = Fraction(INITIAL)
ATTRIBUTION_FIELDS = (
    'original_pnl', 'confirmation_pnl', 'proportional_matched_pnl', 'same_count_matched_mean_pnl',
    'improvement', 'deployment_change', 'selection_change', 'selection_vs_same_count_mean',
)


@dataclass(frozen=True)
class Entry:
    symbol: str
    day: str
    units: int
    signal: str
    price: Fraction
    spend: Fraction


@dataclass(frozen=True)
class Outcome:
    entry: Entry
    day: str
    price: Fraction
    pnl: Fraction
    state: str


def amount(value):
    return Fraction(Decimal(value))


def money(value):
    with localcontext() as context:
        context.prec = 48
        return format(Decimal(value.numerator) / Decimal(value.denominator), '.12f')


def serialized(values):
    return {key: money(value) for key, value in values.items()}


def check_day(account, expected_fills, positions, cash, day):
    if account['fills'] != expected_fills or account['positions'] != positions:
        raise ValueError(f'{day}: fills or share positions do not reconcile')
    fees = sum((amount(fill['fees']) for fill in expected_fills), ZERO)
    values = {symbol: amount(price) for symbol, price in account['close_prices'].items()}
    if positions.keys() != values.keys() or any(price <= 0 for price in values.values()):
        raise ValueError(f'{day}: missing or invalid closing marks')
    nav = cash + sum((values[symbol] * units for symbol, units in positions.items()), ZERO)
    if cash != amount(account['cash']) or fees != amount(account['fees']) or nav != amount(account['nav']):
        raise ValueError(f'{day}: cash, fees or NAV do not reconcile')


def lifecycles(daily):
    open_entries, outcomes = {}, []
    original_cash = confirmation_cash = INITIAL_CASH
    signal_counts = Counter(dict.fromkeys(SIGNALS, 0))
    previous = ''
    for row in daily:
        day = row['date']
        if day <= previous:
            raise ValueError(f'{day}: ledger dates are duplicated or out of order')
        previous = day
        original, confirmation = row['original'], row['confirmation']
        confirmed_fills = []
        decisions = {}
        for decision in row['entry_decisions']:
            symbol = decision['symbol']
            if symbol in decisions or decision['date'] != day or decision['signal'] not in SIGNALS:
                raise ValueError(f'{day} {symbol}: invalid or duplicated decision')
            decisions[symbol] = decision
            signal_counts[decision['signal']] += 1
        if len(decisions) > 10:
            raise ValueError(f'{day}: more than ten original buying opportunities')
        for fill in original['fills']:
            symbol, side, units = fill['symbol'], fill['side'], fill['units']
            if side not in ('buy', 'sell') or type(units) is not int or units <= 0:
                raise ValueError(f'{day} {symbol}: invalid original fill')
            price, delta, fees = (amount(fill[field]) for field in ('price', 'cash_delta', 'fees'))
            if price <= 0 or fees < 0 or delta != (price * units if side == 'sell' else -price * units) - fees:
                raise ValueError(f'{day} {symbol}: execution cash or fees do not reconcile')
            original_cash += delta
            if side == 'buy':
                decision = decisions.pop(symbol, None)
                if symbol in open_entries or decision is None or decision['units'] != units:
                    raise ValueError(f'{day} {symbol}: buy does not match one original decision')
                entry = Entry(symbol, day, units, decision['signal'], price, -delta)
                open_entries[symbol] = entry
            else:
                entry = open_entries.pop(symbol, None)
                if entry is None or entry.units != units or entry.day >= day:
                    raise ValueError(f'{day} {symbol}: sale does not close one earlier lot')
                outcomes.append(Outcome(entry, day, price, delta - entry.spend, 'closed'))
            if entry.signal != BLOCKED:
                confirmed_fills.append(fill)
                confirmation_cash += delta
            if min(original_cash, confirmation_cash) < 0:
                raise ValueError(f'{day} {symbol}: insufficient cash')
        if decisions:
            raise ValueError(f'{day}: decisions without an original buy')
        all_positions = {symbol: entry.units for symbol, entry in open_entries.items()}
        selected_positions = {symbol: entry.units for symbol, entry in open_entries.items() if entry.signal != BLOCKED}
        check_day(original, original['fills'], all_positions, original_cash, day)
        check_day(confirmation, confirmed_fills, selected_positions, confirmation_cash, day)
        if any(amount(confirmation['close_prices'][symbol]) != amount(original['close_prices'][symbol])
               for symbol in selected_positions):
            raise ValueError(f'{day}: confirmation and original marks differ')
    final = daily[-1]
    for entry in open_entries.values():
        price = amount(final['original']['close_prices'][entry.symbol])
        outcomes.append(Outcome(entry, final['date'], price, price * entry.units - entry.spend, 'terminal_marked'))
    return outcomes, dict(signal_counts)


def reconcile_result(result, outcomes, signal_counts):
    summary = result['summary']
    if signal_counts != result['signal_counts']:
        raise ValueError('entry signal counts do not reconcile')
    for name, lots in (
        ('original', outcomes),
        ('confirmation', [lot for lot in outcomes if lot.entry.signal != BLOCKED]),
    ):
        reported_pnl = amount(summary[name]['ending_nav']) - INITIAL_CASH
        if summary[name]['buy_count'] != len(lots) or summary[name]['sell_count'] != sum(
            lot.state == 'closed' for lot in lots
        ) or sum((lot.pnl for lot in lots), ZERO) != reported_pnl:
            raise ValueError(f'{name}: lifecycle amounts do not reconcile to account totals')
        if amount(result['daily'][-1][name]['nav']) != amount(summary[name]['ending_nav']):
            raise ValueError(f'{name}: final ledger NAV differs from account total')
    cash_control = summary['cash_control']
    if amount(cash_control['ending_nav']) != INITIAL_CASH or any(
        amount(cash_control[field]) for field in ('fees', 'net_return', 'max_drawdown', 'fill_count')
    ) or cash_control['ending_positions']:
        raise ValueError('invalid cash control')

    blocked = result['blocked_outcomes']
    reconstructed = {
        (lot.entry.day, lot.entry.symbol):
        (lot.entry.units, lot.entry.price, lot.entry.spend, lot.day, lot.price, lot.state, lot.pnl)
        for lot in outcomes if lot.entry.signal == BLOCKED
    }
    reported = {
        (lot['entry_date'], lot['symbol']):
        (lot['units'], amount(lot['entry_price']), -amount(lot['entry_cash_delta']), lot['outcome_date'],
         amount(lot['outcome_price']), lot['state'], amount(lot['pnl']))
        for lot in blocked['trades']
    }
    difference = -sum((data[-1] for data in reconstructed.values()), ZERO)
    if len(reported) != len(blocked['trades']) or reconstructed != reported or difference != amount(
        blocked['net_pnl_difference']
    ):
        raise ValueError('skipped lots do not reconcile to reported outcomes')


def compare_day(day, outcomes):
    selected = [lot for lot in outcomes if lot.entry.signal != BLOCKED]
    buy_spend = sum((lot.entry.spend for lot in outcomes), ZERO)
    spend = sum((lot.entry.spend for lot in selected), ZERO)
    original = sum((lot.pnl for lot in outcomes), ZERO)
    actual = sum((lot.pnl for lot in selected), ZERO)
    k, n = len(selected), len(outcomes)
    daily = {
        'date': day, 'original_count': n, 'confirmation_count': k,
        'original_spend': money(buy_spend), 'confirmation_spend': money(spend),
        'terminal_count': sum(lot.state == 'terminal_marked' for lot in outcomes),
        'confirmation_terminal_count': sum(lot.state == 'terminal_marked' for lot in selected),
        'unknown_count': sum(lot.entry.signal == 'unknown' for lot in outcomes),
        'choice_count': 0, 'rank': None, 'pnl': None,
    }
    if n == 0:
        return {**daily, 'state': 'no_valued_entries'}, None

    proportional = original * spend / buy_spend
    if not k:
        mean = ZERO
    elif k == n:
        mean = actual
    else:
        normalized = [
            spend * sum((lot.pnl for lot in basket), ZERO) / sum((lot.entry.spend for lot in basket), ZERO)
            for basket in combinations(outcomes, k)
        ]
        daily['rank'] = {
            'higher_than_confirmation': sum(value > actual for value in normalized),
            'equal_to_confirmation': sum(value == actual for value in normalized),
            'lower_than_confirmation': sum(value < actual for value in normalized),
        }
        mean = sum(normalized, ZERO) / len(normalized)
    values = dict(zip(ATTRIBUTION_FIELDS, (
        original, actual, proportional, mean, actual - original, proportional - original,
        actual - proportional, actual - mean,
    ), strict=True))
    daily['choice_count'] = sum(daily['rank'].values()) if daily['rank'] else 1
    return {**daily, 'state': 'comparable' if daily['rank'] else 'no_selection_choice',
            'pnl': serialized(values)}, values


def summarize(dates, lifecycles):
    cohorts = defaultdict(list)
    for lot in lifecycles:
        cohorts[lot.entry.day].append(lot)
    days = []
    totals = dict.fromkeys(ATTRIBUTION_FIELDS, ZERO)
    for day in dates:
        report, values = compare_day(day, cohorts[day])
        days.append(report)
        if values is not None:
            for field in ATTRIBUTION_FIELDS:
                totals[field] += values[field]
    return {
        'days': days,
        'total': serialized(totals) if lifecycles else None,
        'exact_total': {field: str(value) for field, value in totals.items()} if lifecycles else None,
    }


def audit(result):
    """Audit a standard confirmation result already backed by verified observations."""
    daily = result['daily']
    settled = result['settled_sessions']
    if len(daily) != settled:
        raise ValueError('confirmation settlement count does not match daily ledger')
    report = {
        'state': result['state'], 'settled_sessions': settled, 'account_summary': result['summary'],
        'signal_counts': result['signal_counts'], 'all_valued': None, 'closed_only': None,
    }
    for key in ('mode', 'source_artifact', 'source_sha256', 'planned_sessions', 'sessions'):
        if key in result:
            report[key] = result[key]
    if not daily:
        if any(result[key] is not None for key in ('summary', 'signal_counts', 'blocked_outcomes')):
            raise ValueError('unsettled confirmation contains realized metrics')
        return report

    outcomes, signal_counts = lifecycles(daily)
    reconcile_result(result, outcomes, signal_counts)
    dates = [row['date'] for row in daily]
    return {
        **report, 'all_valued': summarize(dates, outcomes),
        'closed_only': summarize(dates, [lot for lot in outcomes if lot.state == 'closed']),
        'lifecycles': [{
            'symbol': lot.entry.symbol, 'entry_date': lot.entry.day, 'outcome_date': lot.day,
            'units': lot.entry.units, 'signal': lot.entry.signal, 'state': lot.state,
            'entry_spend': money(lot.entry.spend), 'net_pnl': money(lot.pnl),
        } for lot in sorted(outcomes, key=lambda lot: (lot.entry.day, lot.entry.symbol))],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', choices=('archive', 'pilot'))
    args = parser.parse_args()
    try:
        result = compare_archive() if args.source == 'archive' else compare_pilot()
        print(json.dumps(audit(result), ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError) as error:
        raise SystemExit(f'Could not verify matched selection: {error}') from error
