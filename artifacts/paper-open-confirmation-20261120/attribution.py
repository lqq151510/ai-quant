"""Explain the original and netted paper NAV changes with verified quote midpoints."""

import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path

sys.dont_write_bytecode = True
NETTING = Path(__file__).resolve().parents[1] / 'paper-order-netting-20261120'
sys.path.insert(0, str(NETTING))

from archive import SOURCE, compare

FIELDS = ('prior_close_to_first_quote', 'inside_opening_quotes', 'last_quote_to_close',
          'execution_vs_midpoint', 'fees', 'unattributed_before_fees')
ZERO = Decimal(0)


def value(quantity, prices, symbol):
    return quantity * Decimal(prices[symbol]) if quantity else ZERO


def quote_midpoint(book):
    try:
        bid, ask = Decimal(book['bid']), Decimal(book['ask'])
        if all(price.is_finite() for price in (bid, ask)) and 0 < bid <= ask:
            return (bid + ask) / 2
    except (KeyError, TypeError, ArithmeticError):
        pass
    return None


def attribute_symbol(symbol, prior, current, quotes, fills):
    start = prior['positions'].get(symbol, 0)
    end = current['positions'].get(symbol, 0)
    mark_delta = value(end, current['close_prices'], symbol) - value(start, prior['close_prices'], symbol)
    cash_delta = sum((Decimal(fill['cash_delta']) for fill in fills), ZERO)
    fees = sum((Decimal(fill['fees']) for fill in fills), ZERO)
    net = mark_delta + cash_delta
    components = dict.fromkeys(FIELDS, ZERO)
    components['fees'] = -fees
    events = defaultdict(set)
    for (code, _), book in quotes.items():
        if code == symbol:
            events[datetime.fromisoformat(book['received_at'])].add(quote_midpoint(book))
    if not events or any(len(prices) != 1 or None in prices for prices in events.values()):
        components['unattributed_before_fees'] = net + fees
        return components

    transactions = defaultdict(list)
    for fill in fills:
        key = symbol, fill['side']
        received = datetime.fromisoformat(quotes[key]['received_at'])
        transactions[received].append(fill)
    quantity = start
    previous = None
    for received, prices in sorted(events.items()):
        midpoint = next(iter(prices))
        if quantity and previous is None:
            components['prior_close_to_first_quote'] += quantity * midpoint - value(
                quantity, prior['close_prices'], symbol,
            )
        elif previous is not None:
            components['inside_opening_quotes'] += quantity * (midpoint - previous)
        for fill in transactions[received]:
            change = fill['units'] if fill['side'] == 'buy' else -fill['units']
            components['execution_vs_midpoint'] += change * (midpoint - Decimal(fill['price']))
            quantity += change
        previous = midpoint
    if quantity != end:
        raise ValueError(f'{symbol}: fill quantities do not reconcile')
    components['last_quote_to_close'] = value(quantity, current['close_prices'], symbol) - quantity * previous
    if sum(components.values(), ZERO) != net:
        raise ValueError(f'{symbol}: attribution does not reconcile')
    return components


def group_report(daily, quotes_by_date, group, starting_cash):
    prior = {'cash': starting_cash, 'nav': starting_cash, 'positions': {}, 'close_prices': {}}
    totals = dict.fromkeys(FIELDS, ZERO)
    symbol_totals = defaultdict(lambda: dict.fromkeys(FIELDS, ZERO))
    days = []
    for row in daily:
        current = row[group]
        fills = defaultdict(list)
        for fill in current['fills']:
            fills[fill['symbol']].append(fill)
        components = dict.fromkeys(FIELDS, ZERO)
        for symbol in sorted(prior['positions'].keys() | current['positions'].keys() | fills.keys()):
            parts = attribute_symbol(symbol, prior, current, quotes_by_date[row['date']], fills[symbol])
            for key in FIELDS:
                components[key] += parts[key]
                symbol_totals[symbol][key] += parts[key]
        nav_change = Decimal(current['nav']) - Decimal(prior['nav'])
        if nav_change != sum(components.values(), ZERO):
            raise ValueError(f'{row["date"]}: attribution does not reconcile to daily NAV')
        for key in FIELDS:
            totals[key] += components[key]
        days.append({'date': row['date'], **strings(components), 'net': str(nav_change)})
        prior = current
    return {
        'daily': days, 'total': {**strings(totals), 'net': str(sum(totals.values(), ZERO))},
        'by_symbol': {code: {**strings(parts), 'net': str(sum(parts.values(), ZERO))}
                      for code, parts in symbol_totals.items()},
    }


def strings(components):
    return {key: str(value) for key, value in components.items()}


def report():
    archive = compare()
    quotes = {}
    for row in archive['daily']:
        date = row['date']
        relative = f'{date}/opening.json'
        payload = (SOURCE / relative).read_bytes()
        if hashlib.sha256(payload).hexdigest() != archive['source_sha256'][relative]:
            raise ValueError(f'{date}: opening evidence changed during attribution')
        decisions = json.loads(payload)['baseline']['decisions']
        quotes[date] = {(item['symbol'], item['side']): item['book']
                        for item in decisions if item['reason'] == 'filled'}
    return {
        'mode': 'retrospective_midpoint_attribution',
        'source_artifact': archive['source_artifact'],
        'source_sha256': archive['source_sha256'],
        'groups': {name: group_report(archive['daily'], quotes, name, archive['initial_virtual_cash'])
                   for name in ('original', 'netting')},
    }


if __name__ == '__main__':
    try:
        print(json.dumps(report(), ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError) as error:
        raise SystemExit(f'Could not reconcile opening attribution: {error}') from error
