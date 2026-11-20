#!/usr/bin/env python3
"""Replay a fixed sequence of paper fills with 1x and 2x whole-lot sizes."""

import hashlib
import json
import sys
from datetime import date
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parent.parent / 'news-filter-forward-20261116'
ROOT = SOURCE.parent.parent
sys.path[:0] = [str(SOURCE), str(ROOT / 'research_v1')]

from book_evidence import verify_capture
from close_monday import closing_price, session_marks, verify_mark_provenance
from continue_open import sale, verify_saved_opening
from monday_open import PROTOCOL, tick_price
from quant_research.ledger import Costs, LOT_SIZE
from reconcile import INITIAL, reconcile

SESSIONS = tuple(date(2026, 11, day) for day in (16, 17, 18))
COSTS = Costs()
SIZE_MULTIPLIERS = {'baseline': 1, 'heavy': 2}
MONEY_FIELDS = ('price', 'commission', 'stamp_tax', 'transfer_fee', 'fees', 'cash_delta')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_input(path, hashes):
    payload = path.read_bytes()
    hashes[str(path.relative_to(SOURCE))] = hashlib.sha256(payload).hexdigest()
    return json.loads(payload)


def verified_session(day, hashes):
    label = str(day)
    folder = SOURCE / label
    opening = read_input(folder / 'opening.json', hashes)
    daily = read_input(folder / 'daily.json', hashes)
    for ledger in (opening, daily):
        require(ledger['date'] == label and ledger['mode'] == 'forward', f'{label}: invalid ledger identity')
    if day != SESSIONS[0]:
        verify_saved_opening(folder / 'opening.json', day)
    base_open, base_close = opening['baseline'], daily['baseline']
    for field in ('cash', 'positions', 'decisions', 'fills'):
        require(base_open[field] == base_close[field], f'{label}: opening/closing {field} mismatch')
    require(not base_close.get('corporate_actions'), f'{label}: corporate action requires separate accounting')
    require(not base_close['valuation_gap'] and base_close['nav'] is not None, f'{label}: missing close valuation')

    quote_dir = folder / 'books'
    held = {symbol for group in opening['expected_sell_symbols'].values() for symbol in group}
    quote_log = read_input(quote_dir / 'books.json', hashes)
    books = verify_capture(quote_log, quote_dir, day, opening['planned_symbols'], held=held,
                           buy_symbols=opening['candidate_symbols'])
    marks, _ = session_marks(read_input(folder / 'close_marks.json', hashes), label)
    prices = {}
    for symbol in base_close['positions']:
        mark = marks.get(symbol)
        price, error = closing_price(mark, label)
        error = error or verify_mark_provenance(mark, folder, require_raw=True, day=label, symbol=symbol,
                                                previous_close=opening['prior_closes'].get(symbol))
        require(error is None, f'{label} {symbol}: {error}')
        require(mark == base_close['close_evidence'][symbol], f'{label} {symbol}: saved close evidence mismatch')
        require(price == Decimal(base_close['close_prices'][symbol]), f'{label} {symbol}: saved close mismatch')
        prices[symbol] = str(price)

    decisions = [decision for decision in base_open['decisions'] if decision['reason'] == 'filled']
    orders = base_open['fills']
    require(len(decisions) == len(orders), f'{label}: accepted decisions and fills differ in number')
    verified_orders = []
    for order, decision in zip(orders, decisions, strict=True):
        symbol, side, units = order['symbol'], order['side'], order['units']
        context = f'{day} {symbol} {side}'
        require((symbol, side, units) == (decision['symbol'], decision['side'], decision['required_units']),
                f'{context}: decision/fill mismatch')
        require(side in ('buy', 'sell'), f'{context}: invalid side')
        quote_key, depth_key = ('selected', 'ask_units') if side == 'buy' else ('exit_selected', 'bid_units')
        book = books[quote_key].get(symbol)
        require(book is not None and book == decision['book'], f'{context}: independent book verification failed')
        require(book[depth_key] == decision['available_units'], f'{context}: visible depth differs')
        verified_orders.append((order, book, opening['prior_closes'][symbol]))
    return base_close, verified_orders, prices


def resize_order(day, order, book, previous_close, multiplier):
    symbol, side, original_units = order['symbol'], order['side'], order['units']
    context = f'{day} {symbol} {side} at {multiplier}x'
    require(type(original_units) is int and original_units > 0 and original_units % LOT_SIZE == 0,
            f'{context}: baseline order is not a positive whole lot')
    units = original_units * multiplier
    kind = 'etf' if symbol.startswith(('15', '51', '56', '58')) else 'stock'
    tick = Decimal('.001' if kind == 'etf' else '.01')
    price = Decimal(order['price'])
    if side == 'buy':
        depth = book['ask_units']
        require(type(depth) is int and depth >= units, f'{context}: requires {units}, ask size {depth}')
        require(Decimal(book['previous']) == Decimal(previous_close), f'{context}: previous close mismatch')
        expected_price = tick_price(Decimal(book['ask']) * Decimal('1.0005'), tick, ROUND_CEILING)
        limit_price = tick_price(Decimal(previous_close) * Decimal('1.01'), tick, ROUND_FLOOR)
        require(price == expected_price and 0 < price <= limit_price, f'{context}: price or limit mismatch')
        gross = price * units
        commission, stamp, transfer = (Decimal(str(value)) for value in COSTS.charges(day, kind, side, float(gross)))
        fee = commission + stamp + transfer
        fill = {'symbol': symbol, 'side': side, 'units': units, 'price': str(price), 'fees': str(fee),
                'commission': str(commission), 'stamp_tax': str(stamp), 'transfer_fee': str(transfer),
                'cash_delta': str(-gross - fee)}
    else:
        depth = book['bid_units']
        reason, fill = sale(day, symbol, units, book, previous_close)
        require(reason == 'filled' and fill is not None, f'{context}: {reason}, requires {units}, bid size {depth}')
        require(price == Decimal(fill['price']), f'{context}: price mismatch')
    if multiplier == 1:
        require(all(Decimal(fill[field]) == Decimal(order[field]) for field in MONEY_FIELDS),
                f'{context}: baseline recalculated prices or costs differ from archived fill')
    return {**fill, 'book_units': depth, 'quote_at': book['at'], 'quote_received_at': book['received_at'],
            'quote_raw_file': book['raw_file'], 'quote_raw_sha256': book['raw_sha256']}


def replay_session(day, account, orders, prices, multiplier):
    cash, positions = Decimal(account['cash']), dict(account['positions'])
    fills = []
    has_buys = False
    for order, book, previous_close in orders:
        fill = resize_order(day, order, book, previous_close, multiplier)
        symbol, side, units = fill['symbol'], fill['side'], fill['units']
        context = f'{day} {symbol} {side} at {multiplier}x'
        if side == 'buy':
            has_buys = True
            require(symbol not in positions, f'{context}: existing position was not exited')
            positions[symbol] = units
        else:
            require(not has_buys and positions.get(symbol) == units, f'{context}: sell is not a full overnight exit')
            del positions[symbol]
        cash += Decimal(fill['cash_delta'])
        require(cash >= 0, f'{context}: negative cash after fill')
        fills.append(fill)
    require(positions.keys() == prices.keys(), f'{day} at {multiplier}x: positions lack paired valuations')
    market_value = sum((Decimal(prices[symbol]) * units for symbol, units in positions.items()), Decimal(0))
    nav = cash + market_value
    return {'cash': str(cash), 'positions': positions, 'fills': fills, 'close_prices': prices,
            'market_value': str(market_value), 'invested_fraction': str(market_value / nav), 'nav': str(nav),
            'fees': str(sum((Decimal(fill['fees']) for fill in fills), Decimal(0)))}


def run():
    protocol_hash = hashlib.sha256((SOURCE / 'protocol.md').read_bytes()).hexdigest()
    require(protocol_hash == PROTOCOL, 'frozen source protocol hash mismatch')
    hashes = {'protocol.md': protocol_hash}
    accounts = {name: {'cash': str(INITIAL), 'positions': {}} for name in SIZE_MULTIPLIERS}
    days = []
    for day in SESSIONS:
        archived, orders, prices = verified_session(day, hashes)
        accounts = {name: replay_session(day, accounts[name], orders, prices, multiplier)
                    for name, multiplier in SIZE_MULTIPLIERS.items()}
        baseline = accounts['baseline']
        require(baseline['positions'] == archived['positions'], f'{day}: baseline positions mismatch')
        for field in ('cash', 'nav', 'fees'):
            require(Decimal(baseline[field]) == Decimal(archived[field]), f'{day}: baseline {field} mismatch')
        days.append({'date': str(day), **accounts})

    # Reuse the existing two-account reconciler. It expects the second key to be 'filtered'.
    reconciled = reconcile([{'date': row['date'], 'baseline': row['baseline'], 'filtered': row['heavy']}
                            for row in days])['groups']
    summary = {'baseline': reconciled['baseline'], 'heavy': reconciled['filtered']}
    for name, metrics in summary.items():
        metrics['fill_count'] = sum(len(row[name]['fills']) for row in days)
        metrics['peak_close_invested_fraction'] = str(max(Decimal(row[name]['invested_fraction']) for row in days))
    return {'mode': 'retrospective_matched_paper_fills', 'initial_virtual_cash': str(INITIAL),
            'quantity_multipliers': SIZE_MULTIPLIERS, 'baseline_reconciled_exactly': True,
            'excluded_sessions': {'2026-11-19': 'missing comparable session ledger',
                                  '2026-11-20': 'missing comparable session ledger'},
            'limitations': ['Only the already accepted original-rule trades are replayed.',
                            'Observable top-of-book depth is checked, with no queue position or market-impact estimate.',
                            'Drawdown uses initial cash and daily closes. Terminal shares are marked to market.',
                            'Only executed fills contribute fees. Open positions will incur costs if sold later.'],
            'source_artifact': SOURCE.name, 'source_sha256': hashes, 'summary': summary, 'daily': days}


if __name__ == '__main__':
    try:
        print(json.dumps(run(), indent=2, ensure_ascii=False))
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError) as exc:
        raise SystemExit(f'Replay failed: {exc}') from exc
