"""Read-only midpoint-at-previous-close confirmation on fixed original paper fills."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

sys.dont_write_bytecode = True
ARTIFACTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARTIFACTS / 'paper-exposure-20261120'))

from replay import PROTOCOL, SESSIONS, SOURCE, replay_session, verified_session
from reconcile import INITIAL, reconcile


def opening_signal(book, previous_close):
    """Assess the verified snapshot used for the original buy, without another quote."""
    try:
        bid, ask, close, book_close = (Decimal(value) for value in
                                      (book['bid'], book['ask'], previous_close, book['previous']))
        if (not all(value.is_finite() and value > 0 for value in (bid, ask, close, book_close))
                or bid > ask or close != book_close
                or any(type(book.get(key)) is not int or book[key] <= 0
                       for key in ('bid_units', 'ask_units'))):
            raise ValueError('incomplete or inconsistent two-sided quote')
    except (KeyError, TypeError, ValueError, InvalidOperation):
        return 'unknown', None
    midpoint = (bid + ask) / 2
    return ('below_previous_close' if midpoint < close else 'confirmed'), midpoint


def close_outcome(lot, day, price, state, cash_proceeds):
    pnl = cash_proceeds + Decimal(lot['entry_cash_delta'])
    return {**lot, 'outcome_date': str(day), 'outcome_price': str(price), 'state': state, 'pnl': str(pnl)}


def account_metrics(records):
    groups = reconcile(records)['groups']
    for name, summary in groups.items():
        accounts = [row[name] for row in records]
        fills = [fill for account in accounts for fill in account['fills']]
        summary.update(
            fill_count=len(fills),
            buy_count=sum(fill['side'] == 'buy' for fill in fills),
            sell_count=sum(fill['side'] == 'sell' for fill in fills),
            entry_notional=str(sum(
                (Decimal(fill['price']) * fill['units'] for fill in fills if fill['side'] == 'buy'), Decimal(0),
            )),
            turnover_notional=str(sum((Decimal(fill['price']) * fill['units'] for fill in fills), Decimal(0))),
            return_before_explicit_fees=str(Decimal(summary['net_return']) + Decimal(summary['fees']) / INITIAL),
            mean_close_invested_fraction=str(sum(
                (Decimal(account['invested_fraction']) for account in accounts), Decimal(0),
            ) / len(accounts)),
        )
    return {
        'original': groups['baseline'], 'confirmation': groups['filtered'],
        'cash_control': {
            'fees': '0', 'ending_positions': {}, 'ending_nav': str(INITIAL), 'net_return': '0',
            'max_drawdown': '0', 'missing_valuation_dates': [], 'fill_count': 0, 'buy_count': 0,
            'sell_count': 0, 'entry_notional': '0', 'turnover_notional': '0',
            'return_before_explicit_fees': '0', 'mean_close_invested_fraction': '0',
        },
    }


def close_values(cash, positions, prices, nav):
    market_value = sum((Decimal(prices[symbol]) * units for symbol, units in positions.items()), Decimal(0))
    if nav <= 0 or nav != cash + market_value:
        raise ValueError('cash and marked positions do not reconcile to NAV')
    return {'market_value': str(market_value), 'invested_fraction': str(market_value / nav)}


def compare_sessions(sessions):
    """Replay the observable subset of original purchases with the same subsequent exits."""
    cash = INITIAL
    positions, blocked = {}, {}
    outcomes, records, decisions = [], [], []
    for day, original, orders in sessions:
        orders = tuple(orders)
        prices = original['close_prices']
        if original.get('corporate_actions') or original.get('valuation_gap') or original['nav'] is None:
            raise ValueError(f'{day}: unsupported action or unavailable original valuation')
        source_positions = original['positions']
        fills, session_decisions = [], []
        for fill, (intent, book, previous_close) in zip(original['fills'], orders, strict=True):
            symbol, side, units = fill['symbol'], fill['side'], fill['units']
            identity = symbol, side, units
            if (identity != (intent['symbol'], intent['side'], intent['units']) or side not in ('buy', 'sell')
                    or type(units) is not int or units <= 0):
                raise ValueError(f'{day} {symbol}: fill is not aligned to original order')
            delta = Decimal(fill['cash_delta'])
            if side == 'buy':
                if symbol in positions or symbol in blocked:
                    raise ValueError(f'{day} {symbol}: new buy overlaps an open lot')
                snapshot = book if isinstance(book, dict) else {}
                signal, midpoint = opening_signal(snapshot, previous_close)
                session_decisions.append({
                    'symbol': symbol, 'date': str(day), 'units': units,
                    'signal': signal, 'midpoint': str(midpoint) if midpoint is not None else None,
                    'previous_close': str(previous_close) if previous_close is not None else None,
                    'quote_received_at': snapshot.get('received_at'),
                    'quote_raw_sha256': snapshot.get('raw_sha256'),
                })
                if signal == 'below_previous_close':
                    blocked[symbol] = {
                        'symbol': symbol, 'entry_date': str(day), 'units': units, 'entry_price': fill['price'],
                        'entry_cash_delta': fill['cash_delta'],
                    }
                    continue
                positions[symbol] = units
            elif symbol in blocked:
                lot = blocked.pop(symbol)
                if lot['units'] != units:
                    raise ValueError(f'{day} {symbol}: original exit does not close the skipped lot')
                outcomes.append(close_outcome(lot, day, fill['price'], 'closed', delta))
                continue
            elif positions.pop(symbol, None) != units:
                raise ValueError(f'{day} {symbol}: original exit does not align to shadow shares')
            cash += delta
            if not cash.is_finite() or cash < 0:
                raise ValueError(f'{day} {symbol}: insufficient shadow cash for unchanged order')
            fills.append(fill)
        if (positions.keys() & blocked.keys() or
                (positions | {symbol: lot['units'] for symbol, lot in blocked.items()}) != source_positions):
            raise ValueError(f'{day}: shadow and skipped shares must partition original shares')
        shadow_prices = {symbol: prices[symbol] for symbol in positions}
        nav = cash + sum((Decimal(price) * positions[symbol] for symbol, price in shadow_prices.items()),
                         Decimal(0))
        original_account = {key: original[key] for key in ('cash', 'positions', 'close_prices', 'fills', 'nav')}
        original_fees = sum((Decimal(fill['fees']) for fill in original['fills']), Decimal(0))
        records.append({
            'date': str(day),
            'baseline': {**original_account, 'fees': str(original_fees),
                         **close_values(Decimal(original['cash']), source_positions, prices, Decimal(original['nav']))},
            'filtered': {'cash': str(cash), 'positions': dict(positions), 'close_prices': shadow_prices,
                         'fills': fills, 'fees': str(sum((Decimal(fill['fees']) for fill in fills), Decimal(0))),
                         'nav': str(nav), **close_values(cash, positions, shadow_prices, nav)},
            'entry_decisions': session_decisions,
        })
        decisions.extend(session_decisions)
    if not records:
        return {'state': 'awaiting_first_settlement', 'settled_sessions': 0, 'summary': None,
                'signal_counts': None, 'blocked_outcomes': None, 'daily': []}

    last = records[-1]
    for symbol, lot in blocked.items():
        price = last['baseline']['close_prices'][symbol]
        outcomes.append(close_outcome(lot, last['date'], price, 'terminal_marked', Decimal(price) * lot['units']))
    net_difference = Decimal(last['filtered']['nav']) - Decimal(last['baseline']['nav'])
    if net_difference != -sum((Decimal(lot['pnl']) for lot in outcomes), Decimal(0)):
        raise ValueError('shadow P&L difference does not reconcile to the skipped trades')
    attribution = {}
    for outcome_state in ('closed', 'terminal_marked'):
        pnls = [Decimal(lot['pnl']) for lot in outcomes if lot['state'] == outcome_state]
        attribution[outcome_state] = {
            'loss_count': sum(pnl < 0 for pnl in pnls), 'gain_count': sum(pnl > 0 for pnl in pnls),
            'flat_count': sum(pnl == 0 for pnl in pnls),
            'avoided_losses': str(-sum((pnl for pnl in pnls if pnl < 0), Decimal(0))),
            'missed_gains': str(sum((pnl for pnl in pnls if pnl > 0), Decimal(0))),
        }
    return {
        'state': 'observations_available', 'settled_sessions': len(records), 'summary': account_metrics(records),
        'signal_counts': dict.fromkeys(('confirmed', 'below_previous_close', 'unknown'), 0)
        | Counter(row['signal'] for row in decisions),
        'blocked_outcomes': {'net_pnl_difference': str(net_difference), **attribution, 'trades': outcomes},
        'daily': [{'date': row['date'], 'original': row['baseline'], 'confirmation': row['filtered'],
                   'entry_decisions': row['entry_decisions']} for row in records],
    }


def compare_archive():
    protocol_hash = hashlib.sha256((SOURCE / 'protocol.md').read_bytes()).hexdigest()
    if protocol_hash != PROTOCOL:
        raise ValueError('source archive protocol hash mismatch')
    hashes = {'protocol.md': protocol_hash}
    prior = {'cash': str(INITIAL), 'positions': {}}
    sessions = []
    for day in SESSIONS:
        original, orders, prices = verified_session(day, hashes)
        replay = replay_session(day, prior, orders, prices, 1)
        if (replay['positions'] != original['positions']
                or any(Decimal(replay[key]) != Decimal(original[key]) for key in ('cash', 'nav', 'fees'))):
            raise ValueError(f'{day}: original cash, positions or fees do not reconcile')
        sessions.append((day, original, orders))
        prior = original
    return {'mode': 'retrospective_fixed_original_intents', 'source_artifact': SOURCE.name,
            'source_sha256': hashes, **compare_sessions(sessions)}


def compare_pilot():
    sys.path.insert(0, str(ARTIFACTS / 'paper-order-netting-20261120'))
    from pilot import CASH_CONTROL, DAYS, observed_days, verified_inputs

    progress, records = observed_days()
    prior = CASH_CONTROL
    sessions = []
    if len(records) > len(DAYS):
        raise ValueError('settled pilot cannot exceed the declared calendar')
    for day, record in zip(DAYS, records):
        if record['date'] != str(day):
            raise ValueError('settled pilot dates do not align with the declared calendar')
        original = record['baseline']
        orders, _ = verified_inputs(day, original, prior)
        sessions.append((day, original, orders))
        prior = original
    return {'mode': 'forward_fixed_original_intents', 'planned_sessions': [str(day) for day in DAYS],
            'sessions': progress['sessions'], **compare_sessions(sessions)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', choices=('archive', 'pilot'))
    args = parser.parse_args()
    try:
        result = compare_archive() if args.source == 'archive' else compare_pilot()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError) as error:
        raise SystemExit(f'Could not verify opening confirmation: {error}') from error
