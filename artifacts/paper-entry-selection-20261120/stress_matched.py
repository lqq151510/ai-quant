"""Reprice verified original lots before comparing equal-budget selection under stress."""

import argparse
import json
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

sys.dont_write_bytecode = True
CONFIRMATION = Path(__file__).resolve().parents[1] / 'paper-open-confirmation-20261120'
sys.path.insert(0, str(CONFIRMATION))

from confirm import INITIAL, compare_archive, compare_pilot, compare_sessions, verified_session
from stress import price_order
from quant_research.ledger import Costs

from matched import audit

STRESSED_COSTS = Costs().stressed()


def stressed_sessions(standard, source):
    if source == 'pilot':
        from pilot import CASH_CONTROL, observed_days, verified_inputs

        prior = CASH_CONTROL
        _, raw_records = observed_days()
        if len(raw_records) != len(standard['daily']):
            raise ValueError('settled pilot coverage changed during repricing')
    cash = prior_nav = INITIAL
    sessions, issues = [], []
    for index, row in enumerate(standard['daily']):
        day = date.fromisoformat(row['date'])
        original = row['original']
        if source == 'archive':
            _, orders, _ = verified_session(day, {})
        else:
            raw_record = raw_records[index]
            raw = raw_record['baseline']
            if (raw_record['date'] != row['date']
                    or any(raw[key] != original[key] for key in ('cash', 'positions', 'close_prices', 'fills', 'nav'))):
                raise ValueError(f'{day}: settled baseline changed during repricing')
            orders, _ = verified_inputs(day, raw, prior)
            prior = raw
        if [fill for fill, _, _ in orders] != original['fills']:
            raise ValueError(f'{day}: original order evidence changed during repricing')
        budget = min(Decimal('5000'), prior_nav * Decimal('.05'))
        repriced = []
        for fill, book, reference in orders:
            symbol, side, units = fill['symbol'], fill['side'], fill['units']
            price, fees, direction, within_limit = price_order(day, fill, book, reference)
            gross = price * units
            kind = 'etf' if symbol.startswith(('15', '51', '56', '58')) else 'stock'
            parts = [Decimal(str(part)) for part in STRESSED_COSTS.charges(day, kind, side, float(gross))]
            if sum(parts) != fees:
                raise ValueError(f'{day} {symbol}: inconsistent stressed fees')
            delta = -direction * gross - fees
            cash += delta
            if not within_limit or cash < 0 or (direction == 1 and -delta > budget):
                issues.append({'date': str(day), 'symbol': symbol, 'side': side,
                               'within_limit': within_limit, 'cash_nonnegative': cash >= 0,
                               'within_buy_budget': direction != 1 or -delta <= budget})
            commission, stamp, transfer = map(str, parts)
            stressed = {**fill, 'price': str(price), 'cash_delta': str(delta), 'fees': str(fees),
                        'commission': commission, 'stamp_tax': stamp, 'transfer_fee': transfer}
            repriced.append((stressed, book, reference))
        nav = cash + Decimal(original['market_value'])
        sessions.append((day, {**original, 'cash': str(cash), 'nav': str(nav),
                               'fills': [fill for fill, _, _ in repriced]}, repriced))
        prior_nav = Decimal(original['nav'])
    return sessions, issues


def compare(source):
    standard = compare_archive() if source == 'archive' else compare_pilot()
    sessions, issues = stressed_sessions(standard, source)
    metadata = {
        'mode': f'{source}_fixed_lot_double_commission_and_slippage',
        'source_sha256': standard.get('source_sha256'),
        'settled_sessions': len(sessions),
        'purchase_budget_or_price_or_cash_issues': issues,
        'market_sessions': standard.get('sessions'),
    }
    if issues:
        return {**metadata, 'state': 'stressed_fills_not_feasible', 'matched': None}
    stressed = compare_sessions(sessions)
    if stressed['signal_counts'] != standard['signal_counts']:
        raise ValueError('stress changed the fixed confirmation decisions')
    matched = audit(stressed)
    return {**metadata, 'state': stressed['state'], 'matched': matched}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', choices=('archive', 'pilot'))
    args = parser.parse_args()
    try:
        print(json.dumps(compare(args.source), ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError) as error:
        raise SystemExit(f'Could not verify cost-stressed matching: {error}') from error
