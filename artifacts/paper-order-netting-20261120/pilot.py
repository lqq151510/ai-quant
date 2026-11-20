"""Compare netting against the settled original-rule forward pilot, read only."""

import json
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

sys.dont_write_bytecode = True
PILOT_ROOT = Path(__file__).resolve().parents[1] / 'paper-forward-20261123'
sys.path.insert(0, str(PILOT_ROOT))

from config import CST, DAYS, load_candidates, previous_session
from run import CASH_CONTROL, read_json, prior_state, status
from book_evidence import verify_capture
from close_monday import closing_price, verify_mark_provenance
from continue_open import run_day
from reconcile import reconcile

from netting import SAVINGS_FIELDS, net_session


def verified_inputs(day, original, prior):
    folder = PILOT_ROOT / str(day)
    manifest = read_json(folder / 'opening-inputs.json')
    prepared = load_candidates(day, folder / manifest['candidates_file'])
    symbols = manifest['symbols']
    if symbols:
        books = folder / 'books'
        capture = verify_capture(read_json(books / 'books.json'), books, day, symbols,
                                 held=manifest['held'], buy_symbols=manifest['candidates'])
    else:
        capture = {'selected': {}, 'exit_selected': {}, 'unavailable': {}, 'attempts': []}

    prior_day = previous_session(day)
    opening = run_day(day, {'baseline': prior}, prepared, capture,
                      {'batches': [], 'errors': [], 'sources': []},
                      prior_day=prior_day, groups=('baseline',))
    if any(opening['baseline'][key] != original[key] for key in ('fills', 'positions', 'cash', 'decisions')):
        raise ValueError(f'{day}: replay does not match the original opening decisions')

    marks = read_json(folder / 'close-progress.json')['marks']
    for symbol, expected in original['close_prices'].items():
        mark = marks.get(symbol)
        price, error = closing_price(mark, str(day))
        if error or price != Decimal(expected):
            raise ValueError(f'{day} {symbol}: close valuation mismatch ({error})')
        error = verify_mark_provenance(mark, folder, True, str(day), symbol,
                                       opening['prior_closes'][symbol], prior_day=prior_day)
        if error:
            raise ValueError(f'{day} {symbol}: close valuation evidence failed ({error})')

    quote_by_order = {
        (row['symbol'], row['side']): row['book'] for row in opening['baseline']['decisions']
        if row['reason'] == 'filled'
    }
    orders = [(order, quote_by_order[order['symbol'], order['side']], opening['prior_closes'][order['symbol']])
              for order in original['fills']]
    return orders, len(prepared['candidates'])


def observed_days():
    progress = status(datetime.now(CST))
    settled = progress['settled_sessions']
    if settled == len(DAYS):
        records, _ = prior_state(DAYS[-1])
        records.append(read_json(PILOT_ROOT / str(DAYS[-1]) / 'daily.json'))
    else:
        records, _ = prior_state(DAYS[settled])
    return progress, records


def compare():
    progress, records = observed_days()
    baseline = CASH_CONTROL
    shadow_cash = baseline['cash']
    daily = []
    for session in records:
        day = DAYS[len(daily)]
        original = session['baseline']
        orders, candidate_count = verified_inputs(day, original, baseline)
        netted = net_session(day, orders, prior_cash=baseline['cash'], prior_nav=baseline['nav'],
                             positions=baseline['positions'], candidate_count=candidate_count,
                             shadow_cash=shadow_cash)
        if netted['positions'] != original['positions']:
            raise ValueError(f'{day}: netting altered end-of-day exposure')
        shadow_cash = netted['cash']
        prices = original['close_prices']
        nav = Decimal(shadow_cash) + sum(Decimal(prices[code]) * count
                                         for code, count in netted['positions'].items())
        daily.append(dict(date=str(day), baseline=original, filtered={
            **netted, 'nav': str(nav), 'close_prices': prices, 'valuation_gap': {},
        }))
        baseline = original

    pairs = sum(len(row['filtered']['matches']) for row in daily)
    summary = reconcile(daily)['groups'] if daily else None
    if summary:
        for group, metrics in summary.items():
            fills = [fill for row in daily for fill in row[group]['fills']]
            metrics['fill_count'] = len(fills)
            metrics['turnover_notional'] = str(sum(
                (Decimal(fill['price']) * fill['units'] for fill in fills), Decimal(0),
            ))
    savings = {
        key: str(sum((Decimal(row['filtered']['savings'][key]) for row in daily), Decimal(0)))
        for key in SAVINGS_FIELDS
    } if daily else None
    return {
        'state': 'observations_available' if daily else 'awaiting_first_settlement',
        'planned_sessions': [str(day) for day in DAYS],
        'settled_sessions': len(daily),
        'same_snapshot_matches': pairs,
        'savings': savings,
        'metrics': dict(original=summary['baseline'], netting=summary['filtered']) if summary else None,
        'sessions': progress['sessions'],
        'daily': [{'date': row['date'], 'original': row['baseline'], 'netting': row['filtered']} for row in daily],
    }


if __name__ == '__main__':
    try:
        print(json.dumps(compare(), ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError) as error:
        raise SystemExit(f'Could not verify forward comparison: {error}') from error
