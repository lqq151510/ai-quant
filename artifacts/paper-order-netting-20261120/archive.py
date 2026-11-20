"""Verify and compare the archived original-rule and netting paper ledgers."""

import hashlib
import json
import sys
from decimal import Decimal
from pathlib import Path

sys.dont_write_bytecode = True
EXPOSURE = Path(__file__).resolve().parents[1] / 'paper-exposure-20261120'
sys.path.insert(0, str(EXPOSURE))

from replay import PROTOCOL, SESSIONS, SOURCE, read_input, replay_session, require, verified_session
from reconcile import INITIAL, reconcile

from netting import SAVINGS_FIELDS, net_session


def compare():
    protocol_hash = hashlib.sha256((SOURCE / 'protocol.md').read_bytes()).hexdigest()
    require(protocol_hash == PROTOCOL, 'frozen source protocol hash mismatch')
    hashes = {'protocol.md': protocol_hash}
    original = {'cash': str(INITIAL), 'nav': str(INITIAL), 'positions': {}}
    netting_cash = original['cash']
    daily = []
    for day in SESSIONS:
        archived, orders, prices = verified_session(day, hashes)
        opening = read_input(SOURCE / str(day) / 'opening.json', hashes)
        reference = replay_session(day, original, orders, prices, 1)
        require(reference['positions'] == archived['positions'], f'{day}: original positions differ')
        for key in ('cash', 'nav', 'fees'):
            require(Decimal(reference[key]) == Decimal(archived[key]), f'{day}: original {key} differs')
        netted = net_session(day, orders, prior_cash=original['cash'], prior_nav=original['nav'],
                             positions=original['positions'], candidate_count=len(opening['candidate_symbols']),
                             shadow_cash=netting_cash)
        require(netted['positions'] == reference['positions'], f'{day}: netting changed end-of-day exposure')
        netting_cash = netted['cash']
        nav = Decimal(netting_cash) + Decimal(reference['market_value'])
        daily.append({'date': str(day), 'original': reference,
                      'netting': {**netted, 'nav': str(nav), 'close_prices': prices}})
        original = reference
    groups = reconcile([{'date': row['date'], 'baseline': row['original'], 'filtered': row['netting']}
                        for row in daily])['groups']
    summary = {'original': groups['baseline'], 'netting': groups['filtered']}
    for name, metrics in summary.items():
        metrics['fill_count'] = sum(len(row[name]['fills']) for row in daily)
        metrics['turnover_notional'] = str(sum(
            (Decimal(fill['price']) * fill['units'] for row in daily for fill in row[name]['fills']), Decimal(0),
        ))
    savings = {field: str(sum((Decimal(row['netting']['savings'][field]) for row in daily), Decimal(0)))
               for field in SAVINGS_FIELDS}
    return {
        'mode': 'retrospective_original_intents_with_same_snapshot_netting',
        'source_artifact': SOURCE.name, 'source_sha256': hashes, 'initial_virtual_cash': str(INITIAL),
        'paired_end_of_day_positions': True, 'matched_pairs': sum(len(row['netting']['matches']) for row in daily),
        'savings': savings, 'summary': summary, 'daily': daily,
    }


if __name__ == '__main__':
    try:
        print(json.dumps(compare(), ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError) as error:
        raise SystemExit(f'Could not verify archived comparison: {error}') from error
