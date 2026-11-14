"""Reconcile consecutive daily records and calculate paired close-to-close metrics."""
import argparse
import json
from decimal import Decimal
from pathlib import Path

INITIAL = Decimal('100000')


def reconcile(records):
    output = {}
    dates = [row['date'] for row in records]
    if dates != sorted(set(dates)):
        raise ValueError('daily records must be chronological and unique')
    for group in ('baseline', 'filtered'):
        cash = peak = INITIAL
        positions = {}
        drawdown = fees = Decimal(0)
        last_nav = None
        incomplete = []
        for row in records:
            account = row[group]
            for order in account['fills']:
                symbol, units = order['symbol'], order['units']
                if type(units) is not int or units <= 0 or order['side'] not in ('buy', 'sell'):
                    raise ValueError('invalid fill')
                direction = 1 if order['side'] == 'buy' else -1
                gross = Decimal(order['price']) * units
                fee = sum(Decimal(order[key]) for key in ('commission', 'stamp_tax', 'transfer_fee'))
                delta = -gross * direction - fee
                if delta != Decimal(order['cash_delta']) or fee != Decimal(order['fees']):
                    raise ValueError(f'{group} {row["date"]}: inconsistent fill cash or fees')
                cash += delta
                fees += fee
                positions[symbol] = positions.get(symbol, 0) + direction * units
                if positions[symbol] < 0:
                    raise ValueError('negative holdings')
                if not positions[symbol]:
                    del positions[symbol]
            # Each verified company action carries a cash and/or share delta.
            for action in account.get('corporate_actions', []):
                if not action.get('source_url'):
                    raise ValueError('corporate action requires source evidence')
                cash += Decimal(action.get('cash_delta', '0'))
                if 'units_delta' in action:
                    symbol = action['symbol']
                    positions[symbol] += action['units_delta']
                    if positions[symbol] < 0:
                        raise ValueError('negative holdings after corporate action')
                    if not positions[symbol]:
                        del positions[symbol]
            if cash != Decimal(account['cash']) or positions != account['positions']:
                raise ValueError(f'{group} {row["date"]}: cash or shares do not reconcile')
            prices = account['close_prices']
            missing = any(prices.get(symbol) is None for symbol in positions)
            if missing or account.get('valuation_gap'):
                if account['nav'] is not None:
                    raise ValueError('unverified NAV must be null')
                incomplete.append(row['date'])
                last_nav = None
                continue
            last_nav = cash + sum(Decimal(prices[symbol]) * n for symbol, n in positions.items())
            if account['nav'] is None or last_nav != Decimal(account['nav']):
                raise ValueError(f'{group} {row["date"]}: NAV does not reconcile')
            peak = max(peak, last_nav)
            drawdown = max(drawdown, (peak - last_nav) / peak)
        output[group] = {
            'fees': str(fees), 'ending_positions': positions, 'ending_nav': str(last_nav) if last_nav is not None else None,
            'net_return': str(last_nav / INITIAL - 1) if last_nav is not None else None,
            'max_drawdown': str(drawdown) if records and not incomplete else None, 'missing_valuation_dates': incomplete,
        }
    return {'dates': dates, 'groups': output}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('daily', nargs='+', type=Path)
    args = parser.parse_args()
    print(json.dumps(reconcile([json.loads(path.read_text()) for path in args.daily]), indent=2))
