"""Reprice confirmed historical fills under the ledger's double-friction scenario."""

import json
from datetime import date
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR

from confirm import INITIAL, compare_archive, verified_session
from monday_open import tick_price
from quant_research.ledger import Costs


def price_order(day, fill, book, prior_close):
    symbol, side, units = fill['symbol'], fill['side'], fill['units']
    direction = 1 if side == 'buy' else -1
    buy = side == 'buy'
    kind = 'etf' if symbol.startswith(('15', '51', '56', '58')) else 'stock'
    tick = Decimal('.001' if kind == 'etf' else '.01')
    costs = Costs().stressed()
    slippage = Decimal(str(costs.slippage_bps)) / 10000
    quote = Decimal(book['ask' if buy else 'bid'])
    price = tick_price(quote * (1 + direction * slippage), tick, ROUND_CEILING if buy else ROUND_FLOOR)
    fees = sum(Decimal(str(part)) for part in costs.charges(day, kind, side, float(price * units)))
    if buy:
        cap = tick_price(Decimal(prior_close) * Decimal('1.01'), tick, ROUND_FLOOR)
        within_limit = 0 < price <= cap
    else:
        within_limit = Decimal(book['lower_limit']) <= price <= Decimal(book['upper_limit'])
    return price, fees, direction, within_limit


def compare():
    archive = compare_archive()
    extra_fees = extra_execution = Decimal(0)
    stress_cash = prior_nav = INITIAL
    issues = []
    by_day = []
    for row in archive['daily']:
        day = date.fromisoformat(row['date'])
        _, originals, _ = verified_session(day, {})
        quotes = {(fill['symbol'], fill['side']): (book, reference) for fill, book, reference in originals}
        account = row['confirmation']
        budget = min(Decimal(5000), Decimal('.05') * prior_nav)
        for fill in account['fills']:
            key = fill['symbol'], fill['side']
            book, reference = quotes[key]
            price, fees, direction, within_limit = price_order(day, fill, book, reference)
            amount = price * fill['units']
            extra_fees += fees - Decimal(fill['fees'])
            extra_execution += direction * (price - Decimal(fill['price'])) * fill['units']
            stress_cash -= direction * amount + fees
            if not within_limit or stress_cash < 0 or (direction == 1 and amount + fees > budget):
                issues.append({'date': str(day), 'symbol': key[0], 'side': key[1]})
        stress_nav = stress_cash + Decimal(account['market_value'])
        by_day.append({'date': str(day), 'stress_nav': str(stress_nav)})
        prior_nav = Decimal(row['original']['nav'])
    normal = Decimal(archive['summary']['confirmation']['ending_nav'])
    if normal - stress_nav != extra_fees + extra_execution:
        raise ValueError('friction sensitivity does not reconcile')
    return {
        'mode': 'fixed_fill_double_friction_sensitivity',
        'normal_net_pnl': str(normal - INITIAL),
        'stressed_net_pnl': str(stress_nav - INITIAL),
        'extra_fees': str(extra_fees), 'extra_execution': str(extra_execution),
        'replayed_fills': archive['summary']['confirmation']['fill_count'],
        'purchase_budget_or_price_or_cash_issues': issues, 'daily': by_day,
    }


if __name__ == '__main__':
    print(json.dumps(compare(), ensure_ascii=False, indent=2))
