"""Pair opposite original-rule instructions from one verified opening quote."""

from decimal import Decimal

from quant_research.ledger import Costs, LOT_SIZE

COSTS = Costs()
SAVINGS_FIELDS = ('spread', 'slippage', 'execution', 'fees', 'total')
FEE_FIELDS = ('commission', 'stamp_tax', 'transfer_fee', 'fees', 'cash_delta')


def reprice(day, fill, units):
    """Charge the original paper fees on the remaining order at its original price."""
    symbol, side = fill['symbol'], fill['side']
    gross = Decimal(fill['price']) * units
    kind = 'etf' if symbol.startswith(('15', '51', '56', '58')) else 'stock'
    commission, stamp, transfer = (Decimal(str(fee)) for fee in COSTS.charges(day, kind, side, float(gross)))
    fees = commission + stamp + transfer
    cash_delta = (gross if side == 'sell' else -gross) - fees
    return {**fill, 'units': units, 'commission': str(commission), 'stamp_tax': str(stamp),
            'transfer_fee': str(transfer), 'fees': str(fees), 'cash_delta': str(cash_delta)}


def validate_orders(day, orders):
    indexed = {}
    for fill, book, _ in orders:
        side, symbol, units = fill['side'], fill['symbol'], fill['units']
        key = side, symbol
        if side not in ('buy', 'sell') or key in indexed:
            raise ValueError(f'{day} {symbol}: invalid side or repeated order')
        if type(units) is not int or units <= 0 or units % LOT_SIZE:
            raise ValueError(f'{day} {symbol}: expected a positive whole-lot order')
        depth = book['ask_units' if side == 'buy' else 'bid_units']
        price = Decimal(fill['price'])
        if type(depth) is not int or depth < units or not price.is_finite() or price <= 0:
            raise ValueError(f'{day} {symbol}: invalid fill price or insufficient observed depth')
        recalculated = reprice(day, fill, units)
        if any(Decimal(fill[field]) != Decimal(recalculated[field]) for field in FEE_FIELDS):
            raise ValueError(f'{day} {symbol}: original order cash or fees differ from the original cost model')
        indexed[key] = fill, book
    return indexed


def exit_batch_gate(indexed, positions, prior_cash, budget, candidate_count):
    if not positions:
        return 'no_carried_positions'
    if prior_cash < budget * candidate_count:
        return 'insufficient_cash_without_exit_proceeds'
    sales = {symbol: (fill, book) for (side, symbol), (fill, book) in indexed.items() if side == 'sell'}
    if {symbol: fill['units'] for symbol, (fill, _) in sales.items()} != positions:
        return 'exit_batch_incomplete'
    batches = {tuple(book.get(key) for key in ('source_url', 'raw_file', 'raw_sha256', 'received_at'))
               for _, book in sales.values()}
    if len(batches) != 1 or not all(next(iter(batches))):
        return 'exit_batch_not_shared'
    return 'shared_exit_batch'


def balance(day, cash, positions, fills):
    remaining = dict(positions)
    for fill in fills:
        symbol, units = fill['symbol'], fill['units']
        delta = units if fill['side'] == 'buy' else -units
        quantity = remaining.get(symbol, 0) + delta
        cash += Decimal(fill['cash_delta'])
        if quantity < 0 or cash < 0:
            raise ValueError(f'{day} {symbol}: negative position or cash after fill')
        if quantity:
            remaining[symbol] = quantity
        else:
            del remaining[symbol]
    return cash, remaining


def net_session(day, orders, *, prior_cash, prior_nav, positions, candidate_count, shadow_cash=None):
    """Execution overlay. Call only with original fills from independently verified quotes."""
    prior_cash, prior_nav = Decimal(prior_cash), Decimal(prior_nav)
    opening_shadow_cash = prior_cash if shadow_cash is None else Decimal(shadow_cash)
    if (not all(value.is_finite() for value in (prior_cash, prior_nav, opening_shadow_cash))
            or prior_cash < 0 or prior_nav <= 0 or opening_shadow_cash < prior_cash):
        raise ValueError(f'{day}: invalid original or netting opening capital')
    if type(candidate_count) is not int or not 0 <= candidate_count <= 10:
        raise ValueError(f'{day}: candidate count outside the original policy')
    if any(type(units) is not int or units <= 0 for units in positions.values()):
        raise ValueError(f'{day}: invalid carried position')
    orders = tuple(orders)
    indexed = validate_orders(day, orders)
    buys = [fill for (side, _), (fill, _) in indexed.items() if side == 'buy']
    budget = min(Decimal(5000), Decimal('.05') * prior_nav)
    if len(buys) > candidate_count or any(-Decimal(fill['cash_delta']) > budget for fill in buys):
        raise ValueError(f'{day}: original purchases exceed the candidate count or funding bound')
    baseline_cash, baseline_positions = balance(day, prior_cash, positions, [row[0] for row in orders])
    gate = exit_batch_gate(indexed, positions, prior_cash, budget, candidate_count)
    replacement = {}
    matches = []
    if gate == 'shared_exit_batch':
        for symbol in sorted(positions):
            if ('buy', symbol) not in indexed:
                continue
            sell, sell_book = indexed['sell', symbol]
            buy, buy_book = indexed['buy', symbol]
            if sell_book != buy_book:
                continue
            bid, ask = Decimal(sell_book['bid']), Decimal(sell_book['ask'])
            sell_price, buy_price = Decimal(sell['price']), Decimal(buy['price'])
            if not all(value.is_finite() for value in (bid, ask)) or not 0 < sell_price <= bid <= ask <= buy_price:
                continue
            matched = min(sell['units'], buy['units'])
            residual_fee = Decimal(0)
            for original in (sell, buy):
                units = original['units'] - matched
                fill = reprice(day, original, units) if units else None
                replacement[original['side'], symbol] = fill
                if fill:
                    residual_fee = Decimal(fill['fees'])
            spread = (ask - bid) * matched
            execution = (buy_price - sell_price) * matched
            fees = Decimal(sell['fees']) + Decimal(buy['fees']) - residual_fee
            matches.append({
                'symbol': symbol, 'sell_units': sell['units'], 'buy_units': buy['units'],
                'netted_units': matched, 'residual_units': abs(buy['units'] - sell['units']),
                'residual_side': ('buy' if buy['units'] > sell['units'] else 'sell')
                if buy['units'] != sell['units'] else None,
                'residual_fee': str(residual_fee), 'quote_received_at': sell_book['received_at'],
                'quote_raw_sha256': sell_book['raw_sha256'],
                'spread': str(spread), 'slippage': str(execution - spread),
                'execution': str(execution), 'fees': str(fees), 'total': str(execution + fees),
            })
    fills = []
    for fill, _, _ in orders:
        revised = replacement.get((fill['side'], fill['symbol']), fill)
        if revised is not None:
            fills.append(revised)
    cash, closing_positions = balance(day, opening_shadow_cash, positions, fills)
    savings = {key: sum((Decimal(pair[key]) for pair in matches), Decimal(0)) for key in SAVINGS_FIELDS}
    if closing_positions != baseline_positions or cash - opening_shadow_cash != baseline_cash - prior_cash + savings['total']:
        raise ValueError(f'{day}: exposure or cash attribution fails to reconcile')
    return {
        'cash': str(cash), 'positions': closing_positions, 'fills': fills, 'matches': matches, 'gate': gate,
        'savings': {key: str(value) for key, value in savings.items()},
    }
