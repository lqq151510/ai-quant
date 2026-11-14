"""Bound first-day paper outcomes across tick-valid quotes in a price stress window."""

import argparse
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR

from monday_open import INITIAL, decide, run, tick_price
from monday_scenario_matrix import frozen_inputs, synthetic_books
from shadow_factors import ARMS, compare, load_features


ZERO = Decimal(0)
STRESS_MOVE = Decimal('0.10')
PREMIUM = Decimal('0.005')
SLIPPAGE_FACTOR = Decimal('1.0005')
BUY_LIMIT_FACTOR = Decimal('1.01')
LABELS = {'baseline': '原规则', 'news': '消息过滤', 'volume': '量价过滤', 'ma': '均线过滤',
          'volume_ma': '量价与均线', 'all': '三项叠加'}


@dataclass(frozen=True)
class QuoteOutcome:
    ask: Decimal
    price: Decimal
    units: int
    fees: Decimal
    pnl_low: Decimal
    pnl_high: Decimal


def at_ask(row, fixture, ask, tick, *, premium=PREMIUM):
    book = {**fixture, 'ask': str(ask)}
    if row['cross_border']:
        price = tick_price(ask * SLIPPAGE_FACTOR, tick, ROUND_CEILING)
        nav = price / (1 + premium)
        book['iopv'] = {**fixture['iopv'], 'nav': str(nav),
                        'vendor_premium_pct': str((ask / nav - 1) * 100)}
        book['iopv_update'] = {**fixture['iopv_update'], 'previous_nav': str(nav - Decimal('0.0001'))}
    return book


def quote_outcomes(row, fixture, news):
    previous, tick = map(Decimal, (row['previous_close'], row['tick']))
    close_low = tick_price(previous * (1 - STRESS_MOVE), tick, ROUND_CEILING)
    close_high = tick_price(previous * (1 + STRESS_MOVE), tick, ROUND_FLOOR)
    buy_limit = tick_price(previous * BUY_LIMIT_FACTOR, tick, ROUND_FLOOR)
    ask_high = tick_price(buy_limit / SLIPPAGE_FACTOR, tick, ROUND_FLOOR)
    if not close_low <= ask_high < close_high:
        raise ValueError(f"{row['symbol']}: stress band does not span the buy boundary")
    outcomes = []
    for i in range(int(close_low / tick), int(ask_high / tick) + 1):
        ask = i * tick
        book = at_ask(row, fixture, ask, tick)
        _, decision, fill, _ = decide(row, book, news)
        if not decision or not decision['baseline']['eligible']:
            raise ValueError(f"{row['symbol']}: an expected eligible quote was rejected at {ask}")
        if ask == ask_high:
            _, above, _, _ = decide(row, at_ask(row, fixture, ask + tick, tick), news)
            if not above or above['baseline']['reason'] != 'ask_above_limit':
                raise ValueError(f"{row['symbol']}: first ask above the derived limit was accepted")
        units = fill['units']
        for depth, reason in ((units, 'within_limit'), (units - 1, 'insufficient_ask_units')):
            _, at_depth, _, _ = decide(row, {**book, 'ask_units': depth}, news)
            if not at_depth or at_depth['baseline']['reason'] != reason:
                raise ValueError(f"{row['symbol']}: unexpected result for sell-one depth {depth}")
        cash_delta, price, fees = map(Decimal, (fill['cash_delta'], fill['price'], fill['fees']))
        if cash_delta != -price * units - fees:
            raise ValueError(f"{row['symbol']}: fill fees do not reconcile")
        outcomes.append(QuoteOutcome(ask, price, units, fees,
                                     close_low * units + cash_delta, close_high * units + cash_delta))
    if row['cross_border']:
        high_premium = at_ask(row, fixture, close_low, tick, premium=Decimal('0.015'))
        reason, _, _, _ = decide(row, high_premium, news)
        if reason != 'premium_above_limit':
            raise ValueError(f"{row['symbol']}: a premium above 1% should cause a skip")
    return (close_low, close_high), outcomes


def calculate():
    prepared, news = frozen_inputs()
    books = synthetic_books(prepared['candidates'], 'eligible')
    opening = run(prepared, books, news)
    fills = opening['baseline']['fills']
    reference = compare(opening, {f['symbol']: f['price'] for f in fills}, load_features(prepared))
    if set(books['selected']) != {f['symbol'] for f in fills} or any(
        result['unknown'] for result in reference.values()
    ):
        raise ValueError('the reference scenario does not resolve all executable names and filter signals')
    quotes = {row['symbol']: quote_outcomes(row, books['selected'][row['symbol']], news)
              for row in prepared['candidates']}
    if Decimal(len(quotes)) * 5000 > INITIAL:
        raise ValueError('cash competition invalidates independent per-symbol bounds')
    pnl_ranges = {symbol: (min(ZERO, *(q.pnl_low for q in outcomes)),
                           max(ZERO, *(q.pnl_high for q in outcomes)))
                  for symbol, (_, outcomes) in quotes.items()}
    return quotes, pnl_ranges, {arm: set(result['blocked']) for arm, result in reference.items()}


def money_span(low, high):
    return f'{low:,.2f} 至 {high:,.2f}'


def render():
    quotes, pnl_ranges, blocked_by_arm = calculate()
    count = sum(len(outcomes) for _, outcomes in quotes.values())
    lines = [
        '# 周一卖一价、挂量和收盘估值压力测试', '',
        f'以冻结名单的周五收盘为基准，卖一价和收盘价各自限定在昨收上下 10% 内。'
        f'区间端点向内取整到报价单位。这是压力测试设定。逐档核对 **{count:,}** 个合格卖一价，'
        '每档用正式买入判断验证挂量恰好够时通过、少一股或一份时拒绝。'
        '不满足报价、挂量或溢价要求时，该标的在对应情景中跳过。', '',
        '513100 的可成交分支假设盘中参考净值已经核实，且模拟买价对应的溢价为 0.5%。'
        '同时核对了 1.5% 溢价会被跳过。实际盘中参考净值和溢价仍待采样。', '',
        '## 逐标的买入和估值', '',
        '| 标的 | 卖一价档数 | 合格卖一价 | 模拟买价 | 成交股数或份额 | 单笔费用（元） | 收盘估值价 | 单标的盈亏（元） |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for symbol, (marks, outcomes) in quotes.items():
        first, last = outcomes[0], outcomes[-1]
        units = [q.units for q in outcomes]
        fees = [q.fees for q in outcomes]
        lines.append(
            f'| {symbol} | {len(outcomes)} | {first.ask} 至 {last.ask} | {first.price} 至 {last.price} | '
            f'{min(units):,} 至 {max(units):,} | {money_span(min(fees), max(fees))} | {marks[0]} 至 {marks[1]} | '
            f'{money_span(*pnl_ranges[symbol])} |'
        )
    lines.extend([
        '', '成交数量与费用列只列可成交的情况。盈亏计入买入滑点、佣金和适用过户费，跳过时贡献为零。',
        '', '## 分组收盘净值范围', '',
        '| 规则 | 含费收盘净值（元） | 相对原组净值差（元） |',
        '|---|---:|---:|',
    ])
    for arm in ARMS:
        blocked = blocked_by_arm[arm]
        kept = [bounds for symbol, bounds in pnl_ranges.items() if symbol not in blocked]
        removed = [bounds for symbol, bounds in pnl_ranges.items() if symbol in blocked]
        nav_low = INITIAL + sum(low for low, _ in kept)
        nav_high = INITIAL + sum(high for _, high in kept)
        delta_low = -sum(high for _, high in removed)
        delta_high = -sum(low for low, _ in removed)
        lines.append(f'| {LABELS[arm]} | {money_span(nav_low, nav_high)} | {money_span(delta_low, delta_high)} |')
    lines.extend([
        '', '各组使用同一盘口和收盘假设。过滤信号根据周五行情和当时已归档的消息确定，拦下的订单留为现金且不补位。'
        '区间允许各标的独立到达最不利或最有利状态，'
        '也允许盘口全部不可用。范围对应周一收盘盯市，后续卖价和卖出费用需等退出时另计。', '',
    ])
    return '\n'.join(lines)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(render(), end='')


if __name__ == '__main__':
    main()
