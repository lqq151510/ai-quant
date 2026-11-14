"""Count a finite Monday state grid with shared execution and market outcomes."""
import argparse
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from itertools import product

from monday_open import INITIAL, run, tick_price
from monday_scenario_matrix import frozen_inputs, synthetic_books
from shadow_factors import ARMS, compare, load_features


def cents(value):
    scaled = value * 100
    if scaled != scaled.to_integral_value():
        raise ValueError('non-cent P&L')
    return int(scaled)


def distribution(states):
    """Convolve integer outcomes. Repeated outcomes retain their multiplicity."""
    outcomes = Counter({0: 1})
    for states_for_symbol in states:
        frequencies = Counter(states_for_symbol)
        updated = Counter()
        for pnl, count in outcomes.items():
            for value, frequency in frequencies.items():
                updated[pnl + value] += count * frequency
        outcomes = updated
    return outcomes


def calculate():
    prepared, news = frozen_inputs()
    features = load_features(prepared)
    candidates = prepared['candidates']
    ticks = {row['symbol']: Decimal(row['tick']) for row in candidates}
    opening = run(prepared, synthetic_books(candidates, 'eligible'), news)
    fills = {f['symbol']: f for f in opening['baseline']['fills']}
    if set(fills) != ticks.keys():
        raise ValueError('eligible fixture did not execute all frozen candidates')
    outcomes = {}
    for symbol, fill in fills.items():
        price = Decimal(fill['price'])
        outcomes[symbol] = (0, *(
            cents(tick_price(price * (1 + move), ticks[symbol], ROUND_HALF_UP) * fill['units']
                  + Decimal(fill['cash_delta']))
            for move in map(Decimal, ('-.05', '0', '.05'))
        ))
    reference = compare(opening, {s: f['price'] for s, f in fills.items()}, features)
    reports = {}
    for arm in ARMS:
        if reference[arm]['unknown']:
            raise ValueError(f'{arm}: unresolved features prevent a complete state grid')
        removed = set(reference[arm]['blocked'])
        by_delta = distribution([tuple(-v if s in removed else 0 for v in row) for s, row in outcomes.items()])
        by_pnl = distribution([tuple(0 if s in removed else v for v in row) for s, row in outcomes.items()])
        if sum(by_delta.values()) != 4 ** len(fills) or sum(by_pnl.values()) != 4 ** len(fills):
            raise ValueError('incomplete combination count')
        # For the blocked symbols, directly enumerate every assignment as an independent check.
        direct = Counter(-sum(combo) for combo in product(*(outcomes[s] for s in sorted(removed))))
        common_multiplicity = 4 ** (len(fills) - len(removed))
        if by_delta != {v: count * common_multiplicity for v, count in direct.items()}:
            raise ValueError('paired P&L distribution disagrees with direct enumeration')
        reports[arm] = (sorted(removed), by_delta, by_pnl)
    return len(fills), reports


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    count, reports = calculate()
    print('# 周一盘口与收盘走势组合\n')
    print(f'冻结名单共 {count} 只。每只标的设四种状态：盘口不可用，或盘口合格且收盘相对模拟买价跌 5%、持平、涨 5%。'
          f'一共计入 **{4**count:,} 种**联合状态。合格盘口用昨收作卖一，513100 按参考净值核实通过设定。'
          '所有规则共用这些状态，成交股数沿用同一买单，'
          '扣除逐笔买入费用，价格按报价单位取整。当天从空仓开始，只有买入和收盘估值。\n')
    print('| 规则 | 净值高于原组 | 等于原组 | 低于原组 | 净值差范围（元） | 最差到最好净值（元） |')
    print('|---|---:|---:|---:|---:|---:|')
    for arm, (_, delta, pnl) in reports.items():
        higher = sum(n for v, n in delta.items() if v > 0)
        lower = sum(n for v, n in delta.items() if v < 0)
        low, high = (INITIAL + Decimal(x) / 100 for x in (min(pnl), max(pnl)))
        print(f'| {arm} | {higher:,} | {delta[0]:,} | {lower:,} | '
              f'{Decimal(min(delta))/100:,.2f} 至 {Decimal(max(delta))/100:,.2f} | {low:,.2f} 至 {high:,.2f} |')
    print('\n计数只描述这套假设。持平时买入仍会付手续费，过滤组在这些状态中会保留这笔现金。'
          '各市场状态的概率未知，实际盘口可能超限、深度不足或缺少参考净值，'
          '这些执行条件在另外的盘口场景和测试中检验。周一收盘净值也不包含后续退出的价格和费用。')


if __name__ == '__main__':
    main()
