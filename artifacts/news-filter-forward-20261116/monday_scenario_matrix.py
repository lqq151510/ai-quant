"""Print a reproducible hypothetical Monday scenario matrix as Markdown.

From the project root:
    PYTHONPATH=research_v1 research_v1/.venv/bin/python \
      artifacts/news-filter-forward-20261116/monday_scenario_matrix.py

Uses Friday shortlist and archived news. All Monday books and closes are
synthetic. Makes no network requests and writes no formal research files.
"""

import hashlib
import json
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal
from itertools import product
from pathlib import Path

from candidate_review import SHORTLIST
from collect_books import IOPV_SYMBOL
from monday_open import DAY, PROTOCOL, run, tick_price
from reconcile import reconcile
from quant_research.ledger import Costs

BASE = Path(__file__).resolve().parent
GROUPS = ('baseline', 'filtered')
QUOTE_AT = f'{DAY}T09:45:20+08:00'
RECEIVED_AT = f'{DAY}T09:45:21+08:00'
PRIOR_AT = f'{DAY}T09:45:10+08:00'


@dataclass(frozen=True)
class Scenario:
    name: str
    quote_mode: str = 'eligible'
    close_mode: str = 'flat'
    move: Decimal = Decimal(0)


SCENARIOS = (
    Scenario('所有标的盘口不可用', quote_mode='absent'),
    Scenario('所有盘口超过 30 秒', quote_mode='stale'),
    Scenario('所有卖一挂量为零', quote_mode='empty'),
    Scenario('所有模拟买价高于限价', quote_mode='above_limit'),
    Scenario('513100 盘口缺失', quote_mode='etf_absent'),
    Scenario('513100 参考净值字段缺失', quote_mode='iopv_missing'),
    Scenario('513100 盘中参考净值更新未核实', quote_mode='iopv_unverified'),
    Scenario('513100 含滑点溢价 1.5%', quote_mode='etf_premium_high'),
    Scenario('513100 含滑点溢价 0.5%，全体按买入价估值'),
    Scenario('满足买入条件的标的统一上涨 3%', close_mode='market', move=Decimal('.03')),
    Scenario('满足买入条件的标的统一下跌 3%', close_mode='market', move=Decimal('-.03')),
    Scenario('被新闻过滤的标的上涨 5%，其余持平', close_mode='blocked', move=Decimal('.05')),
    Scenario('被新闻过滤的标的下跌 5%，其余持平', close_mode='blocked', move=Decimal('-.05')),
    Scenario('被新闻过滤的标的缺收盘价', close_mode='blocked_missing'),
    Scenario('新闻文件无法读取', quote_mode='news_unavailable'),
)


def frozen_inputs():
    if hashlib.sha256((BASE / 'protocol.md').read_bytes()).hexdigest() != PROTOCOL:
        raise ValueError('frozen protocol hash mismatch')
    shortlist = (BASE / 'monday-preparation/candidates.json').read_bytes()
    if hashlib.sha256(shortlist).hexdigest() != SHORTLIST:
        raise ValueError('frozen shortlist hash mismatch')
    prepared = json.loads(shortlist)
    if prepared['date'] != str(DAY):
        raise ValueError('shortlist date does not match the frozen Monday session')
    friday = BASE.parent / 'news-forward/2026-11-13.json'
    return prepared, [json.loads(friday.read_text())]


def synthetic_books(candidates, mode):
    selected, unavailable = {}, {}
    for row in candidates:
        symbol = row['symbol']
        is_etf = symbol == IOPV_SYMBOL
        if mode == 'absent' or (is_etf and mode == 'etf_absent'):
            unavailable[symbol] = 'quote_unavailable'
            continue
        previous, tick = map(Decimal, (row['previous_close'], row['tick']))
        ask = previous
        if mode == 'above_limit':
            ask = tick_price(previous * Decimal('1.02'), tick, ROUND_CEILING)
        source = 'https://qt.gtimg.cn/q=sh513100' if is_etf else 'https://example.invalid/synthetic-book'
        book = {
            'at': f'{DAY}T09:45:00+08:00' if mode == 'stale' else QUOTE_AT,
            'received_at': f'{DAY}T09:45:31+08:00' if mode == 'stale' else RECEIVED_AT,
            'source_url': source,
            'raw_file': 'synthetic-current.txt',
            'previous': str(previous),
            'ask': str(ask) if mode != 'empty' else '0',
            'ask_units': 100_000 if mode != 'empty' else 0,
        }
        if is_etf:
            execution_price = tick_price(ask * Decimal('1.0005'), tick, ROUND_CEILING)
            premium = Decimal('.015') if mode == 'etf_premium_high' else Decimal('.005')
            iopv = execution_price / (1 + premium)
            book['iopv'] = {
                'status': 'verified_snapshot',
                'nav': str(iopv),
                'vendor_premium_pct': str((ask / iopv - 1) * 100),
            }
            book['iopv_update'] = {
                'previous_at': PRIOR_AT,
                'previous_received_at': f'{DAY}T09:45:11+08:00',
                'previous_nav': str(iopv - Decimal('.0001')),
                'previous_raw_file': 'synthetic-prior.txt',
            }
            if mode == 'iopv_missing':
                book['iopv'] = {'status': 'iopv_missing'}
            if mode == 'iopv_unverified':
                del book['iopv_update']
        selected[symbol] = book
    return {'selected': selected, 'unavailable': unavailable}


def simulate_scenario(prepared, news, scenario):
    books = synthetic_books(prepared['candidates'], scenario.quote_mode)
    errors = ('synthetic news read error',) if scenario.quote_mode == 'news_unavailable' else ()
    opening = run(prepared, books, news, news_errors=errors)
    blocked = {item['symbol'] for item in opening['filtered']['decisions']
               if item['reason'] == 'news_abnormal_volatility'}
    ticks = {row['symbol']: Decimal(row['tick']) for row in prepared['candidates']}
    daily = {'date': opening['date']}
    for group in GROUPS:
        account = opening[group]
        closes = {}
        for fill in account['fills']:
            symbol = fill['symbol']
            if scenario.close_mode == 'blocked_missing' and symbol in blocked:
                continue
            move = scenario.move if scenario.close_mode == 'market' or symbol in blocked else Decimal(0)
            marked = tick_price(Decimal(fill['price']) * (1 + move), ticks[symbol], ROUND_HALF_UP)
            closes[symbol] = str(marked)
        nav = None
        if closes.keys() >= account['positions'].keys():
            nav = Decimal(account['cash']) + sum(
                Decimal(closes[symbol]) * units for symbol, units in account['positions'].items()
            )
        daily[group] = {**account, 'close_prices': closes, 'nav': None if nav is None else str(nav)}
    return opening, reconcile([daily])['groups'], blocked


def summarize_etf(opening):
    item = next(row for row in opening['baseline']['decisions'] if row['symbol'] == IOPV_SYMBOL)
    ratio = item['premium']
    if ratio is None:
        return item['reason']
    return f"{item['reason']} / {Decimal(ratio):.2%}"


def summarize_group(opening, metrics, group):
    fills = opening[group]['fills']
    buys = sum(fill['side'] == 'buy' for fill in fills)
    sells = sum(fill['side'] == 'sell' for fill in fills)
    nav = metrics[group]['ending_nav']
    net_return = metrics[group]['net_return']
    drawdown = metrics[group]['max_drawdown']
    valuation = '不可核' if nav is None else f'{Decimal(nav):,.2f}'
    performance = (
        '不可核 / 不可核' if net_return is None or drawdown is None
        else f'{Decimal(net_return):+.4%} / {Decimal(drawdown):.4%}'
    )
    return f"{buys} / {sells} / {Decimal(metrics[group]['fees']):.2f} / {valuation} / {performance}"


def enumerate_availability(prepared, news):
    """Try every availability combination with a qualifying quote when present."""
    candidates = prepared['candidates']
    normal = synthetic_books(candidates, 'eligible')['selected']
    expected = run(prepared, {'selected': normal, 'unavailable': {}}, news)
    blocked = {d['symbol'] for d in expected['filtered']['decisions'] if d['reason'] == 'news_abnormal_volatility'}
    symbols = [row['symbol'] for row in candidates]
    totals = {'combinations': 0, 'same_buy_count': 0, 'news_changed_buy_count': 0}
    for mask in product((False, True), repeat=len(symbols)):
        available = {symbol for symbol, present in zip(symbols, mask) if present}
        result = run(prepared, {'selected': {s: normal[s] for s in available}, 'unavailable': {}}, news)
        for group, expected_symbols in (('baseline', available), ('filtered', available - blocked)):
            account = result[group]
            actual = {fill['symbol'] for fill in account['fills']}
            if actual != expected_symbols or set(account['positions']) != actual:
                raise ValueError(f'{group}: wrong fill set with available quotes {sorted(available)}')
            if Decimal(account['cash']) != Decimal(100000) + sum(Decimal(f['cash_delta']) for f in account['fills']):
                raise ValueError(f'{group}: cash mismatch with available quotes {sorted(available)}')
        totals['combinations'] += 1
        totals['news_changed_buy_count' if available & blocked else 'same_buy_count'] += 1
    return totals


def entry_boundaries(prepared):
    for row in prepared['candidates']:
        close, tick = map(Decimal, (row['previous_close'], row['tick']))
        limit = tick_price(close * Decimal('1.01'), tick, ROUND_FLOOR)
        ask_ceiling = tick_price(limit / Decimal('1.0005'), tick, ROUND_FLOOR)
        assert tick_price(ask_ceiling * Decimal('1.0005'), tick, ROUND_CEILING) <= limit
        assert tick_price((ask_ceiling + tick) * Decimal('1.0005'), tick, ROUND_CEILING) > limit
        units = int(Decimal(5000) // (limit * 100)) * 100
        while units:
            gross = limit * units
            fees = sum(Decimal(str(v)) for v in Costs().charges(DAY, row['kind'], 'buy', float(gross)))
            if gross + fees <= 5000:
                break
            units -= 100
        yield row['symbol'], close, limit, ask_ceiling, units


def main():
    prepared, news = frozen_inputs()
    print('# 周一模拟情景演算\n')
    print(f"使用冻结名单的 {len(prepared['candidates'])} 只候选与 11 月 13 日存档的新闻。"
          '盘口、挂量、盘中参考净值和收盘价均为假设。各场景独立从 10 万元空仓开始，模拟首日无卖出。')
    print('正常盘口以昨收作为卖一，假设挂量 100,000 股或份。默认收盘价等于模拟买价，'
          '上涨和下跌情景在模拟买价基础上按标注比例变动，再按报价单位取整。\n')
    print('每组依次为买入笔数 / 卖出笔数 / 手续费与税费合计（元） / 收盘净值（元） / 扣费收益率 / 当日最大回撤。\n')
    print('| 情景 | 513100 原规则判断 / 溢价 | 新闻额外拦截 | 原规则组 | 新闻过滤组 |')
    print('|---|---|---|---|---|')
    for scenario in SCENARIOS:
        opening, metrics, blocked = simulate_scenario(prepared, news, scenario)
        blocks = '、'.join(sorted(blocked)) or '—'
        print(f'| {scenario.name} | {summarize_etf(opening)} | {blocks} | '
              f'{summarize_group(opening, metrics, "baseline")} | '
              f'{summarize_group(opening, metrics, "filtered")} |')
    print('\n新闻文件无法读取的情景中，新闻组的账面现金和净值只用于核对失败处理。'
          '缺少新闻数据的差异不可归因于新闻过滤。所有场景仅用于验证规则与账本，实际结果需以正式开盘和收盘记录为准。')
    totals = enumerate_availability(prepared, news)
    print(f"\n另枚举 {totals['combinations']} 种盘口可用组合。在有效盘口都满足买入条件的假设下，"
          f"两组买入笔数相同 {totals['same_buy_count']} 种，新闻过滤改变笔数 {totals['news_changed_buy_count']} 种。"
          "每种组合均核对了成交标的、持仓和现金。组合数量不代表发生概率。")
    print('\n## 冻结名单的入场价格边界\n')
    print('可接受的卖一上限已预留滑点及报价取整。最后一列按模拟买价恰好等于限价时计算，'
          '挂量以股或份计。513100 还须通过独立的参考净值核验和 1% 溢价检查。\n')
    print('| 代码 | 昨收 | 模拟买入限价 | 可接受的卖一上限 | 限价时所需挂量 |')
    print('|---|---:|---:|---:|---:|')
    for symbol, close, limit, ask, units in entry_boundaries(prepared):
        print(f'| {symbol} | {close} | {limit} | {ask} | {units:,} |')


if __name__ == '__main__':
    main()
