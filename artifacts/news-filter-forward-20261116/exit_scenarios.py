"""Replay a synthetic Monday entry through the frozen daily exit and weekly report."""

import argparse
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_UP

from close_monday import close_report
from continue_open import run_day
from monday_open import INITIAL, run, tick_price
from monday_scenario_matrix import frozen_inputs, synthetic_books
from weekly_report import DATES, GROUPS, weekly_report


BLOCKED = '603135.SH'
COMMON = '601988.SH'
CENT = Decimal('.01')
TUESDAY = DATES[1]
FRIDAY = DATES[-1]
REASONS = {
    'insufficient_bid_units': '买一挂量不足',
    'quote_unavailable': '盘口缺失',
    'sell_outside_price_limits': '扣滑点后低于模拟跌停价',
    'filled': '成交',
    'fewer_than_five_distinct_verified_filtered_symbols': '有效过滤标的不足 5 只',
    'incomplete_execution_evidence': '部分执行证据缺失',
    'missing_close_valuation': '收盘估值缺失',
}


@dataclass(frozen=True)
class Scenario:
    label: str
    exit_modes: tuple[str, ...]
    sale_price: str
    friday_mark: str | None = '9.40'
    recover_monday_mark: bool = False


SCENARIOS = (
    Scenario('周二上涨卖出', ('sell',), '9.45'),
    Scenario('周二下跌卖出', ('sell',), '8.90'),
    Scenario('连续拒卖后周五卖出', ('shallow', 'missing', 'at_lower_limit', 'sell'), '9.45'),
    Scenario('周一缺估值，周二凭独立昨收卖出', ('sell',), '9.45', recover_monday_mark=True),
    Scenario('周五未卖出，收盘价可核', ('shallow',) * 4, '9.45'),
    Scenario('周五未卖出，收盘价缺失', ('shallow',) * 4, '9.45', None),
)
SELL_REASON = {
    'sell': 'filled', 'shallow': 'insufficient_bid_units',
    'missing': 'quote_unavailable', 'at_lower_limit': 'sell_outside_price_limits',
}


def synthetic_close(opening, prices, records):
    day = opening['date']
    held = {symbol for group in GROUPS for symbol in opening[group]['positions']}
    marks = {symbol: {
        'date': day, 'close': prices[symbol], 'price_type': 'unadjusted_close',
        'source_url': 'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get',
    } for symbol in held if prices.get(symbol) is not None}
    record, _ = close_report(opening, {'date': day, 'marks': marks}, day=day, prior=records)
    return record


def exit_book(day, symbol, previous, bid, units, mode):
    last_close = Decimal(previous)
    lower = tick_price(last_close * Decimal('.90'), CENT, ROUND_HALF_UP)
    upper = tick_price(last_close * Decimal('1.10'), CENT, ROUND_HALF_UP)
    shown_bid = lower if mode == 'at_lower_limit' else Decimal(bid)
    return {
        'at': day + 'T09:45:20+08:00', 'received_at': day + 'T09:45:21+08:00',
        'source_url': f'https://qt.gtimg.cn/q=sh{symbol[:6]}',
        'previous': str(last_close), 'bid': str(shown_bid),
        'lower_limit': str(lower), 'upper_limit': str(upper),
        'bid_units': units - 1 if mode == 'shallow' else units, 'traded_volume': 100_000,
    }


def run_scenario(prepared, news, scenario):
    rows = [row for row in prepared['candidates'] if row['symbol'] in (BLOCKED, COMMON)]
    if {row['symbol'] for row in rows} != {BLOCKED, COMMON}:
        raise ValueError('selected stress-test symbols missing from the frozen shortlist')
    prices = {row['symbol']: row['previous_close'] for row in rows}
    opening = run({**prepared, 'candidates': rows}, synthetic_books(rows, 'eligible'), news)
    opening = {'mode': 'forward', **opening}
    actual = {group: set(opening[group]['positions']) for group in GROUPS}
    if actual != {'baseline': {BLOCKED, COMMON}, 'filtered': {COMMON}}:
        raise ValueError('Monday fixture must yield one common and one news-filtered entry')
    monday_marks = {**prices, BLOCKED: None} if scenario.recover_monday_mark else prices
    records = [synthetic_close(opening, monday_marks, ())]
    snapshot = {'batches': news, 'errors': [], 'sources': []}
    modes = dict(zip(DATES[1:], scenario.exit_modes))
    for day in DATES[1:]:
        mode = modes.get(day)
        previous = records[-1]
        selected, unavailable = {}, {}
        if mode == 'missing':
            unavailable[BLOCKED] = 'quote_unavailable'
        elif mode is not None:
            selected[BLOCKED] = exit_book(day, BLOCKED, prices[BLOCKED], scenario.sale_price,
                                         previous['baseline']['positions'][BLOCKED], mode)
        if day == TUESDAY:
            selected[COMMON] = exit_book(day, COMMON, prices[COMMON], prices[COMMON],
                                         previous['baseline']['positions'][COMMON], 'sell')
        later = {'coverage': 'synthetic', 'candidates': []}
        if day == TUESDAY and scenario.recover_monday_mark:
            later['reviewed'] = [{'symbol': BLOCKED, 'status': 'signal_not_triggered',
                                  'previous_close': prices[BLOCKED],
                                  'sources': ['synthetic-independent-close']}]
        current = run_day(date.fromisoformat(day), previous, later,
                          {'selected': selected, 'unavailable': unavailable}, snapshot)
        marks = {**prices, BLOCKED: scenario.friday_mark} if day == FRIDAY else prices
        records.append(synthetic_close(current, marks, records))
    return records, weekly_report(records)


def verify_accounting(records, report, scenario):
    expected = [SELL_REASON[mode] for mode in scenario.exit_modes]
    attempts = []
    leg_pnl = {}
    for group in GROUPS:
        cash = INITIAL
        total_fees = Decimal(0)
        peak = INITIAL
        max_drawdown = Decimal(0)
        valuation_complete = True
        pnl = dict.fromkeys((BLOCKED, COMMON), Decimal(0))
        for record in records:
            account = record[group]
            for decision in account['decisions']:
                if group == 'baseline' and decision['symbol'] == BLOCKED and decision['side'] == 'sell':
                    attempts.append((record['date'], decision['reason']))
            for fill in account['fills']:
                units, price, side = fill['units'], Decimal(fill['price']), fill['side']
                gross = price * units
                commission = max(Decimal(5), gross * Decimal('.0003')).quantize(CENT, ROUND_HALF_UP)
                stamp = (gross * Decimal('.0005')).quantize(CENT, ROUND_HALF_UP) if side == 'sell' else Decimal(0)
                transfer = (gross * Decimal('.00001')).quantize(CENT, ROUND_HALF_UP)
                fees = commission + stamp + transfer
                delta = (gross if side == 'sell' else -gross) - fees
                components = tuple(Decimal(fill[key]) for key in ('commission', 'stamp_tax', 'transfer_fee'))
                if (components != (commission, stamp, transfer) or fees != Decimal(fill['fees'])
                        or delta != Decimal(fill['cash_delta'])):
                    raise ValueError(f'{group}: fill charges or cash movement did not reconcile')
                if side == 'sell':
                    bid = (Decimal(scenario.sale_price) if fill['symbol'] == BLOCKED
                           else Decimal(records[0][group]['close_prices'][COMMON]))
                    expected_price = (bid * Decimal('.9995')).quantize(CENT, ROUND_FLOOR)
                    if price != expected_price:
                        raise ValueError(f'{group}: sell price did not reflect 5 bp slippage')
                cash += delta
                total_fees += fees
                pnl[fill['symbol']] += delta
            marks = account['close_prices']
            valuation = {symbol: units * Decimal(marks[symbol]) for symbol, units in account['positions'].items()
                         if symbol in marks}
            complete = len(valuation) == len(account['positions'])
            nav = cash + sum(valuation.values()) if complete else None
            valuation_complete &= complete
            if Decimal(account['cash']) != cash or account['nav'] != (str(nav) if nav is not None else None):
                raise ValueError(f'{group}: daily cash or NAV did not reconcile')
            if nav is not None:
                peak = max(peak, nav)
                max_drawdown = max(max_drawdown, (peak - nav) / peak)
            if record['date'] == FRIDAY:
                for symbol in account['positions']:
                    pnl[symbol] = pnl[symbol] + valuation[symbol] if symbol in valuation else None
        group_report = report['groups'][group]
        if Decimal(group_report['fees']) != total_fees:
            raise ValueError(f'{group}: cumulative fees do not reconcile')
        if nav is not None and Decimal(group_report['ending_nav']) != nav:
            raise ValueError(f'{group}: weekly NAV did not reconcile')
        expected_drawdown = Decimal(group_report['max_drawdown']) if group_report['max_drawdown'] is not None else None
        if expected_drawdown != (max_drawdown if valuation_complete else None):
            raise ValueError(f'{group}: weekly drawdown did not reconcile')
        if nav is None and any(group_report[key] is not None for key in ('ending_nav', 'net_return', 'max_drawdown')):
            raise ValueError(f'{group}: unpriced inventory produced an unsupported weekly metric')
        leg_pnl[group] = pnl

    if attempts != list(zip(DATES[1:], expected)):
        raise ValueError('exit attempts do not match the scenario')
    opportunity, = report['effective_opportunities']
    pnl = leg_pnl['baseline'][BLOCKED]
    expected_status = 'sold' if 'sell' in scenario.exit_modes else 'marked_open' if pnl is not None else 'open'
    attributed = None if opportunity['net_pnl'] is None else Decimal(opportunity['net_pnl'])
    if (opportunity['symbol'] != BLOCKED or opportunity['exit_status'] != expected_status
            or attributed != pnl or leg_pnl['baseline'][COMMON] != leg_pnl['filtered'][COMMON]):
        raise ValueError('weekly attribution does not reconcile with the two simulated lots')
    if pnl is not None:
        group_navs = [Decimal(report['groups'][group]['ending_nav']) for group in GROUPS]
        if group_navs[1] - group_navs[0] != -pnl:
            raise ValueError('paired NAV difference does not reconcile with the news-filtered lot')
    if report['verdict'] != 'insufficient_evidence':
        raise ValueError('the one-symbol evidence gate should prevent a weekly effectiveness verdict')
    if scenario.recover_monday_mark:
        if (records[0]['baseline']['nav'] is not None or report['comparison'] is not None
                or report['groups']['baseline']['max_drawdown'] is not None):
            raise ValueError('a later exit must not fill a historical NAV gap or create a full weekly comparison')
    return attempts, opportunity, leg_pnl['baseline'][COMMON]


def money(value):
    return '不可核' if value is None else f'{Decimal(value):,.2f}'


def metric_pair(report, key, formatter=money):
    return ' / '.join(formatter(report['groups'][group][key]) for group in GROUPS)


def render():
    prepared, news = frozen_inputs()
    lines = [
        '# 周一买入后卖出与周五对账压力测试', '',
        '用冻结名单与已归档消息构造周一双组开盘。两组都买入 601988，603135 只在原规则组成交。'
        '周二起仅尝试退出已有持仓。盘口、收盘价和用来检验卖出边界的价格限制均为离线假设。', '',
        '| 情景 | 603135 卖出尝试 | 603135 原组扣费盈亏（元） | 消息组净值差（元） |',
        '|---|---|---:|---:|',
    ]
    weekly_rows = []
    common_pnls = set()
    for scenario in SCENARIOS:
        records, report = run_scenario(prepared, news, scenario)
        attempts, opportunity, common_pnl = verify_accounting(records, report, scenario)
        common_pnls.add(common_pnl)
        trace = '；'.join(f'{day[-2:]} 日{REASONS[reason]}' for day, reason in attempts)
        status = {'sold': '已卖出', 'marked_open': '仍持有，按周五收盘估值', 'open': '仍持有，估值缺失'}
        lines.append(f"| {scenario.label} | {trace} | {money(opportunity['net_pnl'])}"
                     f"（{status[opportunity['exit_status']]}） | {money(opportunity.get('filtered_minus_baseline'))} |")
        reasons = '、'.join(REASONS.get(reason, reason) for reason in report['insufficient_evidence_reasons'])
        orders = ' / '.join(f"{report['groups'][g]['buy_count']} 买 {report['groups'][g]['sell_count']} 卖"
                            for g in GROUPS)
        drawdowns = metric_pair(report, 'max_drawdown', lambda value: '不可核' if value is None
                                else f'{Decimal(value):.4%}')
        weekly_rows.append(f"| {scenario.label} | {orders} | {metric_pair(report, 'fees')} | "
                           f"{metric_pair(report, 'ending_nav')} | {drawdowns} | 证据不足，{reasons} |")
    common_pnl, = common_pnls
    lines.extend([
        '', f'601988 在两组的已实现盈亏均为 {money(common_pnl)} 元，计入各组净值。'
        '净值差可归因于双方共同买入机会中只被消息组拦下的 603135。'
        '周五仍持仓时只扣已付的买入费用，缺收盘价时相应净值和最大回撤不可核。', '',
        '## 周报对账', '',
        '每格按原规则组 / 消息过滤组排列。净值已扣发生的全部费用。', '',
        '| 情景 | 成交笔数 | 手续费（元） | 周五收盘净值（元） | 最大回撤 | 周报判定 |',
        '|---|---|---:|---:|---:|---|', *weekly_rows, '',
    ])
    return '\n'.join(lines)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(render(), end='')


if __name__ == '__main__':
    main()
