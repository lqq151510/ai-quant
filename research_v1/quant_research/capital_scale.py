"""Replay the same pre-holdout ETF momentum baseline at six cash sizes."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
from math import floor
from pathlib import Path

import pandas as pd

from .baseline import build_targets, market_close_returns
from .ledger import LOT_SIZE, Costs, _fill_value, simulate
from .momentum_study import END, load_market
from .protocol import ETF_SYMBOLS, HOLDOUT_START, VALIDATION_START
from .risk import summarize
from .snapshot import file_sha256
from .timeline import sessions


CAPITALS = (1_000, 5_000, 10_000, 50_000, 100_000, 500_000)
HORIZON = 5
FRICTION_BUDGETS_BPS = (25, 50)
MAX_REFERENCE_NOTIONAL = Decimal(1_000_000)
MAX_TARGET_WEIGHT = Decimal('0.35')
SOURCE_FILES = (
    'capital_scale.py', 'baseline.py', 'ledger.py', 'momentum_study.py',
    'protocol.py', 'risk.py', 'snapshot.py', 'timeline.py',
)


def etf_flat_round_trip(day: date, reference_price: Decimal, units: int, cost: Costs) -> dict:
    """Cash loss of buying and selling the same whole lots with unchanged raw price."""
    slip = Decimal(str(cost.slippage_bps)) / 10_000
    buy, buy_fees, debit = _fill_value(day, 'etf', 'buy', units, float(reference_price * (1 + slip)), cost)
    sell, sell_fees, credit = _fill_value(day, 'etf', 'sell', units, float(reference_price * (1 - slip)), cost)
    notional = reference_price * units
    loss = -Decimal(str(debit)) - Decimal(str(credit))
    return {
        'units': units, 'reference_notional': float(notional), 'buy_cash_required': -debit,
        'commission': float(Decimal(str(buy_fees[0])) + Decimal(str(sell_fees[0]))),
        'slippage': float(Decimal(str(buy)) - Decimal(str(sell))),
        'all_in_cost': float(loss), 'all_in_cost_bps': float(loss / notional * 10_000),
    }


def whole_lot_cost_thresholds(day: date, reference_price: Decimal, cost: Costs) -> dict:
    """Scan whole lots in increasing order; commission rounding can create nonmonotone rates."""
    minimums = dict.fromkeys(FRICTION_BUDGETS_BPS)
    max_lots = floor(MAX_REFERENCE_NOTIONAL / (reference_price * LOT_SIZE))
    for lots in range(1, max_lots + 1):
        result = etf_flat_round_trip(day, reference_price, lots * LOT_SIZE, cost)
        loss = Decimal(str(result['all_in_cost']))
        notional = reference_price * result['units']
        for limit in FRICTION_BUDGETS_BPS:
            if minimums[limit] is None and loss * 10_000 <= limit * notional:
                account_floor = (notional / MAX_TARGET_WEIGHT).quantize(Decimal('0.01'), rounding=ROUND_CEILING)
                minimums[limit] = {**result, 'minimum_account_equity_at_weight_cap': float(account_floor)}
        if all(minimums.values()):
            break
    return {'one_lot': etf_flat_round_trip(day, reference_price, LOT_SIZE, cost),
            'minimum_by_cost_budget_bps': minimums}


def run_study(snapshot: Path, output: Path) -> dict:
    audit, frames = load_market(snapshot, end=END)
    calendar = sessions(frames['calendar']['date'])
    days = calendar[calendar >= pd.Timestamp(VALIDATION_START)]
    if days.empty or days[-1].date() != END:
        raise ValueError('the predeclared 2023–2024 window must be complete')
    bars, actions = frames['bars'], frames['actions']
    returns = market_close_returns(bars, actions, calendar)
    targets = build_targets(days, dict.fromkeys(days, ETF_SYMBOLS), returns, 'etf', HORIZON)['momentum']
    signal_records = [{
        'signal_date': days[days.get_loc(execute) - 1].date().isoformat(),
        'entry_date': execute.date().isoformat(), 'target_weights': weights,
    } for execute, weights in targets.items()]
    signal_bytes = (json.dumps(signal_records, sort_keys=True, allow_nan=False, indent=2) + '\n').encode()
    first_signal = next((record for record in signal_records if record['target_weights']), None)
    if first_signal is None:
        raise ValueError('the study window has no positive target')
    signal_sha256 = hashlib.sha256(signal_bytes).hexdigest()
    first_entry = pd.Timestamp(first_signal['entry_date'])
    opens = bars.loc[bars['date'].eq(first_entry)].set_index('symbol')['open']
    reference_closes = bars.loc[bars['date'].eq(pd.Timestamp(END))].set_index('symbol')['close']
    metrics, first_orders, fees_by_symbol = [], [], []
    equity, orders = [], []
    costs = Costs()
    for label, cost in (('normal', costs), ('double_friction', costs.stressed())):
        fees_by_symbol.append({'cost_case': label, 'commission_rate': cost.commission_rate,
                               'minimum_commission': cost.minimum_commission, 'slippage_bps': cost.slippage_bps,
                               'by_symbol': {symbol: whole_lot_cost_thresholds(END, Decimal(str(reference_closes[symbol])), cost)
                                             for symbol in ETF_SYMBOLS}})
        for capital in CAPITALS:
            result = simulate(bars, days, targets, actions, initial_cash=capital, cost=cost)
            nav, log = result.equity, result.orders
            discrepancy = (nav['nav'] - nav['cash'] - nav['market_value'] - nav['receivable']).abs().max()
            if (discrepancy > 1e-6 or nav['cash'].min() < 0
                    or nav['date'].tolist() != days.date.tolist()
                    or not set(log['date']) <= {day.date() for day in targets}):
                raise RuntimeError('ledger balances or execution timing failed reconciliation')
            filled = log.loc[log['filled_qty'].gt(0)]
            if not filled.loc[filled['side'].eq('buy'), 'filled_qty'].mod(LOT_SIZE).eq(0).all():
                raise RuntimeError('buy executed outside whole-lot sizing')
            summary = summarize(nav, log, capital)
            metrics.append({
                'initial_cash': capital, 'cost_case': label, 'signal_sha256': signal_sha256,
                **summary,
                'fees_over_initial_cash': summary['transaction_fees'] / capital,
                'buy_fills': int(filled['side'].eq('buy').sum()), 'sell_fills': int(filled['side'].eq('sell').sum()),
                'order_shortfalls': dict(Counter(log.loc[log['reason'].ne(''), 'reason'])),
                'mean_exposure': float((nav['market_value'] / nav['nav']).mean()),
                'ending_market_value': float(nav.iloc[-1]['market_value']),
                'ledger_max_discrepancy': float(discrepancy), 'min_cash': float(nav['cash'].min()),
            })
            first_log = log.loc[log['date'].eq(first_entry.date())].set_index('symbol')
            for symbol, weight in first_signal['target_weights'].items():
                price = Decimal(str(opens[symbol]))
                one_lot = price * LOT_SIZE
                order = first_log.loc[symbol]
                first_orders.append({'cost_case': label, 'initial_cash': capital, 'symbol': symbol,
                                     'target_weight': weight, 'one_lot_weight': float(one_lot / capital),
                                     'one_lot_reference_notional': float(one_lot),
                                     'one_lot_buy_cash_required': etf_flat_round_trip(first_entry.date(), price, LOT_SIZE, cost)['buy_cash_required'],
                                     'status': order['status'], 'reason': order['reason'],
                                     'filled_units': int(order['filled_qty'])})
            equity.append(nav.assign(initial_cash=capital, cost_case=label))
            orders.append(log.assign(initial_cash=capital, cost_case=label))
    source_sha256 = {name: file_sha256(Path(__file__).with_name(name)) for name in SOURCE_FILES}
    report = {
        'status': 'pre_holdout_historical_research', 'evidence_class': 'audited_retrospective_market',
        'snapshot_manifest_sha256': file_sha256(snapshot / 'manifest.json'),
        'source_sha256': source_sha256, 'audit': audit, 'symbols': list(ETF_SYMBOLS),
        'start': days[0].date().isoformat(), 'end': END.isoformat(),
        'sealed_holdout_start': HOLDOUT_START.isoformat(), 'signal_sha256': signal_sha256,
        'strategy': 'baseline.build_targets momentum / 5-session review / 20-day total return',
        'execution': 'signal after close, next session open with ledger slippage and exchange limit checks',
        'review_count': len(signal_records), 'positive_target_count': sum(len(r['target_weights']) for r in signal_records),
        'first_entry_date': first_entry.date().isoformat(), 'fee_reference_date': END.isoformat(),
        'first_order_details': first_orders, 'flat_round_trip': fees_by_symbol, 'metrics': metrics,
        'cost_budgets_bps': list(FRICTION_BUDGETS_BPS), 'max_target_weight': float(MAX_TARGET_WEIGHT),
        'max_threshold_search_notional': float(MAX_REFERENCE_NOTIONAL),
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / 'signals.json').write_bytes(signal_bytes)
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (output / 'report.md').write_text(render(report), encoding='utf-8')
    (output / 'capital-curve.html').write_text(render_curve(report['metrics']), encoding='utf-8')
    pd.concat(equity, ignore_index=True).to_csv(output / 'equity.csv', index=False)
    pd.concat(orders, ignore_index=True).to_csv(output / 'orders.csv', index=False)
    return report


def render(report: dict) -> str:
    lines = [
        '# ETF 动量基线的资金规模对照', '',
        f"历史区间 {report['start']} 至 {report['end']}，四只通过市场数据审计的境内 ETF。复用既有五交易日轮换基线，"
        '以过去二十个交易日复权收益为正的标的为候选，按逆波动分配，每只最多 35%。'
        '收盘确认信号，下一交易日开盘价加减滑点后模拟成交。期末持仓按收盘盯市。', '',
        f"各组共用 {report['review_count']} 次排定的调仓和 {report['positive_target_count']} 个正权重目标。"
        '正常假设为佣金万三、每边最低 5 元、每边滑点 5 基点。双倍摩擦同时翻倍三个参数。', '',
        '[资金与扣费收益曲线](capital-curve.html)', '',
        '| 虚拟本金（元） | 费用情景 | 扣费收益 | 最大回撤 | 买入 / 卖出 | 整手不足拒单 | 已付手续费（元） | 手续费 / 本金 | 换手 | 平均持仓比例 |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    labels = {'normal': '正常', 'double_friction': '双倍摩擦'}
    for row in report['metrics']:
        lines.append(f"| {row['initial_cash']:,} | {labels[row['cost_case']]} | {row['net_return']:+.2%} | "
                     f"{row['max_drawdown']:.2%} | {row['buy_fills']} / {row['sell_fills']} | "
                     f"{row['order_shortfalls'].get('board_lot', 0)} | {row['transaction_fees']:,.2f} | "
                     f"{row['fees_over_initial_cash']:.2%} | {row['traded_notional_over_average_nav']:.2f} 倍 | "
                     f"{row['mean_exposure']:.2%} |")
    lines += ['', '零成交时，账本的 0% 收益和回撤表示虚拟现金未动用。滑点已反映在成交价，手续费列只记账本实际扣费。'
              '换手为全期买卖成交金额合计除以平均净值。', '',
              '## 整手可执行性', '',
              f"首个模拟买入日为 {report['first_entry_date']}。初始为空仓，按当日开盘参考价核对每只的目标权重和一手占本金。"
              '现金校验另计费用和滑点。下表使用正常费用。', '',
              '| 本金（元） | ETF | 目标权重 | 一手占本金 | 含买费一手需现金（元） | 结果 |',
              '|---:|---|---:|---:|---:|---|']
    for order in report['first_order_details']:
        if order['cost_case'] != 'normal':
            continue
        status = '整手不可执行' if order['reason'] == 'board_lot' else f"成交 {order['filled_units']:,} 份"
        lines.append(f"| {order['initial_cash']:,} | {order['symbol']} | {order['target_weight']:.2%} | "
                     f"{order['one_lot_weight']:.2%} | {order['one_lot_buy_cash_required']:,.2f} | {status} |")
    lines += ['', '## 成本门槛', '',
              f"以 {report['fee_reference_date']} 收盘价为参考，假设买入和卖出时的原价持平。"
              '用同一账本费率、100 份整手、双边滑点与分币取整计算，成本预算为成交额的 25 和 50 个基点。'
              '1 个基点为 0.01 个百分点。表中最低成交额按不含滑点的参考价计算，账户还需留出买入费用与滑点。', '',
              '| 费用情景 | ETF | 一手参考额（元） | 一手双边成本（基点） | ≤25 基点最低成交额（元） | ≤50 基点最低成交额（元） |',
              '|---|---|---:|---:|---:|---:|']
    for row in report['flat_round_trip']:
        for symbol, values in row['by_symbol'].items():
            one = values['one_lot']
            minimums = ['未达' if result is None else f"{result['reference_notional']:,.2f}"
                        for result in values['minimum_by_cost_budget_bps'].values()]
            lines.append(f"| {labels[row['cost_case']]} | {symbol} | {one['reference_notional']:,.2f} | "
                         f"{one['all_in_cost_bps']:.2f} | {minimums[0]} | {minimums[1]} |")
    normal = report['flat_round_trip'][0]['by_symbol']
    accounts = [
        result['minimum_account_equity_at_weight_cap'] for values in normal.values()
        if (result := values['minimum_by_cost_budget_bps'][25]) is not None
    ]
    if accounts:
        lines += ['', f'按单标权重上限 {report["max_target_weight"]:.0%}，要在目标额度内达到正常费用 25 基点预算，'
                  f'示例需要账户权益约 {min(accounts):,.2f} 至 {max(accounts):,.2f} 元。'
                  '这只检查单标额度，多个目标同时买入还需按现金校验，实际目标权重更低时门槛更高。']
    lines += ['', '未达表示搜索至单笔参考成交额 100 万元仍未达到预算。忽略分币取整，比例佣金与滑点的成本下界约为'
              '正常 16 基点、双倍摩擦 32 基点。2 万元时双边最低佣金 10 元对应 5 基点，按正常费率还要考虑比例佣金与滑点。', '',
              '本次是已审阅历史的资金敏感性。行情供应商的历史首次获知时间和修订记录仍不完整，'
              '使用日线开盘价代理成交，缺少盘口、容量冲击和历史新闻。2025 年起封存样本未进入计算。', '']
    return '\n'.join(lines)


def render_curve(metrics: list[dict]) -> str:
    series = {
        case: [[row['initial_cash'], 100 * row['net_return']] for row in metrics if row['cost_case'] == case]
        for case in ('normal', 'double_friction')
    }
    return '''<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>本金与扣费收益</title>
<style>
body { margin: 2rem auto; max-width: 850px; padding: 0 1rem; font: 16px system-ui; color: #182435; }
canvas { width: 100%; height: auto; } a { color: #165a8c; }
</style>
<h1>本金与扣费收益</h1>
<p>2023–2024 年四只境内 ETF 的同一动量基线。</p>
<canvas width="830" height="415" aria-label="六个虚拟本金档位的扣费收益曲线"></canvas>
<p><a href="report.md">各档数值与费用假设</a></p>
<script>
const series = ''' + json.dumps(series, allow_nan=False) + ''';
const ctx = document.querySelector('canvas').getContext('2d');
const capital = series.normal.map(([value]) => value);
const returns = Object.values(series).flat().map(([, value]) => value);
const bottom = 350, top = 50, left = 70, right = 800;
const lower = Math.floor(Math.min(...returns, 0) / 5) * 5;
const upper = Math.max(lower + 5, Math.ceil(Math.max(...returns, 0) / 5) * 5);
const x = value => left + (right - left) * Math.log(value / capital[0]) / Math.log(capital.at(-1) / capital[0]);
const y = value => top + (bottom - top) * (upper - value) / (upper - lower);
ctx.font = '14px system-ui';
ctx.fillStyle = '#26384a';
ctx.textAlign = 'right';
for (let value = lower; value <= upper; value += 5) {
  ctx.fillText(`${value}%`, left - 10, y(value) + 4);
  ctx.strokeStyle = '#dbe3ed';
  ctx.beginPath(); ctx.moveTo(left, y(value)); ctx.lineTo(right, y(value)); ctx.stroke();
}
ctx.textAlign = 'center';
for (const value of capital) ctx.fillText(value.toLocaleString('zh-CN'), x(value), bottom + 22);
ctx.fillText('虚拟本金（元，对数刻度）', (left + right) / 2, bottom + 49);
for (const [i, [name, points]] of Object.entries(series).entries()) {
  const color = i ? '#ab4d32' : '#176087';
  ctx.strokeStyle = ctx.fillStyle = color;
  ctx.textAlign = 'left';
  ctx.fillText(name === 'normal' ? '正常费用' : '双倍摩擦', left + i * 95, 27);
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  points.forEach(([capital, value], index) => {
    if (index) ctx.lineTo(x(capital), y(value)); else ctx.moveTo(x(capital), y(value));
  });
  ctx.stroke();
  for (const [capital, value] of points) {
    ctx.beginPath(); ctx.arc(x(capital), y(value), 4, 0, 2 * Math.PI); ctx.fill();
  }
}
</script>
</html>
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run_study(args.snapshot, args.out)


if __name__ == '__main__':
    main()
