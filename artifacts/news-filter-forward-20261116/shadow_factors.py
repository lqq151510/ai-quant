"""Compare Friday volume and trend filters on the same Monday paper fills."""
import argparse
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime

from candidate_review import rows_of, sessions
from close_monday import daily_ledger_path
from monday_scenario_matrix import BASE, DAY, GROUPS, frozen_inputs, synthetic_books
from monday_open import INITIAL, run, tick_price
from reconcile import reconcile

FEATURES = BASE / 'volume-news-ma'
ARMS = {
    'baseline': (),
    'news': ('news',),
    'volume': ('volume',),
    'ma': ('ma',),
    'volume_ma': ('volume', 'ma'),
    'all': ('news', 'volume', 'ma'),
}
THRESHOLD = Decimal('1.5')
UNVERIFIED_EXECUTION = {'invalid_quote', 'outside_window', 'premium_not_verified',
                        'invalid_ask_or_previous_close_mismatch'}


def load_features(prepared):
    """Rebuild all indicators from date checked, archived raw responses."""
    manifest = json.loads((FEATURES / 'candidate-review.json').read_text())
    capture_time = datetime.fromisoformat(manifest['captured_at'])
    if (manifest['market_close'] != '2026-11-13' or capture_time.utcoffset() is None
            or capture_time >= datetime.fromisoformat(f'{DAY}T09:45:00+08:00')):
        raise ValueError('feature date does not match the decision cutoff')
    rows = {row['symbol']: row for row in manifest['rows']}
    features = {}
    for row in prepared['candidates']:
        symbol = row['symbol']
        code = symbol[-2:].lower() + symbol[:6]
        feature = {}
        features[symbol] = feature
        try:
            evidence = rows[symbol]
            frozen = BASE / 'monday-preparation' / 'bars'
            prices, volumes = rows_of((frozen / f'{symbol}-0.json').read_bytes(), code, 'day', sessions(6))
            adjusted, _ = rows_of((frozen / f'{symbol}-1.json').read_bytes(), code, 'qfqday', sessions(6))
            if adjusted[-1] != prices[-1] or prices[-1] != Decimal(row['previous_close']):
                raise ValueError('price evidence conflicts with frozen shortlist')
            feature['return'] = adjusted[-1] / adjusted[-2] - 1
            feature['volume_ratio'] = volumes[-1] / (sum(volumes[:-1]) / 5)
        except (OSError, ValueError, KeyError, TypeError, IndexError, ArithmeticError) as exc:
            feature['volume_gap'] = str(exc)
            feature['ma_gap'] = 'short adjusted history unverified'
            continue
        try:
            filename = f'{symbol}-qfq-20.json'
            if evidence['extended_raw'] != filename:
                raise ValueError('unexpected adjusted evidence filename')
            raw = (FEATURES / filename).read_bytes()
            if hashlib.sha256(raw).hexdigest() != evidence['extended_sha256']:
                raise ValueError('adjusted price evidence checksum mismatch')
            closes, _ = rows_of(raw, code, 'qfqday', sessions(20))
            if closes[-6:] != adjusted:
                raise ValueError('20-session history conflicts with archived last six')
            feature['above_all_ma'] = all(closes[-1] >= sum(closes[-n:]) / n for n in (5, 10, 20))
        except (OSError, ValueError, KeyError, TypeError, IndexError, ArithmeticError) as exc:
            feature['ma_gap'] = str(exc)
    return features


def signals(feature, news_decision, volume_threshold):
    feature = feature or {}
    volume = None if 'volume_ratio' not in feature else (
        feature['return'] < 0 and feature['volume_ratio'] >= volume_threshold)
    ma = None if 'above_all_ma' not in feature else not feature['above_all_ma']
    reason = (news_decision or {}).get('reason')
    news = True if reason == 'news_abnormal_volatility' else False if reason == 'filled' else None
    return {'news': news, 'volume': volume, 'ma': ma}


def compare(opening, close_prices, features, *, volume_threshold=THRESHOLD):
    """Hold fills, sizing, and prices constant. Drop only the fills an arm vetoes."""
    baseline = opening['baseline']['fills']
    if any(fill['side'] != 'buy' for fill in baseline):
        raise ValueError('this comparison starts flat and only supports Monday buys')
    if len({fill['symbol'] for fill in baseline}) != len(baseline):
        raise ValueError('expected at most one baseline buy per symbol')
    valid_marks = {}
    for symbol, close in close_prices.items():
        try:
            mark = Decimal(close)
        except (TypeError, ArithmeticError, ValueError):
            continue
        if mark.is_finite() and mark > 0:
            valid_marks[symbol] = str(mark)
    news = {row['symbol']: row for row in opening['filtered']['decisions'] if row.get('side') == 'buy'}
    checks = {fill['symbol']: signals(features.get(fill['symbol']),
                                      news.get(fill['symbol']), volume_threshold) for fill in baseline}
    reports = {}
    for arm, filters in ARMS.items():
        kept, blocked, unknown = [], {}, {}
        for fill in baseline:
            symbol = fill['symbol']
            vetoes = [name for name in filters if checks[symbol][name] is True]
            gaps = [name for name in filters if checks[symbol][name] is None]
            if vetoes:
                blocked[symbol] = vetoes
            elif gaps:
                unknown[symbol] = gaps
            else:
                kept.append(fill)
        positions = {fill['symbol']: fill['units'] for fill in kept}
        cash = INITIAL + sum(Decimal(fill['cash_delta']) for fill in kept)
        missing = sorted(positions.keys() - valid_marks.keys())
        prices = {symbol: valid_marks[symbol] for symbol in positions if symbol not in missing}
        nav = None if missing else cash + sum(Decimal(prices[s])*units for s, units in positions.items())
        account = {'fills': kept, 'positions': positions, 'cash': str(cash), 'close_prices': prices,
                   'nav': None if nav is None else str(nav)}
        metrics = reconcile([{'date': str(DAY), **dict.fromkeys(GROUPS, account)}])['groups']['baseline']
        if unknown:
            for field in ('ending_nav', 'net_return', 'max_drawdown'):
                metrics[field] = None
        contribution, break_even = {}, {}
        for fill in baseline:
            symbol = fill['symbol']
            if symbol in blocked:
                close = valid_marks.get(symbol)
                break_even[symbol] = str(-Decimal(fill['cash_delta']) / fill['units'])
                contribution[symbol] = None if close is None else str(
                    -Decimal(close)*fill['units'] - Decimal(fill['cash_delta']))
        reports[arm] = {
            **metrics, 'buys': len(kept), 'kept': sorted(positions),
            'deployed_cash': str(INITIAL - cash), 'blocked': blocked, 'unknown': unknown,
            'missing_marks': missing, 'blocked_contribution': contribution, 'break_even_close': break_even,
            'comparable': not unknown and not missing,
        }
    base_nav = reports['baseline']['ending_nav']
    for arm in reports.values():
        arm['comparable'] = arm['comparable'] and base_nav is not None
        if arm['comparable']:
            arm['delta_vs_baseline'] = str(Decimal(arm['ending_nav']) - Decimal(base_nav))
            if Decimal(arm['delta_vs_baseline']) != sum(Decimal(v) for v in arm['blocked_contribution'].values()):
                raise ValueError('counterfactual delta does not match blocked fill attribution')
        else:
            arm['delta_vs_baseline'] = None
    return reports


def stress(prepared, archived_news, features):
    ticks = {row['symbol']: Decimal(row['tick']) for row in prepared['candidates']}
    cases = []
    for quote_mode in ('eligible', 'etf_absent', 'etf_premium_high'):
        opened = run(prepared, synthetic_books(prepared['candidates'], quote_mode), archived_news)
        for label, moves in (
            ('flat', {}),
            ('risk_names_fall', {'603135.SH': '-.05', '603920.SH': '-.05', '000651.SZ': '.05'}),
            ('risk_names_rebound', {'603135.SH': '.05', '603920.SH': '.05', '000651.SZ': '-.05'}),
            ('ma_blocks_rebound', {'600498.SH': '.04', '513100.SH': '.04', '515880.SH': '.04'}),
            ('mark_missing', {'603135.SH': None}),
        ):
            closes = {}
            for fill in opened['baseline']['fills']:
                symbol = fill['symbol']
                move = moves.get(symbol, '0')
                if move is not None:
                    closes[symbol] = str(tick_price(Decimal(fill['price'])*(1+Decimal(move)), ticks[symbol], ROUND_HALF_UP))
            cases.append({'quotes': quote_mode, 'close': label, 'arms': compare(opened, closes, features)})
    sensitivity = {}
    opened = run(prepared, synthetic_books(prepared['candidates'], 'eligible'), archived_news)
    for level in ('1.0', '1.2', '1.5', '2.0', '2.5'):
        result = compare(opened, {}, features, volume_threshold=Decimal(level))
        sensitivity[level] = sorted(result['volume']['blocked'])
    envelopes = {}
    for move in (Decimal('-.05'), Decimal('.05')):
        prices = {fill['symbol']: str(tick_price(Decimal(fill['price'])*(1+move),
                  ticks[fill['symbol']], ROUND_HALF_UP)) for fill in opened['baseline']['fills']}
        envelopes[str(move)] = {arm: result['delta_vs_baseline']
                                for arm, result in compare(opened, prices, features).items()}
    return {'mode': 'synthetic_shadow', 'date': str(DAY), 'volume_threshold': str(THRESHOLD),
            'cases': cases, 'volume_threshold_sensitivity': sensitivity, 'close_move_envelopes': envelopes}


def observed(prepared, features):
    path = daily_ledger_path(BASE / str(DAY))
    if path is None:
        raise ValueError('no Monday closing ledger available')
    raw = path.read_bytes()
    daily = json.loads(raw)
    if daily.get('mode') != 'forward' or daily.get('date') != str(DAY):
        raise ValueError('expected Monday forward ledger')
    if any(daily[group].get('corporate_actions') for group in GROUPS):
        raise ValueError('shadow comparison requires a Monday ledger without corporate actions')
    official = reconcile([daily])['groups']
    if daily.get('candidate_symbols') != [row['symbol'] for row in prepared['candidates']]:
        raise ValueError('official ledger does not use the frozen shortlist')
    marks = daily['baseline']['close_prices']
    filtered_marks = daily['filtered']['close_prices']
    for symbol in daily['filtered']['positions']:
        baseline_mark, filtered_mark = marks.get(symbol), filtered_marks.get(symbol)
        if (baseline_mark is None) != (filtered_mark is None) or (
            baseline_mark is not None and Decimal(baseline_mark) != Decimal(filtered_mark)
        ):
            raise ValueError(f'{symbol}: official groups disagree on the closing price')
    arms = compare(daily, marks, features)
    for group, arm in (('baseline', arms['baseline']), ('filtered', arms['news'])):
        kept = set(arm['kept'])
        replayed = [fill for fill in daily['baseline']['fills'] if fill['symbol'] in kept]
        if replayed != daily[group]['fills'] or Decimal(arm['fees']) != Decimal(official[group]['fees']):
            raise ValueError(f'{group}: shadow trades or fees disagree with the official ledger')
        if arm['unknown']:
            continue
        official_nav = official[group]['ending_nav']
        shadow_nav = arm['ending_nav']
        if (official_nav is None) != (shadow_nav is None) or (
            official_nav is not None and Decimal(official_nav) != Decimal(shadow_nav)
        ):
            raise ValueError(f'{group}: shadow NAV disagrees with the official ledger')
    rejected = {d['symbol']: d['reason'] for d in daily['baseline'].get('decisions', [])
                if d.get('side') == 'buy' and d.get('reason') != 'filled'}
    if daily.get('book_evidence_errors', {}).get('selected') or any(
        reason.startswith(('quote_', 'iopv_')) or reason in UNVERIFIED_EXECUTION for reason in rejected.values()
    ):
        # A valid subset can still be described; report the unavailable opportunity set.
        scope = 'partial_execution_evidence'
    else:
        scope = 'available_baseline_fills_only'
    return {'mode': 'observed_shadow', 'date': str(DAY), 'scope': scope,
            'close_ledger': str(path.relative_to(BASE)),
            'baseline_unfilled': rejected, 'news_errors': daily.get('news_errors', []),
            'ledger_sha256': hashlib.sha256(raw).hexdigest(),
            'volume_threshold': str(THRESHOLD), 'arms': arms}


def render(result):
    lines = [
        '# 成交量、均线与消息过滤情景对照', '',
        '使用 11 月 13 日收盘行情重算因子，开盘盘口和收盘价格为假设。每组只从同一批原规则买单中移除触发拦截的单，不补位。'
        '量价规则拦截周五下跌且量比至少 1.5 倍的标的，均线规则要求前复权收盘不低于 MA5、MA10、MA20。'
        '消息规则读取正式归档的异常波动公告。没有已匹配记录不代表已证明没有公告。', '',
        '| 盘口条件 | 收盘情景 | 原组净值（元） | 消息 Δ | 量价 Δ | 均线 Δ | 量价+均线 Δ | 三项 Δ |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    labels = {'eligible': '全体合格', 'etf_absent': '513100 缺盘口', 'etf_premium_high': '513100 溢价超限',
              'flat': '买入价持平', 'risk_names_fall': '两只异常波动股跌 5%，格力涨 5%',
              'risk_names_rebound': '两只异常波动股涨 5%，格力跌 5%',
              'ma_blocks_rebound': '三只均线未通过标的涨 4%', 'mark_missing': '中重科技收盘价缺失'}
    for case in result['cases']:
        arms = case['arms']
        def money(value):
            return '不可核' if value is None else f'{Decimal(value):,.2f}'
        delta = ' | '.join(money(arms[key]['delta_vs_baseline']) for key in ARMS if key != 'baseline')
        lines.append(f"| {labels[case['quotes']]} | {labels[case['close']]} | {money(arms['baseline']['ending_nav'])} | {delta} |")
    lines += ['', 'Δ 为该规则组净值减原组净值，包含已发生费用。缺收盘证据时不报净值差。', '',
              '## 固定盘口下的盈亏边界', '',
              '给定每笔买价与股数，被拦截标的若全部在买价上下 5% 内收盘，各规则净值差落在下表区间内，已按报价单位取整。',
              '', '| 规则 | Δ 下界（元） | Δ 上界（元） |', '|---|---:|---:|']
    for arm in ARMS:
        lower = Decimal(result['close_move_envelopes']['0.05'][arm])
        upper = Decimal(result['close_move_envelopes']['-0.05'][arm])
        lines.append(f'| {arm} | {lower:,.2f} | {upper:,.2f} |')
    lines += ['', '阈值敏感性按拦截标的列出。以下是假设对照，不能用来从样本内选择阈值。', '']
    for level, symbols in result['volume_threshold_sensitivity'].items():
        lines.append(f"- {level} 倍：{'、'.join(symbols) if symbols else '无'}")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observed', action='store_true', help='read official Monday ledger without modifying it')
    parser.add_argument('--markdown', action='store_true', help='print the synthetic scenario table')
    args = parser.parse_args()
    try:
        prepared, archived_news = frozen_inputs()
        features = load_features(prepared)
        result = observed(prepared, features) if args.observed else stress(prepared, archived_news, features)
    except (ValueError, TypeError, KeyError, OSError, ArithmeticError) as exc:
        parser.exit(2, f'{exc}\n')
    print(render(result) if args.markdown and not args.observed else json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
