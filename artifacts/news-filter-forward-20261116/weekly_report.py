"""Reconcile official daily ledgers and attribute this week's news exclusions."""

import argparse
import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

from close_monday import AmbiguousPartialLedgers, daily_ledger_path, quote_evidence_detail
from reconcile import reconcile

BASE = Path(__file__).resolve().parent
PROTOCOL_SHA256 = '477e3dcfcf51b4b21ddb3fd51ebee38cdfb126c4fcb5c9fc8b715da4dcf692fe'
DATES = tuple(f'2026-11-{day}' for day in range(16, 21))
GROUPS = ('baseline', 'filtered')
NEWS_BLOCK = 'news_abnormal_volatility'
MISSING_EVIDENCE = {
    'quote_unavailable', 'quote_stale', 'quote_from_future', 'outside_window', 'invalid_quote',
    'invalid_ask_or_previous_close_mismatch', 'quote_received_after_window',
    'premium_not_verified', 'iopv_missing', 'iopv_invalid', 'iopv_layout_unverified',
    'iopv_premium_mismatch', 'iopv_intraday_update_unverified',
    'news_data_unavailable', 'previous_nav_unverified', 'invalid_quote_or_price_limits',
    'corporate_action_or_previous_close_unverified', 'price_limits_unverified',
    'entry_quote_precedes_exit',
}


def cash_value(value):
    amount = Decimal(value)
    if not amount.is_finite():
        raise ValueError('nonfinite ledger amount')
    return amount


def check_fills(records):
    positions = {group: {} for group in GROUPS}
    for row in records:
        for group in GROUPS:
            account = row[group]
            held = dict(positions[group])
            bought = False
            for fill in account['fills']:
                side = fill['side']
                symbol, units = fill['symbol'], fill['units']
                if side not in ('buy', 'sell'):
                    raise ValueError(f'{row["date"]} {group}: unknown fill side')
                if side == 'sell' and bought:
                    raise ValueError(f'{row["date"]} {group}: sells must precede buys')
                bought |= side == 'buy'
                if type(units) is not int or units <= 0:
                    raise ValueError(f'{row["date"]} {group}: invalid fill quantity')
                if side == 'sell':
                    if held.pop(symbol, None) != units:
                        raise ValueError(f'{row["date"]} {group}: sell must close a previously held lot')
                elif symbol in held:
                    raise ValueError(f'{row["date"]} {group}: buy overlaps an open lot')
                else:
                    held[symbol] = units
                if cash_value(fill['price']) <= 0 or any(
                    cash_value(fill[key]) < 0 for key in ('commission', 'stamp_tax', 'transfer_fee')
                ):
                    raise ValueError(f'{row["date"]} {group}: invalid price or fee')
            if any(cash_value(price) <= 0 for price in account['close_prices'].values() if price is not None):
                raise ValueError(f'{row["date"]} {group}: invalid close price')
            positions[group] = account['positions']


def attributed_buys(records, complete_week):
    """Track baseline lots by symbol and pair any exact news exclusion."""
    open_lots, opportunities, excluded = {}, [], []
    for row in records:
        day, baseline, filtered = row['date'], row['baseline'], row['filtered']
        baseline_decisions = {d['symbol']: d for d in baseline.get('decisions', []) if d.get('side') == 'buy'}
        for decision in baseline.get('decisions', []):
            if decision.get('side') != 'sell':
                continue
            lot = open_lots.get(decision['symbol'])
            if lot is None or lot['opportunity'] is None:
                continue
            book = decision.get('book') or {}
            lot['opportunity']['exit_attempts'].append({
                'date': day, 'reason': decision['reason'],
                'required_units': decision.get('required_units', lot['current_units']),
                'available_units': decision.get('available_units', book.get('bid_units')),
            })
        filtered_decisions = {d['symbol']: d for d in filtered.get('decisions', []) if d.get('side') == 'buy'}
        filtered_fills = {fill['symbol'] for fill in filtered['fills'] if fill['side'] == 'buy'}
        baseline_buys = {fill['symbol'] for fill in baseline['fills'] if fill['side'] == 'buy'}
        for symbol, decision in filtered_decisions.items():
            if decision['reason'] not in (NEWS_BLOCK, 'news_data_unavailable'):
                continue
            reason = (
                'news_data_unavailable' if decision['reason'] == 'news_data_unavailable'
                else 'baseline_not_filled' if symbol not in baseline_buys else None
            )
            if reason:
                excluded.append({'date': day, 'symbol': symbol, 'reason': reason})

        for fill in baseline['fills']:
            symbol, units = fill['symbol'], fill['units']
            if fill['side'] == 'sell':
                lot = open_lots.pop(symbol, None)
                if lot is None or lot['date'] >= day or lot['current_units'] != units:
                    raise ValueError(f'{day} {symbol}: exit does not match one previously held lot')
                opportunity = lot['opportunity']
                if opportunity is not None:
                    opportunity['exit_date'] = day
                    opportunity['exit_status'] = 'sold'
                    opportunity['exit_price'] = fill['price']
                    opportunity['exit_fees'] = fill['fees']
                    if not opportunity.get('unresolved_reason'):
                        opportunity['net_pnl'] = str(cash_value(fill['cash_delta']) - lot['entry_cost'])
                continue

            if symbol in open_lots:
                raise ValueError(f'{day} {symbol}: buy overlaps an open lot')
            opportunity = None
            other = filtered_decisions.get(symbol, {})
            if other.get('reason') == NEWS_BLOCK:
                own = baseline_decisions.get(symbol, {})
                if not own.get('book') or own['book'] != other.get('book'):
                    excluded.append({'date': day, 'symbol': symbol, 'reason': 'common_book_unverified'})
                elif (own.get('reason') != 'filled' or own.get('required_units') != units
                      or other.get('required_units') != units or symbol in filtered_fills):
                    excluded.append({'date': day, 'symbol': symbol, 'reason': 'paired_execution_mismatch'})
                else:
                    opportunity = {
                        'date': day, 'symbol': symbol, 'units': units, 'entry_price': fill['price'],
                        'buy_fees': fill['fees'], 'exit_status': 'open', 'exit_attempts': [], 'net_pnl': None,
                    }
                    opportunities.append(opportunity)
            open_lots[symbol] = {
                'date': day, 'current_units': units, 'entry_cost': -cash_value(fill['cash_delta']),
                'opportunity': opportunity,
            }

        for action in baseline.get('corporate_actions', []):
            symbol = action.get('symbol')
            affected = (symbol,) if symbol else tuple(open_lots)
            for code in affected:
                lot = open_lots.get(code)
                if lot is None:
                    continue
                lot['current_units'] += action.get('units_delta', 0)
                opportunity = lot['opportunity']
                if opportunity is not None:
                    opportunity['unresolved_reason'] = 'corporate_action_attribution_unverified'
                if not lot['current_units']:
                    if opportunity is not None:
                        opportunity['exit_status'] = 'closed_by_corporate_action'
                    del open_lots[code]

    final = records[-1]['baseline'] if records else {}
    for symbol, lot in open_lots.items():
        opportunity = lot['opportunity']
        if opportunity is None or opportunity.get('unresolved_reason'):
            continue
        if not complete_week:
            opportunity['unresolved_reason'] = 'week_incomplete'
            continue
        mark = final.get('close_prices', {}).get(symbol)
        gaps = final.get('valuation_gap') or {}
        if mark is None or symbol in gaps:
            opportunity['unresolved_reason'] = 'friday_mark_unverified'
            continue
        price = cash_value(mark)
        if price <= 0:
            raise ValueError(f'{symbol}: invalid Friday mark')
        opportunity['exit_status'] = 'marked_open'
        opportunity['mark_price'] = str(price)
        opportunity['net_pnl'] = str(price * lot['current_units'] - lot['entry_cost'])

    for item in opportunities:
        if item['net_pnl'] is None:
            item['effect'] = 'unresolved'
        else:
            pnl = Decimal(item['net_pnl'])
            item['effect'] = 'avoided_loss' if pnl < 0 else 'missed_gain' if pnl > 0 else 'flat'
            item['filtered_minus_baseline'] = str(-pnl)
    return opportunities, excluded


def resolved_opportunity_totals(opportunities, *, amounts_known):
    """Summarize signed, after-fee effects for opportunities with a known value."""
    avoided_loss = Decimal(0)
    missed_gain = Decimal(0)
    losses = gains = flat = 0
    for opportunity in opportunities:
        delta = cash_value(opportunity['filtered_minus_baseline'])
        if delta > 0:
            losses += 1
            avoided_loss += delta
        elif delta < 0:
            gains += 1
            missed_gain -= delta
        else:
            flat += 1
    return {
        'resolved_count': len(opportunities),
        'avoided_loss_count': losses,
        'missed_gain_count': gains,
        'flat_count': flat,
        'avoided_loss_cny': str(avoided_loss) if amounts_known else None,
        'missed_gain_cny': str(missed_gain) if amounts_known else None,
        'known_filtered_minus_baseline_cny': str(avoided_loss - missed_gain) if amounts_known else None,
    }


def summarize_opportunities(opportunities, *, reconciled_days, has_evidence_gaps, terminal_nav_gap):
    resolved = [opportunity for opportunity in opportunities if opportunity['net_pnl'] is not None]
    unresolved = Counter(opportunity['unresolved_reason'] for opportunity in opportunities
                         if opportunity['net_pnl'] is None)
    amounts_known = bool(resolved) or (bool(reconciled_days) and not unresolved and not has_evidence_gaps)
    totals = resolved_opportunity_totals(resolved, amounts_known=amounts_known)
    known_delta = totals['known_filtered_minus_baseline_cny']
    return {
        **totals,
        'reconciled_days': reconciled_days,
        'realized': resolved_opportunity_totals(
            [opportunity for opportunity in resolved if opportunity['exit_status'] == 'sold'],
            amounts_known=amounts_known,
        ),
        'friday_marked_open': resolved_opportunity_totals(
            [opportunity for opportunity in resolved if opportunity['exit_status'] == 'marked_open'],
            amounts_known=amounts_known,
        ),
        'unresolved_count': sum(unresolved.values()),
        'unresolved_by_reason': dict(sorted(unresolved.items())),
        'account_ending_nav_gap_cny': str(terminal_nav_gap) if terminal_nav_gap is not None else None,
        'unattributed_ending_nav_gap_cny': (
            str(terminal_nav_gap - cash_value(known_delta))
            if terminal_nav_gap is not None and known_delta is not None else None
        ),
    }


def weekly_report(records, *, artifact_gaps=()):
    """Report only the consecutive prefix that can be reconciled from initial cash."""
    dates = [row['date'] for row in records]
    if dates != sorted(set(dates)) or any(day not in DATES for day in dates):
        raise ValueError('daily dates must be unique, chronological and within 2026-11-16 to 2026-11-20')
    by_date = {row['date']: row for row in records}
    prefix = []
    for day in DATES:
        if day not in by_date:
            break
        prefix.append(by_date[day])
    check_fills(prefix)
    groups = reconcile(prefix)['groups']
    complete_week = len(prefix) == len(DATES)
    opportunities, excluded = attributed_buys(prefix, complete_week)
    usable = {row['symbol'] for row in opportunities if row['effect'] != 'unresolved'}
    evidence_gaps, archive_gaps, inconsistent_marks, daily_metrics = [], [], [], []
    previous_positions = {group: {} for group in GROUPS}
    for row in prefix:
        day, baseline, filtered = row['date'], row['baseline'], row['filtered']
        daily = {'date': day}
        if (archive := row.get('news_archive')) is not None:
            daily['news_archive'] = archive
            gaps = {key: archive[key] for key in ('missing_dates', 'invalid_dates', 'incomplete_dates')
                    if archive.get(key)}
            if gaps:
                archive_gaps.append({'date': day, **gaps})
        if row.get('news_errors'):
            evidence_gaps.append({'date': day, 'reason': 'news_files_unavailable',
                                  'details': row['news_errors']})
        if row.get('candidate_data_missing'):
            evidence_gaps.append({'date': day, 'reason': 'candidate_data_missing'})
        for group in GROUPS:
            account = row[group]
            decisions = account.get('decisions', [])
            planned = row.get('candidate_symbols')
            if planned is None and day == DATES[0]:
                planned = row.get('planned_symbols')
            expected = {
                'buy': set(planned or ()),
                'sell': set((row.get('expected_sell_symbols') or {}).get(group, previous_positions[group])),
            }
            observed = {(decision.get('side'), decision['symbol']) for decision in decisions}
            for side, symbols in expected.items():
                for symbol in sorted(symbols):
                    if (side, symbol) not in observed:
                        evidence_gaps.append({'date': day, 'group': group, 'symbol': symbol,
                                              'side': side, 'reason': 'decision_missing'})
            previous_positions[group] = account['positions']
            fills = account['fills']
            daily[group] = {
                'buy_count': sum(fill['side'] == 'buy' for fill in fills),
                'sell_count': sum(fill['side'] == 'sell' for fill in fills),
                'fees': str(sum((cash_value(fill['fees']) for fill in fills), Decimal(0))),
                'close_nav': account['nav'],
            }
            for decision in decisions:
                reason = decision['reason']
                if reason in MISSING_EVIDENCE or reason.startswith('quote_') or reason.startswith('iopv_'):
                    gap = {'date': day, 'group': group, 'symbol': decision['symbol'],
                           'side': decision.get('side'), 'reason': reason}
                    detail = ((decision.get('evaluation') or {}).get('news_error')
                              if reason == 'news_data_unavailable' else
                              quote_evidence_detail(row.get('book_evidence_errors') or {}, decision))
                    if detail:
                        gap['detail'] = detail
                    evidence_gaps.append(gap)
        for symbol in sorted(baseline['positions'].keys() & filtered['positions'].keys()):
            a, b = (account['close_prices'].get(symbol) for account in (baseline, filtered))
            if a is not None and b is not None and cash_value(a) != cash_value(b):
                inconsistent_marks.append({'date': day, 'symbol': symbol, 'baseline': a, 'filtered': b})
        daily_metrics.append(daily)

    for group, metrics in groups.items():
        fills = [fill for row in prefix for fill in row[group]['fills']]
        metrics['buy_count'] = sum(fill['side'] == 'buy' for fill in fills)
        metrics['sell_count'] = sum(fill['side'] == 'sell' for fill in fills)
        if not prefix:
            for key in ('buy_count', 'sell_count', 'fees', 'ending_positions'):
                metrics[key] = None

    incomplete_reasons = []
    if not complete_week:
        incomplete_reasons.append('missing_daily_ledgers')
    if any(item['missing_valuation_dates'] for item in groups.values()):
        incomplete_reasons.append('missing_close_valuation')
    if evidence_gaps:
        incomplete_reasons.append('incomplete_execution_evidence')
    if archive_gaps:
        incomplete_reasons.append('incomplete_news_archive_coverage')
    if inconsistent_marks:
        incomplete_reasons.append('inconsistent_common_close_prices')
    unverified_pairing = any(row['reason'] in ('common_book_unverified', 'paired_execution_mismatch')
                            for row in excluded)
    if unverified_pairing:
        incomplete_reasons.append('unverified_pairing')
    if any(item.get('unresolved_reason') == 'corporate_action_attribution_unverified'
           for item in opportunities):
        incomplete_reasons.append('unresolved_effective_opportunities')
    if len(usable) < 5:
        incomplete_reasons.append('fewer_than_five_distinct_verified_filtered_symbols')

    comparison = None
    baseline, filtered = (groups[group] for group in GROUPS)
    if complete_week and not inconsistent_marks and all(
        item['net_return'] is not None and item['max_drawdown'] is not None for item in groups.values()
    ):
        return_diff = Decimal(filtered['net_return']) - Decimal(baseline['net_return'])
        drawdown_diff = Decimal(filtered['max_drawdown']) - Decimal(baseline['max_drawdown'])
        comparison = {'return_diff': str(return_diff), 'drawdown_diff': str(drawdown_diff)}
    terminal_nav_gap = None
    if (complete_week and not any(mark['date'] == DATES[-1] for mark in inconsistent_marks)
            and baseline['ending_nav'] is not None and filtered['ending_nav'] is not None):
        terminal_nav_gap = cash_value(filtered['ending_nav']) - cash_value(baseline['ending_nav'])
    if not incomplete_reasons:
        improvements = (return_diff > 0, drawdown_diff < 0)
        verdict = 'both_improved' if all(improvements) else 'tradeoff' if any(improvements) else 'neither_improved'
    else:
        verdict = 'insufficient_evidence'
    return {
        'expected_dates': list(DATES), 'available_dates': dates,
        'missing_dates': [day for day in DATES if day not in by_date],
        'reconciled_dates': [row['date'] for row in prefix],
        'unreconciled_available_dates': dates[len(prefix):], 'groups': groups,
        'daily_metrics': daily_metrics,
        'effective_opportunities': opportunities, 'excluded_opportunities': excluded,
        'opportunity_summary': summarize_opportunities(
            opportunities, reconciled_days=len(prefix),
            has_evidence_gaps=bool(evidence_gaps or archive_gaps or unverified_pairing),
            terminal_nav_gap=terminal_nav_gap,
        ),
        'distinct_verified_filtered_symbols': sorted(usable),
        'execution_evidence_gaps': evidence_gaps, 'inconsistent_common_close_prices': inconsistent_marks,
        'ledger_artifact_gaps': list(artifact_gaps), 'news_archive_gaps': archive_gaps,
        'verdict': verdict, 'insufficient_evidence_reasons': incomplete_reasons, 'comparison': comparison,
    }


def cash_only_marks_error(path, folder, row):
    """Confirm a cash-only partial is paired with its archived marks error."""
    if any(row[group]['positions'] for group in GROUPS):
        return None
    summary_path = path.with_name('close-summary.json')
    if summary_path.resolve() != summary_path:
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        issue = summary['input_errors']['close_marks']
        artifacts = summary['artifacts']
        if (summary['date'] != row['date'] or summary['status'] != 'valuation_incomplete'
                or artifacts.get('daily') != str(path) or artifacts.get('summary') != str(summary_path)
                or Path(issue['path']).resolve() != folder / 'close_marks.json'
                or not isinstance(issue['error'], str) or not issue['error']):
            return None
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    payload = {key: value for key, value in summary.items() if key != 'artifacts'}
    encoded = json.dumps({'daily': row, 'summary': payload}, ensure_ascii=False, sort_keys=True)
    return issue if hashlib.sha256(encoded.encode('utf-8')).hexdigest() == path.parent.name else None


def official_ledgers(root):
    """Read forward ledgers from dated folders using the close ledger resolver."""
    if hashlib.sha256((root / 'protocol.md').read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError('frozen protocol hash mismatch')
    ledgers, artifact_gaps = [], []
    for day in DATES:
        folder = root / day
        try:
            path = daily_ledger_path(folder)
        except AmbiguousPartialLedgers as exc:
            artifact_gaps.append({'date': day, 'reason': 'ambiguous_partial_ledgers', 'count': exc.count})
            continue
        if path is None:
            continue
        row = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(row, dict) or row.get('date') != day or row.get('mode') != 'forward':
            raise ValueError(f'{day}: expected an official forward daily ledger')
        if path.parent != folder and all(row[group]['nav'] is not None for group in GROUPS):
            issue = cash_only_marks_error(path, folder, row)
            if issue is None:
                raise ValueError(f'{day}: partial ledger must have an unavailable NAV or a verified cash-only marks error')
            artifact_gaps.append({'date': day, 'reason': 'close_marks_unverified', 'details': issue})
        ledgers.append(row)
    return ledgers, artifact_gaps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        records, gaps = official_ledgers(BASE)
        if not records:
            message = 'No usable official 2026-11-16 to 2026-11-20 daily ledgers found.'
            if gaps:
                message += ' ' + json.dumps(gaps, ensure_ascii=False)
            parser.exit(2, message + '\n')
        print(json.dumps(weekly_report(records, artifact_gaps=gaps), ensure_ascii=False, indent=2))
    except (OSError, TypeError, ValueError, ArithmeticError, KeyError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
