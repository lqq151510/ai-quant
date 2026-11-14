"""Run isolated ledger and attribution scenarios without writing study outputs."""

import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from close_monday import close_report, save_report
from weekly_report import BASE, DATES, GROUPS, official_ledgers, weekly_report


SYMBOLS = tuple(f'{600001 + index}.SH' for index in range(5))


def order(symbol, side, price):
    fee = Decimal('5')
    gross = Decimal(price) * 100
    return {
        'symbol': symbol, 'side': side, 'units': 100, 'price': price,
        'commission': str(fee), 'stamp_tax': '0', 'transfer_fee': '0',
        'fees': str(fee), 'cash_delta': str((-gross if side == 'buy' else gross) - fee),
    }


def ledgers(schedule):
    """Build self-consistent accounts from explicit entries, exits, and marks."""
    cash = dict.fromkeys(GROUPS, Decimal('100000'))
    positions = {group: {} for group in GROUPS}
    records = []
    for day, session in zip(DATES, schedule):
        record = {'date': day, 'mode': 'forward'}
        for group in GROUPS:
            spec = session.get(group, {})
            fills = spec.get('fills', [])
            decisions = spec.get('decisions', [])
            closes = spec.get('close_prices', {})
            for fill in fills:
                symbol = fill['symbol']
                cash[group] += Decimal(fill['cash_delta'])
                if fill['side'] == 'buy':
                    positions[group][symbol] = fill['units']
                else:
                    positions[group].pop(symbol)
            nav = None
            if all(symbol in closes for symbol in positions[group]):
                nav = cash[group] + sum(Decimal(closes[s]) * n for s, n in positions[group].items())
            record[group] = {
                'fills': fills, 'decisions': decisions, 'cash': str(cash[group]),
                'positions': dict(positions[group]), 'close_prices': closes,
                'nav': str(nav) if nav is not None else None,
            }
        records.append(record)
    return records


def buy_spec(symbols=SYMBOLS, reason='news_abnormal_volatility', same_book=True):
    book = {'at': 'synthetic', 'ask': '10'}
    return {
        'baseline': {
            'fills': [order(symbol, 'buy', '10') for symbol in symbols],
            'close_prices': dict.fromkeys(symbols, '10'),
            'decisions': [{'symbol': symbol, 'side': 'buy', 'book': book,
                           'required_units': 100, 'reason': 'filled'} for symbol in symbols],
        },
        'filtered': {
            'decisions': [
                {'symbol': symbol, 'side': 'buy', 'book': book if same_book else {'ask': '10.01'},
                 'required_units': 100, 'reason': reason} for symbol in symbols
            ],
        },
    }


def with_sale(prices, **options):
    purchase = buy_spec(tuple(prices), **options)
    sale = {'baseline': {
        'fills': [order(symbol, 'sell', price) for symbol, price in prices.items()],
        'decisions': [{'symbol': symbol, 'side': 'sell', 'reason': 'filled'} for symbol in prices],
    }}
    return ledgers([purchase, sale, {}, {}, {}])


class WeeklyReportTest(unittest.TestCase):
    def test_five_resolved_symbols_pass_and_include_two_side_fees(self):
        records = with_sale(dict.fromkeys(SYMBOLS, '9.90'))
        result = weekly_report(records)
        self.assertEqual(result['verdict'], 'both_improved')
        self.assertEqual(result['distinct_verified_filtered_symbols'], list(SYMBOLS))
        self.assertEqual(result['groups']['baseline']['fees'], '50')
        self.assertEqual(result['groups']['baseline']['ending_nav'], '99900.00')
        self.assertEqual(result['groups']['baseline']['max_drawdown'], '0.001')
        self.assertEqual(result['comparison'], {'return_diff': '0.001', 'drawdown_diff': '-0.001'})
        self.assertEqual(result['daily_metrics'][0]['baseline']['buy_count'], 5)
        self.assertEqual(result['daily_metrics'][1]['baseline']['sell_count'], 5)
        for row in result['effective_opportunities']:
            self.assertEqual(row['effect'], 'avoided_loss')
            self.assertEqual(row['exit_date'], DATES[1])
            self.assertEqual(Decimal(row['net_pnl']), Decimal('-20'))

    def test_winner_flat_loser_and_tradeoff(self):
        prices = dict(zip(SYMBOLS, ('10.50', '10.10', '9.90', '10.50', '10.50')))
        result = weekly_report(with_sale(prices))
        self.assertEqual(result['verdict'], 'tradeoff')
        self.assertEqual([row['effect'] for row in result['effective_opportunities']],
                         ['missed_gain', 'flat', 'avoided_loss', 'missed_gain', 'missed_gain'])

    def test_attribution_tracks_failed_exit_until_sale_and_keeps_both_fees(self):
        symbol = SYMBOLS[0]
        purchase = buy_spec((symbol,))
        retry = {'baseline': {
            'decisions': [{'symbol': symbol, 'side': 'sell', 'reason': 'insufficient_bid_units',
                           'required_units': 100, 'available_units': 99}],
            'close_prices': {symbol: '10.01'},
        }}
        sale = {'baseline': {
            'decisions': [{'symbol': symbol, 'side': 'sell', 'reason': 'filled', 'book': {'bid_units': 100}}],
            'fills': [order(symbol, 'sell', '10.05')],
        }}
        [lot] = weekly_report(ledgers([purchase, retry, sale, {}, {}]))['effective_opportunities']
        self.assertEqual(lot['exit_attempts'], [
            {'date': DATES[1], 'reason': 'insufficient_bid_units', 'required_units': 100, 'available_units': 99},
            {'date': DATES[2], 'reason': 'filled', 'required_units': 100, 'available_units': 100},
        ])
        self.assertEqual((lot['buy_fees'], lot['exit_fees'], lot['exit_price']), ('5', '5', '10.05'))
        self.assertEqual(Decimal(lot['net_pnl']), Decimal('-5'))

    def test_attribution_summary_separates_sales_friday_marks_and_unknowns(self):
        loser, winner, flat, marked, unpriced = SYMBOLS
        purchase = buy_spec((loser, winner, flat))
        sale = {'baseline': {
            'fills': [order(symbol, 'sell', price) for symbol, price in (
                (loser, '9.90'), (winner, '10.50'), (flat, '10.10'))],
        }}
        friday = buy_spec((marked, unpriced))
        friday['baseline']['close_prices'] = {marked: '10.20'}
        result = weekly_report(ledgers([purchase, sale, {}, {}, friday]))
        summary = result['opportunity_summary']
        self.assertEqual((summary['resolved_count'], summary['flat_count']), (4, 1))
        self.assertEqual((summary['avoided_loss_count'], summary['missed_gain_count']), (1, 2))
        self.assertEqual(summary['realized'], {
            'resolved_count': 3, 'avoided_loss_count': 1, 'missed_gain_count': 1, 'flat_count': 1,
            'avoided_loss_cny': '20.00', 'missed_gain_cny': '40.00',
            'known_filtered_minus_baseline_cny': '-20.00',
        })
        self.assertEqual(summary['friday_marked_open'], {
            'resolved_count': 1, 'avoided_loss_count': 0, 'missed_gain_count': 1, 'flat_count': 0,
            'avoided_loss_cny': '0', 'missed_gain_cny': '15.00',
            'known_filtered_minus_baseline_cny': '-15.00',
        })
        self.assertEqual(summary['known_filtered_minus_baseline_cny'], '-35.00')
        self.assertEqual(summary['unresolved_count'], 1)
        self.assertEqual(summary['unresolved_by_reason'], {'friday_mark_unverified': 1})
        self.assertIsNone(summary['account_ending_nav_gap_cny'])
        self.assertIsNone(summary['unattributed_ending_nav_gap_cny'])

    def test_corporate_action_blocks_attribution_verdict_but_retains_account_result(self):
        action_symbol = '600006.SH'
        records = with_sale(dict.fromkeys((*SYMBOLS, action_symbol), '9.90'))
        records[0]['baseline']['corporate_actions'] = [
            {'symbol': action_symbol, 'cash_delta': '3', 'source_url': 'https://example.invalid/cash-action'},
        ]
        for row in records:
            account = row['baseline']
            for field in ('cash', 'nav'):
                account[field] = str(Decimal(account[field]) + Decimal('3'))
        result = weekly_report(records)
        self.assertEqual(result['verdict'], 'insufficient_evidence')
        self.assertEqual(result['insufficient_evidence_reasons'], ['unresolved_effective_opportunities'])
        self.assertEqual(result['distinct_verified_filtered_symbols'], list(SYMBOLS))
        self.assertEqual(result['comparison'], {'return_diff': '0.00117', 'drawdown_diff': '-0.00117'})
        summary = result['opportunity_summary']
        self.assertEqual(summary['known_filtered_minus_baseline_cny'], '100.00')
        self.assertEqual(summary['account_ending_nav_gap_cny'], '117.00')
        self.assertEqual(summary['unattributed_ending_nav_gap_cny'], '17.00')
        self.assertEqual(summary['unresolved_by_reason'], {'corporate_action_attribution_unverified': 1})

    def test_friday_open_lot_only_deducts_paid_buy_fee(self):
        records = ledgers([{}, {}, {}, {}, buy_spec((SYMBOLS[0],))])
        result = weekly_report(records)
        lot, = result['effective_opportunities']
        self.assertEqual(lot['exit_status'], 'marked_open')
        self.assertEqual(Decimal(lot['net_pnl']), Decimal('-5'))
        self.assertEqual(result['groups']['baseline']['ending_positions'], {SYMBOLS[0]: 100})
        self.assertEqual(result['verdict'], 'insufficient_evidence')

    def test_unpriced_lot_does_not_count_toward_distinct_gate(self):
        spec = buy_spec()
        del spec['baseline']['close_prices'][SYMBOLS[0]]
        result = weekly_report(ledgers([{}, {}, {}, {}, spec]))
        self.assertEqual(result['verdict'], 'insufficient_evidence')
        self.assertIn('missing_close_valuation', result['insufficient_evidence_reasons'])
        self.assertEqual(result['distinct_verified_filtered_symbols'], list(SYMBOLS[1:]))
        self.assertEqual(result['effective_opportunities'][0]['unresolved_reason'], 'friday_mark_unverified')
        self.assertIsNone(result['groups']['baseline']['max_drawdown'])

    def test_missing_day_only_reconciles_consecutive_prefix(self):
        records = with_sale(dict.fromkeys(SYMBOLS, '9.90'))
        result = weekly_report([records[0], *records[2:]])
        self.assertEqual(result['reconciled_dates'], [DATES[0]])
        self.assertEqual(result['missing_dates'], [DATES[1]])
        self.assertEqual(result['unreconciled_available_dates'], list(DATES[2:]))
        self.assertEqual(result['verdict'], 'insufficient_evidence')
        self.assertIsNone(result['comparison'])
        summary = result['opportunity_summary']
        self.assertEqual(summary['reconciled_days'], 1)
        self.assertEqual(summary['unresolved_by_reason'], {'week_incomplete': len(SYMBOLS)})
        self.assertIsNone(summary['known_filtered_minus_baseline_cny'])
        self.assertIsNone(summary['account_ending_nav_gap_cny'])
        self.assertIsNone(weekly_report([])['opportunity_summary']['known_filtered_minus_baseline_cny'])

    def test_account_terminal_gap_survives_an_earlier_valuation_gap(self):
        records = with_sale({SYMBOLS[0]: '9.90'})
        records[0]['baseline']['close_prices'] = {}
        records[0]['baseline']['nav'] = None
        result = weekly_report(records)
        self.assertIsNone(result['comparison'])
        self.assertIsNone(result['groups']['baseline']['max_drawdown'])
        self.assertEqual(result['opportunity_summary']['account_ending_nav_gap_cny'], '20.00')
        self.assertEqual(result['opportunity_summary']['unattributed_ending_nav_gap_cny'], '0.00')

    def test_unavailable_news_or_different_book_is_not_an_effective_filter(self):
        for options, reason in (
            ({'reason': 'news_data_unavailable'}, 'news_data_unavailable'),
            ({'same_book': False}, 'common_book_unverified'),
        ):
            with self.subTest(reason=reason):
                result = weekly_report(with_sale(dict.fromkeys(SYMBOLS, '9.90'), **options))
                self.assertEqual(result['effective_opportunities'], [])
                self.assertIsNone(result['opportunity_summary']['known_filtered_minus_baseline_cny'])
                self.assertEqual({item['reason'] for item in result['excluded_opportunities']}, {reason})
                self.assertEqual(result['verdict'], 'insufficient_evidence')

    def test_selling_then_rebuying_same_symbol_counts_one_distinct_symbol(self):
        symbol = SYMBOLS[0]
        first = buy_spec((symbol,))
        second = buy_spec((symbol,))
        second['baseline']['fills'].insert(0, order(symbol, 'sell', '9.90'))
        sell_decision = {'symbol': symbol, 'side': 'sell', 'reason': 'filled'}
        second['baseline']['decisions'].insert(0, sell_decision)
        third = {'baseline': {'fills': [order(symbol, 'sell', '10.50')], 'decisions': [sell_decision]}}
        records = ledgers([first, second, third, {}, {}])
        result = weekly_report(records)
        self.assertEqual([item['date'] for item in result['effective_opportunities']], list(DATES[:2]))
        self.assertEqual(result['distinct_verified_filtered_symbols'], [symbol])
        records[1]['baseline']['fills'].reverse()
        with self.assertRaisesRegex(ValueError, 'sells must precede buys|overlaps an open lot'):
            weekly_report(records)

    def test_held_only_symbols_require_sell_decisions_not_buy_decisions(self):
        records = with_sale(dict.fromkeys(SYMBOLS, '9.90'))
        records[1]['planned_symbols'] = list(SYMBOLS)
        records[1]['candidate_symbols'] = []
        records[1]['expected_sell_symbols'] = {'baseline': list(SYMBOLS), 'filtered': []}
        result = weekly_report(records)
        self.assertEqual(result['verdict'], 'both_improved')
        self.assertEqual(result['execution_evidence_gaps'], [])
        records[1]['baseline']['decisions'].pop()
        result = weekly_report(records)
        self.assertEqual(result['verdict'], 'insufficient_evidence')
        self.assertEqual(result['execution_evidence_gaps'], [
            {'date': DATES[1], 'group': 'baseline', 'symbol': SYMBOLS[-1],
             'side': 'sell', 'reason': 'decision_missing'},
        ])

    def test_news_evidence_gap_retains_the_decision_error(self):
        spec = buy_spec((SYMBOLS[0],), reason='news_data_unavailable')
        spec['filtered']['decisions'][0]['evaluation'] = {'news_error': 'saved snapshot is late'}
        report = weekly_report(ledgers([spec]))
        self.assertEqual(report['execution_evidence_gaps'], [
            {'date': DATES[0], 'group': 'filtered', 'symbol': SYMBOLS[0], 'side': 'buy',
             'reason': 'news_data_unavailable', 'detail': 'saved snapshot is late'},
        ])

    def test_weekly_quote_evidence_gap_uses_the_matching_direction(self):
        symbol = SYMBOLS[0]
        spec = buy_spec((symbol,), reason='quote_evidence_unverified')
        records = ledgers([spec])
        records[0]['book_evidence_errors'] = {
            'selected': {symbol: 'raw response checksum mismatch'},
            'exit_selected': {symbol: 'different sell error'},
        }
        report = weekly_report(records)
        self.assertEqual(report['execution_evidence_gaps'], [
            {'date': DATES[0], 'group': 'filtered', 'symbol': symbol, 'side': 'buy',
             'reason': 'quote_evidence_unverified', 'detail': 'raw response checksum mismatch'},
        ])

    def test_complete_zero_exclusion_week_cannot_support_verdict(self):
        result = weekly_report(ledgers([{}, {}, {}, {}, {}]))
        self.assertEqual(result['effective_opportunities'], [])
        self.assertEqual(result['verdict'], 'insufficient_evidence')
        self.assertEqual(result['groups']['filtered']['ending_nav'], '100000')
        self.assertEqual(result['comparison'], {'return_diff': '0', 'drawdown_diff': '0'})
        self.assertEqual(result['opportunity_summary']['known_filtered_minus_baseline_cny'], '0')
        self.assertEqual(result['opportunity_summary']['account_ending_nav_gap_cny'], '0')

    def test_unverified_exit_evidence_is_distinct_from_no_buyers(self):
        symbol = SYMBOLS[0]
        for reason in ('corporate_action_or_previous_close_unverified', 'price_limits_unverified', 'no_bid_liquidity'):
            with self.subTest(reason=reason):
                opening = buy_spec((symbol,))
                hold = {'baseline': {'close_prices': {symbol: '9.90'}, 'decisions': [
                    {'symbol': symbol, 'side': 'sell', 'reason': reason},
                ]}}
                result = weekly_report(ledgers([opening, hold]))
                gaps = result['execution_evidence_gaps']
                self.assertEqual(any(gap['reason'] == reason for gap in gaps), reason != 'no_bid_liquidity')

    def test_marks_disagree_on_common_holding(self):
        spec = {
            'baseline': {'fills': [order(SYMBOLS[0], 'buy', '10')], 'close_prices': {SYMBOLS[0]: '10'}},
            'filtered': {'fills': [order(SYMBOLS[0], 'buy', '10')], 'close_prices': {SYMBOLS[0]: '10.01'}},
        }
        result = weekly_report(ledgers([{}, {}, {}, {}, spec]))
        self.assertIn('inconsistent_common_close_prices', result['insufficient_evidence_reasons'])
        self.assertEqual(result['inconsistent_common_close_prices'][0]['symbol'], SYMBOLS[0])

    def test_official_loader_rejects_synthetic_or_outside_evidence(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch).resolve()
            protocol = root / 'protocol.md'
            protocol.write_bytes((BASE / 'protocol.md').read_bytes())
            self.assertEqual(official_ledgers(root), ([], []))
            day = root / DATES[0]
            day.mkdir()
            daily = day / 'daily.json'
            daily.write_text(json.dumps({'mode': 'synthetic', 'date': DATES[0]}))
            with self.assertRaisesRegex(ValueError, 'official forward'):
                official_ledgers(root)
            daily.unlink()
            outside = root / 'external.json'
            outside.write_text(json.dumps({'mode': 'forward', 'date': DATES[0]}))
            daily.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, 'not canonical'):
                official_ledgers(root)

    def test_matching_partials_preserve_verified_fees_and_conflicts_are_reported(self):
        spec = buy_spec((SYMBOLS[0],))
        spec['baseline']['close_prices'] = {}
        record = ledgers([spec])[0]
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch).resolve()
            (root / 'protocol.md').write_bytes((BASE / 'protocol.md').read_bytes())
            parent = root / DATES[0] / 'close-partials'
            for digest in ('a' * 64, 'b' * 64):
                partial = parent / digest / 'daily.json'
                partial.parent.mkdir(parents=True)
                partial.write_text(json.dumps(record))
                loaded, gaps = official_ledgers(root)
                self.assertEqual((loaded, gaps), ([record], []))
                result = weekly_report(loaded, artifact_gaps=gaps)
                self.assertEqual(result['groups']['baseline']['fees'], '5')
                self.assertIsNone(result['groups']['baseline']['ending_nav'])
                self.assertEqual(result['verdict'], 'insufficient_evidence')
            conflict = {**record, 'baseline': {**record['baseline'], 'decisions': []}}
            partial.write_text(json.dumps(conflict))
            self.assertEqual(official_ledgers(root), ([], [
                {'date': DATES[0], 'reason': 'ambiguous_partial_ledgers', 'count': 2},
            ]))

    def test_cash_only_partial_requires_its_archived_marks_error(self):
        day = DATES[0]
        record = ledgers([{}])[0]
        daily, summary = close_report(record)
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch).resolve()
            (root / 'protocol.md').write_bytes((BASE / 'protocol.md').read_bytes())
            folder = root / day
            issue = {'path': str(folder / 'close_marks.json'), 'error': 'invalid JSON'}
            stored = save_report(folder, daily, {
                **summary, 'status': 'valuation_incomplete', 'input_errors': {'close_marks': issue},
            }, official=True)
            rows, gaps = official_ledgers(root)
            self.assertEqual(rows, [daily])
            self.assertEqual(gaps, [{'date': day, 'reason': 'close_marks_unverified', 'details': issue}])
            result = weekly_report(rows, artifact_gaps=gaps)
            self.assertEqual(result['groups']['baseline']['ending_nav'], '100000')
            self.assertEqual(result['groups']['filtered']['fees'], '0')
            self.assertEqual(result['ledger_artifact_gaps'], gaps)
            path = Path(stored['artifacts']['summary'])
            stored['input_errors']['close_marks']['error'] = 'edited error'
            path.write_text(json.dumps(stored))
            with self.assertRaisesRegex(ValueError, 'verified cash-only marks error'):
                official_ledgers(root)


if __name__ == '__main__':
    unittest.main()
