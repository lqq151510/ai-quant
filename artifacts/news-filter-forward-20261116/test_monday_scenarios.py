"""Exercise Monday's frozen simulation with synthetic order books and news.

Run from the project root with:
    PYTHONPATH=research_v1 research_v1/.venv/bin/python -m unittest discover \
      -s artifacts/news-filter-forward-20261116 -p test_monday_scenarios.py -v

These fixtures perform no network requests and produce no official ledgers.
"""

import unittest
from decimal import Decimal
from pathlib import Path

from close_monday import etf_decision
from collect_books import parse_books
from monday_open import run
from reconcile import reconcile


ETF = '513100.SH'
STOCK = '601988.SH'
GROUPS = ('baseline', 'filtered')
ETF_ROW = dict(symbol=ETF, kind='etf', cross_border=True, previous_close='1.010', tick='.001')
STOCK_ROW = dict(symbol=STOCK, kind='stock', cross_border=False, previous_close='10.00', tick='.01')


def stock_book(**overrides):
    return {
        'at': '2026-11-16T09:45:10+08:00',
        'received_at': '2026-11-16T09:45:11+08:00',
        'source_url': 'https://hq.sinajs.cn/list=sh601988',
        'raw_file': '001-sina.txt',
        'previous': '10.00',
        'ask': '10.00',
        'ask_units': 100_000,
        **overrides,
    }


def etf_book(**overrides):
    return {
        'at': '2026-11-16T09:45:20+08:00',
        'received_at': '2026-11-16T09:45:21+08:00',
        'source_url': 'https://qt.gtimg.cn/q=sh513100',
        'raw_file': '003-tencent.txt',
        'previous': '1.010',
        'ask': '1.009',
        'ask_units': 100_000,
        'iopv': {'status': 'verified_snapshot', 'nav': '1.0000', 'vendor_premium_pct': '0.90'},
        'iopv_update': {
            'previous_at': '2026-11-16T09:45:10+08:00',
            'previous_received_at': '2026-11-16T09:45:11+08:00',
            'previous_nav': '0.9999',
            'previous_raw_file': '002-tencent.txt',
        },
        **overrides,
    }


def news_batch(observed_at, symbol=STOCK):
    return {'date': '2026-11-16', 'events': [{
        'event_id': 'synthetic-event',
        'symbols': [symbol],
        'title': '股票交易异常波动公告',
        'first_observed_at': observed_at,
        'source_url': 'https://example.invalid/synthetic-event',
    }]}


def simulate(selected, unavailable=None, news=None, rows=None, **options):
    candidates = [ETF_ROW, STOCK_ROW] if rows is None else rows
    prepared = {'candidates': candidates, 'coverage': 'synthetic_scenarios'}
    books = {'selected': selected, 'unavailable': unavailable or {}}
    return run(prepared, books, news or [], **options)


def decision(result, group, symbol):
    return next(row for row in result[group]['decisions'] if row['symbol'] == symbol)


class MondayScenarios(unittest.TestCase):
    def assert_paired_reason(self, result, symbol, reason):
        for group in GROUPS:
            self.assertEqual(decision(result, group, symbol)['reason'], reason)

    def assert_stock_still_fills(self, result):
        self.assert_paired_reason(result, STOCK, 'filled')
        for group in GROUPS:
            self.assertEqual([fill['symbol'] for fill in result[group]['fills']], [STOCK])

    def test_missing_etf_quote_or_reference_keeps_stock_executable(self):
        for reason in (
            'quote_unavailable',
            'quote_stale',
            'iopv_missing',
            'iopv_premium_mismatch',
            'iopv_intraday_update_unverified',
        ):
            with self.subTest(reason=reason):
                result = simulate({STOCK: stock_book()}, {ETF: reason})
                self.assert_paired_reason(result, ETF, reason)
                self.assert_stock_still_fills(result)

    def test_etf_requires_verified_intraday_reference(self):
        prior = etf_book()['iopv_update']
        cases = [
            ({'iopv_update': None}, 'iopv_intraday_update_unverified'),
            ({'iopv': None}, 'iopv_missing'),
            ({'iopv': []}, 'iopv_missing'),
            ({'iopv': {'status': 'iopv_missing'}}, 'iopv_missing'),
            ({'source_url': 'https://hq.sinajs.cn/list=sh513100'}, 'premium_not_verified'),
            ({'at': '2026-11-16T09:44:59+08:00'}, 'quote_stale'),
            ({'iopv_update': {**prior, 'previous_received_at': '2026-11-16T09:45:21+08:00'}},
             'iopv_intraday_update_unverified'),
            ({'iopv_update': {**prior, 'previous_nav': '1.000'}}, 'iopv_intraday_update_unverified'),
            ({'iopv_update': {**prior, 'previous_raw_file': '003-tencent.txt'}},
             'iopv_intraday_update_unverified'),
        ]
        for overrides, reason in cases:
            with self.subTest(reason=reason):
                result = simulate({ETF: etf_book(**overrides), STOCK: stock_book()})
                self.assert_paired_reason(result, ETF, reason)
                self.assert_stock_still_fills(result)
                for group in GROUPS:
                    self.assertIsNone(decision(result, group, ETF)['premium'])

    def test_etf_premium_uses_slipped_buy_price_and_includes_exact_limit(self):
        # A 1.009 quote slips and rounds to 1.010. A 1.010 quote reaches 1.011.
        for ask, reason, premium in (
            ('1.008', 'filled', '.009'),
            ('1.009', 'filled', '.010'),
            ('1.010', 'premium_above_limit', '.011'),
        ):
            with self.subTest(ask=ask):
                result = simulate({ETF: etf_book(ask=ask), STOCK: stock_book()})
                self.assert_paired_reason(result, ETF, reason)
                self.assert_paired_reason(result, STOCK, 'filled')
                for group in GROUPS:
                    self.assertEqual(Decimal(decision(result, group, ETF)['premium']), Decimal(premium))

    def test_verified_premium_still_needs_enough_ask_units(self):
        for available, reason in ((4899, 'insufficient_ask_units'), (4900, 'filled')):
            with self.subTest(available=available):
                result = simulate({ETF: etf_book(ask_units=available), STOCK: stock_book()})
                self.assert_paired_reason(result, ETF, reason)
                self.assert_paired_reason(result, STOCK, 'filled')
                for group in GROUPS:
                    row = decision(result, group, ETF)
                    self.assertEqual(row['available_units'], available)
                    summary = etf_decision(result[group])
                    self.assertEqual(summary['status'], reason)
                    self.assertEqual(Decimal(summary['premium']), Decimal('.010'))
                    self.assertEqual((summary['required_units'], summary['available_units']), (4900, available))

    def test_bad_etf_numbers_do_not_abort_other_candidates(self):
        for override, reason in (
            ({'ask': 'NaN'}, 'invalid_quote'),
            ({'ask_units': -1}, 'invalid_quote'),
            ({'iopv': {'status': 'verified_snapshot', 'nav': '0'}}, 'iopv_invalid'),
            ({'iopv': {'status': 'verified_snapshot', 'nav': 'NaN'}}, 'iopv_invalid'),
        ):
            with self.subTest(override=override):
                result = simulate({ETF: etf_book(**override), STOCK: stock_book()})
                self.assert_paired_reason(result, ETF, reason)
                self.assert_stock_still_fills(result)
                self.assertIsNone(decision(result, 'baseline', ETF)['premium'])

    def test_quote_age_capture_window_depth_limit_and_lot(self):
        cases = [
            ({'received_at': '2026-11-16T09:45:40+08:00'}, 'filled'),
            ({'received_at': '2026-11-16T09:45:41+08:00'}, 'quote_stale'),
            ({'at': '2026-11-16T09:45:12+08:00'}, 'quote_from_future'),
            ({'received_at': '2026-11-16T09:50:01+08:00'}, 'outside_window'),
            ({'ask': '0', 'ask_units': 0}, 'no_ask_liquidity'),
            ({'ask_units': 399}, 'insufficient_ask_units'),
            ({'ask_units': 400}, 'filled'),
            ({'ask': '10.10'}, 'ask_above_limit'),
            ({'previous': '9.99'}, 'invalid_ask_or_previous_close_mismatch'),
        ]
        for overrides, reason in cases:
            with self.subTest(reason=reason, overrides=overrides):
                result = simulate({STOCK: stock_book(**overrides)}, rows=[STOCK_ROW])
                self.assert_paired_reason(result, STOCK, reason)
        pricey = {**STOCK_ROW, 'previous_close': '50.00'}
        result = simulate({STOCK: stock_book(previous='50.00', ask='50.00')}, rows=[pricey])
        self.assert_paired_reason(result, STOCK, 'budget_below_one_lot')

    def test_news_only_blocks_matching_visible_event(self):
        cases = [
            ('2026-11-13T09:45:11+08:00', STOCK, 'news_abnormal_volatility'),
            ('2026-11-13T09:45:10+08:00', STOCK, 'filled'),
            ('2026-11-16T09:45:11+08:00', STOCK, 'news_abnormal_volatility'),
            ('2026-11-16T09:45:12+08:00', STOCK, 'filled'),
            ('2026-11-16T09:45:11+08:00', '600000.SH', 'filled'),
        ]
        for observed, symbol, expected in cases:
            with self.subTest(observed=observed, symbol=symbol):
                result = simulate({ETF: etf_book(), STOCK: stock_book()},
                                  news=[news_batch(observed, symbol)])
                self.assertEqual(decision(result, 'baseline', STOCK)['reason'], 'filled')
                self.assertEqual(decision(result, 'filtered', STOCK)['reason'], expected)
                self.assert_paired_reason(result, ETF, 'filled')

    def test_news_must_be_observed_by_snapshot_without_reanchoring_the_72_hour_window(self):
        captured_at = '2026-11-16T09:45:02+08:00'
        for event_time, expected in (
            ('2026-11-16T09:45:02+08:00', 'news_abnormal_volatility'),
            ('2026-11-16T09:45:03+08:00', 'filled'),
            ('2026-11-13T09:45:07+08:00', 'filled'),
        ):
            with self.subTest(event_time=event_time):
                result = simulate({STOCK: stock_book()}, news=[news_batch(event_time)],
                                  news_captured_at=captured_at)
                self.assertEqual(decision(result, 'baseline', STOCK)['reason'], 'filled')
                self.assertEqual(decision(result, 'filtered', STOCK)['reason'], expected)

    def test_cross_day_duplicate_cannot_restart_the_72_hour_window(self):
        friday = news_batch('2026-11-13T09:45:10+08:00')
        weekend = news_batch('2026-11-15T18:00:00+08:00')
        friday['date'], weekend['date'] = '2026-11-13', '2026-11-15'
        for batches in ([friday, weekend], [weekend, friday]):
            result = simulate({STOCK: stock_book()}, news=batches)
            self.assert_paired_reason(result, STOCK, 'filled')
        result = simulate({STOCK: stock_book()}, news=[weekend])
        self.assertEqual(decision(result, 'filtered', STOCK)['reason'], 'news_abnormal_volatility')

    def test_duplicate_first_observation_is_stable_across_timezones(self):
        local = news_batch('2026-11-15T18:00:00+08:00')
        utc = news_batch('2026-11-15T10:00:00+00:00')
        for batches in ([local, utc], [utc, local]):
            result = simulate({STOCK: stock_book()}, news=batches)
            matched = decision(result, 'filtered', STOCK)['evaluation']['news']['matched']
            self.assertEqual([row['first_observed_at'] for row in matched], ['2026-11-15T18:00:00+08:00'])

    def test_distinct_announcement_is_retained_and_future_bad_fields_are_ignored(self):
        friday = news_batch('2026-11-13T09:45:10+08:00')
        weekend = news_batch('2026-11-15T18:00:00+08:00')
        weekend['events'][0]['event_id'] = 'new-announcement'
        future = news_batch('2026-11-16T18:00:00+08:00')
        future['events'][0]['published_at'] = 'bad-time'
        result = simulate({STOCK: stock_book()}, news=[future, weekend, friday])
        filtered = decision(result, 'filtered', STOCK)
        self.assertEqual(filtered['reason'], 'news_abnormal_volatility')
        matched = filtered['evaluation']['news']['matched']
        self.assertEqual([row['event_id'] for row in matched], ['new-announcement'])

    def test_bad_news_for_one_symbol_preserves_baseline_and_unaffected_symbol(self):
        broken = news_batch('bad-timestamp')
        result = simulate({ETF: etf_book(), STOCK: stock_book()}, news=[broken])
        self.assertEqual(decision(result, 'baseline', STOCK)['reason'], 'filled')
        self.assertEqual(decision(result, 'filtered', STOCK)['reason'], 'news_data_unavailable')
        self.assert_paired_reason(result, ETF, 'filled')

    def test_news_file_failure_preserves_baseline_but_blocks_unverified_comparison(self):
        result = simulate({ETF: etf_book(), STOCK: stock_book()},
                          news_errors=['unreadable synthetic news file'])
        for symbol in (ETF, STOCK):
            self.assertEqual(decision(result, 'baseline', symbol)['reason'], 'filled')
            self.assertEqual(decision(result, 'filtered', symbol)['reason'], 'news_data_unavailable')

    def test_first_day_paired_fills_reconcile_costs_and_nav(self):
        result = simulate({ETF: etf_book(), STOCK: stock_book()},
                          news=[news_batch('2026-11-13T18:00:00+08:00')])
        self.assertEqual(result['baseline']['positions'], {ETF: 4900, STOCK: 400})
        self.assertEqual(result['filtered']['positions'], {ETF: 4900})
        closes = {ETF: '1.005', STOCK: '9.80'}
        expected_nav = {'baseline': '99881.46', 'filtered': '99970.50'}
        daily = {'date': result['date']}
        for group in GROUPS:
            account = result[group]
            self.assertTrue(all(fill['side'] == 'buy' for fill in account['fills']))
            daily[group] = {**account, 'close_prices': closes, 'nav': expected_nav[group]}
        report = reconcile([daily])['groups']
        self.assertEqual(Decimal(report['baseline']['fees']), Decimal('10.04'))
        self.assertEqual(Decimal(report['filtered']['fees']), Decimal('5'))
        for group in GROUPS:
            self.assertEqual(Decimal(report[group]['ending_nav']), Decimal(expected_nav[group]))
        self.assertEqual(Decimal(report['baseline']['max_drawdown']), Decimal('.0011854'))
        self.assertEqual(Decimal(report['filtered']['max_drawdown']), Decimal('.000295'))

    def test_missing_close_does_not_become_cash_only_nav(self):
        result = simulate({STOCK: stock_book()}, rows=[STOCK_ROW])
        daily = {
            'date': result['date'],
            **{group: {**result[group], 'close_prices': {}, 'nav': None} for group in GROUPS},
        }
        report = reconcile([daily])['groups']
        for group in GROUPS:
            self.assertEqual(Decimal(report[group]['fees']), Decimal('5.04'))
            for metric in ('ending_nav', 'net_return', 'max_drawdown'):
                self.assertIsNone(report[group][metric])


class CollectorScenarios(unittest.TestCase):
    symbols = {'sh513100': ETF, 'sh601988': STOCK}
    fixture_dir = Path(__file__).resolve().parent / 'monday-preparation/quote-probe'

    def archived_quote(self, filename):
        try:
            return (self.fixture_dir / filename).read_bytes()
        except FileNotFoundError:
            self.skipTest(f'local archive missing: {self.fixture_dir.name}/{filename}')

    def test_friday_both_sources_preserve_symbol_and_quantity_units(self):
        sina_raw = self.archived_quote('001-sina.txt')
        tencent_raw = self.archived_quote('002-tencent.txt')
        sina, sina_errors = parse_books('sina', sina_raw, self.symbols)
        tencent, tencent_errors = parse_books('tencent', tencent_raw, self.symbols)
        self.assertEqual(sina_errors, {})
        self.assertEqual(tencent_errors, {})
        self.assertEqual(set(sina), {ETF, STOCK})
        self.assertEqual(set(tencent), {ETF, STOCK})
        self.assertTrue(all(book['at'].startswith('2026-11-13') for book in tencent.values()))
        for symbol in self.symbols.values():
            for side in ('bid', 'ask'):
                self.assertEqual(Decimal(sina[symbol][side]), Decimal(tencent[symbol][side]))
                field = f'{side}_units'
                self.assertLessEqual(abs(sina[symbol][field] - tencent[symbol][field]), 100)
        self.assertEqual(tencent[ETF]['iopv']['status'], 'verified_snapshot')

    def test_bad_or_duplicate_etf_row_does_not_poison_other_quote(self):
        payload = self.archived_quote('002-tencent.txt')
        original, _ = parse_books('tencent', payload, self.symbols)
        etf_row = next(line for line in payload.splitlines() if line.startswith(b'v_sh513100='))
        valid_mixed_rows = payload.replace(etf_row, b'v_sh513100="bad";')
        for mixed in (valid_mixed_rows, payload + b'\n' + etf_row):
            with self.subTest(duplicate=mixed.endswith(etf_row)):
                books, errors = parse_books('tencent', mixed, self.symbols)
                self.assertEqual(books, {STOCK: original[STOCK]})
                self.assertEqual(set(errors), {ETF})


if __name__ == '__main__':
    unittest.main()
