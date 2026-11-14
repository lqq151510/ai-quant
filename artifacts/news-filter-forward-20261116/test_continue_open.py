"""Exercise next-session exits and independent balances under the frozen rules."""
import unittest
from datetime import date

from continue_open import run_day

DAY = date(2026, 11, 17)
STOCK = '601988.SH'
ETF = '513100.SH'
PREPARED = {'coverage': 'scenario', 'candidates': [
    {'symbol': STOCK, 'tick': '.01', 'kind': 'stock', 'cross_border': False, 'previous_close': '7.18'},
]}
NEWS = {'batches': [], 'errors': [], 'sources': []}


def prior(symbol=STOCK, nav='100000', close='7.18'):
    return {
        'baseline': {'positions': {symbol: 100}, 'close_prices': {symbol: close}, 'cash': '99282', 'nav': nav},
        'filtered': {'positions': {}, 'close_prices': {}, 'cash': '100000', 'nav': '100000'},
    }


def quote(**overrides):
    return {
        'at': '2026-11-17T09:45:10+08:00', 'received_at': '2026-11-17T09:45:11+08:00',
        'bid': '7.18', 'ask': '7.18', 'bid_units': 100, 'ask_units': 10000, 'previous': '7.18',
        'lower_limit': '6.46', 'upper_limit': '7.90',
        'source_url': 'https://qt.gtimg.cn/q=sh601988',
        **overrides,
    }


def process(book, previous=None):
    return run_day(DAY, previous or prior(), PREPARED,
                   {'selected': {STOCK: book}, 'unavailable': {}}, NEWS)


class ContinuationScenarios(unittest.TestCase):
    def test_exit_then_buy(self):
        opening = process(quote())
        self.assertEqual([f['side'] for f in opening['baseline']['fills']], ['sell', 'buy'])
        self.assertEqual([f['side'] for f in opening['filtered']['fills']], ['buy'])
        self.assertEqual(opening['baseline']['fills'][0]['fees'], '5.37')

    def test_failed_exit_preserves_holdings_without_double_entry(self):
        for fields, reason in (
            ({'bid_units': 0}, 'no_bid_liquidity'),
            ({'bid_units': 99}, 'insufficient_bid_units'),
            ({'bid': '6.46'}, 'sell_outside_price_limits'),
            ({'previous': '7.01'}, 'corporate_action_or_previous_close_unverified'),
        ):
            with self.subTest(reason=reason):
                group = process(quote(**fields))['baseline']
                self.assertEqual([d['reason'] for d in group['decisions']], [reason, 'already_held'])
                self.assertEqual(group['positions'], {STOCK: 100})
                self.assertFalse(group['fills'])

    def test_missing_nav_does_not_prevent_verified_exit(self):
        opening = process(quote(), prior(nav=None))
        self.assertEqual([d['reason'] for d in opening['baseline']['decisions']],
                         ['filled', 'previous_nav_unverified'])
        self.assertEqual(opening['baseline']['positions'], {})
        self.assertEqual(len(opening['filtered']['fills']), 1)

    def test_source_backed_close_recovers_held_only_exit_when_ledger_mark_is_missing(self):
        previous = prior(nav=None, close=None)
        prepared = {'coverage': 'scenario', 'candidates': [], 'reviewed': [
            {'symbol': STOCK, 'previous_close': '7.18', 'status': 'eligible',
             'sources': ['https://web.ifzq.gtimg.cn/raw', 'https://quotes.sina.cn/raw']},
        ]}
        capture = {'selected': {}, 'exit_selected': {STOCK: quote()}, 'unavailable': {}}
        opening = run_day(DAY, previous, prepared, capture, NEWS)
        self.assertEqual(opening['prior_closes'][STOCK], '7.18')
        self.assertEqual([fill['side'] for fill in opening['baseline']['fills']], ['sell'])
        self.assertEqual(opening['baseline']['positions'], {})

    def test_shared_prior_close_uses_agreement_and_rejects_conflicts(self):
        for a, b, expected in ((None, '7.18', 'filled'), ('7.180', '7.18', 'filled'),
                               ('7.17', '7.18', 'corporate_action_or_previous_close_unverified')):
            with self.subTest(marks=(a, b)):
                previous = prior(nav=None, close=a)
                previous['filtered'] = prior(nav=None, close=b)['baseline']
                opening = run_day(DAY, previous, None, {'selected': {STOCK: quote()}, 'unavailable': {}}, NEWS)
                for group in ('baseline', 'filtered'):
                    self.assertEqual(opening[group]['decisions'][0]['reason'], expected)
        self.assertEqual(process(quote(), prior(close='7.17'))['baseline']['decisions'][0]['reason'],
                         'corporate_action_or_previous_close_unverified')

    def test_etf_exit_does_not_depend_on_entry_iopv(self):
        etf_book = quote(bid='2.23', previous='2.23', lower_limit='2.00', upper_limit='2.45')
        opening = run_day(DAY, prior(ETF, close='2.23'), None,
                          {'selected': {}, 'exit_selected': {ETF: etf_book}, 'unavailable': {}}, NEWS)
        self.assertEqual(opening['baseline']['fills'][0]['side'], 'sell')
        self.assertEqual(opening['baseline']['fills'][0]['fees'], '5.0')
        self.assertTrue(opening['candidate_data_missing'])

    def test_entry_does_not_reuse_quote_received_before_exit(self):
        entry = quote(received_at='2026-11-17T09:45:11+08:00')
        exit_book = quote(at='2026-11-17T09:45:13+08:00', received_at='2026-11-17T09:45:14+08:00')
        opening = run_day(DAY, prior(), PREPARED,
                          {'selected': {STOCK: entry}, 'exit_selected': {STOCK: exit_book}, 'unavailable': {}},
                          NEWS)
        self.assertEqual([row['reason'] for row in opening['baseline']['decisions']],
                         ['filled', 'entry_quote_precedes_exit'])
        self.assertEqual([fill['side'] for fill in opening['filtered']['fills']], ['buy'])

    def test_source_outage_and_valid_empty_shortlist_are_distinct(self):
        books = {'selected': {}, 'unavailable': {}}
        for reviewed, missing in (([{'status': 'data_gap'}], True), ([{'status': 'signal_not_triggered'}], False)):
            prepared = {**PREPARED, 'reviewed': reviewed, 'candidates': []}
            result = run_day(DAY, prior(), prepared, books, NEWS)
            self.assertEqual(result['candidate_data_missing'], missing)


if __name__ == '__main__':
    unittest.main()
