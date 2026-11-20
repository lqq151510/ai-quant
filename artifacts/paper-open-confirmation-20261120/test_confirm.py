"""Test the opening-confirmation candidate and its original-intent constraints."""

import unittest
from decimal import Decimal

from confirm import SESSIONS, SOURCE, compare_archive, compare_sessions, opening_signal


BOOK = {'bid': '9.99', 'ask': '10.01', 'previous': '10.00', 'bid_units': 100, 'ask_units': 100}
SYMBOL = '600000.SH'
ARCHIVE_AVAILABLE = any((SOURCE / str(day)).exists() for day in SESSIONS)


def fill(side, units, price):
    delta = Decimal(price) * units * (-1 if side == 'buy' else 1) - 5
    return {'symbol': SYMBOL, 'side': side, 'units': units, 'price': price, 'cash_delta': str(delta),
            'commission': '5', 'stamp_tax': '0', 'transfer_fee': '0', 'fees': '5'}


def session(day, order, book, cash, positions, previous_close='10.00'):
    prices = {symbol: '10' for symbol in positions}
    nav = Decimal(cash) + sum(Decimal(prices[code]) * quantity for code, quantity in positions.items())
    return day, {'fills': [order], 'cash': cash, 'positions': positions, 'close_prices': prices,
                 'nav': str(nav), 'fees': order['fees']}, [(order, book, previous_close)]


class ConfirmationTests(unittest.TestCase):
    def test_equality_passes_and_a_strictly_lower_midpoint_blocks(self):
        self.assertEqual(opening_signal(BOOK, '10.000'), ('confirmed', Decimal('10.00')))
        self.assertEqual(opening_signal({**BOOK, 'bid': '9.98'}, '10'),
                         ('below_previous_close', Decimal('9.995')))

    def test_missing_crossed_or_inconsistent_evidence_is_unknown(self):
        for book, close in (
            ({}, '10'), ({**BOOK, 'bid': 'NaN'}, '10'),
            ({**BOOK, 'bid': '10.02'}, '10'), ({**BOOK, 'bid_units': 0}, '10'),
            (BOOK, None), ({**BOOK, 'previous': '9.98'}, '10'),
        ):
            with self.subTest(book=book, close=close):
                self.assertEqual(opening_signal(book, close), ('unknown', None))

    def test_unknown_leaves_original_order_and_exit_unchanged(self):
        buy = session('2026-11-16', fill('buy', 100, '10'), None, '98995', {SYMBOL: 100})
        sell = session('2026-11-17', fill('sell', 100, '10.1'), BOOK, '100000', {})
        result = compare_sessions([buy, sell])
        self.assertEqual(result['signal_counts'], {'confirmed': 0, 'below_previous_close': 0, 'unknown': 1})
        self.assertEqual(result['summary']['original'], result['summary']['confirmation'])
        self.assertEqual(result['blocked_outcomes']['trades'], [])

    def test_blocked_lot_does_not_generate_a_shadow_exit(self):
        buy = session('2026-11-16', fill('buy', 100, '10'), {**BOOK, 'bid': '9.98'}, '98995', {SYMBOL: 100})
        sell = session('2026-11-17', fill('sell', 100, '10.1'), BOOK, '100000', {})
        result = compare_sessions([buy, sell])
        self.assertEqual(result['summary']['confirmation']['fill_count'], 0)
        self.assertEqual(result['blocked_outcomes']['closed']['flat_count'], 1)
        self.assertEqual(Decimal(result['blocked_outcomes']['net_pnl_difference']), Decimal(0))

    def test_pilot_shape_without_aggregate_fees_and_equal_cash_control(self):
        buy = session('2026-11-16', fill('buy', 100, '10'), BOOK, '98995', {SYMBOL: 100})
        del buy[1]['fees']
        result = compare_sessions([buy])
        self.assertEqual(result['daily'][0]['original']['fees'], '5')
        self.assertEqual(result['summary']['cash_control']['ending_nav'], '100000')
        self.assertEqual(result['summary']['cash_control']['net_return'], '0')

    def test_refuses_misaligned_exit_and_insufficient_funding(self):
        buy = session('2026-11-16', fill('buy', 100, '10'), BOOK, '98995', {SYMBOL: 100})
        mismatched = session('2026-11-17', fill('sell', 200, '10.1'), BOOK, '101010', {})
        with self.assertRaisesRegex(ValueError, 'does not align to shadow shares'):
            compare_sessions([buy, mismatched])
        oversized = session('2026-11-16', fill('buy', 10000, '10'), BOOK, '-5', {SYMBOL: 10000})
        with self.assertRaisesRegex(ValueError, 'insufficient shadow cash'):
            compare_sessions([oversized])

    @unittest.skipUnless(ARCHIVE_AVAILABLE, 'local original-market archive is absent')
    def test_archived_intents_and_skipped_trade_attribution_reconcile(self):
        result = compare_archive()
        baseline = result['summary']['original']
        confirmation = result['summary']['confirmation']
        self.assertEqual((baseline['buy_count'], confirmation['buy_count']), (21, 6))
        self.assertEqual((baseline['sell_count'], confirmation['sell_count']), (16, 4))
        self.assertEqual(Decimal(baseline['ending_nav']), Decimal('98751.590'))
        self.assertEqual(Decimal(confirmation['ending_nav']), Decimal('100051.390'))
        outcomes = result['blocked_outcomes']
        difference = Decimal(confirmation['ending_nav']) - Decimal(baseline['ending_nav'])
        self.assertEqual(Decimal(outcomes['net_pnl_difference']), difference)
        self.assertEqual(len(outcomes['trades']), result['signal_counts']['below_previous_close'])
        self.assertEqual(outcomes['terminal_marked']['loss_count'], 3)

    def test_an_unsettled_pilot_has_no_performance_observation(self):
        result = compare_sessions([])
        self.assertEqual(result['state'], 'awaiting_first_settlement')
        self.assertEqual(result['settled_sessions'], 0)
        self.assertIsNone(result['summary'])
        self.assertIsNone(result['blocked_outcomes'])


if __name__ == '__main__':
    unittest.main()
