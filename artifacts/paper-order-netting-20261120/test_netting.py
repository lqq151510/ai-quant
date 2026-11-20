"""Tests for the fixed-intent netting candidate using archived paper evidence."""

import unittest
from decimal import Decimal

from archive import SOURCE, compare, SESSIONS, verified_session
from netting import net_session, reprice


ARCHIVE_AVAILABLE = any((SOURCE / str(day)).exists() for day in SESSIONS)


def fixture_session(day):
    symbols = ('000001.SZ', '000651.SZ', '512800.SH', '600000.SH', '600036.SH', '512480.SH')
    units = 400
    book = {'bid': '10', 'ask': '10.01', 'bid_units': units, 'ask_units': units,
            'source_url': 'fixture', 'raw_file': 'book.json', 'raw_sha256': 'fixture',
            'received_at': '2026-11-17T09:46:00+08:00'}
    orders = [
        (reprice(day, {'symbol': symbol, 'side': side, 'price': price}, units), book, '10')
        for side, price, selected in (('sell', '9.99', symbols), ('buy', '10.02', symbols[:-1]))
        for symbol in selected
    ]
    prior = {'cash': '70000', 'nav': '100000', 'positions': dict.fromkeys(symbols, units)}
    return orders, prior


class NettingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.day = SESSIONS[1]
        if ARCHIVE_AVAILABLE:
            cls.result = compare()
            _, cls.orders, _ = verified_session(cls.day, {})
            cls.prior = cls.result['daily'][0]['original']
        else:
            cls.orders, cls.prior = fixture_session(cls.day)

    def run_session(self, orders=None, **overrides):
        account = dict(prior_cash=self.prior['cash'], prior_nav=self.prior['nav'],
                       positions=self.prior['positions'], candidate_count=10)
        return net_session(self.day, self.orders if orders is None else orders, **(account | overrides))

    @unittest.skipUnless(ARCHIVE_AVAILABLE, 'local original-market archive is absent')
    def test_reconciles_existing_books_and_prices(self):
        result = self.result
        original, netting = (result['summary'][name] for name in ('original', 'netting'))
        self.assertEqual((original['fill_count'], netting['fill_count'], result['matched_pairs']), (37, 18, 10))
        self.assertEqual(Decimal(result['savings']['total']), Decimal('234.38'))
        self.assertEqual(Decimal(result['savings']['fees']), Decimal('114.48'))
        self.assertEqual(Decimal(result['savings']['execution']), Decimal('119.90'))
        for session in result['daily']:
            self.assertEqual(session['original']['positions'], session['netting']['positions'])
        self.assertEqual(Decimal(netting['ending_nav']) - Decimal(original['ending_nav']), Decimal('234.38'))

    @unittest.skipUnless(ARCHIVE_AVAILABLE, 'local original-market archive is absent')
    def test_reprices_residual_with_minimum_commission(self):
        final = self.result['daily'][-1]['netting']
        pair, = [pair for pair in final['matches'] if pair['symbol'] == '512800.SH']
        self.assertEqual((pair['sell_units'], pair['buy_units'], pair['residual_units']), (5400, 5300, 100))
        fill, = [fill for fill in final['fills'] if fill['symbol'] == pair['symbol']]
        self.assertEqual((fill['side'], fill['units'], Decimal(fill['fees']), Decimal(fill['cash_delta'])),
                         ('sell', 100, Decimal('5'), Decimal('86.40')))

    def test_different_quote_is_not_used_to_cancel_earlier_sell(self):
        symbol = '000001.SZ'
        orders = [(fill, {**book, 'received_at': '2026-11-17T09:46:08+08:00'}, close)
                  if (fill['symbol'], fill['side']) == (symbol, 'buy') else (fill, book, close)
                  for fill, book, close in self.orders]
        output = self.run_session(orders)
        self.assertNotIn(symbol, {pair['symbol'] for pair in output['matches']})
        self.assertEqual(len(output['matches']), 4)

    def test_later_seller_batch_forces_unchanged_original_trades(self):
        orders = [(fill, {**book, 'received_at': '2026-11-17T09:46:08+08:00'}, close)
                  if (fill['symbol'], fill['side']) == ('512480.SH', 'sell') else (fill, book, close)
                  for fill, book, close in self.orders]
        output = self.run_session(orders)
        self.assertEqual(output['gate'], 'exit_batch_not_shared')
        self.assertEqual(output['fills'], [order for order, _, _ in orders])

    def test_missing_exit_forces_unchanged_original_trades(self):
        orders = [(fill, book, close) for fill, book, close in self.orders
                  if (fill['symbol'], fill['side']) != ('512480.SH', 'sell')]
        output = self.run_session(orders)
        self.assertEqual(output['gate'], 'exit_batch_incomplete')
        self.assertEqual(output['fills'], [order for order, _, _ in orders])

    def test_cash_reserve_excludes_proceeds_of_planned_exits(self):
        output = self.run_session(prior_cash='30000', shadow_cash='80000')
        self.assertEqual(output['gate'], 'insufficient_cash_without_exit_proceeds')
        self.assertEqual(output['fills'], [order for order, _, _ in self.orders])

    def test_validates_candidate_count_and_buy_budget(self):
        with self.assertRaisesRegex(ValueError, 'funding bound'):
            self.run_session(candidate_count=1)
        with self.assertRaisesRegex(ValueError, 'funding bound'):
            self.run_session(prior_nav='50000')


if __name__ == '__main__':
    unittest.main()
