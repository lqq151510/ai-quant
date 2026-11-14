"""Test paired attribution, data gaps and feature boundaries for research arms."""
import json
import unittest
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from close_monday import close_report
from monday_scenario_matrix import DAY, frozen_inputs, synthetic_books
from monday_open import INITIAL, run
from shadow_factors import BASE, FEATURES, compare, load_features, observed, signals, stress


def archived_shadow_inputs():
    try:
        prepared, news = frozen_inputs()
        bars = BASE / 'monday-preparation' / 'bars'
        for row in prepared['candidates']:
            symbol = row['symbol']
            for path in (bars / f'{symbol}-0.json', bars / f'{symbol}-1.json',
                         FEATURES / f'{symbol}-qfq-20.json'):
                path.stat()
        features = load_features(prepared)
    except FileNotFoundError as exc:
        raise unittest.SkipTest(f'local archive missing: {exc.filename}') from exc
    return prepared, news, features


class ShadowFactorsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared, cls.news, cls.features = archived_shadow_inputs()
        cls.opening = run(cls.prepared, synthetic_books(cls.prepared['candidates'], 'eligible'), cls.news)
        cls.flat = {fill['symbol']: fill['price'] for fill in cls.opening['baseline']['fills']}

    def test_friday_data_and_thresholds(self):
        report = compare(self.opening, self.flat, self.features)
        self.assertEqual(set(report['volume']['blocked']), {'603135.SH'})
        self.assertEqual(report['volume']['delta_vs_baseline'], report['news']['delta_vs_baseline'])
        self.assertNotIn('000651.SZ', report['volume']['blocked'])
        self.assertEqual(report['news']['break_even_close']['603135.SH'], '9.1801')
        for move, expected in (('9.18', '.05'), ('9.19', '-4.95')):
            report = compare(self.opening, {**self.flat, '603135.SH': move}, self.features)
            self.assertEqual(Decimal(report['news']['delta_vs_baseline']), Decimal(expected))

    def test_missing_ma_does_not_disable_verified_volume(self):
        features = {**self.features, '000651.SZ': {
            'volume_ratio': Decimal('1.5'), 'return': Decimal('-.01'), 'ma_gap': 'missing 20-day history'}}
        report = compare(self.opening, self.flat, features)
        self.assertIn('000651.SZ', report['volume']['blocked'])
        self.assertEqual(report['ma']['unknown']['000651.SZ'], ['ma'])
        self.assertIsNone(report['ma']['ending_nav'])
        self.assertIsNone(report['ma']['delta_vs_baseline'])
        self.assertTrue(report['volume']['comparable'])

    def test_missing_news_does_not_become_no_news(self):
        opening = run(self.prepared, synthetic_books(self.prepared['candidates'], 'eligible'), [],
                      news_errors=['missing'])
        report = compare(opening, self.flat, self.features)
        self.assertEqual(len(report['news']['unknown']), 10)
        self.assertIsNone(report['news']['ending_nav'])
        self.assertTrue(report['volume']['comparable'])

    def test_missing_or_bad_baseline_mark_makes_delta_unknown(self):
        for invalid in (None, '0', 'NaN', 'bad'):
            with self.subTest(invalid=invalid):
                report = compare(self.opening, {**self.flat, '603135.SH': invalid}, self.features)
                self.assertIsNone(report['baseline']['ending_nav'])
                self.assertIsNotNone(report['news']['ending_nav'])
                self.assertFalse(report['news']['comparable'])
                self.assertIsNone(report['news']['delta_vs_baseline'])

    def test_unavailable_etf_is_not_an_attributed_veto(self):
        opening = run(self.prepared, synthetic_books(self.prepared['candidates'], 'etf_absent'), self.news)
        report = compare(opening, self.flat, self.features)
        self.assertNotIn('513100.SH', report['ma']['blocked'])
        self.assertNotIn('513100.SH', report['ma']['kept'])

    def test_shock_envelope_reverses_and_all_arms_share_fills(self):
        report = stress(self.prepared, self.news, self.features)
        self.assertEqual(len(report['cases']), 15)
        self.assertLess(Decimal(report['close_move_envelopes']['0.05']['all']), 0)
        self.assertGreater(Decimal(report['close_move_envelopes']['-0.05']['all']), 0)
        for case in report['cases']:
            for arm in case['arms'].values():
                self.assertLessEqual(set(arm['kept']), set(case['arms']['baseline']['kept']))

    def read_observed(self, opening, *, omit=(), replace_filtered=False, change_mark=False):
        day = str(DAY)
        marks = {
            symbol: {'date': day, 'close': price, 'price_type': 'unadjusted_close',
                     'source_url': 'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get'}
            for symbol, price in self.flat.items() if symbol not in omit
        }
        daily, _ = close_report({'mode': 'forward', **opening}, {'date': day, 'marks': marks})
        if replace_filtered:
            daily['filtered'] = {**daily['filtered'], 'cash': str(INITIAL), 'nav': str(INITIAL),
                                 'positions': {}, 'fills': [], 'close_prices': {}}
        if change_mark:
            daily['filtered']['close_prices']['603920.SH'] = '38.11'
        with TemporaryDirectory() as temp:
            base = Path(temp).resolve()
            path = base / day / 'daily.json'
            path.parent.mkdir()
            path.write_text(json.dumps(daily), encoding='utf-8')
            with patch('shadow_factors.BASE', base):
                return observed(self.prepared, self.features)

    def test_observed_replays_official_fills_with_partial_marks(self):
        arms = self.read_observed(self.opening, omit=('603135.SH',))['arms']
        self.assertIsNone(arms['baseline']['ending_nav'])
        self.assertEqual(Decimal(arms['news']['ending_nav']), Decimal('99954.80'))
        self.assertFalse(arms['news']['comparable'])

    def test_observed_uses_the_partial_with_more_verified_marks(self):
        day = str(DAY)
        missing = {'603135.SH', '000651.SZ'}
        marks = {
            symbol: {'date': day, 'close': price, 'price_type': 'unadjusted_close',
                     'source_url': 'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get'}
            for symbol, price in self.flat.items() if symbol not in missing
        }
        earlier, _ = close_report({'mode': 'forward', **self.opening}, {'date': day, 'marks': marks})
        price = self.flat['000651.SZ']
        mark = {**next(iter(marks.values())), 'close': price}
        later, _ = close_report({'mode': 'forward', **self.opening},
                                {'date': day, 'marks': {**marks, '000651.SZ': mark}})
        with TemporaryDirectory() as temp:
            base = Path(temp).resolve()
            for digest, ledger in (('a' * 64, earlier), ('b' * 64, later)):
                path = base / day / 'close-partials' / digest / 'daily.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(ledger), encoding='utf-8')
            with patch('shadow_factors.BASE', base):
                report = observed(self.prepared, self.features)
            self.assertEqual(report['close_ledger'], str(path.relative_to(base)))
            self.assertIsNone(report['arms']['baseline']['ending_nav'])
            self.assertEqual(Decimal(report['arms']['news']['ending_nav']), Decimal('99954.80'))

    def test_observed_accepts_news_gap_but_cannot_attribute_it(self):
        opening = run(self.prepared, synthetic_books(self.prepared['candidates'], 'eligible'), [],
                      news_errors=['missing'])
        arms = self.read_observed(opening)['arms']
        self.assertEqual(arms['news']['buys'], 0)
        self.assertIsNone(arms['news']['delta_vs_baseline'])

    def test_observed_rejects_a_balanced_but_different_official_trade_set(self):
        with self.assertRaisesRegex(ValueError, 'filtered: shadow trades or fees disagree'):
            self.read_observed(self.opening, replace_filtered=True)

    def test_observed_checks_common_marks_even_when_both_navs_are_unknown(self):
        with self.assertRaisesRegex(ValueError, '603920.SH: official groups disagree'):
            self.read_observed(self.opening, omit=('000651.SZ',), change_mark=True)


class ShadowSignalTests(unittest.TestCase):
    def test_equal_volume_boundary_and_zero_return(self):
        feature = {'return': Decimal('0'), 'volume_ratio': Decimal('1.5'), 'above_all_ma': True}
        self.assertFalse(signals(feature, {'reason': 'filled'}, Decimal('1.5'))['volume'])
        self.assertTrue(signals({**feature, 'return': Decimal('-.01')}, None, Decimal('1.5'))['volume'])
        self.assertFalse(signals(feature, None, Decimal('1.5'))['ma'])


if __name__ == '__main__':
    unittest.main()
