"""Verify matching rules and the raw-backed opening-confirmation audit."""

import copy
import unittest
from fractions import Fraction

from matched import BLOCKED, Entry, Outcome, amount, audit, compare_archive, compare_day
from confirm import SESSIONS, SOURCE


DAY = '2026-11-16'
ARCHIVE_AVAILABLE = any((SOURCE / str(day)).exists() for day in SESSIONS)


def outcome(symbol, spend, pnl, signal):
    entry = Entry(symbol, DAY, 100, signal, Fraction(spend, 100), Fraction(spend))
    return Outcome(entry, '2026-11-17', Fraction(10), Fraction(pnl), 'closed')


class MatchedTests(unittest.TestCase):
    def test_same_count_controls_scale_each_basket_to_the_same_spend(self):
        report, exact = compare_day(DAY, [
            outcome('600001.SH', 100, 10, 'confirmed'),
            outcome('600002.SH', 200, 20, BLOCKED),
            outcome('600003.SH', 100, -10, BLOCKED),
        ])
        self.assertEqual(exact['proportional_matched_pnl'], Fraction(5))
        self.assertEqual(exact['same_count_matched_mean_pnl'], Fraction(10, 3))
        self.assertEqual(report['rank'], {
            'higher_than_confirmation': 0, 'equal_to_confirmation': 2, 'lower_than_confirmation': 1,
        })

    def test_no_pool_and_no_selection_choice_are_explicit(self):
        empty, values = compare_day(DAY, [])
        self.assertIsNone(values)
        self.assertEqual((empty['state'], empty['choice_count'], empty['rank']), ('no_valued_entries', 0, None))
        for signal, expected in ((BLOCKED, 0), ('unknown', 10)):
            with self.subTest(signal=signal):
                row, values = compare_day(DAY, [outcome('600001.SH', 100, 10, signal)])
                self.assertEqual((row['state'], row['choice_count'], row['rank']), ('no_selection_choice', 1, None))
                self.assertEqual(values['same_count_matched_mean_pnl'], expected)
                self.assertEqual(values['selection_change'], 0)

    def test_empty_unsettled_pilot_has_no_pnl_or_lifecycles(self):
        report = audit({
            'state': 'awaiting_first_settlement', 'settled_sessions': 0, 'daily': [],
            'summary': None, 'signal_counts': None, 'blocked_outcomes': None,
        })
        self.assertIsNone(report['all_valued'])
        self.assertIsNone(report['closed_only'])
        self.assertIsNone(report['account_summary'])
        self.assertNotIn('lifecycles', report)


@unittest.skipUnless(ARCHIVE_AVAILABLE, 'local original-market archive is absent')
class MatchedArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = compare_archive()

    def test_raw_backed_archive_reconciles_selection_and_deployment(self):
        report = audit(self.original)
        total = report['all_valued']['total']
        exact = {key: Fraction(value) for key, value in report['all_valued']['exact_total'].items()}
        self.assertEqual((amount(total['original_pnl']), amount(total['confirmation_pnl'])),
                         (Fraction('-1248.41'), Fraction('51.39')))
        self.assertEqual(total['proportional_matched_pnl'], '-329.144872105274')
        self.assertEqual(total['same_count_matched_mean_pnl'], '-330.085751087377')
        self.assertEqual(exact['deployment_change'] + exact['selection_change'], exact['improvement'])
        self.assertEqual([row['choice_count'] for row in report['all_valued']['days']], [28, 28, 10])
        self.assertEqual([row['terminal_count'] for row in report['all_valued']['days']], [0, 0, 5])
        self.assertEqual(report['closed_only']['total']['confirmation_pnl'], '83.460000000000')
        self.assertEqual(report['closed_only']['days'][-1]['state'], 'no_valued_entries')

    def test_skipped_outcome_must_match_the_original_trade_lifecycle(self):
        corrupted = copy.deepcopy(self.original)
        corrupted['blocked_outcomes']['trades'][0]['pnl'] = '12.34'
        with self.assertRaisesRegex(ValueError, 'skipped lots do not reconcile'):
            audit(corrupted)

    def test_refuses_confirmation_fill_or_closing_mark_drift(self):
        changed_fill = copy.deepcopy(self.original)
        fills = changed_fill['daily'][0]['confirmation']['fills']
        fills[0] = {**fills[0], 'units': fills[0]['units'] + 100}
        with self.assertRaisesRegex(ValueError, 'fills or share positions'):
            audit(changed_fill)
        changed_mark = copy.deepcopy(self.original)
        prices = changed_mark['daily'][0]['confirmation']['close_prices']
        del prices[next(iter(prices))]
        with self.assertRaisesRegex(ValueError, 'closing marks'):
            audit(changed_mark)

if __name__ == '__main__':
    unittest.main()
