"""Verify state multiplicities and consistency with the separate stress test."""
from collections import Counter
from decimal import Decimal
from itertools import product
import unittest

from joint_scenarios import calculate, distribution
from shadow_factors import stress
from test_shadow_factors import archived_shadow_inputs


class JointScenarioTests(unittest.TestCase):
    def test_duplicate_outcomes_retain_multiplicity(self):
        states = [(0, 0, -3, 5), (0, -5, 2, 4), (0, 0, 2, 3)]
        self.assertEqual(distribution(states), Counter(map(sum, product(*states))))

    def test_joint_counts_and_paired_bounds(self):
        prepared, news, features = archived_shadow_inputs()
        envelopes = stress(prepared, news, features)['close_move_envelopes']
        count, reports = calculate()
        self.assertEqual(count, 10)
        self.assertEqual(reports['news'], reports['volume'])
        for name, (_, delta, pnl) in reports.items():
            with self.subTest(arm=name):
                self.assertEqual(sum(delta.values()), 4 ** count)
                self.assertEqual(sum(pnl.values()), 4 ** count)
                self.assertEqual(Decimal(min(delta)) / 100, Decimal(envelopes['0.05'][name]))
                self.assertEqual(Decimal(max(delta)) / 100, Decimal(envelopes['-0.05'][name]))
