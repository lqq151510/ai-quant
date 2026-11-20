"""Check the forward repricing adapter with saved real evidence and no new session."""

import copy
import sys
import unittest
from decimal import Decimal
from types import ModuleType
from unittest.mock import patch

import stress_matched
from confirm import SESSIONS, SOURCE, compare_archive, verified_session


@unittest.skipUnless(any((SOURCE / str(day)).exists() for day in SESSIONS),
                     'local original-market archive is absent')
class StressPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.standard = compare_archive()
        cls.inputs = {day: verified_session(day, {}) for day in SESSIONS}

    def pilot_module(self):
        pilot = ModuleType('pilot')
        pilot.CASH_CONTROL = {'cash': '100000', 'positions': {}}
        raw_records = [{'date': str(day), 'baseline': self.inputs[day][0]} for day in SESSIONS]
        pilot.observed_days = lambda: ({}, raw_records)

        def verify(day, original, prior):
            raw, orders, _ = self.inputs[day]
            self.assertIn('decisions', original)
            self.assertEqual(original, raw)
            previous = SESSIONS.index(day) - 1
            self.assertEqual(prior, self.inputs[SESSIONS[previous]][0] if previous >= 0 else pilot.CASH_CONTROL)
            return orders, 10

        pilot.verified_inputs = verify
        return pilot

    def test_settled_pilot_preserves_full_original_decisions(self):
        pilot = self.pilot_module()
        with patch.dict(sys.modules, {'pilot': pilot}), patch.object(
            stress_matched, 'compare_pilot', return_value=self.standard,
        ):
            result = stress_matched.compare('pilot')
        self.assertEqual(result['settled_sessions'], 3)
        self.assertEqual(result['purchase_budget_or_price_or_cash_issues'], [])
        self.assertEqual(Decimal(result['matched']['all_valued']['total']['confirmation_pnl']), Decimal('-18.61'))

    def test_changed_settlement_is_rejected_before_repricing(self):
        pilot = self.pilot_module()
        standard = copy.deepcopy(self.standard)
        standard['daily'][0]['original']['cash'] = '1'
        with patch.dict(sys.modules, {'pilot': pilot}), patch.object(
            stress_matched, 'compare_pilot', return_value=standard,
        ):
            with self.assertRaisesRegex(ValueError, 'settled baseline changed'):
                stress_matched.compare('pilot')


if __name__ == '__main__':
    unittest.main()
