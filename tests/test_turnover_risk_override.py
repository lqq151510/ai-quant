"""Risk exits override turnover friction without changing ordinary rebalancing."""

import ast
from pathlib import Path
import unittest

import numpy as np

from src.application.walk_forward_engine import apply_turnover_deadband


class TestTurnoverRiskOverride(unittest.TestCase):
    def test_explicit_cap_reduces_existing_exposure_inside_deadband(self):
        previous = np.array([0.4, 0.4])
        target, trade = apply_turnover_deadband(np.array([0.35, 0.35]), previous, max_exposure=0.7)
        self.assertTrue(trade)
        np.testing.assert_allclose(target, [0.35, 0.35])
        self.assertLessEqual(target.sum(), 0.7 + 1e-12)

    def test_explicit_zero_target_clears_all_inertia(self):
        previous = np.array([0.4, 0.4])
        target, trade = apply_turnover_deadband(np.zeros(2), previous)
        self.assertTrue(trade)
        np.testing.assert_array_equal(target, np.zeros(2))
        target, trade = apply_turnover_deadband(np.zeros(2), np.zeros(2))
        self.assertFalse(trade)
        np.testing.assert_array_equal(target, np.zeros(2))

    def test_proposed_exposure_above_cap_is_bounded_despite_current_safe_holdings(self):
        target, trade = apply_turnover_deadband(np.array([0.4, 0.4]), np.array([0.3, 0.3]), max_exposure=0.7)
        self.assertTrue(trade)
        np.testing.assert_allclose(target, [0.35, 0.35])
        initial, trade = apply_turnover_deadband(np.array([0.5, 0.5]), np.zeros(2), max_exposure=0.7)
        self.assertTrue(trade)
        np.testing.assert_allclose(initial, [0.35, 0.35])

    def test_same_exposure_micro_changes_and_small_increases_still_skip(self):
        previous = np.array([0.3, 0.3])
        for proposal in (np.array([0.31, 0.29]), np.array([0.32, 0.32])):
            target, trade = apply_turnover_deadband(proposal, previous, max_exposure=0.7)
            self.assertFalse(trade)
            np.testing.assert_array_equal(target, previous)

    def test_nonzero_decrease_without_explicit_cap_preserves_old_deadband(self):
        previous = np.array([0.4, 0.4])
        target, trade = apply_turnover_deadband(np.array([0.35, 0.35]), previous)
        self.assertFalse(trade)
        np.testing.assert_array_equal(target, previous)

    def test_roundoff_at_cap_does_not_force_a_micro_trade(self):
        previous = np.array([0.1, 0.2])
        target, trade = apply_turnover_deadband(np.array([0.1001, 0.1999]), previous, max_exposure=0.3)
        self.assertFalse(trade)
        np.testing.assert_array_equal(target, previous)

    def test_replacement_of_holdings_obeys_cap_and_does_not_mutate_inputs(self):
        previous = np.array([0.4, 0.4, 0.0, 0.0])
        proposal = np.array([0.0, 0.0, 0.4, 0.4])
        before_previous, before_proposal = previous.copy(), proposal.copy()
        target, trade = apply_turnover_deadband(proposal, previous, max_exposure=0.7)
        self.assertTrue(trade)
        self.assertLessEqual(target.sum(), 0.7 + 1e-12)
        self.assertTrue((target[2:] > 0).all())
        np.testing.assert_array_equal(previous, before_previous)
        np.testing.assert_array_equal(proposal, before_proposal)
        self.assertFalse(np.shares_memory(target, previous))
        self.assertFalse(np.shares_memory(target, proposal))

    def test_zero_cap_and_invalid_limits(self):
        for previous in (np.array([0.4, 0.4]), np.zeros(2)):
            target, trade = apply_turnover_deadband(np.array([0.5, 0.5]), previous, max_exposure=0)
            np.testing.assert_array_equal(target, np.zeros(2))
            self.assertEqual(trade, bool(np.any(previous)))
        for invalid in (-0.1, 1.01, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "max_exposure"):
                apply_turnover_deadband(np.zeros(2), np.zeros(2), max_exposure=invalid)

    def test_stage2_keeps_default_cap_while_hrp_and_rl_enable_hmm(self):
        """Inspect wiring without training networks or creating trading outputs."""
        source = Path(__file__).resolve().parents[1] / "src/application/walk_forward_engine.py"
        tree = ast.parse(source.read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "close_policy"]
        wiring = {node.args[0].id: {keyword.arg: keyword.value for keyword in node.keywords}
                  for node in calls if node.args and isinstance(node.args[0], ast.Name)}
        self.assertNotIn("enforce_hmm_cap", wiring["stage2_proposal"])
        self.assertTrue(wiring["hrp_proposal"]["enforce_hmm_cap"].value)
        self.assertTrue(wiring["propose"]["enforce_hmm_cap"].value)


if __name__ == "__main__":
    unittest.main()
