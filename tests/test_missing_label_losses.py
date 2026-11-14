"""Regression checks for fitting and calibration with unobserved returns."""

import unittest

import numpy as np
import torch

from src.domain.timeseries_dl.losses import CrossSectionalICLoss
from src.domain.uncertainty.quantile_conformal import fit_and_calibrate_conformal_quantile


class MissingLabelLossTests(unittest.TestCase):
    def test_ic_and_regularization_use_only_eligible_cross_sections(self):
        predictions = torch.tensor([
            [0.0, 1.0, 900.0],
            [100.0, 0.0, 200.0],
            [800.0, 900.0, 300.0],
            [0.0, 1.0, 800.0],
        ], requires_grad=True)
        labels = torch.tensor([
            [0.0, 1.0, float("nan")],
            [float("nan"), 1.0, float("nan")],
            [float("nan"), float("nan"), float("nan")],
            [1.0, 1.0, float("nan")],
        ])
        weights = torch.tensor([1.0, 2.0, 50.0, 3.0])
        criterion = CrossSectionalICLoss(mse_weight=0.2)
        observed_ic_loss = criterion(predictions[:1, :2], labels[:1, :2], weights[:1])

        loss = criterion(predictions, labels, weights)
        expected_reg = (2 * 0.5 + 3 * 0.25) / (6 + criterion.eps)
        self.assertAlmostEqual(loss.item(), observed_ic_loss.item() + 0.2 * expected_reg, places=6)

        loss.backward()
        self.assertTrue(torch.isfinite(predictions.grad).all())
        self.assertTrue(predictions.grad[~torch.isfinite(labels)].eq(0).all())
        self.assertNotEqual(predictions.grad[1, 1].item(), 0)

    def test_ic_loss_can_backpropagate_with_no_observed_labels(self):
        predictions = torch.tensor([[1.0, 2.0]], requires_grad=True)
        loss = CrossSectionalICLoss()(predictions, torch.full_like(predictions, float("nan")))

        loss.backward()

        self.assertEqual(loss.item(), 0)
        self.assertTrue(predictions.grad.eq(0).all())

    def test_quantile_fit_and_calibration_ignore_features_without_labels(self):
        rng = np.random.default_rng(91)
        train_x = rng.normal(size=(4, 3, 2, 2))
        val_x = rng.normal(size=(3, 3, 2, 2))
        eval_x = rng.normal(size=(2, 3, 2, 2))
        train_y = rng.normal(size=(4, 3))
        val_y = rng.normal(size=(3, 3))
        train_y[[0, 2], 1] = np.nan
        val_y[1] = np.nan

        torch.manual_seed(23)
        expected = fit_and_calibrate_conformal_quantile(train_x, train_y, val_x, val_y, eval_x, epochs=2)
        train_x[~np.isfinite(train_y)] = np.nan
        val_x[~np.isfinite(val_y)] = np.nan
        torch.manual_seed(23)
        actual = fit_and_calibrate_conformal_quantile(train_x, train_y, val_x, val_y, eval_x, epochs=2)

        self.assertAlmostEqual(actual.conformal_q_hat, expected.conformal_q_hat)
        self.assertEqual(actual.interval_width.shape, (2, 3))
        self.assertTrue(np.isfinite(actual.interval_width).all())
        np.testing.assert_allclose(actual.q50_median, expected.q50_median)

    def test_quantile_requires_support_in_both_partitions(self):
        features = np.zeros((2, 2, 2, 2))
        observed = np.ones((2, 2))
        missing = np.full((2, 2), np.nan)
        for train, val, stage in ((missing, observed, "training"), (observed, missing, "validation")):
            with self.subTest(stage=stage), self.assertRaisesRegex(ValueError, f"observed {stage} labels"):
                fit_and_calibrate_conformal_quantile(features, train, features, val, features)
