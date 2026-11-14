"""Regression coverage for unavailable returns in the tree ensemble."""

import unittest

import numpy as np

from src.domain.ensemble.gbdt_stacking import OrthogonalTreeGNNStacker, iter_observed_cross_sections


class GBDTStackingTests(unittest.TestCase):
    def test_unlabeled_training_features_do_not_change_test_predictions(self) -> None:
        rng = np.random.default_rng(47)
        train_x = rng.normal(size=(48, 4, 5, 2))
        train_y = train_x[:, :, -1, 0].copy()
        train_y[::3, 1] = np.nan
        val_x = rng.normal(size=(8, 4, 5, 2))
        val_y = val_x[:, :, -1, 0].copy()
        val_y[:, 0] = np.nan
        test_x = rng.normal(size=(6, 4, 5, 2))
        gnn_preds = {"train_preds": train_y.copy(), "test_preds": rng.normal(size=(6, 4))}
        gnn_preds["train_preds"][~np.isfinite(train_y)] = 0

        def predict(features):
            model = OrthogonalTreeGNNStacker()
            blended, tree = model.fit_and_blend(features, train_y, val_x, val_y, test_x, gnn_preds)
            self.assertEqual(blended["train_preds"].shape, train_y.shape)
            self.assertTrue(np.isfinite(blended["train_preds"]).all())
            self.assertEqual(tree.shape, (6, 4))
            self.assertGreater(float(np.abs(tree).max()), 0)
            return blended["test_preds"]

        perturbed = train_x.copy()
        perturbed[~np.isfinite(train_y)] *= 100
        np.testing.assert_array_equal(predict(train_x), predict(perturbed))

        model = OrthogonalTreeGNNStacker()
        with self.assertRaisesRegex(ValueError, "at least one observed target"):
            model.fit_and_blend(train_x, np.full_like(train_y, np.nan), val_x, val_y, test_x, gnn_preds)

    def test_cross_section_correlations_require_two_varying_observed_pairs(self) -> None:
        predictions = np.array([[1.0, 2.0, 50.0], [1.0, 4.0, np.nan], [1.0, 1.0, 5.0]])
        targets = np.array([[2.0, 5.0, np.nan], [2.0, np.nan, 5.0], [2.0, 5.0, np.nan]])
        [(prediction, target)] = list(iter_observed_cross_sections(predictions, targets))
        np.testing.assert_array_equal(prediction, [1.0, 2.0])
        np.testing.assert_array_equal(target, [2.0, 5.0])
