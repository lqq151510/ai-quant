import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from quant_research.regime_study import filtered_regimes


def test_regime_probabilities_are_causal_with_fixed_training_cutoff():
    days = pd.bdate_range("2024-01-01", periods=44)
    sample = pd.DataFrame(np.random.default_rng(42).normal(size=(len(days), 4)), index=days)
    full = filtered_regimes(sample, days[30])
    prefix = filtered_regimes(sample.iloc[:37], days[30])
    np.testing.assert_allclose(full.iloc[:37].to_numpy(), prefix.to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(full.sum(axis=1), np.ones(len(days)))
