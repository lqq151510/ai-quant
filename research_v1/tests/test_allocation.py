import numpy as np
import pandas as pd
import pytest

from quant_research.allocation import allocation_targets


@pytest.fixture
def allocation_dependencies():
    pytest.importorskip("scipy")
    pytest.importorskip("sklearn")


def test_allocation_excludes_ineligible_names_and_can_hold_all_cash():
    recent = np.tile([0.01, 0.02], 10)
    history = pd.DataFrame({
        "valid": recent,
        "missing_return": np.r_[recent[:-1], np.nan],
        "infinite_return": np.r_[recent[:-1], np.inf],
        "negative_price": np.r_[recent[:-1], -1.0],
        "flat": 0.02,
        "zero_score": recent,
        "negative_score": recent,
        "infinite_score": recent,
    })
    scores = pd.Series(1.0, index=[*history.columns, "no_history"])
    scores.loc[["zero_score", "negative_score", "infinite_score"]] = [0.0, -0.1, np.inf]
    targets = allocation_targets(scores, history, "etf")
    assert set(targets) == {"allocation_equal", "allocation_inverse_vol", "allocation_hrp"}
    for weights in targets.values():
        assert weights == pytest.approx({"valid": 0.28})
    assert all(not weights for weights in allocation_targets(scores, history.iloc[1:], "etf").values())
    assert all(not weights for weights in allocation_targets(scores.drop("valid"), history, "etf").values())


@pytest.mark.parametrize("kind, count, cap", [("etf", 2, 0.35), ("etf", 4, 0.35), ("stock", 12, 0.10)])
def test_methods_share_capped_gross_and_break_score_ties_by_code(allocation_dependencies, kind, count, cap):
    symbols = [f"{index:02}" for index in range(count)]
    returns = np.random.default_rng(34).normal(0, 1, (60, count)) * np.geomspace(0.0001, 0.04, count)
    history = pd.DataFrame(returns, columns=symbols)
    scores = pd.Series(1.0, index=symbols)
    targets = allocation_targets(scores, history, kind)
    reversed_targets = allocation_targets(scores.iloc[::-1], history.loc[:, symbols[::-1]], kind)
    selected = symbols[:10] if kind == "stock" else symbols
    gross = 0.80 * min(1.0, len(selected) * cap)
    assert targets == reversed_targets
    for weights in targets.values():
        assert list(weights) == selected
        assert sum(weights.values()) == pytest.approx(gross)
        assert all(0 < weight <= cap for weight in weights.values())
    assert targets["allocation_inverse_vol"][symbols[0]] == cap
    assert targets["allocation_inverse_vol"] != targets["allocation_equal"]
    assert targets["allocation_hrp"] != targets["allocation_equal"]


def test_hrp_allocates_less_to_a_correlated_pair_at_equal_volatility(allocation_dependencies):
    history = pd.DataFrame({
        "a": np.tile([0.01, 0.01, -0.01, -0.01], 14),
        "b": np.tile([0.01, 0.01, -0.01, -0.01], 14),
        "c": np.tile([0.01, -0.01, 0.01, -0.01], 14),
        "d": np.tile([0.01, -0.01, -0.01, 0.01], 14),
    })
    scores = pd.Series(1.0, index=history.columns)
    targets = allocation_targets(scores, history, "etf")
    assert targets["allocation_inverse_vol"] == targets["allocation_equal"]
    hrp = targets["allocation_hrp"]
    assert hrp["a"] == pytest.approx(hrp["b"])
    assert hrp["a"] + hrp["b"] < 0.4 < hrp["c"] + hrp["d"]


def test_hrp_uses_at_most_60_rows_and_keeps_only_the_common_contiguous_suffix(allocation_dependencies):
    history = pd.DataFrame(np.random.default_rng(67).normal(0, 0.01, (70, 4)), columns=list("abcd"))
    scores = pd.Series(1.0, index=history.columns)
    assert allocation_targets(scores, history, "etf") == allocation_targets(scores, history.iloc[-60:], "etf")
    history.loc[32, "a"] = np.nan
    assert allocation_targets(scores, history, "etf") == allocation_targets(scores, history.iloc[33:], "etf")
