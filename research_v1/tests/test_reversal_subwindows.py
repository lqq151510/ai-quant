from datetime import date

import pandas as pd
import pytest

from quant_research.reversal_subwindows import subwindow_metrics


def test_continuation_returns_compound_and_include_first_session_loss():
    equity = pd.DataFrame({
        "date": [date(2023, 12, 29), date(2024, 1, 2), date(2024, 1, 3)],
        "nav": [120.0, 108.0, 112.0], "market_value": [0.0, 0.0, 0.0],
    })
    orders = pd.DataFrame(columns=["date", "reason"])

    first = subwindow_metrics(equity, orders, 2023, 100.0)
    second = subwindow_metrics(equity, orders, 2024, 100.0)

    assert second["starting_nav"] == 120.0
    assert second["max_drawdown"] == pytest.approx(0.10)
    assert second["calmar"] == pytest.approx(second["annualized_return"] / 0.10)
    assert (1 + first["net_return"]) * (1 + second["net_return"]) - 1 == pytest.approx(0.12)
    assert first["calmar"] is None
