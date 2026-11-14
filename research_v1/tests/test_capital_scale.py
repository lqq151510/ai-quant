from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from quant_research.baseline import build_targets
from quant_research.capital_scale import (
    attribute_actual_fills, etf_flat_round_trip, reversal_exposure_targets, whole_lot_cost_thresholds,
)
from quant_research.ledger import LOT_SIZE, Costs


def test_cost_threshold_is_first_whole_lot_meeting_budget():
    day = date(2024, 12, 31)
    price = Decimal('4.022')
    cost = Costs()
    table = whole_lot_cost_thresholds(day, price, cost)
    assert table['one_lot']['commission'] == 10.0
    assert table['one_lot']['buy_cash_required'] == 407.40
    assert table['one_lot']['all_in_cost'] == 10.40
    target = table['minimum_by_cost_budget_bps'][25]
    assert target['units'] == 1700
    assert target['reference_notional'] == 6837.40
    assert target['minimum_account_equity_at_weight_cap'] == 19535.43
    previous = etf_flat_round_trip(day, price, target['units'] - LOT_SIZE, cost)
    assert target['all_in_cost_bps'] <= 25 < previous['all_in_cost_bps']


def test_two_sided_minimum_does_not_ignore_proportional_rate_or_slippage():
    result = etf_flat_round_trip(date(2024, 12, 31), Decimal('4.000'), 5000, Costs())
    assert result['reference_notional'] == 20_000.0
    assert result['commission'] == 12.0
    assert result['slippage'] == 20.0
    assert result['all_in_cost_bps'] == 16.0


def test_actual_fills_attribute_buy_and_sell_slippage_without_changing_quantity():
    days = [date(2024, 1, 2), date(2024, 1, 3)]
    fills = pd.DataFrame({
        'date': days, 'symbol': ['510300'] * 2, 'side': ['buy', 'sell'], 'filled_qty': [100, 100],
        'notional': [1001.0, 1099.0], 'commission': [5.0, 5.0],
        'stamp_tax': [0.0, 0.0], 'transfer_fee': [0.0, 0.0],
    })
    openings = pd.Series([10.0, 11.0], index=pd.MultiIndex.from_arrays([days, fills['symbol']]))

    result = attribute_actual_fills(fills, openings, final_nav=2088.0, capital=2000)

    assert result == {
        'fees': 10.0, 'slippage': 2.0, 'same_fill_before_fees_return': 0.049,
        'same_fill_raw_open_return': 0.05, 'fill_sessions': 2,
    }
    with pytest.raises(ValueError, match='opening reference price'):
        attribute_actual_fills(fills, openings.iloc[:1], final_nav=2088.0, capital=2000)


def test_reverse_momentum_uses_negative_compound_returns_on_same_dates():
    days = pd.bdate_range("2024-01-02", periods=26)
    noise = np.tile([0.001, -0.001], 13)
    returns = pd.DataFrame({
        "winner": 0.01 + noise, "loser": -0.01 + noise, "missing": np.nan,
    }, index=days)
    cohort = dict.fromkeys(days, returns.columns)

    momentum = build_targets(days, cohort, returns, "etf", 5)["momentum"]
    reverse = build_targets(days, cohort, returns, "etf", 5, reverse_momentum=True)["momentum"]

    assert momentum.keys() == reverse.keys()
    assert momentum[days[16]] == reverse[days[16]] == {}
    assert momentum[days[21]] == {"winner": 0.35}
    assert reverse[days[21]] == {"loser": 0.35}


def test_calendar_monthly_reviews_use_first_exchange_session_and_next_open():
    days = pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-05",
                           "2024-02-06", "2024-03-04", "2024-03-05", "2024-04-01"])
    returns = pd.DataFrame({"etf": -0.01}, index=days)
    targets = build_targets(
        days, dict.fromkeys(days, returns.columns), returns, "etf", 20,
        reverse_momentum=True, calendar_monthly=True,
    )

    assert list(targets["momentum"]) == list(days[[1, 3, 6]])
    assert list(targets["equal_weight"]) == list(targets["momentum"])


def test_exposure_controls_separate_allocation_from_cash_timing():
    days = list(pd.to_datetime(["2024-02-02", "2024-05-13", "2024-08-12"]))
    original = {
        days[0]: {"510050": 0.35},
        days[1]: {"510300": 0.2, "510880": 0.35},
        days[2]: {},
    }

    controls = reversal_exposure_targets(original)
    matched = controls["equal_weight_match_reverse"]
    full = controls["reverse_full"]
    on_signal = controls["equal_weight_on_reverse"]

    assert all(list(schedule) == days and schedule[days[2]] == {} for schedule in controls.values())
    for day in days[:2]:
        assert len(matched[day]) == 4 and len(set(matched[day].values())) == 1
        assert sum(matched[day].values()) == pytest.approx(sum(original[day].values()))
        assert set(full[day]) == set(original[day])
        assert sum(full[day].values()) == pytest.approx(1.0)
        assert set(on_signal[day].values()) == {0.25}
    assert full[days[0]] == {"510050": 1.0}
    assert full[days[1]]["510300"] / full[days[1]]["510880"] == pytest.approx(0.2 / 0.35)
