from datetime import date
from decimal import Decimal

from quant_research.capital_scale import etf_flat_round_trip, whole_lot_cost_thresholds
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
