from datetime import date

import pandas as pd
import pytest

from quant_research.ledger import Costs, LedgerError, simulate


FRIDAY = "2023-08-25"
MONDAY = "2023-08-28"
TUESDAY = "2023-08-29"
ETF = "510300"
STOCK = "600519"


def bar(day, symbol=ETF, **changes):
    row = {
        "date": day,
        "symbol": symbol,
        "kind": "etf" if symbol == ETF else "stock",
        "open": 10.0,
        "high": 10.0,
        "low": 10.0,
        "close": 10.0,
        "volume": 1_000,
        "amount": 10_000.0,
        "tradable": True,
        "upper_limit": 11.0,
        "lower_limit": 9.0,
    }
    return row | changes


def test_costs_and_t_plus_one_reconcile_across_weekend():
    bars = pd.DataFrame([bar(FRIDAY, STOCK), bar(MONDAY, STOCK)])
    result = simulate(bars, [FRIDAY, MONDAY], {FRIDAY: {STOCK: 1}, MONDAY: {}}, initial_cash=10_000)
    buy, sell = list(result.orders.itertuples(index=False))
    assert (buy.requested_qty, buy.filled_qty, buy.status, buy.reason) == (1_000, 900, "partial", "insufficient_cash")
    assert (buy.notional, buy.commission, buy.stamp_tax, buy.transfer_fee) == (9004.50, 5.0, 0.0, 0.09)
    assert (sell.filled_qty, sell.notional, sell.stamp_tax, sell.transfer_fee) == (900, 8995.50, 4.50, 0.09)
    assert result.positions["sellable_quantity"].tolist() == [0]
    assert result.equity["cash"].tolist() == [990.41, 9976.32]
    assert result.equity["nav"].tolist() == [9990.41, 9976.32]
    assert (result.equity["cash"] >= 0).all()


def test_close_policy_receives_filled_close_state_and_trades_next_open():
    wednesday = "2023-08-30"
    calendar = [FRIDAY, MONDAY, TUESDAY, wednesday]
    bars = pd.DataFrame([bar(FRIDAY), bar(MONDAY, close=9.8, low=9.8), bar(TUESDAY), bar(wednesday)])
    actions = pd.DataFrame([{
        "symbol": ETF, "record_date": MONDAY, "ex_date": TUESDAY, "pay_date": wednesday,
        "cash_per_share": 0.10, "share_multiplier": 1.0,
    }])

    def run(data):
        states = []

        def policy(state):
            states.append(state)
            if state.date == date.fromisoformat(FRIDAY):
                return {ETF: 1.0}
            if state.date == date.fromisoformat(TUESDAY):
                return {}
            return None

        result = simulate(data, calendar, {}, actions, initial_cash=10_000, close_policy=policy)
        return states, result

    states, result = run(bars)
    friday, monday, tuesday = states
    assert [state.date for state in states] == [date.fromisoformat(day) for day in calendar[:-1]]
    assert (friday.nav, friday.cash, friday.drawdown, friday.weights) == (10_000, 10_000, 0, {})
    assert (monday.nav, monday.cash) == (9810.50, 990.50)
    assert monday.weights == {ETF: 8820 / 9810.50}
    assert monday.drawdown == pytest.approx(1.0 - 9810.50 / 10_000)
    assert tuesday.cash == monday.cash
    assert tuesday.nav == tuesday.cash + 9000 + 90
    assert tuesday.weights == {ETF: 9000 / tuesday.nav}
    assert result.orders["date"].tolist() == [date.fromisoformat(MONDAY), date.fromisoformat(wednesday)]
    assert result.orders["filled_qty"].tolist() == [900, 900]

    perturbed = bars.copy()
    perturbed.loc[perturbed["date"] == TUESDAY, ["open", "close", "low"]] = 9.5
    changed_states, _ = run(perturbed)
    assert changed_states[:2] == states[:2]
    assert changed_states[2].nav != tuesday.nav


def test_close_policy_rejects_static_targets_and_invalid_weights():
    bars = pd.DataFrame([bar(FRIDAY), bar(MONDAY)])
    with pytest.raises(LedgerError, match="combined with static"):
        simulate(bars, [FRIDAY, MONDAY], {MONDAY: {}}, initial_cash=10_000, close_policy=lambda _: None)
    with pytest.raises(LedgerError, match="leverage"):
        simulate(bars, [FRIDAY, MONDAY], {}, initial_cash=10_000, close_policy=lambda _: {ETF: 1.01})


def test_stock_fee_changes_leave_etf_stamp_and_transfer_untaxed():
    costs = Costs()
    assert costs.charges(date(2022, 4, 28), "stock", "sell", 100_000) == (30, 100, 2)
    assert costs.charges(date(2022, 4, 29), "stock", "sell", 100_000) == (30, 100, 1)
    assert costs.charges(date(2023, 8, 28), "stock", "sell", 100_000) == (30, 50, 1)
    assert costs.stressed().charges(date(2023, 8, 28), "etf", "sell", 100_000) == (60, 0, 0)
    with pytest.raises(LedgerError, match="missing historical stock fee schedule"):
        costs.charges(date(2015, 7, 31), "stock", "buy", 100_000)


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"tradable": False, "volume": 0}, "suspended"),
        ({"open": 11.0, "high": 11.0}, "limit_up"),
        ({"upper_limit": float("nan")}, "missing_price_limits"),
        ({"open": 10.999, "high": 10.999}, "slippage_outside_limit"),
    ],
)
def test_no_fill_when_open_cannot_support_order(changes, reason):
    result = simulate(pd.DataFrame([bar(FRIDAY, **changes)]), [FRIDAY],
                      {FRIDAY: {ETF: 1}}, initial_cash=10_000)
    order = result.orders.iloc[0]
    assert (order["status"], order["reason"], order["filled_qty"]) == ("rejected", reason, 0)
    assert result.equity.iloc[0]["cash"] == 10_000


def test_limit_down_keeps_position_and_records_rejection():
    bars = pd.DataFrame([bar(FRIDAY), bar(MONDAY, open=9.0, low=9.0, close=9.0)])
    result = simulate(bars, [FRIDAY, MONDAY], {FRIDAY: {ETF: 1}, MONDAY: {}}, initial_cash=10_000)
    assert result.orders.iloc[-1]["reason"] == "limit_down"
    assert result.positions.iloc[-1]["quantity"] == result.positions.iloc[-1]["sellable_quantity"] == 900
    assert result.equity.iloc[-1]["market_value"] == 8100


def test_split_and_dividend_are_separate_from_open_orders():
    # A 2:1 split and a 1 yuan dividend leave a 4.5 yuan ex price.
    ex_bar = {"open": 4.5, "high": 4.5, "low": 4.5, "close": 4.5, "preclose": 4.5,
              "upper_limit": 4.95, "lower_limit": 4.05}
    bars = pd.DataFrame([bar(FRIDAY), bar(MONDAY, **ex_bar), bar(TUESDAY, **ex_bar)])
    actions = pd.DataFrame([{
        "symbol": ETF, "record_date": FRIDAY, "ex_date": MONDAY, "pay_date": TUESDAY,
        "share_available_date": TUESDAY, "cash_per_share": 1.0, "share_multiplier": 2.0,
    }])
    result = simulate(bars, [FRIDAY, MONDAY, TUESDAY], {FRIDAY: {ETF: 1}, MONDAY: {}},
                      actions, initial_cash=10_000, cost=Costs(0, 0, 0))
    assert result.equity["nav"].tolist() == [10_000, 10_000, 10_000]
    assert result.equity["cash"].tolist() == [0, 4500, 5500]
    assert result.equity["receivable"].tolist() == [0, 1000, 0]
    assert result.positions["sellable_quantity"].tolist() == [0, 0, 1000]
    assert result.orders.iloc[-1]["filled_qty"] == 1000
    assert result.orders.iloc[-1]["reason"] == "t_plus_one"
    assert result.cash_events["event"].tolist() == ["dividend_receivable", "share_change", "dividend_paid"]


def test_missing_bar_does_not_fill_and_missing_held_mark_fails():
    bars = pd.DataFrame([bar(FRIDAY)])
    result = simulate(bars, [FRIDAY, MONDAY], {MONDAY: {ETF: 1}}, initial_cash=10_000)
    assert result.orders.iloc[0]["reason"] == "missing_bar"
    with pytest.raises(LedgerError, match="missing valuation for held"):
        simulate(bars, [FRIDAY, MONDAY], {FRIDAY: {ETF: 1}}, initial_cash=10_000)


def test_unexplained_ex_rights_reference_is_not_used_as_clean_market_return():
    bars = pd.DataFrame([bar(FRIDAY), bar(MONDAY, preclose=9.5)])
    with pytest.raises(LedgerError, match="unexplained ex-rights reference"):
        simulate(bars, [FRIDAY, MONDAY], {FRIDAY: {ETF: 1}}, initial_cash=10_000)


def test_weight_cannot_use_leverage_and_small_target_is_visible():
    bars = pd.DataFrame([bar(FRIDAY)])
    with pytest.raises(LedgerError, match="leverage"):
        simulate(bars, [FRIDAY], {FRIDAY: {ETF: 1.01}}, initial_cash=1000)
    result = simulate(bars, [FRIDAY], {FRIDAY: {ETF: 0.35}}, initial_cash=1000)
    assert (result.orders.iloc[0]["requested_qty"], result.orders.iloc[0]["reason"]) == (35, "board_lot")


def test_etf_fractional_split_truncates_to_integer_shares_while_stock_rejects():
    friday_bar = {"open": 2.243, "high": 2.249, "low": 2.151, "close": 2.243, "upper_limit": 2.467, "lower_limit": 2.019}
    ex_bar = {"open": 8.001, "high": 8.100, "low": 7.790, "close": 7.818, "preclose": 8.001,
              "upper_limit": 8.801, "lower_limit": 7.201}
    bars = pd.DataFrame([bar(FRIDAY, **friday_bar), bar(MONDAY, **ex_bar)])
    actions = pd.DataFrame([{
        "symbol": ETF, "record_date": FRIDAY, "ex_date": MONDAY, "pay_date": None,
        "share_available_date": MONDAY, "cash_per_share": 0.0, "share_multiplier": 0.2803,
    }])
    result = simulate(bars, [FRIDAY, MONDAY], {FRIDAY: {ETF: 1.0}}, actions, initial_cash=1_000, cost=Costs(0, 0, 0))
    # 400 shares * 0.2803 = 112.12 -> truncated to 112 integer shares
    assert result.positions["quantity"].tolist() == [400, 112]
    assert result.cash_events.iloc[0]["quantity_delta"] == -288

    stock_bars = pd.DataFrame([bar(FRIDAY, STOCK), bar(MONDAY, STOCK)])
    stock_actions = actions.assign(symbol=STOCK)
    with pytest.raises(LedgerError, match="fractional units"):
        simulate(stock_bars, [FRIDAY, MONDAY], {FRIDAY: {STOCK: 1.0}}, stock_actions, initial_cash=10_000, cost=Costs(0, 0, 0))
