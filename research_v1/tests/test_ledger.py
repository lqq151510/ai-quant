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


def test_510500_2015_halt_uses_split_mark_and_odd_units_sell_after_resume():
    symbol = "510500"
    last_trade, record, ex, resume = ("2015-04-10", "2015-04-13", "2015-04-14", "2015-04-15")
    bars = pd.DataFrame([
        bar(last_trade, symbol, kind="etf", open=2.165, high=2.249, low=2.151, close=2.243,
            preclose=2.176, lower_limit=1.958, upper_limit=2.394, volume=37690283),
        bar(record, symbol, kind="etf", open=2.243, high=2.243, low=2.243, close=2.243,
            preclose=2.243, tradable=False, volume=0, amount=0),
        bar(ex, symbol, kind="etf", open=2.243, high=2.243, low=2.243, close=2.243,
            preclose=2.243, tradable=False, volume=0, amount=0, valuation_close=8.001),
        bar(ex, open=4.395, high=4.458, low=4.370, close=4.406,
            preclose=4.404, lower_limit=3.964, upper_limit=4.844, volume=984846384),
        bar(resume, symbol, kind="etf", open=8.100, high=8.100, low=7.790, close=7.818,
            preclose=8.001, lower_limit=7.201, upper_limit=8.801, volume=26064505),
        bar(resume, open=4.406, high=4.450, low=4.364, close=4.369,
            preclose=4.406, lower_limit=3.965, upper_limit=4.847, volume=1191779369),
    ])
    actions = pd.DataFrame([{
        "symbol": symbol, "record_date": record, "ex_date": ex, "pay_date": None,
        "share_available_date": resume, "cash_per_share": 0.0, "share_multiplier": 0.28032483,
        "split_rounding": "floor",
    }])
    result = simulate(
        bars, [last_trade, record, ex, resume],
        {last_trade: {symbol: 0.7}, record: {}, ex: {ETF: 0.25}, resume: {}}, actions,
        initial_cash=10_000, cost=Costs(0, 0, 0),
    )
    holdings = result.positions.loc[result.positions["symbol"].eq(symbol)]
    assert holdings["quantity"].tolist() == [3200, 3200, 897]
    assert holdings["sellable_quantity"].tolist() == [0, 3200, 0]
    assert holdings["close"].tolist() == [2.243, 2.243, 8.001]
    assert holdings["market_value"].tolist() == pytest.approx([7177.6, 7177.6, 7176.90])
    orders = result.orders.loc[result.orders["symbol"].eq(symbol)]
    assert orders["reason"].tolist() == ["", "suspended", "suspended", ""]
    assert orders["filled_qty"].tolist() == [3200, 0, 0, 897]
    assert orders.iloc[-1]["price"] == 8.100
    # Sizing another ETF includes the adjusted value of the halted position.
    other_orders = result.orders.loc[result.orders["symbol"].eq(ETF)]
    assert other_orders.iloc[0]["filled_qty"] == 500
    assert bars.loc[(bars["symbol"] == symbol) & (bars["date"] == ex), "close"].item() == 2.243


def test_510500_2022_rounds_account_once_after_record_close():
    symbol = "510500"
    first, record, ex, last = ("2022-08-25", "2022-08-26", "2022-08-29", "2022-08-30")
    bars = pd.DataFrame([
        bar(first, symbol, kind="etf", open=7.196, high=7.240, low=7.140, close=7.236,
            preclose=7.197, lower_limit=6.477, upper_limit=7.917),
        bar(record, symbol, kind="etf", open=7.228, high=7.303, low=7.195, close=7.220,
            preclose=7.236, lower_limit=6.512, upper_limit=7.960),
        bar(ex, symbol, kind="etf", open=6.200, high=6.310, low=6.199, close=6.300,
            preclose=6.304, lower_limit=5.674, upper_limit=6.934),
        bar(last, symbol, kind="etf", open=6.300, high=6.310, low=6.243, close=6.291,
            preclose=6.300, lower_limit=5.670, upper_limit=6.930),
    ])
    action = {
        "symbol": symbol, "record_date": record, "ex_date": ex, "pay_date": None,
        "share_available_date": ex, "cash_per_share": 0.0, "share_multiplier": 1.14539,
    }
    targets = {first: {symbol: 0.10}, record: {symbol: 0.23}, last: {}}
    days = [first, record, ex, last]
    result = simulate(bars, days, targets, pd.DataFrame([action | {"split_rounding": "ceil"}]),
                      initial_cash=10_000, cost=Costs(0, 0, 0))
    assert result.orders["filled_qty"].tolist() == [100, 200, 344]
    assert result.positions["quantity"].tolist() == [100, 300, 344]
    assert result.positions["sellable_quantity"].tolist() == [0, 100, 344]
    assert result.positions["close"].tolist() == [7.236, 7.220, 6.300]
    assert result.positions["market_value"].tolist() == pytest.approx([723.6, 2166.0, 2167.2])
    assert result.cash_events.iloc[0]["date"].isoformat() == ex
    assert result.cash_events.iloc[0]["quantity_delta"] == 44
    with pytest.raises(LedgerError, match="split creates fractional"):
        simulate(bars, days, targets, pd.DataFrame([action]), initial_cash=10_000, cost=Costs(0, 0, 0))


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


def test_close_policy_executes_at_next_open_and_only_earns_after_fill():
    bars = pd.DataFrame([
        bar(FRIDAY),
        bar(MONDAY, open=10.5, high=10.5, low=9.45, close=9.45),
        bar(TUESDAY, open=9.45, high=9.45, low=9.45, close=9.45),
    ])
    observed = []

    def policy(state):
        observed.append(state)
        return {ETF: 1.0} if state.day.isoformat() == FRIDAY else {}

    result = simulate(
        bars, [FRIDAY, MONDAY, TUESDAY], {},
        initial_cash=10_500, cost=Costs(0, 0, 0), on_close=policy,
    )
    buy, sell = result.orders.itertuples(index=False)
    assert (buy.date.isoformat(), buy.price, buy.filled_qty) == (MONDAY, 10.5, 1000)
    assert (sell.date.isoformat(), sell.price, sell.filled_qty) == (TUESDAY, 9.45, 1000)
    assert result.positions["sellable_quantity"].tolist() == [0]
    assert result.equity["nav"].tolist() == [10_500, 9_450, 9_450]
    assert result.equity["daily_return"].tolist() == pytest.approx([0, -0.1, 0])
    assert [state.day.isoformat() for state in observed] == [FRIDAY, MONDAY]
    assert observed[0].cash_weight == 1.0
    assert dict(observed[1].weights) == {ETF: 1.0}
    assert observed[1].drawdown == pytest.approx(0.1)
    assert result.close_state.day.isoformat() == TUESDAY
    assert result.close_state.cash_weight == 1.0
    assert result.close_state.nav == 9_450


def test_terminal_close_separates_receivable_from_available_cash():
    bars = pd.DataFrame([bar(FRIDAY), bar(MONDAY), bar(TUESDAY, open=9.0, low=9.0, close=9.0)])
    actions = pd.DataFrame([{
        "symbol": ETF, "record_date": MONDAY, "ex_date": TUESDAY, "pay_date": "2023-08-30",
        "cash_per_share": 1.0, "share_multiplier": 1.0,
    }])

    def policy(state):
        return {ETF: 1.0} if state.day.isoformat() == FRIDAY else None

    result = simulate(
        bars, [FRIDAY, MONDAY, TUESDAY], {}, actions,
        initial_cash=10_000, cost=Costs(0, 0, 0), on_close=policy,
    )
    assert len(result.orders) == 1
    assert result.close_state.nav == 10_000
    assert result.close_state.cash_weight == 0
    assert dict(result.close_state.weights) == {ETF: 0.9}
    with pytest.raises(TypeError):
        result.close_state.weights[ETF] = 1.0


def test_close_policy_uses_target_validation_and_rejects_mixed_schedules():
    bars = pd.DataFrame([bar(FRIDAY), bar(MONDAY)])
    with pytest.raises(LedgerError, match="leverage"):
        simulate(bars, [FRIDAY, MONDAY], {}, initial_cash=10_000, on_close=lambda _: {ETF: 1.01})
    with pytest.raises(LedgerError, match="cannot be combined"):
        simulate(bars, [FRIDAY, MONDAY], {FRIDAY: {}}, initial_cash=10_000, on_close=lambda _: None)
