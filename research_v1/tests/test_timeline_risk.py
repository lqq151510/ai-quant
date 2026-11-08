import numpy as np
import pandas as pd
import pytest

from quant_research.risk import block_resample, summarize
from quant_research.timeline import execution_schedule, label_windows, open_labels, purge_training, select_stock_cohort


def test_signal_does_not_fill_on_same_close_and_gap_uses_sessions():
    dates = pd.bdate_range("2024-09-23", periods=40).difference(pd.bdate_range("2024-10-01", "2024-10-07"))
    schedule = execution_schedule(dates, {pd.Timestamp("2024-09-30"): {"510300": 0.35}})
    assert list(schedule) == [pd.Timestamp("2024-10-08")]
    for h in (5, 20):
        window = label_windows(dates, h)
        assert window.loc[0, "entry"] == dates[1]
        assert window.loc[0, "label_end"] == dates[h + 1]
        cutoff = dates[h + 4]
        train = purge_training(window, cutoff)
        assert (train["label_end"] < cutoff).all()
        assert cutoff not in train["label_end"].to_list()


def test_actions_in_open_labels_no_double_count_or_future_fill():
    dates = pd.bdate_range("2024-01-02", periods=30)
    bars = pd.DataFrame({"date": dates, "symbol": "510300", "open": [10.] * 3 + [9.] * 27, "tradable": True})
    actions = pd.DataFrame([{"symbol": "510300", "ex_date": dates[3], "pay_date": dates[9],
                             "record_date": dates[2], "cash_per_share": 1., "share_multiplier": 1.}])
    labels = open_labels(bars, dates, 5, actions)
    assert labels.loc[0, "target_return"] == pytest.approx(0)
    assert labels.loc[2, "target_return"] == pytest.approx(0)  # buy on ex date has no dividend
    assert labels.tail(6)["target_return"].isna().all()


def test_labels_and_features_reject_conflicting_corporate_actions():
    from quant_research.baseline import market_close_returns

    dates = pd.bdate_range("2024-01-02", periods=30)
    bars = pd.DataFrame({"date": dates, "symbol": "510300", "open": [10.] * 3 + [9.] * 27,
                         "close": [10.] * 3 + [9.] * 27, "tradable": True})
    action = pd.DataFrame([{"symbol": "510300", "ex_date": dates[3], "pay_date": dates[9],
                            "record_date": dates[2], "cash_per_share": 1., "share_multiplier": 1.}])
    for invalid in (pd.concat([action, action], ignore_index=True), action.assign(record_date=dates[10])):
        with pytest.raises(ValueError):
            open_labels(bars, dates, 5, invalid)
        with pytest.raises(ValueError):
            market_close_returns(bars, invalid, dates)


def test_halted_split_mark_carries_forward_without_an_artificial_return():
    from quant_research.baseline import market_close_returns

    dates = pd.to_datetime(["2015-04-10", "2015-04-13", "2015-04-14", "2015-04-15"])
    bars = pd.DataFrame({"date": dates, "symbol": "510500", "close": [2.243, 2.243, 2.243, 7.818],
                         "valuation_close": [np.nan, np.nan, 8.001, np.nan]})
    actions = pd.DataFrame([{"symbol": "510500", "ex_date": dates[2], "pay_date": pd.NaT,
                             "record_date": dates[1], "cash_per_share": 0., "share_multiplier": 0.28032483}])
    returns = market_close_returns(bars, actions, dates)["510500"]
    assert returns.iloc[1:3].tolist() == [0., 0.]
    assert returns.iloc[3] == pytest.approx(7.818 / 8.001 - 1)


def test_future_membership_and_todays_turnover_cannot_change_past_cohort():
    dates = pd.bdate_range("2024-01-02", periods=90)
    rows = [dict(date=d, symbol=s, kind="stock", amount=a) for d in dates
            for s, a in (("600000", 100), ("600001", 50))]
    bars = pd.DataFrame(rows)
    membership = pd.DataFrame([{"symbol": "600000", "effective_date": dates[0], "known_at": "2023-12-31T00:00:00Z"}])
    original = select_stock_cohort(bars, membership, dates)
    future = pd.DataFrame([{"symbol": "600001", "effective_date": dates[0], "known_at": "2025-01-01T00:00:00Z"}])
    changed = select_stock_cohort(bars, pd.concat([membership, future], ignore_index=True), dates)
    assert original == changed
    assert original[dates[-1]] == ("600000",)


def test_drawdown_includes_initial_cash():
    summary = summarize(pd.DataFrame({"nav": [9000.]}), pd.DataFrame(), 10000)
    assert summary["max_drawdown"] == pytest.approx(0.1)
    assert summary["net_return"] == pytest.approx(-0.1)


def test_buy_and_hold_does_not_invent_free_daily_rebalancing():
    from quant_research.ledger import Costs, simulate

    dates = pd.bdate_range("2024-01-02", periods=4)
    rows = []
    for symbol, prices in (("510050", [10., 20., 20., 20.]), ("510300", [10., 10., 20., 20.])):
        for day, price in zip(dates, prices, strict=True):
            rows.append(dict(date=day, symbol=symbol, kind="etf", open=price, high=price, low=price,
                             close=price, volume=100000, amount=price*100000, tradable=True,
                             upper_limit=1000., lower_limit=0.01))
    result = simulate(pd.DataFrame(rows), dates, {dates[0]: {"510050": 0.5, "510300": 0.5}},
                      initial_cash=10000, cost=Costs(commission_rate=0, minimum_commission=0, slippage_bps=0))
    assert result.equity["nav"].tolist() == [10000, 15000, 20000, 20000]
    assert len(result.orders) == 2
    assert set(result.positions["quantity"]) == {500}


def test_paired_bootstrap_deterministic_and_same_path_for_same_strategy():
    r = pd.Series(np.random.default_rng(9).normal(0, 0.01, 120))
    result = block_resample(r, r, block=20, paths=200)
    assert result == block_resample(r, r, block=20, paths=200)
    assert result["paired_terminal_excess_interval"] == [0, 0]
    assert 0 <= result["loss_path_fraction"] <= 1
    with pytest.raises(ValueError):
        block_resample(r, r.iloc[::-1], block=20)


def test_cash_bootstrap_has_no_loss_or_drawdown():
    r = pd.Series(np.zeros(100))
    result = block_resample(r, r, block=20, paths=100)
    assert result["loss_path_fraction"] == 0
    assert result["drawdown_above_15pct_fraction"] == 0
