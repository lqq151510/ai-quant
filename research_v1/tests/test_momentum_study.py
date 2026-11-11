from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant_research.ledger import Costs
from quant_research.momentum_study import (
    ETF_SYMBOLS, load_market, momentum_scores, momentum_target, require_unsealed,
    review_days, run_policy, run_study, verify_prior,
)


def fixture_market(count=170):
    days = pd.bdate_range("2020-01-02", periods=count)
    rows = []
    for i, day in enumerate(days):
        for j, symbol in enumerate(ETF_SYMBOLS):
            close = 10 * (1 + 0.001 * (j + 1)) ** i
            rows.append({
                "date": day, "symbol": symbol, "kind": "etf", "open": close,
                "close": close, "high": close, "low": close, "volume": 1_000_000,
                "amount": close * 1_000_000, "tradable": True,
                "upper_limit": close * 1.1, "lower_limit": close * 0.9,
            })
    actions = pd.DataFrame(columns=["symbol", "ex_date", "pay_date", "cash_per_share", "share_multiplier"])
    return days, {"bars": pd.DataFrame(rows), "actions": actions, "calendar": pd.DataFrame({"date": days})}


def test_126_returns_require_127_closes_without_filling_missing_prices():
    days, frames = fixture_market()
    scores = momentum_scores(frames["bars"], frames["actions"], days)
    assert scores.iloc[:126].isna().all().all()
    assert scores.iloc[126].notna().all()
    assert scores.iloc[126, 0] == pytest.approx(1.001 ** 126 - 1)
    missing = frames["bars"].loc[~(frames["bars"]["date"].eq(days[40]) & frames["bars"]["symbol"].eq(ETF_SYMBOLS[0]))]
    changed = momentum_scores(missing, frames["actions"], days)
    assert np.isnan(changed.loc[days[126], ETF_SYMBOLS[0]])
    assert ETF_SYMBOLS[0] not in momentum_target(changed.loc[days[126]])


def test_positive_only_slots_and_deterministic_ties():
    scores = pd.Series({ETF_SYMBOLS[3]: 0.1, ETF_SYMBOLS[1]: 0.1, ETF_SYMBOLS[2]: 0.1, ETF_SYMBOLS[0]: 0.1})
    assert momentum_target(scores) == {ETF_SYMBOLS[0]: 0.35, ETF_SYMBOLS[1]: 0.35}
    assert momentum_target(pd.Series({ETF_SYMBOLS[0]: 0.1, ETF_SYMBOLS[1]: -0.1})) == {ETF_SYMBOLS[0]: 0.35}
    assert momentum_target(pd.Series({ETF_SYMBOLS[0]: 0, ETF_SYMBOLS[1]: -0.1, ETF_SYMBOLS[2]: np.nan})) == {}


def test_future_bars_and_future_actions_do_not_change_past_scores_or_targets():
    days, frames = fixture_market()
    cutoff = days[140]
    original = momentum_scores(frames["bars"], frames["actions"], days)
    changed = frames["bars"].copy()
    changed.loc[changed["date"].gt(cutoff), "close"] *= 0.5
    action = pd.DataFrame([{
        "symbol": ETF_SYMBOLS[0], "ex_date": days[150], "pay_date": days[151],
        "cash_per_share": 0.1, "share_multiplier": 2.0,
    }])
    perturbed = momentum_scores(changed, action, days)
    pd.testing.assert_frame_equal(original.loc[:cutoff], perturbed.loc[:cutoff])
    assert momentum_target(original.loc[cutoff]) == momentum_target(perturbed.loc[cutoff])
    truncated = momentum_scores(frames["bars"].loc[frames["bars"]["date"].le(cutoff)], action, days[days <= cutoff])
    pd.testing.assert_frame_equal(original.loc[:cutoff], truncated)


def test_split_and_cash_dividend_preserve_true_total_return():
    days, frames = fixture_market(140)
    bars = frames["bars"].copy()
    symbol = ETF_SYMBOLS[0]
    mask = bars["symbol"].eq(symbol)
    bars.loc[mask, "close"] = 10.0
    bars.loc[mask & bars["date"].ge(days[80]), "close"] = 5.0
    bars.loc[mask & bars["date"].ge(days[110]), "close"] = 4.9
    actions = pd.DataFrame([
        {"symbol": symbol, "ex_date": days[80], "pay_date": pd.NaT,
         "cash_per_share": 0.0, "share_multiplier": 2.0},
        {"symbol": symbol, "ex_date": days[110], "pay_date": days[111],
         "cash_per_share": 0.1, "share_multiplier": 1.0},
    ])
    scores = momentum_scores(bars, actions, days)
    assert scores.loc[days[126]:, symbol].to_numpy() == pytest.approx(np.zeros(14), abs=1e-12)


def test_monthly_review_and_next_open_execution_use_common_anchor():
    full_days, frames = fixture_market()
    days = full_days[126:170]
    scores = momentum_scores(frames["bars"], frames["actions"], full_days)
    reviews = review_days(days)
    assert reviews.tolist() == [days[0], days[20], days[40]]
    stats, result, decisions = run_policy(frames, days, reviews, 100_000, Costs(), "dual_momentum126_monthly", scores)
    assert stats["review_count"] == 3
    assert decisions["eligible_open_date"].tolist() == [days[1].date(), days[21].date(), days[41].date()]
    assert set(result.orders["date"]) <= set(decisions["eligible_open_date"])
    assert result.orders["date"].min() == days[1].date()
    assert decisions.iloc[0]["target_gross"] == pytest.approx(0.7)
    assert stats["ledger_reconciliation_error"] < 1e-6
    assert stats["min_cash"] >= 0


def test_no_positive_scores_exit_holdings_instead_of_skipping():
    full_days, frames = fixture_market()
    days = full_days[126:150]
    scores = pd.DataFrame(0.1, index=full_days, columns=ETF_SYMBOLS)
    scores.loc[days[20]:] = -0.1
    stats, result, decisions = run_policy(frames, days, review_days(days), 100_000, Costs(), "dual_momentum126_monthly", scores)
    assert decisions["target_gross"].tolist() == pytest.approx([0.7, 0])
    sells = result.orders.loc[result.orders["side"].eq("sell")]
    assert len(sells) == 2
    assert set(sells["date"]) == {days[21].date()}
    assert result.equity.iloc[-1]["market_value"] == 0


@pytest.mark.parametrize("end", ["2025-01-01", "2026-11-11"])
def test_holdout_is_rejected_before_files_are_opened(tmp_path, end):
    with pytest.raises(ValueError, match="sealed holdout"):
        require_unsealed(end)
    with pytest.raises(ValueError, match="sealed holdout"):
        load_market(tmp_path / "missing", end=end)


def test_prior_baseline_mismatch_fails_closed():
    verify_prior({"net_return": 0.1, "gross": {"mean": 0.7}}, {"net_return": 0.1, "gross": {"mean": 0.7}})
    with pytest.raises(ValueError, match="mismatch"):
        verify_prior({"net_return": 0.099}, {"net_return": 0.1})
    with pytest.raises(ValueError, match="missing metric"):
        verify_prior({}, {"net_return": 0.1})


def test_failed_run_preserved_and_output_cannot_be_overwritten(tmp_path):
    out = tmp_path / "run"
    with pytest.raises(FileNotFoundError):
        run_study(tmp_path / "missing-snapshot", out)
    assert (out / "failure.json").is_file()
    before = (out / "failure.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_study(tmp_path / "missing-snapshot", out)
    assert (out / "failure.json").read_bytes() == before
