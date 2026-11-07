"""Focused tests for the isolated research experiment; no network or real accounts."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance

from src.research_upgrades.data import (
    _FIVE_MINUTE_ENDS, _market_for_symbol, _normalize_flow, fetch_akshare_inputs, intraday_variance, prepare_feature_panel,
)
from src.research_upgrades.experiment import (
    Dataset, build_dataset, fit_lowdin, mask_unseen_feature_groups, run_experiment, split_dataset, synthetic_bars, weighted_daily_ic_loss,
)
from src.research_upgrades.data import ALL_FEATURES
from src.research_upgrades.transforms import (
    LowdinTransform, WassersteinRegimeMonitor, exponential_time_weights, macro_distribution_observations,
)


def minutes_for_day(symbol="600900", day="2026-01-05"):
    timestamps = pd.to_datetime([f"{day} {clock}" for clock in _FIVE_MINUTE_ENDS])
    # A two-sided path, with the first five-minute open independent of yesterday.
    steps = np.array([0.002, -0.001] * 24)
    prices = 10 * np.exp(np.cumsum(steps))
    return pd.DataFrame({"symbol": symbol, "timestamp": timestamps, "open": np.r_[10, prices[:-1]], "close": prices})


class TransformTests(unittest.TestCase):
    def test_lowdin_whitens_training_covariance_and_reuses_parameters(self):
        rng = np.random.default_rng(19)
        x = rng.normal(size=(500, 3)) @ np.array([[1, 0.8, 0.2], [0, 0.5, 0.2], [0, 0, 0.3]])
        fitted = LowdinTransform().fit(x)
        z = fitted.transform(x)
        np.testing.assert_allclose(z.T @ z / len(z), np.eye(3), atol=1e-10)
        before = fitted.matrix.copy()
        fitted.transform(x * 20 + 7)
        np.testing.assert_array_equal(fitted.matrix, before)
        self.assertEqual(fitted.rank, 3)

    def test_lowdin_rank_deficiency_gives_projection_not_identity(self):
        a = np.arange(20, dtype=float)
        x = np.column_stack([a, a, np.ones_like(a)])
        fitted = LowdinTransform().fit(x)
        z = fitted.transform(x)
        self.assertEqual(fitted.rank, 1)
        np.testing.assert_allclose(z.T @ z / len(z), fitted.projection, atol=1e-12)
        self.assertFalse(np.allclose(fitted.projection, np.eye(3)))

    def test_half_life_and_minibatch_weight_are_effective(self):
        weights = exponential_time_weights(pd.bdate_range("2025-01-01", periods=121), 120)
        self.assertAlmostEqual(weights[-1] / weights[0], 2.0)
        self.assertAlmostEqual(weights.mean(), 1.0)
        pred = torch.tensor([[1.0, 0.0, -1.0]])
        target = torch.tensor([[-1.0, 0.0, 1.0]])
        one = weighted_daily_ic_loss(pred, target, torch.tensor([1.0]))
        half = weighted_daily_ic_loss(pred, target, torch.tensor([0.5]))
        self.assertAlmostEqual(float(half), float(one) / 2, places=6)
        with self.assertRaises(ValueError):
            exponential_time_weights(["2026-01-02", "2026-01-01"])

    def test_lowdin_dataset_fits_only_training_and_keeps_other_features(self):
        panel = prepare_feature_panel(synthetic_bars(260))
        train, val, test = split_dataset(build_dataset(panel))
        untouched = train.x[..., 6:].copy()
        original_train = train.x.copy()
        fitted = fit_lowdin(train, val, test)
        self.assertEqual(fitted.rank, 6)
        z = train.x[:, :, -1, :6].reshape(-1, 6).astype(float)
        np.testing.assert_allclose(z.T @ z / len(z), np.eye(6), atol=1e-5)
        np.testing.assert_array_equal(train.x[..., 6:], untouched)
        np.testing.assert_allclose(fitted.mean, original_train[:, :, -1, :6].reshape(-1, 6).mean(axis=0), atol=1e-6)

    def test_test_only_flow_is_reported_but_cannot_reach_untrained_model(self):
        shape = (2, 3, 5, len(ALL_FEATURES))
        def dataset():
            return Dataset(pd.date_range("2026-01-01", periods=2), ["a", "b", "c"],
                           np.zeros(shape, dtype=np.float32), np.zeros((2, 3), dtype=np.float32))
        train, val, test = dataset(), dataset(), dataset()
        flag, feature = (ALL_FEATURES.index(n) for n in ("flow_available", "flow_super_ratio"))
        test.x[..., flag] = 1
        test.x[..., feature] = 10
        status = mask_unseen_feature_groups(train, val, test)
        self.assertEqual(status["flow"]["coverage_before_mask"]["test"], 1)
        self.assertFalse(status["flow"]["enabled"])
        self.assertTrue((test.x[..., [flag, feature]] == 0).all())


class FeatureTests(unittest.TestCase):
    def test_akshare_percent_units(self):
        raw = pd.DataFrame({"日期": ["2026-01-05"], "超大单净流入-净占比": ["3.5"], "大单净流入-净占比": ["-2"]})
        flow = _normalize_flow(raw, "600900")
        self.assertAlmostEqual(flow.iloc[0]["flow_super_ratio"], 0.035)
        self.assertAlmostEqual(flow.iloc[0]["flow_large_ratio"], -0.02)

    def test_complete_intraday_ratio_excludes_overnight(self):
        frame = minutes_for_day()
        output = intraday_variance(frame)
        self.assertEqual(len(output), 1)
        expected = np.log(((24 * 0.002 ** 2) + 1e-12) / ((24 * 0.001 ** 2) + 1e-12))
        self.assertAlmostEqual(output.iloc[0]["intraday_log_up_down_variance"], expected, places=9)
        self.assertTrue(intraday_variance(frame.iloc[:-1]).empty)
        self.assertTrue(intraday_variance(pd.concat([frame, frame.iloc[[0]]])).empty)
        wrong_seconds = frame.copy()
        wrong_seconds.loc[0, "timestamp"] += pd.Timedelta(seconds=30)
        self.assertTrue(intraday_variance(wrong_seconds).empty)

    def test_exchange_selection_including_beijing(self):
        self.assertEqual(_market_for_symbol("600900"), "sh")
        self.assertEqual(_market_for_symbol("159001"), "sz")
        self.assertEqual(_market_for_symbol("920001"), "bj")

    def test_missing_flow_is_distinct_from_observed_zero_and_does_not_fill(self):
        bars = synthetic_bars(240)
        date = bars.trade_date.min()
        flow = pd.DataFrame({"symbol": ["600900", "510300"], "trade_date": [date, date],
                             "flow_super_ratio": [0.0, 0.1], "flow_large_ratio": [0.0, -0.2]})
        panel = prepare_feature_panel(bars, flow)
        observed = panel[(panel.symbol == "600900") & (panel.trade_date == date)].iloc[0]
        missing = panel[(panel.symbol == "002594") & (panel.trade_date == date)].iloc[0]
        tomorrow = panel[(panel.symbol == "600900") & (panel.trade_date > date)].iloc[0]
        self.assertEqual(observed.flow_available, 1)
        self.assertEqual(missing.flow_available, 0)
        self.assertEqual(missing.flow_super_ratio, 0)
        self.assertEqual(tomorrow.flow_available, 0)

    def test_same_count_different_symbols_is_not_balanced(self):
        bars = synthetic_bars(240)
        first = bars.trade_date.min()
        mask = (bars.trade_date == first) & (bars.symbol == "600900")
        bars.loc[mask, "symbol"] = "600901"
        with self.assertRaisesRegex(ValueError, "balanced"):
            prepare_feature_panel(bars)

    def test_features_do_not_change_when_future_bars_change(self):
        bars = synthetic_bars(240)
        cutoff = sorted(bars.trade_date.unique())[150]
        first = prepare_feature_panel(bars)
        changed = bars.copy()
        for col in ("open", "high", "low", "close"):
            changed.loc[changed.trade_date > cutoff, col] *= 1.9
        second = prepare_feature_panel(changed)
        cols = ["ret_3d", "ret_20d", "clv", "flow_available"]
        np.testing.assert_allclose(first.loc[first.trade_date <= cutoff, cols], second.loc[second.trade_date <= cutoff, cols])

    def test_rate_limit_stops_all_remaining_requests_and_restores_timeout(self):
        import akshare
        from akshare.utils.context import config
        from akshare.utils.http import Push2RateLimitError

        previous = config.timeout
        with patch.object(akshare, "stock_individual_fund_flow", side_effect=Push2RateLimitError("limited")) as get_flow, \
             patch.object(akshare, "fund_etf_hist_min_em") as get_minute:
            flow, minute, status = fetch_akshare_inputs(["510300", "600900"], timeout=4)
        self.assertEqual(get_flow.call_count, 1)
        get_minute.assert_not_called()
        self.assertTrue(flow.empty and minute.empty)
        self.assertEqual(status["results"][0]["status"], "rate_limited")
        self.assertTrue(all(s["status"] == "skipped_rate_limit" for s in status["results"][1:]))
        self.assertEqual(config.timeout, previous)

    def test_stock_and_etf_use_expected_akshare_interfaces(self):
        import akshare

        raw_flow = pd.DataFrame({"日期": ["2026-01-05"], "超大单净流入-净占比": ["1.5"],
                                 "大单净流入-净占比": ["-2.0"]})
        minute = minutes_for_day()
        raw_minute = minute.rename(columns={"timestamp": "时间", "open": "开盘", "close": "收盘"})
        with patch.object(akshare, "stock_individual_fund_flow", return_value=raw_flow) as flow_call, \
             patch.object(akshare, "stock_zh_a_hist_min_em", return_value=raw_minute) as stock_call, \
             patch.object(akshare, "fund_etf_hist_min_em", return_value=raw_minute) as etf_call:
            flow, minute_out, status = fetch_akshare_inputs(["600900", "510300"])
        self.assertEqual(flow_call.call_count, 2)
        stock_call.assert_called_once_with(symbol="600900", period="5", adjust="")
        etf_call.assert_called_once_with(symbol="510300", period="5", adjust="")
        self.assertEqual(len(flow), 2)
        self.assertEqual(len(minute_out), 96)
        self.assertTrue(all(result["status"] == "ok" for result in status["results"]))
        self.assertEqual(set(intraday_variance(minute_out).symbol), {"600900", "510300"})

    def test_complete_minute_session_is_joined_by_symbol_and_date(self):
        bars = synthetic_bars(240)
        day = pd.Timestamp(bars.trade_date.iloc[0])
        minute = minutes_for_day(day=day.strftime("%Y-%m-%d"))
        panel = prepare_feature_panel(bars, minutes=minute)
        one = panel[(panel.symbol == "600900") & (panel.trade_date == day)].iloc[0]
        another = panel[(panel.symbol == "510300") & (panel.trade_date == day)].iloc[0]
        self.assertEqual(one.intraday_available, 1)
        self.assertEqual(another.intraday_available, 0)


class DistributionTests(unittest.TestCase):
    def test_wasserstein_uses_non_overlapping_history_through_current_day(self):
        bars = synthetic_bars(240)
        obs = macro_distribution_observations(bars, reference_days=12, recent_days=5)
        raw = bars.sort_values(["symbol", "trade_date"]).copy()
        raw["r"] = raw.groupby("symbol").close.pct_change(fill_method=None)
        groups = [g.r.dropna().to_numpy() for _, g in raw.groupby("trade_date") if g.r.notna().sum() > 1]
        expected = wasserstein_distance(np.concatenate(groups[:12]), np.concatenate(groups[12:17]))
        self.assertAlmostEqual(obs.iloc[0].wasserstein, expected, places=12)

    def test_future_changes_cannot_change_earlier_distances_or_training_threshold(self):
        bars = synthetic_bars(260)
        cutoff = sorted(bars.trade_date.unique())[200]
        first = macro_distribution_observations(bars, reference_days=40, recent_days=10)
        changed = bars.copy()
        changed.loc[changed.trade_date > cutoff, "close"] *= np.linspace(1, 3, (changed.trade_date > cutoff).sum())
        second = macro_distribution_observations(changed, reference_days=40, recent_days=10)
        original_train = first[first.trade_date <= cutoff]
        changed_train = second[second.trade_date <= cutoff]
        pd.testing.assert_frame_equal(original_train, changed_train)
        monitor = WassersteinRegimeMonitor().fit(original_train)
        threshold = monitor.threshold
        before = monitor.filter_sequence(first)
        after = monitor.filter_sequence(second)
        self.assertEqual(monitor.threshold, threshold)
        for a, b in zip(before, after):
            if a.trade_date <= cutoff:
                np.testing.assert_allclose(a.probabilities, b.probabilities)
                self.assertEqual(a.distribution_alert, b.distribution_alert)

    def test_gap_in_symbol_history_is_not_a_single_day_return(self):
        bars = synthetic_bars(240)
        dates = sorted(bars.trade_date.unique())
        t0, t1 = dates[100], dates[101]
        bars = bars[~((bars.symbol == "600900") & (bars.trade_date == t0))].copy()
        obs = macro_distribution_observations(bars, reference_days=12, recent_days=5)
        day_return = []
        for _, group in bars[bars.symbol != "600900"].groupby("symbol"):
            previous = float(group.loc[group.trade_date == t0, "close"].iloc[0])
            current = float(group.loc[group.trade_date == t1, "close"].iloc[0])
            day_return.append(current / previous - 1)
        actual = obs.loc[obs.trade_date == t1, "mkt_ret"].iloc[0]
        self.assertAlmostEqual(actual, float(np.mean(day_return)), places=12)


class IntegrationTests(unittest.TestCase):
    def test_training_pipeline_uses_observed_mock_flow_and_minutes(self):
        bars = synthetic_bars(240)
        dates = sorted(bars.trade_date.unique())
        symbols = sorted(bars.symbol.unique())
        flow = pd.DataFrame([
            {"symbol": symbol, "trade_date": dates[day], "flow_super_ratio": 0.01 * index,
             "flow_large_ratio": -0.02 * index}
            for day in (60, 220) for index, symbol in enumerate(symbols)
        ])
        minutes = pd.concat([minutes_for_day(day=pd.Timestamp(dates[day]).strftime("%Y-%m-%d"))
                             for day in (60, 220)], ignore_index=True)
        extra_market_symbol = bars[bars.symbol == "600900"].assign(symbol="600901")
        macro_bars = pd.concat([bars, extra_market_symbol], ignore_index=True)
        old_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            report, _, _, _, monitor = run_experiment(bars, flow, minutes, epochs=1, macro_bars=macro_bars)
        finally:
            torch.set_num_threads(old_threads)
        for group in ("flow", "intraday"):
            self.assertTrue(report["feature_groups"][group]["enabled"])
            self.assertGreater(report["feature_groups"][group]["coverage_before_mask"]["train"], 0)
            self.assertGreater(report["feature_groups"][group]["coverage_before_mask"]["test"], 0)
        self.assertEqual(monitor.threshold, report["distribution_monitor"]["threshold"])
        self.assertEqual(report["distribution_monitor"]["universe_symbols"], 7)
        self.assertTrue(report["distribution_monitor"]["separate_macro_input"])


if __name__ == "__main__":
    unittest.main()
