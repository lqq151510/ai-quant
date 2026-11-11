"""针对方向 A (Walk-Forward 样本外 + 真实份额账本 RL 消融) 与方向 C (双时点 Bitemporal KG-RAG) 的回归测试."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import wraps
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.application import walk_forward_engine
from src.application.walk_forward_engine import (
    apply_turnover_deadband,
    build_open_to_open_panel,
    run_walk_forward_ablation_experiment,
    select_buffered_top_k,
)
from src.domain.features.pipeline import compute_raw_feature_columns
from src.infrastructure.wide_universe_builder import (
    WIDE_CSI300_2015_COHORT,
    build_supply_chain_and_sector_edges,
)
from src.domain.kg_gnn.bitemporal_kg_bridge import (
    BitemporalEvent,
    BitemporalKGBundle,
    BitemporalRelationVersion,
    build_asof_directed_graph,
    build_bitemporal_bundle_from_news_and_edges,
    compute_bitemporal_3hop_features_and_adjacency,
)
from src.domain.kg_gnn.graph_model import SpatioTemporalKGAlphaModel


class BitemporalAndWalkForwardTests(unittest.TestCase):
    def test_bitemporal_graph_blocks_future_relations_and_withdrawn_edges_do_not_resurface(self) -> None:
        t0 = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2024, 6, 1, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2024, 9, 1, 0, 0, tzinfo=timezone.utc)

        rels = [
            BitemporalRelationVersion(
                relation_id="rel_semi",
                src_symbol="510050",
                dst_symbol="510300",
                relation_type="chain",
                valid_from=t0,
                known_at=t0,
                weight=0.60,
            ),
            # 2024-06-01 修订权重至 0.90 (在 2024-03-01 截面不可见)
            BitemporalRelationVersion(
                relation_id="rel_semi",
                src_symbol="510050",
                dst_symbol="510300",
                relation_type="chain",
                valid_from=t0,
                known_at=t1,
                weight=0.90,
            ),
            # 2024-09-01 撤回该关系 (不应回退到 0.90 或 0.60)
            BitemporalRelationVersion(
                relation_id="rel_semi",
                src_symbol="510050",
                dst_symbol="510300",
                relation_type="chain",
                valid_from=t0,
                known_at=t2,
                weight=0.90,
                withdrawn=True,
            ),
        ]

        g_march = build_asof_directed_graph(rels, "2024-03-01T07:00:00+00:00")
        self.assertAlmostEqual(g_march["510050"]["510300"]["weight"], 0.60)

        g_july = build_asof_directed_graph(rels, "2024-07-01T07:00:00+00:00")
        self.assertAlmostEqual(g_july["510050"]["510300"]["weight"], 0.90)

        g_oct = build_asof_directed_graph(rels, "2024-10-01T07:00:00+00:00")
        self.assertFalse(g_oct.has_edge("510050", "510300"))

    def test_3hop_features_deduplicate_reposts_and_isolate_post_close_events(self) -> None:
        t_base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        # 2跳图关系: 510050 -> 510300 -> 510500 (配合事件提及 510050 构成 3 跳传导)
        rels = (
            BitemporalRelationVersion(
                relation_id="r1",
                src_symbol="510050",
                dst_symbol="510300",
                relation_type="hop1",
                valid_from=t_base,
                known_at=t_base,
                weight=0.80,
                confidence=0.90,
            ),
            BitemporalRelationVersion(
                relation_id="r2",
                src_symbol="510300",
                dst_symbol="510500",
                relation_type="hop2",
                valid_from=t_base,
                known_at=t_base,
                weight=0.75,
                confidence=0.90,
            ),
        )
        # 同一事件 ev_1 的两条转载观测 (ready_at 均在 2024-05-10 14:00 CST，早于 15:00 收盘)
        ev_repost_a = BitemporalEvent(
            event_id="ev_1",
            source_url="https://example.com/a",
            entity_symbols=("510050",),
            category="policy",
            published_at=datetime.fromisoformat("2024-05-10T13:00:00+08:00"),
            observed_at=datetime.fromisoformat("2024-05-10T13:30:00+08:00"),
            ready_at=datetime.fromisoformat("2024-05-10T14:00:00+08:00"),
            confidence=0.8,
            sentiment=0.5,
        )
        ev_repost_b = BitemporalEvent(
            event_id="ev_1",
            source_url="https://example.com/b",
            entity_symbols=("510050",),
            category="policy",
            published_at=datetime.fromisoformat("2024-05-10T13:10:00+08:00"),
            observed_at=datetime.fromisoformat("2024-05-10T13:40:00+08:00"),
            ready_at=datetime.fromisoformat("2024-05-10T14:10:00+08:00"),
            confidence=0.9,
            sentiment=0.5,
        )
        # 盘后 16:30 才就绪的事件 ev_after_close (不得泄漏到 2024-05-10 15:00 截面)
        ev_after_close = BitemporalEvent(
            event_id="ev_after",
            source_url="https://example.com/c",
            entity_symbols=("510050",),
            category="earnings",
            published_at=datetime.fromisoformat("2024-05-10T16:00:00+08:00"),
            observed_at=datetime.fromisoformat("2024-05-10T16:10:00+08:00"),
            ready_at=datetime.fromisoformat("2024-05-10T16:30:00+08:00"),
            confidence=0.95,
            sentiment=-0.9,
        )

        bundle = BitemporalKGBundle(events=(ev_repost_a, ev_repost_b, ev_after_close), relations=rels)
        dates = [pd.Timestamp("2024-05-10"), pd.Timestamp("2024-05-13")]
        symbols = ["510050", "510300", "510500"]
        dyn_adj, feats = compute_bitemporal_3hop_features_and_adjacency(dates, symbols, bundle)

        self.assertEqual(dyn_adj.shape, (2, 3, 3))
        # 2024-05-10 15:00 CST 仅可见 ev_1 (去重后计 1 次 -> log1p(1)), 三跳传导至 510500 情绪为 0.5 * 0.8 * 0.75 = 0.30
        self.assertAlmostEqual(float(feats[0, 2, 0]), float(np.log1p(1)), places=5)
        self.assertAlmostEqual(float(feats[0, 2, 1]), 0.30, places=5)
        # 2024-05-13 15:00 CST 同时可见 ev_1 与 ev_after (计 2 次 -> log1p(2))
        self.assertAlmostEqual(float(feats[1, 2, 0]), float(np.log1p(2)), places=5)

    def test_spatio_temporal_kg_model_supports_dynamic_adjacency_batches(self) -> None:
        adj_prior = torch.tensor([[0.0, 0.8, 0.0], [0.5, 0.0, 0.6], [0.0, 0.7, 0.0]], dtype=torch.float32)
        model = SpatioTemporalKGAlphaModel(adj_prior=adj_prior, in_features=15, seq_len=20, d_model=32, n_heads=4)
        x = torch.randn(4, 3, 20, 15)
        dyn_adj = torch.rand(4, 3, 3)
        out = model(x, dynamic_adj=dyn_adj)
        self.assertEqual(tuple(out.shape), (4, 3))
        self.assertTrue(torch.isfinite(out).all())

    def test_walk_forward_targets_and_risk_caps_ignore_future_bars(self) -> None:
        rng = np.random.default_rng(123)
        dates = pd.bdate_range("2023-01-02", periods=130)
        symbols = ("510050", "510300", "510500")
        rows = []
        for idx, sym in enumerate(symbols):
            close = (3.0 + idx) * np.cumprod(1.0 + rng.normal(0.0004, 0.008, size=len(dates)))
            prev = np.r_[close[0], close[:-1]]
            open_ = prev * (1.0 + rng.normal(0.0, 0.002, size=len(dates)))
            high = np.maximum(open_, close) * 1.003
            low = np.minimum(open_, close) * 0.997
            vol = rng.integers(200_000, 900_000, size=len(dates))
            rows.append(
                pd.DataFrame({
                    "date": dates,
                    "symbol": sym,
                    "kind": "etf",
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "preclose": prev,
                    "volume": vol,
                    "amount": close * vol,
                    "tradable": True,
                    "upper_limit": np.round(prev * 1.1, 3),
                    "lower_limit": np.round(prev * 0.9, 3),
                })
            )
        bars_df = pd.concat(rows, ignore_index=True)
        actions_df = pd.DataFrame(
            columns=["symbol", "record_date", "ex_date", "pay_date", "cash_per_share", "share_multiplier"]
        )
        edges_df = pd.DataFrame(
            [("510050", "510300", "chain", 0.8), ("510300", "510500", "chain", 0.75)],
            columns=["src_symbol", "dst_symbol", "relation", "weight"],
        )
        bundle = build_bitemporal_bundle_from_news_and_edges(edges_df)

        panel = build_open_to_open_panel(bars_df, dates, actions_df, bundle, horizon=5, seq_len=20)
        self.assertGreater(len(panel.dates), 60)

        def run_and_capture(bars):
            policy_history = []
            risk_caps = {}
            simulate = walk_forward_engine.simulate
            generate_stream = walk_forward_engine.generate_walk_forward_oos_stream

            def capture_stream(*args, **kwargs):
                stream = generate_stream(*args, **kwargs)
                risk_caps.update(zip(stream.dates, stream.oos_hmm_exposure, strict=True))
                return stream

            def capture_policy(*args, **kwargs):
                close_policy = kwargs.get("close_policy")
                if close_policy is not None:
                    history = {}
                    policy_history.append(history)

                    def observe(state):
                        target = close_policy(state)
                        history[pd.Timestamp(state.date)] = None if target is None else dict(target)
                        return target

                    kwargs["close_policy"] = observe
                return simulate(*args, **kwargs)

            with (
                patch.object(walk_forward_engine, "simulate", side_effect=capture_policy),
                patch.object(walk_forward_engine, "generate_walk_forward_oos_stream", side_effect=capture_stream),
                patch.object(torch.backends.mps, "is_available", wraps(torch.backends.mps.is_available)(lambda: False)),
            ):
                report = run_walk_forward_ablation_experiment(
                    bars_df=bars,
                    calendar_dates=dates,
                    actions_df=actions_df,
                    kg_bundle=bundle,
                    base_edges_df=edges_df,
                    horizon=5,
                    top_k=2,
                    initial_cash=50_000.0,
                    min_train_sessions=35,
                    val_sessions=15,
                    oos_step_sessions=25,
                    epochs_per_fold=2,
                    ppo_seeds=(42, 101),
                )
            return report, risk_caps, policy_history

        res, caps, history = run_and_capture(bars_df)
        self.assertEqual(len(history), 12)
        cutoff = sorted(history[0])[5]
        perturbed = bars_df.copy()
        future = perturbed["date"].gt(cutoff) & perturbed["symbol"].eq("510500")
        price_columns = ["open", "high", "low", "close", "preclose", "lower_limit", "upper_limit"]
        perturbed.loc[future, price_columns] *= 1.06
        perturbed.loc[perturbed.index[future][0], "preclose"] = np.nan
        _, perturbed_caps, perturbed_history = run_and_capture(perturbed)
        self.assertEqual(caps.keys(), perturbed_caps.keys())
        for day, cap in caps.items():
            if day <= cutoff:
                self.assertAlmostEqual(cap, perturbed_caps[day], places=10)
        self.assertTrue(any(abs(cap - perturbed_caps[day]) > 1e-6 for day, cap in caps.items() if day > cutoff))
        for policy, perturbed_policy in zip(history, perturbed_history, strict=True):
            self.assertEqual(policy.keys(), perturbed_policy.keys())
            self.assertIsNotNone(policy[min(policy)])
            for day, target in policy.items():
                if day > cutoff:
                    break
                other = perturbed_policy[day]
                if target is None:
                    self.assertIsNone(other)
                else:
                    self.assertIsNotNone(other)
                    self.assertEqual(target.keys(), other.keys())
                    for symbol, weight in target.items():
                        self.assertAlmostEqual(weight, other[symbol], places=7)

        normal_ledgers = res["share_ledger_ablation"]["normal"]
        for key in ("equal_weight", "wf_stage2_bitemporal_kg_gat", "wf_hrp_cqr_hmm", "rl_rule_prior_only", "rl_offline_iql"):
            self.assertLess(normal_ledgers[key]["ledger_reconciliation_error"], 1e-6)
            self.assertGreaterEqual(normal_ledgers[key]["min_cash"], 0.0)
        ppo_summary = normal_ledgers["rl_ppo_5seed_summary"]
        self.assertEqual(len(ppo_summary["per_seed"]), 2)
        self.assertLess(ppo_summary["max_reconciliation_error"], 1e-6)

    def test_causal_preclose_feature_pipeline_eliminates_split_jump_distortion(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=30)
        # 构造第 25 日发生 1:2 份额拆分 (价格减半，但 preclose 同步除权减半)
        closes = np.full(30, 10.0)
        closes[25:] = 5.0
        precloses = np.full(30, 10.0)
        precloses[25:] = 5.0
        df = pd.DataFrame({
            "symbol": "510500",
            "trade_date": dates,
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "preclose": precloses,
            "volume": np.full(30, 1_000_000.0),
        })
        feat = compute_raw_feature_columns(df)
        # 拆分当日及后续日的真实收益率与动量必须严格为 0，不能产生 -50% 伪暴跌！
        self.assertAlmostEqual(float(feat.loc[25, "ret_1d_raw"]), 0.0, places=9)
        self.assertAlmostEqual(float(feat.loc[26, "ret_5d"]), 0.0, places=9)
        self.assertAlmostEqual(float(feat.loc[28, "ret_20d"]), 0.0, places=9)
        self.assertAlmostEqual(float(feat.loc[28, "ma_bias_20"]), 0.0, places=6)
        # 一条未来坏参考价不能把此前的份额拆分重新解释成暴跌。
        df.loc[29, "preclose"] = np.nan
        with_future_gap = compute_raw_feature_columns(df)
        pd.testing.assert_frame_equal(feat.loc[:28], with_future_gap.loc[:28], check_exact=True)
        self.assertAlmostEqual(float(with_future_gap.loc[29, "ret_1d_raw"]), 0.0, places=9)

    def test_hysteresis_buffer_and_turnover_deadband_suppress_micro_churn(self) -> None:
        # 标的 1 已持仓且分数排第 3 (位于 Top-M 缓冲带内)，即使标的 0/2 分数略高也应保留已持仓标的
        scores = np.array([0.85, 0.72, 0.80, -0.40, -0.90])
        prev_w = np.array([0.0, 0.50, 0.0, 0.50, 0.0])
        selected = select_buffered_top_k(scores, prev_w, top_k=2, buffer_mult=1.6)
        self.assertIn(1, selected.tolist())
        self.assertIn(0, selected.tolist())

        # 当新旧目标仓位 L1 偏差小于死区阈值时，apply_turnover_deadband 应返回 should_trade=False
        prop_w = np.array([0.52, 0.48, 0.0, 0.0, 0.0])
        old_w = np.array([0.50, 0.50, 0.0, 0.0, 0.0])
        out_w, should_trade = apply_turnover_deadband(prop_w, old_w, l1_deadband=0.16)
        self.assertFalse(should_trade)
        np.testing.assert_allclose(out_w, old_w)

    def test_wide28_supply_chain_graph_and_cohort_integrity(self) -> None:
        self.assertEqual(len(WIDE_CSI300_2015_COHORT), 24)
        edges_df = build_supply_chain_and_sector_edges()
        self.assertGreaterEqual(len(edges_df), 80)
        self.assertTrue((edges_df["weight"] > 0.0).all() and (edges_df["weight"] <= 1.0).all())



if __name__ == "__main__":
    unittest.main()
