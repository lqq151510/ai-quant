"""控制器层：真实 A股/ETF 历史行情 + 动态知识图谱 Graph-Agent + 5大高阶正交算法全链路消融与实盘计划主入口."""

import random
import numpy as np
import torch

from src.application.graph_agent import KnowledgeGraphAgent
from src.application.trading_plan_generator import generate_semi_auto_trading_plan
from src.application.trainer import (
    evaluate_predictions_direct,
    get_device,
    train_and_eval_iql_hrp_hmm,
    train_and_eval_rl_portfolio,
    train_and_evaluate_alpha_model,
)
from src.domain.ensemble.gbdt_stacking import OrthogonalTreeGNNStacker
from src.domain.features.pipeline import (
    FEATURE_COLUMNS,
    build_spatio_temporal_dataset,
    chronological_split,
)
from src.domain.kg_gnn.graph_model import (
    SpatioTemporalKGAlphaModel,
    build_hybrid_kg_adjacency,
)
from src.domain.regime.market_hmm import StickyMarketRegimeHMM
from src.domain.timeseries_dl.models import (
    FinancialiTransformer,
    MASTERAlphaModel,
    PureTemporalAlphaModel,
)
from src.domain.uncertainty.quantile_conformal import fit_and_calibrate_conformal_quantile
from src.infrastructure.market_data import (
    CORE_UNIVERSE,
    build_initial_kg_edges,
    fetch_or_generate_market_data,
)
from src.infrastructure.storage import QuantDuckDBStorage


def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    set_all_seeds(42)
    print("=" * 84)
    print("🚀 AI 量化实战全链路工作台 (真实 A股/ETF 行情 + 动态 KG-Agent + DL/GNN/PPO)")
    print("=" * 84)

    # 1. 基础设施层：优先复用 DuckDB 已入库的真实前复权日线，若存在未复权拆分跳空则自动重刷
    storage = QuantDuckDBStorage("data/quant_market.duckdb")
    bars_df = storage.load_daily_bars()
    has_unadjusted_split = (
        not bars_df.empty
        and (bars_df.groupby("symbol")["close"].pct_change().abs() > 0.25).any()
    )
    if bars_df.empty or bars_df["symbol"].nunique() < len(CORE_UNIVERSE) or has_unadjusted_split:
        bars_df = fetch_or_generate_market_data(n_days=750, try_online_akshare=True, seed=42)
        storage.replace_all_daily_bars(bars_df)
    init_edges_df = build_initial_kg_edges()
    storage.save_kg_edges(init_edges_df)
    n_bars = len(bars_df)
    min_dt = bars_df["trade_date"].min().strftime("%Y-%m-%d")
    max_dt = bars_df["trade_date"].max().strftime("%Y-%m-%d")
    print(
        f"[Step 1 | 真实行情底座] DuckDB 已入库 {n_bars} 条真实前复权日线 "
        f"({bars_df['symbol'].nunique()} 只核心 ETF/龙头, 区间: {min_dt} -> {max_dt})"
    )

    # 2. Stage 2 动态知识图谱 Graph-Agent：抓取东方财富实时新闻并动态更新 kg_edges 权重
    kg_agent = KnowledgeGraphAgent(storage=storage)
    kg_update_res = kg_agent.run_daily_kg_update(top_news_per_asset=3)
    print(
        f"[Step 2 | Graph-Agent ] 抓取东方财富最新资讯 {len(kg_update_res.scored_news_df)} 条 "
        f"| 模式: {kg_update_res.llm_mode_used}"
    )
    for cat in kg_update_res.active_catalysts[:3]:
        print(f"   ↳ 动态图谱演化: {cat}")

    loaded_bars = storage.load_daily_bars()
    loaded_edges = storage.load_kg_edges()

    # 3. 零未来函数特征工程与严格时间切分 (含 Purge Gap 隔离带)
    pred_horizon = 3
    dataset = build_spatio_temporal_dataset(loaded_bars, seq_len=20, pred_horizon=pred_horizon)
    train_ds, val_ds, test_ds = chronological_split(
        dataset, train_ratio=0.65, val_ratio=0.15, purge_gap=5, label_horizon=pred_horizon
    )
    print(
        f"[Step 3 | 时空特征张量] X={dataset.x_seq.shape} (含主力资金流CMF+上下行半方差) | "
        f"Löwdin正交化共线性: {dataset.collinearity_before:.4f} -> {dataset.collinearity_after:.4f} | "
        f"Train={len(train_ds.dates)}d, Val={len(val_ds.dates)}d, Test={len(test_ds.dates)}d (PurgeGap=5d)"
    )

    # 4. Stage 1 多模型对比：PatchTST vs iTransformer vs MASTER
    set_all_seeds(42)
    m_patchtst_model = PureTemporalAlphaModel(
        in_features=len(FEATURE_COLUMNS), seq_len=20, d_model=64
    )
    m_patchtst, preds_patchtst = train_and_evaluate_alpha_model(
        m_patchtst_model, train_ds, val_ds, test_ds, epochs=25, lr=8e-4
    )

    set_all_seeds(42)
    m_itrans_model = FinancialiTransformer(
        in_features=len(FEATURE_COLUMNS), seq_len=20, d_model=64
    )
    m_itrans, _ = train_and_evaluate_alpha_model(
        m_itrans_model, train_ds, val_ds, test_ds, epochs=25, lr=8e-4
    )

    set_all_seeds(42)
    m_master_model = MASTERAlphaModel(
        in_features=len(FEATURE_COLUMNS), seq_len=20, d_model=64
    )
    m_master, _ = train_and_evaluate_alpha_model(
        m_master_model, train_ds, val_ds, test_ds, epochs=25, lr=8e-4
    )

    # 5. Stage 2 知识图谱时空图网络：Prior-Guided Stock GAT (融合动态更新后的 kg_edges)
    set_all_seeds(42)
    hybrid_adj = build_hybrid_kg_adjacency(
        symbols=dataset.symbols,
        kg_edges_df=loaded_edges,
        train_x_seq=train_ds.x_seq,
    )
    stage2_kg_model = SpatioTemporalKGAlphaModel(
        adj_prior=hybrid_adj,
        in_features=len(FEATURE_COLUMNS),
        seq_len=20,
        d_model=64,
        n_heads=4,
    )
    stage2_kg_model.itrans_branch.load_state_dict(m_itrans_model.state_dict())
    stage2_kg_model.temporal_encoder.load_state_dict(m_patchtst_model.encoder.state_dict())
    for p in stage2_kg_model.itrans_branch.parameters():
        p.requires_grad = False
    for p in stage2_kg_model.temporal_encoder.parameters():
        p.requires_grad = False
    m_kg_gat, preds_kg_gat = train_and_evaluate_alpha_model(
        stage2_kg_model, train_ds, val_ds, test_ds, epochs=15, lr=1e-3
    )

    # 5B. 新增算法 1：Sticky-HMM 市场状态识别器 (仅在训练集宏观序列拟合，测试集执行纯因果前向滤波)
    macro_df = StickyMarketRegimeHMM.extract_macro_observations(loaded_bars)
    train_end_dt = train_ds.dates[-1]
    train_macro_df = macro_df[macro_df["trade_date"] <= train_end_dt].reset_index(drop=True)
    hmm_model = StickyMarketRegimeHMM(self_transition_prob=0.82, random_state=42).fit(train_macro_df)
    all_regime_snaps = hmm_model.filter_sequence(macro_df)
    regime_map = {snap.trade_date: snap for snap in all_regime_snaps}
    test_hmm_exposures = [
        regime_map[dt].recommended_max_exposure if dt in regime_map else 0.78
        for dt in test_ds.dates
    ]
    latest_regime = all_regime_snaps[-1]
    print(
        f"[Step 4 | HMM+最优传输] 最新交易日 ({latest_regime.trade_date.strftime('%Y-%m-%d')}) 状态: "
        f"{latest_regime.state_name} | 1D-Wasserstein W2={latest_regime.wasserstein_w2:.4f} "
        f"(左尾偏移={latest_regime.ot_left_tail_shift:+.3f}) | 建议最大权益敞口: {latest_regime.recommended_max_exposure * 100:.1f}%"
    )

    # 5C. 新增算法 4 + 数据升级 1 + 算法升级 A/C：Löwdin 正交化 + 概念漂移衰减 GBDT + 主力资金流微结构 + KG-GAT
    stacker = OrthogonalTreeGNNStacker(random_state=42)
    preds_gbdt_kg, _ = stacker.fit_and_blend(
        train_x=train_ds.x_seq,
        train_y=train_ds.y_alpha,
        val_x=val_ds.x_seq,
        val_y=val_ds.y_alpha,
        test_x=test_ds.x_seq,
        gnn_preds=preds_kg_gat,
        train_micro=train_ds.micro_feats,
        test_micro=test_ds.micro_feats,
    )
    m_gbdt_kg = evaluate_predictions_direct(preds_gbdt_kg["test_preds"], test_ds, top_k=3)

    # 5D. 新增算法 2：多分位数回归 [q10, q50, q90] + Split Conformal Prediction 共形预测校准
    set_all_seeds(42)
    uncertainty_res = fit_and_calibrate_conformal_quantile(
        train_x=train_ds.x_seq,
        train_y=train_ds.y_alpha,
        val_x=val_ds.x_seq,
        val_y=val_ds.y_alpha,
        eval_x=test_ds.x_seq,
        alpha_miscoverage=0.20,
        epochs=22,
        device=get_device(),
    )
    # 将共形校准后的悲观下界与不确定性宽度融合进 GBDT+KG-GAT 正交预测
    cqr_test_preds = (
        0.82 * preds_gbdt_kg["test_preds"]
        + 0.18 * uncertainty_res.robust_alpha
    )
    m_cqr_alpha = evaluate_predictions_direct(cqr_test_preds, test_ds, top_k=3)
    print(
        f"[Step 5 | Conformal  ] 验证集 80% 共形校准半径 q_hat={uncertainty_res.conformal_q_hat:.3f} | "
        f"GBDT 正交融合权重={stacker.tree_weight:.2f}"
    )

    # 6A. Stage 3A 强化学习：在线 PPO (GAE + Clipped Surrogate Objective) 动态仓位控制
    set_all_seeds(42)
    m_ppo, _ = train_and_eval_rl_portfolio(
        train_alpha=preds_kg_gat["train_preds"],
        train_ds=train_ds,
        test_alpha=preds_kg_gat["test_preds"],
        test_ds=test_ds,
        episodes=35,
    )

    # 6B. 新增算法 3 + 5：HRP 层次风险平价 + Sticky-HMM + Offline IQL (隐式Q学习) 终极组合风控
    set_all_seeds(42)
    m_iql_hrp, _ = train_and_eval_iql_hrp_hmm(
        train_alpha=preds_gbdt_kg["train_preds"],
        train_ds=train_ds,
        test_alpha=cqr_test_preds,
        uncertainty_width=uncertainty_res.interval_width,
        test_ds=test_ds,
        hmm_max_exposures=test_hmm_exposures,
    )

    # 7. 打印真实 A 股样本外测试集消融对比表
    print("\n" + "-" * 90)
    print(f"📊 真实 A股/ETF 样本外测试集 ({test_ds.dates[0].strftime('%Y-%m-%d')} -> {test_ds.dates[-1].strftime('%Y-%m-%d')}) 全链路消融实验:")
    print("-" * 90)
    header = (
        f"{'阶段与模型架构':<36} | {'IC均值':>7} | {'RankIC':>7} | {'ICIR':>6} | "
        f"{'年化收益':>8} | {'夏普比':>6} | {'最大回撤':>8}"
    )
    print(header)
    print("-" * 90)
    for name, m in [
        ("Baseline: 13只核心标的等权基准", None),
        ("Stage 1A: PatchTST (局部时序分块)", m_patchtst),
        ("Stage 1B: iTransformer (跨变量注意力)", m_itrans),
        ("Stage 1C: MASTER (全连接跨股注意力)", m_master),
        ("Stage 2A: 动态知识图谱 KG-GAT", m_kg_gat),
        ("Stage 2B: GBDT + KG-GAT (树与图正交集成)", m_gbdt_kg),
        ("Stage 2C: GBDT + KG-GAT + CQR共形不确定性", m_cqr_alpha),
        ("Stage 3A: KG-GAT + 在线 PPO 动态风控", m_ppo),
        ("Stage 3B: HRP + Sticky-HMM + Offline IQL", m_iql_hrp),
    ]:
        if m is None:
            print(
                f"{name:<36} | {'--':>7} | {'--':>7} | {'--':>6} | "
                f"{m_patchtst.bench_annual_return * 100:>7.2f}% | {'--':>6} | {'--':>8}"
            )
        else:
            print(
                f"{name:<36} | {m.ic_mean:>7.4f} | {m.rank_ic_mean:>7.4f} | {m.icir:>6.2f} | "
                f"{m.annual_return * 100:>7.2f}% | {m.sharpe_ratio:>6.2f} | {m.max_drawdown * 100:>7.2f}%"
            )
    print("-" * 90)
    print(
        f"💡 防守仓位对比: Stage 3A 在线 PPO 平均现金比例 = {m_ppo.avg_cash_weight * 100:.1f}% | "
        f"Stage 3B (HRP + HMM + IQL) 平均现金比例 = {m_iql_hrp.avg_cash_weight * 100:.1f}%"
    )

    # 8. Stage 4：生成基于最新收盘日的半自动实盘交易计划单 (融合 HMM 状态、CQR 区间与 HRP 权重)
    plan_md = generate_semi_auto_trading_plan(
        bars_df=loaded_bars,
        kg_model=stage2_kg_model,
        asset_sentiments=kg_update_res.asset_sentiment_map,
        news_df=kg_update_res.scored_news_df,
        avg_rl_cash_weight=m_iql_hrp.avg_cash_weight,
        latest_regime=latest_regime,
        latest_q10=uncertainty_res.q10_lower[-1],
        latest_q90=uncertainty_res.q90_upper[-1],
        hist_raw_rets=test_ds.y_raw_ret,
        output_path="data/latest_trading_plan.md",
    )
    print("\n" + "=" * 84)
    print("✅ 已生成最新《A股/ETF 半自动实盘辅助交易计划单》-> data/latest_trading_plan.md")
    print("=" * 84)
    # 打印计划单前 18 行核心摘要
    for line in plan_md.splitlines()[:25]:
        print(line)


if __name__ == "__main__":
    main()
