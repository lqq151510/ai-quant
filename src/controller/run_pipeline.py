"""控制器层：AI 量化四阶段渐进式实验与消融验证主入口."""

import random
import numpy as np
import torch

from src.application.trainer import (
    train_and_eval_rl_portfolio,
    train_and_evaluate_alpha_model,
)
from src.domain.features.pipeline import (
    FEATURE_COLUMNS,
    build_spatio_temporal_dataset,
    chronological_split,
)
from src.domain.kg_gnn.graph_model import (
    SpatioTemporalKGAlphaModel,
    build_hybrid_kg_adjacency,
)
from src.domain.timeseries_dl.models import PureTemporalAlphaModel
from src.infrastructure.market_data import (
    build_initial_kg_edges,
    generate_realistic_ashare_panel,
)
from src.infrastructure.storage import QuantDuckDBStorage


def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    set_all_seeds(42)
    print("=" * 76)
    print("🚀 AI 量化研究工作台：时序 DL -> 知识图谱 GNN -> 强化学习 RL 渐进消融实验")
    print("=" * 76)

    # 1. 基础设施层：初始化 DuckDB 与行情/知识图谱入库
    storage = QuantDuckDBStorage("data/quant_market.duckdb")
    bars_df = generate_realistic_ashare_panel(n_days=750, seed=42)
    kg_edges_df = build_initial_kg_edges()

    n_bars = storage.save_daily_bars(bars_df)
    n_edges = storage.save_kg_edges(kg_edges_df)
    print(f"[Infra] DuckDB 入库完成: {n_bars} 条日线记录, {n_edges} 条知识图谱产业链/板块边")

    loaded_bars = storage.load_daily_bars()
    loaded_edges = storage.load_kg_edges()

    # 2. 领域层 Stage 1A：零未来函数特征工程与严格时间切分 (含 Purge Gap 隔离带)
    dataset = build_spatio_temporal_dataset(loaded_bars, seq_len=20, pred_horizon=3)
    train_ds, val_ds, test_ds = chronological_split(
        dataset, train_ratio=0.65, val_ratio=0.15, purge_gap=5
    )
    print(
        f"[Data ] 时空张量形状: X={dataset.x_seq.shape} | "
        f"Train={len(train_ds.dates)}d, Val={len(val_ds.dates)}d, Test={len(test_ds.dates)}d (PurgeGap=5d)"
    )

    # 3. Stage 1B：训练纯时序深度学习模型 (RevIN + PatchTST + IC Loss)
    stage1_model = PureTemporalAlphaModel(
        in_features=len(FEATURE_COLUMNS), seq_len=20, d_model=64
    )
    m_stage1, _ = train_and_evaluate_alpha_model(
        stage1_model, train_ds, val_ds, test_ds, epochs=25
    )

    # 4. Stage 2：构建混合知识图谱邻接矩阵 + 训练时空图注意力网络 (PatchTST + Prior-Guided GAT)
    set_all_seeds(42)
    hybrid_adj = build_hybrid_kg_adjacency(
        symbols=dataset.symbols,
        kg_edges_df=loaded_edges,
        train_x_seq=train_ds.x_seq,
    )
    stage2_model = SpatioTemporalKGAlphaModel(
        adj_prior=hybrid_adj,
        in_features=len(FEATURE_COLUMNS),
        seq_len=20,
        d_model=64,
        n_heads=4,
    )
    # 迁移 Stage 1 预训练好的时序编码器与预测头作为热启动，让 GAT 专注学习外部图谱残差溢出
    stage2_model.temporal_encoder.load_state_dict(stage1_model.encoder.state_dict())
    stage2_model.head.load_state_dict(stage1_model.head.state_dict())
    m_stage2, preds_stage2 = train_and_evaluate_alpha_model(
        stage2_model, train_ds, val_ds, test_ds, epochs=20, lr=5e-4
    )

    # 5. Stage 3：基于 Stage 2 时空预测特征训练 A股 RL 动态仓位与风控智能体
    m_stage3 = train_and_eval_rl_portfolio(
        train_alpha=preds_stage2["train_preds"],
        train_ds=train_ds,
        test_alpha=preds_stage2["test_preds"],
        test_ds=test_ds,
        episodes=45,
    )

    print("\n" + "-" * 76)
    print("📊 样本外测试集 (Out-of-Sample Test) 消融实验对比结果:")
    print("-" * 76)
    header = (
        f"{'阶段与模型架构':<28} | {'IC均值':>7} | {'RankIC':>7} | {'ICIR':>6} | "
        f"{'年化收益':>8} | {'夏普比':>6} | {'最大回撤':>8}"
    )
    print(header)
    print("-" * 76)
    for name, m in [
        ("Baseline (等权持有基准)", None),
        ("Stage 1: 纯时序 PatchTST", m_stage1),
        ("Stage 2: 时序 + 知识图谱 GAT", m_stage2),
        ("Stage 3: KG-GAT + RL 仓位风控", m_stage3),
    ]:
        if m is None:
            print(
                f"{name:<28} | {'--':>7} | {'--':>7} | {'--':>6} | "
                f"{m_stage1.bench_annual_return * 100:>7.2f}% | {'--':>6} | {'--':>8}"
            )
        else:
            print(
                f"{name:<28} | {m.ic_mean:>7.4f} | {m.rank_ic_mean:>7.4f} | {m.icir:>6.2f} | "
                f"{m.annual_return * 100:>7.2f}% | {m.sharpe_ratio:>6.2f} | {m.max_drawdown * 100:>7.2f}%"
            )
    print("-" * 76)
    print(f"💡 Stage 3 RL 平均现金防守仓位比例: {m_stage3.avg_cash_weight * 100:.1f}%")
    print("=" * 76)


if __name__ == "__main__":
    main()
