"""领域层 Stage 2：A股知识图谱构建与先验引导时空图注意力网络 (Spatio-Temporal KG-GAT)."""

from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.domain.timeseries_dl.models import FinancialPatchTSTEncoder, FinancialiTransformer


def build_hybrid_kg_adjacency(
    symbols: List[str],
    kg_edges_df: pd.DataFrame,
    train_x_seq: np.ndarray,
    implicit_weight: float = 0.35,
) -> torch.Tensor:
    """构建融合「显式知识图谱边」与「训练集历史 Lead-Lag 动量溢出隐式边」的混合邻接矩阵.

    注意防泄漏：隐式相关性仅在 train_x_seq (训练集) 的单日收益率序列上统计，绝不触碰验证/测试集！
    """
    n = len(symbols)
    sym_to_idx = {s: i for i, s in enumerate(symbols)}
    adj_explicit = np.eye(n, dtype=np.float32)

    for _, row in kg_edges_df.iterrows():
        u, v = str(row["src_symbol"]), str(row["dst_symbol"])
        if u in sym_to_idx and v in sym_to_idx:
            # message passing: dst(v) 接收来自 src(u) 的信号 -> adj[v, u]
            adj_explicit[sym_to_idx[v], sym_to_idx[u]] = max(
                adj_explicit[sym_to_idx[v], sym_to_idx[u]],
                float(row["weight"]),
            )

    # 在训练集上计算 Lag-1 领先滞后互相关 (资产 j 在 t-1 的收益对资产 i 在 t 的收益预测性)
    # train_x_seq shape: (T_train, N, L, F), 第 0 维特征为 ret_1d
    daily_ret = train_x_seq[:, :, -1, 0]  # (T_train, N)
    adj_implicit = np.zeros((n, n), dtype=np.float32)
    if len(daily_ret) > 10:
        r_prev = daily_ret[:-1]  # t-1
        r_curr = daily_ret[1:]   # t
        r_prev_norm = (r_prev - r_prev.mean(0)) / (r_prev.std(0) + 1e-8)
        r_curr_norm = (r_curr - r_curr.mean(0)) / (r_curr.std(0) + 1e-8)
        # lead_lag[i, j] 表示 j(t-1) -> i(t) 的相关性
        lead_lag = (r_curr_norm.T @ r_prev_norm) / len(r_curr)
        adj_implicit = np.clip(lead_lag, 0.0, 1.0)

    hybrid_adj = (1.0 - implicit_weight) * adj_explicit + implicit_weight * adj_implicit
    # 移除自环：自身时序信息由残差主干 h_self 保留，GAT 专注于聚合纯外部邻居 (j != i) 的溢出信号
    np.fill_diagonal(hybrid_adj, 0.0)
    return torch.tensor(hybrid_adj, dtype=torch.float32)


class PriorGuidedStockGATLayer(nn.Module):
    """知识图谱先验引导的多头图注意力层 (Prior-Guided Graph Attention Layer).

    通过小初值门控残差机制融合：
    h_final = h_self + res_scale * Gate * GAT_Aggregate(External_Neighbors | KG_Prior)
    既保证下界不低于纯时序模型，又精准捕捉产业链上游/同板块龙头的纯外部动量溢出增量。
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # 知识图谱先验强度缩放因子与小初值残差系数 (ReZero 思想)
        self.prior_scale = nn.Parameter(torch.ones(1, n_heads, 1, 1) * 2.5)
        self.res_scale = nn.Parameter(torch.tensor(0.12))
        self.gate_layer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.msg_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_prior: torch.Tensor) -> torch.Tensor:
        """执行截面图注意力消息传递.

        Args:
            h: 节点时序特征, shape = (Batch_days, N_assets, d_model)
            adj_prior: 知识图谱无自环先验邻接矩阵, shape = (N_assets, N_assets)
        """
        b_days, n_assets, _ = h.shape
        q = self.w_q(h).view(b_days, n_assets, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(h).view(b_days, n_assets, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(h).view(b_days, n_assets, self.n_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数并注入知识图谱拓扑软偏置 (Graphormer 风格):
        # 既强化同产业链/同板块邻居的动量溢出，又保留跨板块 (如科技 vs 红利/黄金) 的全局资金跷跷板感知
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        prior_bias = adj_prior.unsqueeze(0).unsqueeze(0) * self.prior_scale
        attn_logits = attn_logits + prior_bias

        # 仅屏蔽自环 (i == j)，迫使图注意力层专注提取外部标的的截面传导与轮动信号
        eye_mask = torch.eye(n_assets, device=h.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        attn_logits = attn_logits.masked_fill(eye_mask, -1e4)

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 聚合外部邻居溢出特征
        neighbor_msg = torch.matmul(attn_weights, v)
        neighbor_msg = neighbor_msg.transpose(1, 2).contiguous().view(b_days, n_assets, self.d_model)
        neighbor_msg = self.msg_norm(neighbor_msg)

        # ReZero 门控残差融合：不破坏原始时序表征分布
        gate = self.gate_layer(torch.cat([h, neighbor_msg], dim=-1))
        return h + self.res_scale * gate * neighbor_msg


class SpatioTemporalKGAlphaModel(nn.Module):
    """Stage 2 时空知识图谱网络 (iTransformer 跨指标时序编码 + PatchTST + 动态知识图谱拓扑传导).

    融合 iTransformer 跨变量耦合表征与显式知识图谱产业链/跨市场映射边 (norm_adj) 消息传递。
    """

    def __init__(
        self,
        adj_prior: torch.Tensor,
        in_features: int = 8,
        seq_len: int = 20,
        d_model: int = 64,
        n_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        row_sum = adj_prior.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        norm_adj = adj_prior / row_sum
        self.register_buffer("norm_adj", norm_adj)

        self.itrans_branch = FinancialiTransformer(
            in_features=in_features,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.temporal_encoder = FinancialPatchTSTEncoder(
            in_features=in_features,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.kg_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.kg_scale = nn.Parameter(torch.tensor(0.22))

    def forward(self, x: torch.Tensor, dynamic_adj: torch.Tensor | None = None) -> torch.Tensor:
        """输入 (B, N, L, F) 与可选时点动态邻接张量 (B, N, N) -> 输出时空融合截面 Alpha 分数 (B, N)."""
        frozen_backbones = not any(p.requires_grad for p in self.itrans_branch.parameters())
        if frozen_backbones:
            with torch.no_grad():
                self.itrans_branch.eval()
                self.temporal_encoder.eval()
                base_alpha = self.itrans_branch(x)  # (B, N)
                h_intra = self.temporal_encoder(x)  # (B, N, d_model)
        else:
            base_alpha = self.itrans_branch(x)      # (B, N)
            h_intra = self.temporal_encoder(x)      # (B, N, d_model)
        # 沿双时点 (Bitemporal) 动态知识图谱或静态先验邻接矩阵聚合邻居特征
        if dynamic_adj is not None:
            row_sum = dynamic_adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            norm_dyn = dynamic_adj / row_sum
            h_kg = torch.bmm(norm_dyn, h_intra)
        else:
            h_kg = torch.matmul(self.norm_adj.unsqueeze(0), h_intra)
        kg_alpha = self.kg_proj(h_kg).squeeze(-1)
        return base_alpha + self.kg_scale * kg_alpha
