"""领域层 Stage 1B：带 RevIN 非平稳抑制的金融时序深度学习编码器 (PatchTST / AttentiveGRU)."""

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """保留截面水位锚点的可逆实例归一化 (Level-Preserving RevIN).

    为什么在截面选股中不能做 100% 去均值的纯 RevIN？
    因为输入特征已在每日截面上做过 Z-Score，窗口均值代表「中期持续跑赢大盘的截面水位」。
    若完全减去窗口均值，会抹杀中期趋势动量，只剩短期反转噪音。
    因此采用 0.75 * 截面原始水位 + 0.25 * 局部窗口形态归一化的可学习融合。
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.level_gate = nn.Parameter(torch.full((1, 1, num_features), 0.75))
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (B_total, L_window, F_features)."""
        mean = x.mean(dim=1, keepdim=True)
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)
        x_local = (x - mean) / std
        g = torch.clamp(self.level_gate, 0.2, 0.95)
        x_norm = g * x + (1.0 - g) * x_local
        if self.affine:
            x_norm = x_norm * self.weight + self.bias
        return x_norm


class FinancialPatchTSTEncoder(nn.Module):
    """针对金融低信噪比序列定制的 PatchTST 时序编码器.

    なぜ单点 Token 的原生 Transformer 在金融日线上容易失效？
    单日 K 线噪点极大，单点注意力容易记住随机毛刺。
    将连续 P 个交易日打包为一个局部语义 Patch (例如 5 日周线形态)，能大幅提升信噪比并降低过拟合。
    """

    def __init__(
        self,
        in_features: int = 8,
        seq_len: int = 20,
        patch_len: int = 5,
        stride: int = 3,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.revin = RevIN(in_features)
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1

        patch_dim = patch_len * in_features
        self.patch_proj = nn.Sequential(
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.temporal_pool = nn.Sequential(
            nn.Linear(d_model * self.num_patches, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # 最新截面因子残差锚点投影，防止多层注意力在中小样本上过拟合丢失中期动量主干
        self.last_step_skip = nn.Linear(in_features, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """提取每个标的的时序表征向量 h_i(t).

        Args:
            x: shape = (Batch_days, N_assets, L_window, F_features)

        Returns:
            h_temporal: shape = (Batch_days, N_assets, d_model)
        """
        b_days, n_assets, seq_len, n_feats = x.shape
        x_flat = x.reshape(b_days * n_assets, seq_len, n_feats)
        x_norm = self.revin(x_flat)

        # 切分局部时序 Patch: (B*N, num_patches, patch_len, F)
        patches = x_norm.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.contiguous().view(b_days * n_assets, self.num_patches, -1)

        tokens = self.patch_proj(patches) + self.pos_embed
        encoded = self.transformer(tokens)
        pooled = self.temporal_pool(encoded.reshape(b_days * n_assets, -1))
        skip = self.last_step_skip(x_flat[:, -1, :])
        return (0.5 * pooled + 0.5 * skip).view(b_days, n_assets, -1)


class PureTemporalAlphaModel(nn.Module):
    """Stage 1 纯时序深度学习选股模型 (不含知识图谱空间消息传递，用作 Stage 2 的消融基线)."""

    def __init__(self, in_features: int = 8, seq_len: int = 20, d_model: int = 64) -> None:
        super().__init__()
        self.encoder = FinancialPatchTSTEncoder(
            in_features=in_features,
            seq_len=seq_len,
            d_model=d_model,
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输出截面 Alpha 预测分: (Batch_days, N_assets)."""
        h = self.encoder(x)
        return self.head(h).squeeze(-1)


class FinancialiTransformer(nn.Module):
    """Stage 1 进阶对比模型：iTransformer (Inverted Transformer for Financial Time-Series).

    核心原理：
    将每个量价特征整条长度为 L 的历史序列映射为一个 Variate Token (F_features 个 Token)，
    在特征变量维度做 Multi-Head Self-Attention，专门捕捉「收益率-换手量比-波动率」之间的跨指标非线性交互。
    """

    def __init__(
        self,
        in_features: int = 8,
        seq_len: int = 20,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.revin = RevIN(in_features)
        # Inverted Projection: 将长度为 seq_len 的单变量序列直接投影为 d_model 维度的 Variate Token
        self.variate_proj = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.variate_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.head = nn.Sequential(
            nn.Linear(in_features * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 (Batch_days, N_assets, L_window, F_features) -> 输出 (Batch_days, N_assets)."""
        b_days, n_assets, seq_len, n_feats = x.shape
        x_flat = x.reshape(b_days * n_assets, seq_len, n_feats)
        x_norm = self.revin(x_flat)

        # 反转维度: (B*N, F_features, L_window)
        x_inv = x_norm.transpose(1, 2)
        var_tokens = self.variate_proj(x_inv)  # (B*N, F_features, d_model)
        encoded_vars = self.variate_transformer(var_tokens)
        out = self.head(encoded_vars.reshape(b_days * n_assets, -1))
        return out.view(b_days, n_assets)


class MASTERAlphaModel(nn.Module):
    """Stage 1/2 进阶对比模型：MASTER (Market-Guided Stock Transformer).

    核心原理：
    1. 市场环境门控 (Market-Guided Gate)：用当日全市场截面均值与波动率生成动态特征缩放门控；
    2. 交替执行 Intra-Stock 时序编码与 Inter-Stock 全连接截面自注意力 (无显式知识图谱拓扑约束)。
    用于与带显式产业链拓扑约束的 KG-GAT 进行严格对比。
    """

    def __init__(
        self,
        in_features: int = 8,
        seq_len: int = 20,
        d_model: int = 64,
        n_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.market_gate = nn.Sequential(
            nn.Linear(in_features * 2, in_features),
            nn.Sigmoid(),
        )
        self.temporal_encoder = FinancialPatchTSTEncoder(
            in_features=in_features,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.cross_stock_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 (Batch_days, N_assets, L_window, F_features) -> 输出 (Batch_days, N_assets)."""
        # 1. 提取全市场截面均值与标准差作为宏观环境引导信号
        mkt_mean = x.mean(dim=1, keepdim=True)
        mkt_std = x.std(dim=1, keepdim=True) + 1e-6
        mkt_stat = torch.cat([mkt_mean, mkt_std], dim=-1)
        gate = self.market_gate(mkt_stat) * 2.0  # 缩放范围 [0, 2]
        x_gated = x * gate

        # 2. Intra-Stock 时序表征提取
        h_intra = self.temporal_encoder(x_gated)  # (B, N, d_model)

        # 3. Inter-Stock 跨标的全连接自注意力
        h_inter, _ = self.cross_stock_attn(h_intra, h_intra, h_intra)
        h_fused = self.norm(h_intra + 0.25 * h_inter)
        return self.head(h_fused).squeeze(-1)
