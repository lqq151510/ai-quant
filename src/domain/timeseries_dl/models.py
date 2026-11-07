"""领域层 Stage 1B：带 RevIN 非平稳抑制的金融时序深度学习编码器 (PatchTST / AttentiveGRU)."""

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """可逆实例归一化 (Reversible Instance Normalization).

    针对每个标的在长度为 L 的局部窗口内去均值除标准差，抑制金融序列的非平稳分布漂移。
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (B_total, L_window, F_features)."""
        mean = x.mean(dim=1, keepdim=True)
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)
        x_norm = (x - mean) / std
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
        return pooled.view(b_days, n_assets, -1)


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
