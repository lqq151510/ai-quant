"""领域层 Stage 1B：金融量化专属截面排序相关性损失函数 (Differentiable IC Loss)."""

import torch
import torch.nn as nn


class CrossSectionalICLoss(nn.Module):
    """混合截面皮尔逊相关系数损失 (1 - IC) 与轻量级 MSE 正则.

    为什么不用纯 MSE？
    金融收益率信噪比极低，纯 MSE 会导致神经网络输出全部坍缩为 0 附近常数。
    直接优化每个交易日截面上的皮尔逊相关系数 (IC)，强迫模型拉开截面多空相对分差。
    """

    def __init__(self, mse_weight: float = 0.15, eps: float = 1e-6) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.eps = eps
        self.mse = nn.SmoothL1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算批次内每个交易日截面的平均 IC Loss.

        Args:
            pred: 预测截面分数, shape = (Batch_days, N_assets)
            target: 真实截面标准化超额收益, shape = (Batch_days, N_assets)
        """
        pred_centered = pred - pred.mean(dim=-1, keepdim=True)
        target_centered = target - target.mean(dim=-1, keepdim=True)

        cov = (pred_centered * target_centered).sum(dim=-1)
        std_pred = torch.sqrt((pred_centered ** 2).sum(dim=-1) + self.eps)
        std_target = torch.sqrt((target_centered ** 2).sum(dim=-1) + self.eps)

        daily_ic = cov / (std_pred * std_target + self.eps)
        ic_loss = (1.0 - daily_ic).mean()
        reg_loss = self.mse(pred, target)
        return ic_loss + self.mse_weight * reg_loss
