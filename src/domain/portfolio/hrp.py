"""领域层新增算法 3：基于 Ledoit-Wolf 收缩协方差与凝聚聚类树的层次风险平价 (HRP, Hierarchical Risk Parity)."""

from typing import List
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf


class HierarchicalRiskParityOptimizer:
    """Marcos López de Prado (2016) 层次风险平价组合优化器.

    三阶段核心步骤：
    1. Tree Clustering: 计算 Ledoit-Wolf 收缩协方差与相关距离 d_ij = sqrt(0.5 * (1 - rho_ij))，生成层次聚类树；
    2. Quasi-Diagonalization: 按聚类树叶节点顺序重排协方差矩阵，使高相关同板块资产相邻；
    3. Recursive Bisection: 自顶向下递归二分簇，按逆方差分配簇间风险预算，防止 Top-K 全部扎堆同一赛道。
    """

    @staticmethod
    def _get_cluster_var(cov: np.ndarray, cluster_items: List[int]) -> float:
        """计算子簇内部的逆方差组合方差."""
        sub_cov = cov[np.ix_(cluster_items, cluster_items)]
        inv_diag = 1.0 / np.maximum(np.diag(sub_cov), 1e-8)
        w = inv_diag / np.sum(inv_diag)
        return float(w.T @ sub_cov @ w)

    @classmethod
    def allocate_hrp_weights(
        cls,
        hist_returns: np.ndarray,
        candidate_indices: List[int],
        alpha_tilts: np.ndarray | None = None,
    ) -> np.ndarray:
        """在候选标的池上计算 HRP 层次风险平价权重 (融合温和 Alpha 倾角).

        Args:
            hist_returns: 历史收益率矩阵 (仅使用 <= t 的历史窗口), shape = (T_hist, N_total)
            candidate_indices: 选中的 Top-K 候选标的下标列表
            alpha_tilts: 候选标的的预测 Alpha 分数, shape = (N_total,)

        Returns:
            weights: 全资产维度权重向量 (未选中标的为 0, 选中标的和为 1.0), shape = (N_total,)
        """
        n_total = hist_returns.shape[1]
        k = len(candidate_indices)
        if k == 0:
            return np.zeros(n_total, dtype=np.float32)
        if k == 1:
            out = np.zeros(n_total, dtype=np.float32)
            out[candidate_indices[0]] = 1.0
            return out

        sub_ret = hist_returns[:, candidate_indices]
        # 1. Ledoit-Wolf 收缩协方差估计 (解决小样本协方差矩阵病态求逆问题)
        try:
            lw = LedoitWolf().fit(sub_ret)
            cov = lw.covariance_
        except Exception:
            cov = np.cov(sub_ret, rowvar=False) + np.eye(k) * 1e-5

        std = np.sqrt(np.maximum(np.diag(cov), 1e-8))
        corr = cov / np.outer(std, std)
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)

        # 2. 相关系数距离矩阵与层次聚类准对角化
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
        np.fill_diagonal(dist, 0.0)
        condensed_dist = squareform(dist, checks=False)
        link = linkage(condensed_dist, method="single")
        sort_ix = leaves_list(link).tolist()

        # 3. 递归二分法 (Recursive Bisection)
        hrp_w = np.ones(k, dtype=np.float64)
        clusters = [sort_ix]
        while len(clusters) > 0:
            next_clusters = []
            for c in clusters:
                if len(c) > 1:
                    mid = len(c) // 2
                    left_c, right_c = c[:mid], c[mid:]
                    var_left = cls._get_cluster_var(cov, left_c)
                    var_right = cls._get_cluster_var(cov, right_c)
                    alpha_split = 1.0 - var_left / (var_left + var_right + 1e-12)
                    hrp_w[left_c] *= alpha_split
                    hrp_w[right_c] *= (1.0 - alpha_split)
                    next_clusters.extend([left_c, right_c])
            clusters = next_clusters

        # 4. 融合温和的 Alpha 信号倾角 (既保持 HRP 跨赛道风险分散主干，又向高 Alpha 龙头适度倾斜)
        if alpha_tilts is not None:
            sub_alpha = alpha_tilts[candidate_indices]
            z_alpha = (sub_alpha - np.mean(sub_alpha)) / (np.std(sub_alpha) + 1e-6)
            tilt_mult = np.exp(np.clip(0.20 * z_alpha, -0.35, 0.35))
            hrp_w = hrp_w * tilt_mult

        hrp_w = hrp_w / np.sum(hrp_w)
        full_w = np.zeros(n_total, dtype=np.float32)
        for idx_local, idx_global in enumerate(candidate_indices):
            full_w[idx_global] = float(hrp_w[idx_local])
        return full_w
