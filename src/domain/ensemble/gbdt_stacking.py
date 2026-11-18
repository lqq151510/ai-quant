"""领域层新增算法 4：Purged Ridge + 梯度提升树 (Hist-GBDT) + 时空图网络 (KG-GAT) 正交 Stacking 集成器."""

from collections.abc import Iterator
from typing import Dict, Tuple
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from src.domain.timeseries_dl.losses import build_exponential_time_decay_weights


def iter_observed_cross_sections(
    predictions: np.ndarray, targets: np.ndarray,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield comparable cross sections using only finite prediction/target pairs."""
    if predictions.ndim != 2 or predictions.shape != targets.shape:
        raise ValueError("cross-sectional predictions and targets must share a (days, assets) shape")
    for daily_pred, daily_target in zip(predictions, targets, strict=True):
        observed = np.isfinite(daily_pred) & np.isfinite(daily_target)
        pred, target = daily_pred[observed], daily_target[observed]
        if len(target) >= 2 and pred.std() > 1e-8 and target.std() > 1e-8:
            yield pred, target


class OrthogonalTreeGNNStacker:
    """结合 GBDT 非线性硬阈值切分与 KG-GAT 连续流形表征的正交融合模型.

    为什么 Tree + GNN 能显著提升 ICIR？
    树模型擅长在表格统计量上做轴平行阈值切分 (如「量比 > 1.5 且 20日乖离率处于区间 [a, b]」)，
    而时空图网络擅长捕捉跨标的连续注意力与产业链溢出。二者残差高度正交。
    """

    def __init__(self, random_state: int = 42) -> None:
        self.gbdt = HistGradientBoostingRegressor(
            max_iter=45,
            max_depth=3,
            min_samples_leaf=30,
            l2_regularization=5.0,
            learning_rate=0.04,
            random_state=random_state,
        )
        self.tree_weight: float = 0.25
        self.expert_weights: Dict[str, float] = {"ridge": 0.40, "tree": 0.35, "gnn": 0.25}

    @staticmethod
    def extract_tabular_features(x_seq: np.ndarray) -> np.ndarray:
        """从 (T, N, L, F) 窗口提取 4 组汇总统计量 -> (T*N, 4*F)."""
        t_days, n_assets, _, n_feats = x_seq.shape
        last_val = x_seq[:, :, -1, :]                          # 当日截面值
        mean_5d = x_seq[:, :, -5:, :].mean(axis=2)             # 5日均值
        mean_20d = x_seq.mean(axis=2)                          # 20日均值
        slope_5d = x_seq[:, :, -1, :] - x_seq[:, :, -5, :]     # 5日动量斜率
        feat_stack = np.concatenate([last_val, mean_5d, mean_20d, slope_5d], axis=-1)
        return feat_stack.reshape(t_days * n_assets, 4 * n_feats)

    @staticmethod
    def _row_zscore(mat: np.ndarray) -> np.ndarray:
        mean = mat.mean(axis=-1, keepdims=True)
        std = mat.std(axis=-1, keepdims=True) + 1e-6
        return (mat - mean) / std

    @staticmethod
    def _eval_skill(pred_2d: np.ndarray, target_2d: np.ndarray) -> float:
        """在观察到的验证截面上计算 0.5 * PearsonIC + 0.5 * SpearmanRankIC."""
        scores = []
        for pred, target in iter_observed_cross_sections(pred_2d, target_2d):
            ic = float(np.corrcoef(pred, target)[0, 1])
            ric = float(spearmanr(pred, target).statistic)
            if np.isfinite(ic) and np.isfinite(ric):
                scores.append(0.5 * ic + 0.5 * ric)
        return float(np.mean(scores)) if scores else 0.0

    def fit_and_blend(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray,
        test_x: np.ndarray,
        gnn_preds: Dict[str, np.ndarray],
        train_micro: np.ndarray | None = None,
        test_micro: np.ndarray | None = None,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """训练 Purged Time-Decayed Ridge 与 Löwdin GBDT，结合验证集非负收缩权重融合三专家预测."""
        t_tr, n_assets = train_y.shape
        t_va = val_y.shape[0]
        t_te = test_x.shape[0]
        y_tr_flat = train_y.reshape(-1)
        observed = np.isfinite(y_tr_flat)
        if not observed.any():
            raise ValueError("GBDT training requires at least one observed target")

        x_tr_raw = self.extract_tabular_features(train_x)
        x_va_raw = self.extract_tabular_features(val_x)
        x_te_raw = self.extract_tabular_features(test_x)

        # 所有统计量、正交化矩阵与时间衰减权重严格仅在已观测训练集行上拟合，绝不触碰未观测样本
        x_tr_observed = x_tr_raw[observed]
        day_w = build_exponential_time_decay_weights(t_tr, half_life=320.0)
        sample_w_flat = np.repeat(day_w, n_assets)
        w_obs = sample_w_flat[observed]

        # Expert 1: Purged Time-Decayed Ridge 线性专家
        mu_obs = np.nanmean(x_tr_observed, axis=0, keepdims=True)
        sd_obs = np.nanstd(x_tr_observed, axis=0, keepdims=True) + 1e-6
        x_tr_std_obs = np.nan_to_num((x_tr_observed - mu_obs) / sd_obs, nan=0.0, posinf=0.0, neginf=0.0)
        x_tr_std = np.nan_to_num((x_tr_raw - mu_obs) / sd_obs, nan=0.0, posinf=0.0, neginf=0.0)
        x_va_std = np.nan_to_num((x_va_raw - mu_obs) / sd_obs, nan=0.0, posinf=0.0, neginf=0.0)
        x_te_std = np.nan_to_num((x_te_raw - mu_obs) / sd_obs, nan=0.0, posinf=0.0, neginf=0.0)

        best_alpha = 800.0
        best_ridge_skill = -float("inf")
        best_ridge_model: Ridge | None = None
        for alpha in (25.0, 150.0, 800.0, 3000.0):
            cand = Ridge(alpha=alpha)
            cand.fit(x_tr_std_obs, y_tr_flat[observed], sample_weight=w_obs)
            cand_va = self._row_zscore(cand.predict(x_va_std).reshape(t_va, n_assets))
            if any(True for _ in iter_observed_cross_sections(cand_va, val_y)):
                skill = self._eval_skill(cand_va, val_y)
                if skill > best_ridge_skill:
                    best_ridge_skill = skill
                    best_alpha = alpha
                    best_ridge_model = cand
        if best_ridge_model is None:
            best_ridge_model = Ridge(alpha=best_alpha)
            best_ridge_model.fit(x_tr_std_obs, y_tr_flat[observed], sample_weight=w_obs)

        ridge_tr = self._row_zscore(best_ridge_model.predict(x_tr_std).reshape(t_tr, n_assets))
        ridge_va = self._row_zscore(best_ridge_model.predict(x_va_std).reshape(t_va, n_assets))
        ridge_te = self._row_zscore(best_ridge_model.predict(x_te_std).reshape(t_te, n_assets))

        # Expert 2: Löwdin 对称正交化 + Hist-GBDT
        cov_tr = (x_tr_observed.T @ x_tr_observed) / len(x_tr_observed)
        shrunk_cov = 0.70 * cov_tr + 0.30 * np.eye(x_tr_raw.shape[1], dtype=np.float64)
        eigvals, eigvecs = np.linalg.eigh(shrunk_cov)
        lowdin_mat = (eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-4))) @ eigvecs.T).astype(np.float32)

        x_tr_tab = 0.75 * x_tr_raw + 0.25 * (x_tr_raw @ lowdin_mat)
        x_va_tab = 0.75 * x_va_raw + 0.25 * (x_va_raw @ lowdin_mat)
        x_te_tab = 0.75 * x_te_raw + 0.25 * (x_te_raw @ lowdin_mat)

        self.gbdt.fit(x_tr_tab[observed], y_tr_flat[observed], sample_weight=w_obs)

        tree_tr = self._row_zscore(self.gbdt.predict(x_tr_tab).reshape(t_tr, n_assets))
        tree_va = self._row_zscore(self.gbdt.predict(x_va_tab).reshape(t_va, n_assets))
        tree_te = self._row_zscore(self.gbdt.predict(x_te_tab).reshape(t_te, n_assets))

        gnn_tr = self._row_zscore(gnn_preds["train_preds"])
        gnn_te = self._row_zscore(gnn_preds["test_preds"])

        # 验证集非负收缩融合（绝不翻转任何专家的预测符号）
        s_ridge = self._eval_skill(ridge_va, val_y)
        s_tree = self._eval_skill(tree_va, val_y)
        if "val_preds" in gnn_preds:
            gnn_va = self._row_zscore(gnn_preds["val_preds"])
            s_gnn = self._eval_skill(gnn_va, val_y)
            raw_ridge = max(0.0, s_ridge) + 0.025
            raw_tree = max(0.0, s_tree) + 0.020
            raw_gnn = max(0.0, s_gnn) + 0.015
            total_w = raw_ridge + raw_tree + raw_gnn
            w_ridge = raw_ridge / total_w
            w_tree = raw_tree / total_w
            w_gnn = raw_gnn / total_w
        else:
            raw_ridge = max(0.0, s_ridge) + 0.02
            raw_tree = max(0.0, s_tree) + 0.02
            w_gnn = 0.50
            w_ridge = 0.50 * (raw_ridge / (raw_ridge + raw_tree))
            w_tree = 0.50 * (raw_tree / (raw_ridge + raw_tree))

        self.tree_weight = float(w_tree)
        self.expert_weights = {
            "ridge": float(w_ridge),
            "tree": float(w_tree),
            "gnn": float(w_gnn),
        }

        tr_blend = self._row_zscore(w_ridge * ridge_tr + w_tree * tree_tr + w_gnn * gnn_tr)
        te_blend = self._row_zscore(w_ridge * ridge_te + w_tree * tree_te + w_gnn * gnn_te)

        # 数据升级 1：融入主力资金流承接强度 (cmf_20d) 与上下行半方差不对称性 (asym_semi_20d)
        if train_micro is not None and test_micro is not None:
            micro_tr = self._row_zscore(0.60 * train_micro[:, :, 0] + 0.40 * train_micro[:, :, 1])
            micro_te = self._row_zscore(0.60 * test_micro[:, :, 0] + 0.40 * test_micro[:, :, 1])
            tr_blend = 0.86 * tr_blend + 0.14 * micro_tr
            te_blend = 0.86 * te_blend + 0.14 * micro_te

        blended = {
            "train_preds": tr_blend,
            "test_preds": te_blend,
        }
        return blended, tree_te
