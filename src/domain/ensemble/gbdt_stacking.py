"""领域层新增算法 4：梯度提升树 (Hist-GBDT) 与时空图网络 (KG-GAT) 正交 Stacking 集成器."""

from collections.abc import Iterator
from typing import Dict, Tuple
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

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
            max_iter=55,
            max_depth=3,
            min_samples_leaf=30,
            l2_regularization=3.0,
            learning_rate=0.04,
            random_state=random_state,
        )
        self.tree_weight: float = 0.25

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
        """训练带概念漂移衰减与 Löwdin 正交化的 GBDT，融合主力资金流微结构因子与 KG-GAT 预测."""
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

        # Fit the Löwdin transform on the same observed training rows as the tree.
        x_tr_observed = x_tr_raw[observed]
        cov_tr = (x_tr_observed.T @ x_tr_observed) / len(x_tr_observed)
        shrunk_cov = 0.70 * cov_tr + 0.30 * np.eye(x_tr_raw.shape[1], dtype=np.float64)
        eigvals, eigvecs = np.linalg.eigh(shrunk_cov)
        lowdin_mat = (eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-4))) @ eigvecs.T).astype(np.float32)

        x_tr_tab = 0.75 * x_tr_raw + 0.25 * (x_tr_raw @ lowdin_mat)
        x_va_tab = 0.75 * x_va_raw + 0.25 * (x_va_raw @ lowdin_mat)
        x_te_tab = 0.75 * x_te_raw + 0.25 * (x_te_raw @ lowdin_mat)

        # Keep time decay tied to the original training dates before dropping labels.
        day_w = build_exponential_time_decay_weights(t_tr, half_life=220.0)
        sample_w_flat = np.repeat(day_w, n_assets)
        self.gbdt.fit(x_tr_tab[observed], y_tr_flat[observed], sample_weight=sample_w_flat[observed])

        tree_tr = self._row_zscore(self.gbdt.predict(x_tr_tab).reshape(t_tr, n_assets))
        tree_va = self._row_zscore(self.gbdt.predict(x_va_tab).reshape(t_va, n_assets))
        tree_te = self._row_zscore(self.gbdt.predict(x_te_tab).reshape(t_te, n_assets))

        gnn_tr = self._row_zscore(gnn_preds["train_preds"])
        gnn_te = self._row_zscore(gnn_preds["test_preds"])

        # 根据验证集 IC 自适应确定 GBDT 辅助权重 (限制在 [0.12, 0.30] 之间以保持时空图网络主干)
        va_ic_list = [
            np.corrcoef(pred, target)[0, 1]
            for pred, target in iter_observed_cross_sections(tree_va, val_y)
            if pred.std() > 1e-6
        ]
        mean_va_ic = float(np.mean(va_ic_list)) if va_ic_list else 0.0
        self.tree_weight = float(np.clip(0.18 + 1.5 * max(0.0, mean_va_ic), 0.15, 0.30))

        w = self.tree_weight
        tr_blend = (1.0 - w) * gnn_tr + w * tree_tr
        te_blend = (1.0 - w) * gnn_te + w * tree_te

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
