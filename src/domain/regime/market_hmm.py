"""领域层新增算法 1：基于粘性转移矩阵与因果前向滤波的市场状态隐马尔可夫模型 (Sticky Market Regime HMM)."""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


REGIME_NAMES: Dict[int, str] = {
    0: "🟢 牛市主升/高景气 (Bull_Trend)",
    1: "🟡 结构性震荡/轮动 (Oscillation)",
    2: "🔴 危机退潮/外盘冲击 (Crisis_Panic)",
}


@dataclass
class MarketRegimeSnapshot:
    """单日市场隐状态识别结果 (融合 Sticky-HMM 与 1D-Wasserstein 最优传输突变检测)."""
    trade_date: pd.Timestamp
    state_id: int
    state_name: str
    state_probs: np.ndarray          # [p_bull, p_osc, p_crisis]
    recommended_max_exposure: float  # HMM + Wasserstein OT 建议总权益仓位上限
    wasserstein_w2: float = 0.0      # 1D-Wasserstein 最优传输几何距离
    ot_left_tail_shift: float = 0.0  # 最优传输左尾分位数偏移 (负值代表左尾黑天鹅增厚)


def compute_1d_wasserstein_w2(u_samples: np.ndarray, v_samples: np.ndarray, n_quantiles: int = 25) -> tuple[float, float]:
    """算法升级 B：计算两个一维经验分布之间的 2-Wasserstein 最优传输距离与左尾偏移.

    数学原理：
    W_2(mu_t, mu_ref) = ( int_0^1 |F_t^{-1}(q) - F_ref^{-1}(q)|^2 dq )^{1/2}
    无需高斯分布假设，直接捕捉收益分布的高阶厚尾形变与黑天鹅偏度坍塌。
    """
    q_grid = np.linspace(0.04, 0.96, n_quantiles)
    q_u = np.quantile(u_samples, q_grid)
    q_v = np.quantile(v_samples, q_grid)
    w2 = float(np.sqrt(np.mean((q_u - q_v) ** 2)))
    # 15% 左尾分位数偏移: 近期左尾相比历史基准是否发生坍塌
    left_shift = float(np.quantile(u_samples, 0.15) - np.quantile(v_samples, 0.15))
    return w2, left_shift


class StickyMarketRegimeHMM:
    """3-状态高斯发射 + 粘性马尔可夫转移矩阵 + 1D-Wasserstein 最优传输的因果市场状态识别器.

    严格防未来函数：
    1. 高斯混合发射分布与 Wasserstein 基准阈值仅在训练集 (train_bars) 上拟合；
    2. 样本外推理采用纯前向滤波 (Causal Forward Algorithm): alpha_t = P(Z_t | O_{1:t})，绝不使用平滑后向算子 (Viterbi/Backward)。
    """

    def __init__(self, self_transition_prob: float = 0.82, random_state: int = 42) -> None:
        self.self_transition_prob = self_transition_prob
        self.random_state = random_state
        self.gmm = GaussianMixture(
            n_components=3,
            covariance_type="full",
            reg_covar=1e-4,
            random_state=random_state,
        )
        self.state_perm: np.ndarray = np.array([0, 1, 2])
        self.trans_mat: np.ndarray = self._build_sticky_transition(3, self_transition_prob)
        self.feat_mean: np.ndarray = np.zeros(4)
        self.feat_std: np.ndarray = np.ones(4)
        self.w2_mean: float = 0.01
        self.w2_std: float = 0.005

    @staticmethod
    def _build_sticky_transition(k: int, p_stay: float) -> np.ndarray:
        off_diag = (1.0 - p_stay) / (k - 1)
        mat = np.full((k, k), off_diag, dtype=np.float64)
        np.fill_diagonal(mat, p_stay)
        return mat

    @staticmethod
    def extract_macro_observations(bars_df: pd.DataFrame) -> pd.DataFrame:
        """从全截面日线提取宏观观测序列与 1D-Wasserstein 最优传输距离."""
        df = bars_df.sort_values(["symbol", "trade_date"]).copy()
        df["ret_1d"] = df.groupby("symbol")["close"].pct_change(1)

        daily = df.groupby("trade_date").agg(
            mkt_ret_1d=("ret_1d", "mean"),
            cross_dispersion=("ret_1d", "std"),
        ).dropna().sort_index()

        daily["mkt_ret_5d"] = daily["mkt_ret_1d"].rolling(5, min_periods=2).mean()
        daily["mkt_vol_10d"] = daily["mkt_ret_1d"].rolling(10, min_periods=3).std()
        daily = daily.dropna().reset_index()

        # 逐日计算严格因果的 1D-Wasserstein 分布距离: 近期 5 日全截面收益 vs 过去 [t-45, t-5] 基准分布
        piv_ret = df.pivot(index="trade_date", columns="symbol", values="ret_1d").reindex(daily["trade_date"]).to_numpy()
        w2_list, left_shift_list = [], []
        for t in range(len(daily)):
            if t < 20:
                w2_list.append(0.008)
                left_shift_list.append(0.0)
            else:
                recent_win = piv_ret[max(0, t - 4): t + 1].reshape(-1)
                ref_win = piv_ret[max(0, t - 45): max(1, t - 4)].reshape(-1)
                recent_win = recent_win[~np.isnan(recent_win)]
                ref_win = ref_win[~np.isnan(ref_win)]
                if len(recent_win) >= 10 and len(ref_win) >= 20:
                    w2, l_shift = compute_1d_wasserstein_w2(recent_win, ref_win)
                else:
                    w2, l_shift = 0.008, 0.0
                w2_list.append(w2)
                left_shift_list.append(l_shift)

        daily["wasserstein_w2"] = w2_list
        daily["ot_left_tail_shift"] = left_shift_list
        return daily

    def fit(self, train_macro_df: pd.DataFrame) -> "StickyMarketRegimeHMM":
        """在训练集宏观序列上拟合 3 个隐状态，并标定 Wasserstein 最优传输距离基准."""
        obs_cols = ["mkt_ret_1d", "mkt_ret_5d", "mkt_vol_10d", "cross_dispersion"]
        x_raw = train_macro_df[obs_cols].to_numpy(dtype=np.float64)
        self.feat_mean = x_raw.mean(axis=0)
        self.feat_std = x_raw.std(axis=0) + 1e-6
        x_norm = (x_raw - self.feat_mean) / self.feat_std

        self.gmm.fit(x_norm)
        # 根据每个高斯分量的 (5日收益均值 - 0.8 * 10日波动率均值) 从高到低映射为 [0: Bull, 1: Osc, 2: Crisis]
        means = self.gmm.means_
        regime_score = means[:, 0] + means[:, 1] - 0.8 * means[:, 2]
        self.state_perm = np.argsort(regime_score)[::-1]

        w2_arr = train_macro_df["wasserstein_w2"].to_numpy(dtype=np.float64)
        self.w2_mean = float(np.mean(w2_arr))
        self.w2_std = float(np.std(w2_arr) + 1e-6)
        return self

    def filter_sequence(self, macro_df: pd.DataFrame) -> List[MarketRegimeSnapshot]:
        """对任意时间序列执行因果前向马尔可夫滤波 + Wasserstein 最优传输厚尾校准."""
        obs_cols = ["mkt_ret_1d", "mkt_ret_5d", "mkt_vol_10d", "cross_dispersion"]
        x_raw = macro_df[obs_cols].to_numpy(dtype=np.float64)
        x_norm = (x_raw - self.feat_mean) / self.feat_std

        # 计算原始 GMM 各分量对数似然并重排为语义顺序 [Bull, Osc, Crisis]
        log_prob_raw = self.gmm._estimate_log_prob(x_norm)  # (T, 3)
        log_emission = log_prob_raw[:, self.state_perm]
        w2_vals = macro_df["wasserstein_w2"].to_numpy(dtype=np.float64)
        left_shifts = macro_df["ot_left_tail_shift"].to_numpy(dtype=np.float64)

        alpha = np.array([0.30, 0.50, 0.20], dtype=np.float64)
        snapshots: List[MarketRegimeSnapshot] = []

        for t in range(len(macro_df)):
            # 1. 一步预测先验: prior = alpha_{t-1} @ A
            prior = alpha @ self.trans_mat
            # 2. 贝叶斯似然更新 + Wasserstein 最优传输左尾黑天鹅增益
            ll = (log_emission[t] - np.max(log_emission[t])) / 3.5
            w2_z = (w2_vals[t] - self.w2_mean) / self.w2_std
            if w2_z > 0.6 and left_shifts[t] < -0.005:
                # 当最优传输距离放大且左尾坍塌时，非参数化提升 Crisis_Panic 对数似然
                ll[2] += 0.85 * min(w2_z, 3.0)
            likelihood = np.exp(np.clip(ll - np.max(ll), -20.0, 0.0))
            post = prior * likelihood
            alpha = post / (np.sum(post) + 1e-12)

            state_id = int(np.argmax(alpha))
            # 根据隐状态后验概率加权计算建议最大权益敞口: Bull=0.92, Osc=0.75, Crisis=0.45
            rec_exposure = float(alpha[0] * 0.92 + alpha[1] * 0.75 + alpha[2] * 0.45)
            snapshots.append(
                MarketRegimeSnapshot(
                    trade_date=pd.Timestamp(macro_df["trade_date"].iloc[t]),
                    state_id=state_id,
                    state_name=REGIME_NAMES[state_id],
                    state_probs=alpha.copy(),
                    recommended_max_exposure=rec_exposure,
                    wasserstein_w2=float(w2_vals[t]),
                    ot_left_tail_shift=float(left_shifts[t]),
                )
            )
        return snapshots
