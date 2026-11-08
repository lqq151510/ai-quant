"""Causal shadow comparison of a sticky Gaussian regime model with and without W1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import market_close_returns
from .demo import fixture
from .feature_upgrades import wasserstein_drift
from .protocol import ETF_SYMBOLS
from .timeline import sessions


def filtered_regimes(features: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Fit Gaussian emissions on past rows, then filter a fixed sticky transition."""
    from sklearn.mixture import GaussianMixture

    if not isinstance(features.index, pd.DatetimeIndex) or not features.index.is_unique or not features.index.is_monotonic_increasing:
        raise ValueError("regime observations must have unique ascending session timestamps")
    training = features.loc[features.index < cutoff].to_numpy(dtype=float)
    observed = features.to_numpy(dtype=float)
    if len(training) < 20 or features.shape[1] < 3 or not np.isfinite(observed).all():
        raise ValueError("regime study requires at least 20 complete training sessions")
    mean = training.mean(axis=0)
    scale = training.std(axis=0)
    scale[scale <= 1e-12] = 1.0
    model = GaussianMixture(n_components=3, covariance_type="full", reg_covar=1e-4,
                            random_state=42).fit((training - mean) / scale)
    # Mixture posteriors are proportional to likelihood times mixture weight.
    log_likelihood = np.log(np.maximum(model.predict_proba((observed - mean) / scale), 1e-300))
    log_likelihood -= np.log(model.weights_)
    ordering = np.argsort(-(model.means_[:, 0] + model.means_[:, 1] - 0.8 * model.means_[:, 2]))
    log_likelihood = log_likelihood[:, ordering]
    transition = np.full((3, 3), 0.09)
    np.fill_diagonal(transition, 0.82)
    probabilities = []
    posterior = np.array([0.30, 0.50, 0.20])
    for row in log_likelihood:
        likelihood = np.exp(np.clip((row - row.max()) / 3.5, -20, 0))
        posterior = (posterior @ transition) * likelihood
        posterior /= posterior.sum()
        probabilities.append(posterior)
    return pd.DataFrame(probabilities, index=features.index, columns=["higher_trend", "middle", "lower_trend"])


def run_demo() -> dict:
    frames = fixture()
    calendar = sessions(frames["calendar"]["date"])
    panel = market_close_returns(frames["bars"], frames["actions"], calendar).loc[:, list(ETF_SYMBOLS)]
    market_return = panel.mean(axis=1, skipna=False)
    daily = pd.DataFrame({"mean": market_return, "trend": market_return.rolling(5).mean(),
                          "volatility": market_return.rolling(10).std(), "dispersion": panel.std(axis=1),
                          "wasserstein": wasserstein_drift(panel)})
    complete = daily.dropna()
    if not complete.index.equals(calendar[calendar >= complete.index[0]]):
        raise ValueError("regime observations lack a continuous complete session window")
    cutoff = calendar[int(len(calendar) * 0.72)]
    baseline = filtered_regimes(complete.drop(columns="wasserstein"), cutoff)
    candidate = filtered_regimes(complete, cutoff)
    path = pd.DataFrame({"wasserstein": complete["wasserstein"],
                         "baseline_lower_trend": baseline["lower_trend"],
                         "candidate_lower_trend": candidate["lower_trend"]})
    path = path.loc[path.index >= cutoff].rename_axis("session").reset_index()
    path["session"] = path["session"].dt.strftime("%Y-%m-%d")
    return {"origin": "synthetic_fixture", "status": "engineering_only",
            "market_proxy": list(ETF_SYMBOLS), "fit_end_exclusive": cutoff.date().isoformat(),
            "training_sessions": int(sum(complete.index < cutoff)),
            "transition_self_probability": 0.82, "wasserstein_windows_sessions": [60, 10],
            "interpretation": "Filtered lower-trend probabilities on a synthetic panel. No crisis detection or trading evidence.",
            "path": path.to_dict(orient="records")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run_demo()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "sessions": len(result["path"]), "report": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
