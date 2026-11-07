"""An isolated research run: no accounts, orders, portfolio plans, or database writes."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingRegressor
from torch.utils.data import DataLoader, TensorDataset

from src.domain.timeseries_dl.models import PureTemporalAlphaModel
from .data import ALL_FEATURES, BASE_FEATURES, EXTRA_FEATURES, prepare_feature_panel
from .transforms import LowdinTransform, WassersteinRegimeMonitor, exponential_time_weights, macro_distribution_observations


@dataclass
class Dataset:
    dates: pd.DatetimeIndex
    symbols: list[str]
    x: np.ndarray
    y: np.ndarray
    horizon: int = 3


def build_dataset(panel: pd.DataFrame, seq_len: int = 20, horizon: int = 3) -> Dataset:
    if seq_len < 5 or horizon < 1:
        raise ValueError("seq_len must be at least five and horizon must be positive")
    symbols = sorted(panel["symbol"].unique())
    dates = pd.DatetimeIndex(sorted(panel["trade_date"].unique()))
    if len(dates) <= 20 + seq_len + horizon:
        raise ValueError("Not enough days after feature warmup and label horizon")
    panel = panel.sort_values(["trade_date", "symbol"])
    cube = panel[ALL_FEATURES].to_numpy(dtype=np.float32).reshape(len(dates), len(symbols), -1)
    close = panel["close"].to_numpy(dtype=float).reshape(len(dates), len(symbols))
    xs, ys, sample_dates = [], [], []
    for t in range(20 + seq_len - 1, len(dates) - horizon):
        future = close[t + horizon] / close[t] - 1.0
        std = future.std()
        target = (future - future.mean()) / std if std > 1e-12 else np.zeros_like(future)
        xs.append(cube[t - seq_len + 1:t + 1].transpose(1, 0, 2))
        ys.append(target)
        sample_dates.append(dates[t])
    return Dataset(pd.DatetimeIndex(sample_dates), symbols, np.stack(xs), np.asarray(ys, dtype=np.float32), horizon)


def split_dataset(data: Dataset, purge_gap: int = 5):
    if purge_gap < data.horizon:
        raise ValueError("Purge gap must cover the target horizon")
    total = len(data.dates)
    a, b = int(total * 0.65), int(total * 0.80)

    def part(start, end):
        return Dataset(data.dates[start:end], data.symbols, data.x[start:end].copy(), data.y[start:end].copy(), data.horizon)

    train, val, test = part(0, a), part(a + purge_gap, b), part(b + purge_gap, total)
    if min(len(train.dates), len(val.dates), len(test.dates)) == 0:
        raise ValueError("The chronological train/validation/test partitions must be nonempty")
    return train, val, test


def fit_lowdin(train: Dataset, val: Dataset, test: Dataset):
    """Fit only the six original multi-horizon features on unique train dates."""
    positions = np.arange(6)
    train_rows = train.x[:, :, -1, positions].reshape(-1, len(positions))
    transform = LowdinTransform().fit(train_rows)
    for part in (train, val, test):
        part.x[..., positions] = transform.transform(part.x[..., positions]).astype(np.float32)
    return transform


def mask_unseen_feature_groups(train: Dataset, val: Dataset, test: Dataset):
    """Prevent random, untrained input weights from acting on test-only data."""
    groups = {
        "flow": (["flow_super_ratio", "flow_large_ratio"], "flow_available"),
        "intraday": (["intraday_log_up_down_variance"], "intraday_available"),
    }
    diagnostics = {}
    for group, (values, available) in groups.items():
        flag = ALL_FEATURES.index(available)
        coverage = {name: float((part.x[:, :, -1, flag] > 0).mean())
                    for name, part in (("train", train), ("validation", val), ("test", test))}
        enabled = coverage["train"] > 0
        if not enabled:
            indices = [ALL_FEATURES.index(name) for name in values + [available]]
            for part in (train, val, test):
                part.x[..., indices] = 0.0
        diagnostics[group] = {"enabled": enabled, "coverage_before_mask": coverage,
                              "reason": "observed_in_training" if enabled else "no_training_observations"}
    return diagnostics


def weighted_daily_ic_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Mean per-day loss with weights normalized over the whole training set."""
    if pred.shape != target.shape or pred.ndim != 2 or weights.shape != (len(pred),):
        raise ValueError("Expected [days, assets] predictions and one weight per day")
    if not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Weights must be positive and finite")
    p = pred - pred.mean(dim=1, keepdim=True)
    y = target - target.mean(dim=1, keepdim=True)
    ic = (p * y).sum(dim=1) / (torch.sqrt((p * p).sum(dim=1) + 1e-6) *
                                      torch.sqrt((y * y).sum(dim=1) + 1e-6) + 1e-6)
    daily_reg = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=1)
    return ((1.0 - ic + 0.15 * daily_reg) * weights).mean()


def _mean_ic(pred: np.ndarray, target: np.ndarray):
    scores = []
    for p, y in zip(pred, target):
        if p.std() > 1e-10 and y.std() > 1e-10:
            scores.append(float(np.corrcoef(p, y)[0, 1]))
    return float(np.mean(scores)) if scores else None


def _predict(model, x):
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(x, dtype=torch.float32)).cpu().numpy()


def train_models(train: Dataset, val: Dataset, test: Dataset, epochs: int = 2, half_life: float = 120.0):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    torch.manual_seed(42)
    np.random.seed(42)
    weights = exponential_time_weights(train.dates, half_life)
    model = PureTemporalAlphaModel(in_features=len(ALL_FEATURES), seq_len=train.x.shape[2], d_model=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.tensor(train.x), torch.tensor(train.y),
                                     torch.tensor(weights, dtype=torch.float32)),
                        batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(42))
    best, state = -np.inf, None
    for _ in range(epochs):
        model.train()
        for x_batch, y_batch, w_batch in loader:
            optimizer.zero_grad()
            loss = weighted_daily_ic_loss(model(x_batch), y_batch, w_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_ic = _mean_ic(_predict(model, val.x), val.y)
        score = val_ic if val_ic is not None else -np.inf
        if state is None or score > best:
            best = score
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    neural_test_ic = _mean_ic(_predict(model, test.x), test.y)

    # The same daily weights are repeated for each asset, preserving row order.
    tree = HistGradientBoostingRegressor(max_iter=40, max_depth=3, min_samples_leaf=20,
                                         l2_regularization=3.0, random_state=42)
    tree.fit(train.x[:, :, -1, :].reshape(-1, len(ALL_FEATURES)), train.y.reshape(-1),
             sample_weight=np.repeat(weights, len(train.symbols)))
    tree_pred = tree.predict(test.x[:, :, -1, :].reshape(-1, len(ALL_FEATURES))).reshape(test.y.shape)
    return model, tree, {"validation_ic": None if not np.isfinite(best) else float(best),
                         "test_ic": neural_test_ic, "tree_test_ic": _mean_ic(tree_pred, test.y),
                         "half_life_sample_dates": half_life,
                         "oldest_weight": float(weights[0]), "newest_weight": float(weights[-1])}


def run_experiment(bars, flow=None, minutes=None, epochs=2, half_life=120.0, macro_bars=None):
    panel = prepare_feature_panel(bars, flow, minutes)
    data = build_dataset(panel)
    train, val, test = split_dataset(data)
    feature_groups = mask_unseen_feature_groups(train, val, test)
    lowdin = fit_lowdin(train, val, test)
    model, tree, model_metrics = train_models(train, val, test, epochs=epochs, half_life=half_life)
    macro_input = bars if macro_bars is None else macro_bars
    macro = macro_distribution_observations(macro_input)
    train_macro = macro[macro["trade_date"] <= train.dates[-1]]
    monitor = WassersteinRegimeMonitor().fit(train_macro)
    snapshots = monitor.filter_sequence(macro)
    test_dates = set(test.dates)
    test_snapshots = [s for s in snapshots if s.trade_date in test_dates]
    latest = snapshots[-1]
    report = {
        "scope": "research_only_no_trading_or_portfolio_recommendations",
        "dates": {"train_start": str(train.dates[0].date()), "train_end": str(train.dates[-1].date()),
                  "validation_start": str(val.dates[0].date()), "validation_end": str(val.dates[-1].date()),
                  "test_start": str(test.dates[0].date()), "test_end": str(test.dates[-1].date())},
        "symbols": data.symbols, "days": {"train": len(train.dates), "validation": len(val.dates), "test": len(test.dates)},
        "features": ALL_FEATURES, "base_features": BASE_FEATURES, "new_features": EXTRA_FEATURES,
        "coverage": {col: float(panel[col].mean()) for col in ("clv_available", "flow_available", "intraday_available")},
        "feature_groups": feature_groups,
        "lowdin": {"features": BASE_FEATURES[:6], "rank": lowdin.rank, "dimensions": 6,
                   "training_covariance_target": "identity" if lowdin.rank == 6 else "rank_projection"},
        "supervised_models": model_metrics,
        "distribution_monitor": {"universe": "supplied_symbols_only", "universe_symbols": int(macro_input["symbol"].nunique()),
             "separate_macro_input": macro_bars is not None, "reference_days": 120, "recent_days": 20,
             "threshold_train_quantile": 0.95, "threshold": monitor.threshold,
             "test_alert_count": sum(s.distribution_alert for s in test_snapshots),
             "latest_date": str(latest.trade_date.date()), "latest_distance": latest.distribution_distance,
             "latest_alert": latest.distribution_alert, "latest_state_probabilities": latest.probabilities.tolist()},
    }
    return report, model, tree, lowdin, monitor


def synthetic_bars(days: int = 360, seed: int = 42) -> pd.DataFrame:
    if days < 240:
        raise ValueError("Use at least 240 synthetic days")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2026-10-30", periods=days)
    symbols = ["510300", "510500", "512480", "600900", "002594", "300750"]
    rows = []
    market = rng.normal(0, 0.012, days)
    for symbol in symbols:
        returns = market * rng.uniform(0.5, 1.1) + rng.normal(0, 0.01, days)
        close = 10 * np.exp(np.cumsum(returns))
        opening = close * np.exp(rng.normal(0, 0.003, days))
        spread = rng.uniform(0.002, 0.015, days)
        for date, op, c, s in zip(dates, opening, close, spread):
            rows.append({"symbol": symbol, "trade_date": date, "open": op, "close": c,
                         "high": max(op, c) * (1 + s), "low": min(op, c) * (1 - s),
                         "volume": float(rng.integers(10_000, 100_000))})
    return pd.DataFrame(rows)
