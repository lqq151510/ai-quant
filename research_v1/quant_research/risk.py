"""Net equity and paired block-resampling statistics for research reports."""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize(equity: pd.DataFrame, orders: pd.DataFrame, initial_cash: float) -> dict:
    nav = equity["nav"].to_numpy(dtype=float)
    if initial_cash <= 0 or len(nav) == 0 or not np.isfinite(nav).all() or np.any(nav <= 0):
        raise ValueError("a positive, complete equity series is required")
    wealth = np.r_[float(initial_cash), nav]
    r = wealth[1:] / wealth[:-1] - 1
    net = float(nav[-1] / initial_cash - 1)
    sd = float(r.std(ddof=1)) if len(r) > 1 else 0
    totals = {name: float(orders[name].sum()) if name in orders else 0.0
              for name in ("commission", "stamp_tax", "transfer_fee", "notional")}
    return {
        "sessions": len(r), "final_nav": float(nav[-1]), "net_return": net,
        "annualized_return": float(np.expm1(np.log1p(net) * 252 / len(r))),
        "annualized_volatility": float(sd * np.sqrt(252)),
        "sharpe_zero_cash_rate": float(r.mean() / sd * np.sqrt(252)) if sd > 0 else None,
        "max_drawdown": float((1 - wealth / np.maximum.accumulate(wealth)).max()),
        "filled_orders": int(orders["filled_qty"].gt(0).sum()) if "filled_qty" in orders else 0,
        "not_fully_filled_orders": int(orders["status"].isin(("partial", "rejected")).sum()) if "status" in orders else 0,
        "transaction_fees": totals["commission"] + totals["stamp_tax"] + totals["transfer_fee"],
        "traded_notional_over_average_nav": float(totals["notional"] / nav.mean()),
    }


def block_resample(
    strategy: pd.Series, benchmark: pd.Series, *, block: int, paths: int = 2000,
    days: int = 250, seed: int = 0, confidence: float = 0.95,
) -> dict:
    """Use the same contiguous sampled blocks for both aligned strategies."""
    if not strategy.index.equals(benchmark.index) or not strategy.index.is_unique or not strategy.index.is_monotonic_increasing:
        raise ValueError("strategy and benchmark must share a unique ordered index")
    if min(block, paths, days) < 1 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap parameters")
    a, b = strategy.to_numpy(float), benchmark.to_numpy(float)
    if min(len(a), days) < block:
        raise ValueError("insufficient sessions for requested block size")
    if not (np.isfinite(a).all() and np.isfinite(b).all()) or (a <= -1).any() or (b <= -1).any():
        raise ValueError("invalid returns in paired samples")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(a) - block + 1, size=(paths, (days + block - 1) // block))
    sample = (starts[..., None] + np.arange(block)).reshape(paths, -1)[:, :days]
    wealth = np.c_[np.ones(paths), np.cumprod(1 + a[sample], axis=1)]
    final = wealth[:, -1]
    other = np.cumprod(1 + b[sample], axis=1)[:, -1]
    drawdowns = (1 - wealth / np.maximum.accumulate(wealth, axis=1)).max(axis=1)
    losses = 1 - final
    tail = (1 - confidence) / 2
    cutoff = np.quantile(losses, confidence)
    return {
        "block": block, "paths": paths, "days": days, "seed": seed,
        "terminal_wealth_quantiles_05_50_95": np.quantile(final, [0.05, 0.5, 0.95]).tolist(),
        "loss_path_fraction": float(np.mean(final < 1)),
        "drawdown_above_15pct_fraction": float(np.mean(drawdowns > 0.15)),
        "terminal_loss_var": float(cutoff),
        "terminal_loss_expected_shortfall": float(np.sort(losses)[-max(1, int(np.ceil(round((1-confidence) * paths, 10)))):].mean()),
        "paired_terminal_excess_interval": np.quantile(final - other, [tail, 1 - tail]).tolist(),
        "interpretation": "historical resampling scenario, not a forecast probability",
    }
