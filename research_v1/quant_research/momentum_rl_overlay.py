"""Offline IQL risk overlay for the frozen 126-session ETF dual-momentum policy.

The momentum selector and 35% per-slot rule stay fixed. IQL only chooses a
discrete multiplier on the candidate's gross exposure. All executions are
scored by the independent integer-share ledger. The historical evaluation is
already inspected and is not independent out-of-sample evidence.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .baseline import market_close_returns
from .ledger import CloseState, Costs, simulate
from .momentum_study import load_market, momentum_scores, momentum_target, review_days, run_policy, verify_prior
from .protocol import ETF_SYMBOLS, HOLDOUT_START, MAX_DRAWDOWN
from .risk import block_resample, summarize
from .snapshot import file_sha256

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT = ROOT / "research_v1/artifacts/market-etf-verified-20261108"
DEFAULT_MOMENTUM_REPORT = ROOT / "research_v1/artifacts/momentum-study-20261111-a/report.json"
TRAIN_END = date(2022, 12, 30)
INITIAL_CASH = 10_000.0
REBALANCE_INTERVAL = 20
LOOKBACK_VOL = 20
ACTION_MULTIPLIERS = np.asarray([0.0, 0.35, 0.60, 0.85, 1.0], dtype=np.float32)
SEEDS = (42, 101, 202, 303, 404)
IQL_EPOCHS = 120
IQL_GAMMA = 0.97
IQL_EXPECTILE = 0.70
IQL_AWR_TEMPERATURE = 3.0
IQL_LEARNING_RATE = 1e-3
TURNOVER_PENALTY = 0.45
DRAWDOWN_PENALTY = 2.5
TARGET_VOLATILITY = 0.10


Chooser = Callable[[np.ndarray, CloseState, pd.Series, pd.Series, Mapping[str, float]], int]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _nearest_action(multiplier: float) -> int:
    value = float(np.clip(multiplier, 0.0, 1.0))
    return int(np.abs(ACTION_MULTIPLIERS - value).argmin())


def _state_vector(
    state: CloseState,
    score_row: pd.Series,
    volatility_row: pd.Series,
) -> np.ndarray:
    momentum = score_row.reindex(ETF_SYMBOLS).to_numpy(dtype=np.float64)
    volatility = volatility_row.reindex(ETF_SYMBOLS).to_numpy(dtype=np.float64)
    weights = np.asarray([state.weights.get(symbol, 0.0) for symbol in ETF_SYMBOLS], dtype=np.float64)
    if not (np.isfinite(momentum).all() and np.isfinite(volatility).all() and np.isfinite(weights).all()):
        raise ValueError("non-finite momentum overlay state")
    cash_weight = float(state.cash / state.nav)
    return np.concatenate([momentum, volatility, weights, [cash_weight, float(state.drawdown)]]).astype(np.float32)


def _base_portfolio_volatility(
    target: Mapping[str, float],
    volatility_row: pd.Series,
) -> float:
    # Conservative diagonal estimate; the ledger remains the final risk measure.
    risk_terms = [
        float(weight) * float(volatility_row[symbol])
        for symbol, weight in target.items()
    ]
    return float(np.sqrt(np.square(risk_terms).sum())) if risk_terms else 0.0


def _behavior_choosers() -> dict[str, Chooser]:
    fixed = {
        "fixed_full": len(ACTION_MULTIPLIERS) - 1,
        "fixed_85": 3,
        "fixed_60": 2,
        "fixed_35": 1,
        "all_cash": 0,
    }
    policies: dict[str, Chooser] = {
        name: (lambda _obs, _state, _score, _vol, _target, idx=idx: idx)
        for name, idx in fixed.items()
    }

    def drawdown_guard(_obs, state, _score, _vol, _target) -> int:
        if state.drawdown < 0.05:
            return 4
        if state.drawdown < 0.10:
            return 3
        if state.drawdown < MAX_DRAWDOWN:
            return 2
        return 0

    def volatility_target(_obs, _state, _score, vol, target) -> int:
        base_vol = _base_portfolio_volatility(target, vol)
        return _nearest_action(TARGET_VOLATILITY / max(base_vol, 1e-6))

    def breadth_guard(_obs, _state, _score, _vol, target) -> int:
        if len(target) >= 2:
            return 4
        if len(target) == 1:
            return 2
        return 0

    policies["drawdown_guard"] = drawdown_guard
    policies["volatility_target_10pct"] = volatility_target
    policies["momentum_breadth"] = breadth_guard
    return policies


def _simulate_overlay(
    bars: pd.DataFrame,
    calendar: Sequence[pd.Timestamp],
    actions: pd.DataFrame,
    scores: pd.DataFrame,
    volatility: pd.DataFrame,
    reviews: pd.DatetimeIndex,
    *,
    chooser: Chooser,
    cost: Costs,
    initial_cash: float = INITIAL_CASH,
) -> tuple[object, list[dict[str, object]]]:
    review_dates = {pd.Timestamp(day).date() for day in reviews}
    decisions: list[dict[str, object]] = []

    def close_policy(state: CloseState) -> Mapping[str, float] | None:
        if state.date not in review_dates:
            return None
        timestamp = pd.Timestamp(state.date)
        score_row = scores.loc[timestamp].reindex(ETF_SYMBOLS)
        vol_row = volatility.loc[timestamp].reindex(ETF_SYMBOLS)
        base_target = momentum_target(score_row)
        observation = _state_vector(state, score_row, vol_row)
        action_index = int(chooser(observation, state, score_row, vol_row, base_target))
        if not 0 <= action_index < len(ACTION_MULTIPLIERS):
            raise ValueError("policy returned an exposure action outside the frozen action set")
        multiplier = float(ACTION_MULTIPLIERS[action_index])
        target = {symbol: weight * multiplier for symbol, weight in base_target.items()}
        turnover = sum(abs(target.get(symbol, 0.0) - state.weights.get(symbol, 0.0)) for symbol in ETF_SYMBOLS)
        decisions.append({
            "date": state.date.isoformat(),
            "action_index": action_index,
            "exposure_multiplier": multiplier,
            "baseline_gross_target": float(sum(base_target.values())),
            "target_gross": float(sum(target.values())),
            "drawdown": float(state.drawdown),
            "turnover_l1": float(turnover),
            "observation": observation.tolist(),
            **{f"momentum_{symbol}": float(score_row[symbol]) for symbol in ETF_SYMBOLS},
            **{f"volatility_{symbol}": float(vol_row[symbol]) for symbol in ETF_SYMBOLS},
            **{f"target_{symbol}": float(target.get(symbol, 0.0)) for symbol in ETF_SYMBOLS},
        })
        # Empty mapping is an explicit all-cash target when no positive score wins.
        return target

    result = simulate(
        bars, calendar, {}, actions, initial_cash=initial_cash, cost=cost, close_policy=close_policy,
    )
    if len(decisions) != len(reviews):
        raise RuntimeError("overlay did not observe every scheduled review close")
    return result, decisions


def _nav_maps(result: object) -> tuple[dict[date, float], dict[date, float]]:
    equity = result.equity
    nav = {pd.Timestamp(day).date(): float(value) for day, value in zip(equity["date"], equity["nav"], strict=True)}
    drawdown = {
        pd.Timestamp(day).date(): float(value)
        for day, value in zip(equity["date"], equity["drawdown"], strict=True)
    }
    return nav, drawdown


def _build_transitions(
    episodes: Sequence[tuple[str, object, list[dict[str, object]]]],
    baseline_result: object,
    end_day: date,
) -> list[dict[str, object]]:
    baseline_nav, _ = _nav_maps(baseline_result)
    transitions: list[dict[str, object]] = []
    for behavior_name, result, decisions in episodes:
        nav, drawdown = _nav_maps(result)
        for index, row in enumerate(decisions):
            start_day = date.fromisoformat(str(row["date"]))
            terminal = index + 1 == len(decisions)
            next_day = end_day if terminal else date.fromisoformat(str(decisions[index + 1]["date"]))
            if next_day <= start_day:
                continue
            start_nav = nav[start_day]
            end_nav = nav[next_day]
            base_start = baseline_nav[start_day]
            base_end = baseline_nav[next_day]
            policy_return = end_nav / start_nav - 1.0
            baseline_return = base_end / base_start - 1.0
            next_drawdown = drawdown[next_day]
            excess = policy_return - baseline_return
            dd_excess = max(0.0, next_drawdown - MAX_DRAWDOWN)
            reward = excess * 100.0 - TURNOVER_PENALTY * float(row["turnover_l1"])
            reward -= DRAWDOWN_PENALTY * (dd_excess ** 2) * 100.0
            next_observation = (
                decisions[index + 1]["observation"] if not terminal else np.zeros_like(row["observation"])
            )
            transitions.append({
                "behavior": behavior_name,
                "date": start_day.isoformat(),
                "action_index": int(row["action_index"]),
                "reward": float(reward),
                "done": float(terminal),
                "state": row["observation"],
                "next_state": next_observation,
                "policy_return": float(policy_return),
                "baseline_return": float(baseline_return),
                "next_drawdown": float(next_drawdown),
                "turnover_l1": float(row["turnover_l1"]),
            })
    if not transitions:
        raise ValueError("behavior policies produced no complete training transitions")
    return transitions


class DiscreteIQL(nn.Module):
    """Small discrete-action IQL model for the five pre-registered exposure choices."""

    def __init__(self, observation_dim: int, action_count: int, hidden_dim: int = 64) -> None:
        super().__init__()

        def mlp(output_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(observation_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            )

        self.q_net = mlp(action_count)
        self.v_net = mlp(1)
        self.policy_net = mlp(action_count)


def _expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weights = torch.where(diff >= 0, expectile, 1.0 - expectile)
    return (weights * diff.square()).mean()


def _train_iql(transitions: Sequence[Mapping[str, object]], seed: int) -> tuple[DiscreteIQL, np.ndarray, np.ndarray, dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    states = np.asarray([row["state"] for row in transitions], dtype=np.float32)
    next_states = np.asarray([row["next_state"] for row in transitions], dtype=np.float32)
    mean = states.mean(axis=0)
    scale = states.std(axis=0)
    scale[scale < 1e-6] = 1.0
    states = (states - mean) / scale
    next_states = (next_states - mean) / scale
    actions = np.asarray([row["action_index"] for row in transitions], dtype=np.int64)
    rewards = np.asarray([row["reward"] for row in transitions], dtype=np.float32)
    reward_scale = max(float(rewards.std()), 1e-6)
    rewards = rewards / reward_scale
    done = np.asarray([row["done"] for row in transitions], dtype=np.float32)

    s = torch.from_numpy(states)
    sp = torch.from_numpy(next_states)
    a = torch.from_numpy(actions).unsqueeze(-1)
    r = torch.from_numpy(rewards).unsqueeze(-1)
    d = torch.from_numpy(done).unsqueeze(-1)
    model = DiscreteIQL(states.shape[1], len(ACTION_MULTIPLIERS))
    q_optimizer = torch.optim.AdamW(model.q_net.parameters(), lr=IQL_LEARNING_RATE, weight_decay=1e-4)
    v_optimizer = torch.optim.AdamW(model.v_net.parameters(), lr=IQL_LEARNING_RATE, weight_decay=1e-4)
    pi_optimizer = torch.optim.AdamW(model.policy_net.parameters(), lr=IQL_LEARNING_RATE, weight_decay=1e-4)
    last_losses: dict[str, float] = {}

    for _ in range(IQL_EPOCHS):
        with torch.no_grad():
            q_for_value = model.q_net(s).gather(1, a)
        value = model.v_net(s)
        value_loss = _expectile_loss(q_for_value - value, IQL_EXPECTILE)
        v_optimizer.zero_grad()
        value_loss.backward()
        v_optimizer.step()

        with torch.no_grad():
            target = r + IQL_GAMMA * (1.0 - d) * model.v_net(sp)
        q_sa = model.q_net(s).gather(1, a)
        q_loss = nn.SmoothL1Loss()(q_sa, target)
        q_optimizer.zero_grad()
        q_loss.backward()
        q_optimizer.step()

        with torch.no_grad():
            advantage = model.q_net(s).gather(1, a) - model.v_net(s)
            weights = torch.exp(IQL_AWR_TEMPERATURE * advantage).clamp(max=20.0)
        log_prob = torch.log_softmax(model.policy_net(s), dim=-1).gather(1, a)
        policy_loss = -(weights * log_prob).mean()
        pi_optimizer.zero_grad()
        policy_loss.backward()
        pi_optimizer.step()
        last_losses = {
            "value": float(value_loss.detach()),
            "q": float(q_loss.detach()),
            "policy": float(policy_loss.detach()),
        }
    model.eval()
    last_losses["reward_scale"] = reward_scale
    return model, mean, scale, last_losses


def _model_chooser(model: DiscreteIQL, mean: np.ndarray, scale: np.ndarray) -> Chooser:
    def choose(observation, _state, _score, _vol, _target) -> int:
        normalized = (observation - mean) / scale
        with torch.no_grad():
            logits = model.policy_net(torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0))
        return int(torch.argmax(logits, dim=-1).item())

    return choose


def _daily_returns(result: object, initial_cash: float) -> pd.Series:
    equity = result.equity
    nav = equity["nav"].to_numpy(dtype=np.float64)
    returns = np.empty_like(nav)
    returns[0] = nav[0] / initial_cash - 1.0
    returns[1:] = nav[1:] / nav[:-1] - 1.0
    return pd.Series(returns, index=pd.to_datetime(equity["date"]), dtype=np.float64)


def _markdown(report: Mapping[str, object]) -> str:
    metrics = report["share_ledger_metrics"]
    config = report["protocol"]["config"]
    lines = [
        "# 双动量 IQL 风险叠加研究",
        "",
        f"证据等级：{report['evidence_class']}。2023—2024 区间此前已查看，不是新的独立样本外证据。",
        "",
        f"固定策略：过去 {config['selector_lookback_sessions']} 个交易日总收益，正动量前两只，每只 35%，每 {config['review_interval_sessions']} 日复核。RL 只能把该组合总仓位乘以离散风险倍率；不能改选标的。",
        f"训练区间：{report['protocol']['train_window']}；评估区间：{report['protocol']['evaluation_window']}。封存区间从 {report['protocol']['sealed_holdout_start']} 开始，未读取。",
        "",
        "|费用情景|策略|净收益|最大回撤|年化收益|费用|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for cost_case, rows in metrics.items():
        for name, row in rows.items():
            if name == "paired_block_resampling_vs_dual_momentum":
                continue
            if name == "iql_five_seed_summary":
                lines.append(
                    f"|{cost_case}|IQL 五种子均值|{row['net_return_mean']:.2%}|{row['max_drawdown_mean']:.2%}|{row['annualized_return_mean']:.2%}|—|"
                )
            else:
                lines.append(
                    f"|{cost_case}|{name}|{row['net_return']:.2%}|{row['max_drawdown']:.2%}|{row['annualized_return']:.2%}|{row['transaction_fees']:.2f}|"
                )
    lines.extend([
        "",
        f"全部种子及费用情景均超过原双动量候选、且最大回撤不超过 15%：**{report['acceptance']['all_seeds_outperform_and_pass_drawdown']}**。这只是预先定义的历史筛选条件，不是收益保证。",
        "",
        "区块重采样与原候选的配对终值区间：",
    ])
    for cost_case, rows in metrics.items():
        for block_result in rows["paired_block_resampling_vs_dual_momentum"]:
            lo, hi = block_result["paired_terminal_excess_interval"]
            lines.append(f"- {cost_case}，块长 {block_result['block']}：[{lo:.2%}, {hi:.2%}]。")
    lines.extend([
        "",
        "行为数据由固定仓位、回撤降仓、波动目标和动量广度规则在训练区间的完整账本轨迹构成，不是用户实盘记录。每日特征在收盘生成，订单进入次日开盘份额账本；训练奖励按 20 日区间相对双动量基线的净收益、换手及超过 15% 回撤惩罚计算。",
        "历史价格快照缺完整的点时获知与修订记录；封存区间没有参与特征、训练或筛选。IQL 增量需与固定仓位和简单风控行为规则对照阅读。",
        "",
        "区块重采样仅是已看历史序列的情景分析，不是未来盈利概率。策略仍需独立前向验证，不能自动替换原候选。",
        "",
    ])
    return "\n".join(lines)


def run_study(snapshot: Path, momentum_report_path: Path, out: Path) -> dict[str, object]:
    out.mkdir(parents=True, exist_ok=False)
    try:
        momentum_report = json.loads(momentum_report_path.read_text(encoding="utf-8"))
        if momentum_report.get("status") != "historical_exploration":
            raise ValueError("expected the frozen dual-momentum historical report")
        study_window = momentum_report["protocol"]["windows"]["primary_2023_2024"]
        eval_start, eval_end = pd.Timestamp(study_window["start"]), pd.Timestamp(study_window["end"])
        eval_reviews = pd.DatetimeIndex(pd.to_datetime(study_window["monthly_review_dates"]))

        audit, frames = load_market(snapshot, end=eval_end.date())
        calendar = pd.DatetimeIndex(pd.to_datetime(frames["calendar"]["date"])).sort_values().unique()
        scores = momentum_scores(frames["bars"], frames["actions"], calendar).reindex(columns=ETF_SYMBOLS)
        close_returns = market_close_returns(frames["bars"], frames["actions"], calendar).reindex(columns=ETF_SYMBOLS)
        volatility = close_returns.rolling(LOOKBACK_VOL, min_periods=LOOKBACK_VOL).std() * math.sqrt(252.0)
        complete = scores.index[np.isfinite(scores.to_numpy(dtype=float)).all(axis=1)]
        if complete.empty:
            raise ValueError("snapshot has insufficient common history for the frozen momentum lookback")
        common_start = complete[0]
        train_days = calendar[(calendar >= common_start) & (calendar.date <= TRAIN_END)]
        eval_days = calendar[(calendar >= eval_start) & (calendar <= eval_end)]
        train_reviews = review_days(train_days, REBALANCE_INTERVAL)
        expected_sessions = int(study_window["sessions"])
        if len(eval_days) != expected_sessions or eval_days.empty:
            raise ValueError("evaluation calendar differs from the frozen momentum study")
        if not eval_reviews.is_unique or not eval_reviews.isin(eval_days[:-1]).all():
            raise ValueError("frozen evaluation reviews are invalid or lack a next exchange open")
        if not train_days.size or not train_reviews.size or eval_reviews.empty:
            raise ValueError("training or evaluation period is empty")

        config = {
            "action_multipliers_of_frozen_35pct_slots": ACTION_MULTIPLIERS.tolist(),
            "selector_lookback_sessions": 126,
            "selector_top_k": 2,
            "positive_momentum_only": True,
            "slot_weight": 0.35,
            "review_interval_sessions": REBALANCE_INTERVAL,
            "train_end": TRAIN_END.isoformat(),
            "initial_cash": INITIAL_CASH,
            "behavior_policies": [
                "fixed_full", "fixed_85", "fixed_60", "fixed_35", "all_cash",
                "drawdown_guard", "volatility_target_10pct", "momentum_breadth",
            ],
            "reward": {
                "baseline": "same-period net return of the frozen dual-momentum candidate",
                "excess_return_scale": 100.0,
                "turnover_penalty": TURNOVER_PENALTY,
                "drawdown_threshold": MAX_DRAWDOWN,
                "drawdown_penalty": DRAWDOWN_PENALTY,
            },
            "iql": {
                "seeds": list(SEEDS), "epochs": IQL_EPOCHS, "gamma": IQL_GAMMA,
                "expectile": IQL_EXPECTILE, "awr_temperature": IQL_AWR_TEMPERATURE,
                "learning_rate": IQL_LEARNING_RATE, "hidden_dim": 64,
            },
            "stress_costs": "double commission, minimum commission and slippage; stamp/transfer schedules unchanged",
            "thresholds_searched": False,
        }
        source_paths = [
            Path(__file__), Path(__file__).with_name("momentum_study.py"),
            Path(__file__).with_name("baseline.py"), Path(__file__).with_name("ledger.py"),
            Path(__file__).with_name("protocol.py"), Path(__file__).with_name("risk.py"),
            Path(__file__).with_name("snapshot.py"),
        ]
        protocol = {
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            "source_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in source_paths},
            "snapshot_manifest_sha256": file_sha256(snapshot / "manifest.json"),
            "momentum_report_sha256": file_sha256(momentum_report_path),
            "audit": audit,
            "train_window": f"{train_days[0].date()} — {train_days[-1].date()} ({len(train_days)} sessions)",
            "training_review_sessions": len(train_reviews),
            "evaluation_window": f"{eval_days[0].date()} — {eval_days[-1].date()} ({len(eval_days)} sessions)",
            "evaluation_review_sessions": len(eval_reviews),
            "sealed_holdout_start": HOLDOUT_START.isoformat(),
            "execution": "signal at close; target executes at next exchange open in the integer-share ledger",
            "features": "126-session causal total-return momentum, trailing 20-session volatility, current integer-ledger weights/cash/drawdown",
            "behavior_data": "simulated historical trajectories of predeclared exposure rules; no user-account trades",
            "evidence_class": "previously inspected historical exploration; not new independent OOS",
            "data_limitation": "historical provider snapshot lacks complete known_at and revision evidence",
            "python": __import__("sys").version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": str(torch.__version__),
        }
        _write_json(out / "protocol.json", protocol)

        behavior_policies = _behavior_choosers()
        baseline_episodes: dict[str, tuple[object, list[dict[str, object]]]] = {}
        normal_cost = Costs()
        baseline_result, baseline_decisions = _simulate_overlay(
            frames["bars"], train_days, frames["actions"], scores, volatility, train_reviews,
            chooser=behavior_policies["fixed_full"], cost=normal_cost,
        )
        episodes: list[tuple[str, object, list[dict[str, object]]]] = []
        for name, chooser in behavior_policies.items():
            if name == "fixed_full":
                result, decisions = baseline_result, baseline_decisions
            else:
                result, decisions = _simulate_overlay(
                    frames["bars"], train_days, frames["actions"], scores, volatility, train_reviews,
                    chooser=chooser, cost=normal_cost,
                )
            episodes.append((name, result, decisions))
            baseline_episodes[name] = (result, decisions)
        transitions = _build_transitions(episodes, baseline_result, train_days[-1].date())
        pd.DataFrame([
            {key: value for key, value in row.items() if key not in {"state", "next_state"}}
            for row in transitions
        ]).to_csv(out / "behavior_transitions.csv", index=False)

        # Reproduce the frozen baseline before comparing learned policies.
        eval_rows: dict[str, dict[str, object]] = {"normal": {}, "double_friction": {}}
        eval_results: dict[str, dict[str, object]] = {"normal": {}, "double_friction": {}}
        decision_frames: list[pd.DataFrame] = []
        ledger_frames: list[pd.DataFrame] = []
        order_frames: list[pd.DataFrame] = []
        reference_rows = [
            row for row in momentum_report["metrics"]
            if row["window"] == "primary_2023_2024" and row["initial_cash"] == int(INITIAL_CASH)
        ]

        for cost_case, cost in (("normal", normal_cost), ("double_friction", normal_cost.stressed())):
            for strategy_name, review_key in (
                ("equal70_monthly", "monthly_review_dates"),
                ("band70_weekly", "weekly_review_dates"),
                ("dual_momentum126_monthly", "monthly_review_dates"),
            ):
                report_reviews = pd.DatetimeIndex(pd.to_datetime(study_window[review_key]))
                stats, result, decisions = run_policy(
                    frames, eval_days, report_reviews, int(INITIAL_CASH), cost,
                    strategy_name, scores,
                )
                key = strategy_name
                eval_rows[cost_case][key] = stats
                eval_results[cost_case][key] = result
                if strategy_name == "dual_momentum126_monthly":
                    expected = next(
                        row for row in reference_rows
                        if row["cost_case"] == cost_case
                        and row["strategy"] == "dual_momentum126_monthly"
                    )
                    verify_prior(stats, {key: value for key, value in expected.items() if key in stats})
                decision_copy = decisions.copy()
                decision_copy["scenario"] = strategy_name
                decision_copy["cost_case"] = cost_case
                decision_frames.append(decision_copy)
                ledger_copy = result.equity.copy()
                ledger_copy["scenario"] = strategy_name
                ledger_copy["cost_case"] = cost_case
                ledger_frames.append(ledger_copy)
                orders_copy = result.orders.copy()
                orders_copy["scenario"] = strategy_name
                orders_copy["cost_case"] = cost_case
                order_frames.append(orders_copy)

            for exposure_name in ("fixed_85", "fixed_60", "fixed_35", "drawdown_guard", "volatility_target_10pct", "momentum_breadth"):
                result, decisions = _simulate_overlay(
                    frames["bars"], eval_days, frames["actions"], scores, volatility, eval_reviews,
                    chooser=behavior_policies[exposure_name], cost=cost,
                )
                stats = summarize(result.equity, result.orders, INITIAL_CASH)
                eval_rows[cost_case][f"overlay_{exposure_name}"] = stats
                eval_results[cost_case][f"overlay_{exposure_name}"] = result
                decision_copy = pd.DataFrame(decisions)
                decision_copy["scenario"] = f"overlay_{exposure_name}"
                decision_copy["cost_case"] = cost_case
                decision_frames.append(decision_copy)
                ledger_copy = result.equity.copy()
                ledger_copy["scenario"] = f"overlay_{exposure_name}"
                ledger_copy["cost_case"] = cost_case
                ledger_frames.append(ledger_copy)
                orders_copy = result.orders.copy()
                orders_copy["scenario"] = f"overlay_{exposure_name}"
                orders_copy["cost_case"] = cost_case
                order_frames.append(orders_copy)

            # The 100% action is the original fixed strategy and serves as an explicit control.
            full_result, full_decisions = _simulate_overlay(
                frames["bars"], eval_days, frames["actions"], scores, volatility, eval_reviews,
                chooser=behavior_policies["fixed_full"], cost=cost,
            )
            full_stats = summarize(full_result.equity, full_result.orders, INITIAL_CASH)
            reference = eval_rows[cost_case]["dual_momentum126_monthly"]
            verify_prior(full_stats, {key: value for key, value in reference.items() if key in full_stats})

            seed_metrics = []
            seed_returns = []
            for seed in SEEDS:
                model, mean, scale, losses = _train_iql(transitions, seed)
                chooser = _model_chooser(model, mean, scale)
                result, decisions = _simulate_overlay(
                    frames["bars"], eval_days, frames["actions"], scores, volatility, eval_reviews,
                    chooser=chooser, cost=cost,
                )
                stats = summarize(result.equity, result.orders, INITIAL_CASH)
                stats["seed"] = seed
                stats["last_training_losses"] = losses
                seed_metrics.append(stats)
                seed_returns.append(_daily_returns(result, INITIAL_CASH))
                scenario = f"iql_seed_{seed}"
                eval_rows[cost_case][scenario] = stats
                eval_results[cost_case][scenario] = result
                decision_copy = pd.DataFrame(decisions)
                decision_copy["scenario"] = scenario
                decision_copy["cost_case"] = cost_case
                decision_frames.append(decision_copy)
                ledger_copy = result.equity.copy()
                ledger_copy["scenario"] = scenario
                ledger_copy["cost_case"] = cost_case
                ledger_frames.append(ledger_copy)
                orders_copy = result.orders.copy()
                orders_copy["scenario"] = scenario
                orders_copy["cost_case"] = cost_case
                order_frames.append(orders_copy)
            eval_rows[cost_case]["iql_five_seed_summary"] = {
                "net_return_mean": float(np.mean([row["net_return"] for row in seed_metrics])),
                "net_return_std": float(np.std([row["net_return"] for row in seed_metrics])),
                "annualized_return_mean": float(np.mean([row["annualized_return"] for row in seed_metrics])),
                "annualized_return_std": float(np.std([row["annualized_return"] for row in seed_metrics])),
                "max_drawdown_mean": float(np.mean([row["max_drawdown"] for row in seed_metrics])),
                "max_drawdown_std": float(np.std([row["max_drawdown"] for row in seed_metrics])),
                "all_seed_metrics": seed_metrics,
            }
            baseline_returns = _daily_returns(eval_results[cost_case]["dual_momentum126_monthly"], INITIAL_CASH)
            mean_returns = pd.concat(seed_returns, axis=1).mean(axis=1)
            eval_rows[cost_case]["paired_block_resampling_vs_dual_momentum"] = [
                block_resample(mean_returns, baseline_returns, block=block, paths=500)
                for block in (20, 40, 60)
            ]

        pd.concat(decision_frames, ignore_index=True, sort=False).to_csv(out / "decisions.csv", index=False)
        pd.concat(ledger_frames, ignore_index=True, sort=False).to_csv(out / "daily_ledgers.csv", index=False)
        pd.concat(order_frames, ignore_index=True, sort=False).to_csv(out / "orders.csv", index=False)

        all_seed_pass = all(
            row["net_return"] > eval_rows[cost_case]["dual_momentum126_monthly"]["net_return"]
            and row["max_drawdown"] <= MAX_DRAWDOWN
            for cost_case in ("normal", "double_friction")
            for row in eval_rows[cost_case]["iql_five_seed_summary"]["all_seed_metrics"]
        )
        report: dict[str, object] = {
            "status": "historical_exploration",
            "evidence_class": protocol["evidence_class"],
            "protocol": protocol,
            "share_ledger_metrics": eval_rows,
            "acceptance": {
                "criteria": "every IQL seed beats dual momentum net return in both fee cases and keeps max drawdown at or below 15% in both fee cases",
                "all_seeds_outperform_and_pass_drawdown": bool(all_seed_pass),
                "strategy_replacement_authorized": False,
            },
            "new_independent_oos_improvement_proven": False,
        }
        _write_json(out / "report.json", report)
        (out / "report.md").write_text(_markdown(report), encoding="utf-8")
        return report
    except Exception as exc:
        _write_json(out / "failed.json", {
            "status": "failed", "exception_type": type(exc).__name__, "message": str(exc),
        })
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a paper-only IQL risk overlay on frozen ETF momentum")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--momentum-report", type=Path, default=DEFAULT_MOMENTUM_REPORT)
    parser.add_argument("--out", type=Path, required=True, help="new experiment output directory")
    args = parser.parse_args(argv)
    report = run_study(args.snapshot, args.momentum_report, args.out)
    summary = {
        "status": report["status"],
        "evaluation_window": report["protocol"]["evaluation_window"],
        "all_seeds_outperform_and_pass_drawdown": report["acceptance"]["all_seeds_outperform_and_pass_drawdown"],
        "report_path": str((args.out / "report.md").resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
