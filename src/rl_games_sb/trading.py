"""Portfolio trading environment and reproducible market-analysis reports."""
from __future__ import annotations

import csv
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import gym_anytrading
import gymnasium as gym
import numpy as np
from gymnasium import spaces

TRADING_ENV_IDS = ("stocks-v0", "forex-v0")
gym.register_envs(gym_anytrading)
ACTION_EXPOSURES = (0.0, 0.5, 1.0)
OBSERVATION_FEATURES = (
    "return_1",
    "momentum_5",
    "momentum_20",
    "volatility_20",
    "price_vs_sma_20",
    "rsi_14",
    "current_exposure",
)


def market_data(env_id: str) -> tuple[Any, np.ndarray, int, int, int]:
    """Read the market frame bundled with a registered gym-anytrading env."""
    if env_id not in TRADING_ENV_IDS:
        raise ValueError(f"unsupported trading environment {env_id!r}")

    source = gym.make(env_id, disable_env_checker=True)
    try:
        base = source.unwrapped
        frame = base.df.copy()
        frame_start, frame_end = base.frame_bound
        window_size = int(base.window_size)
    finally:
        source.close()

    if "Close" not in frame:
        raise ValueError(f"{env_id} market data must contain a 'Close' column")
    prices = frame["Close"].to_numpy(dtype=np.float64)
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError(f"{env_id} closing prices must be finite and positive")
    return frame, prices, int(frame_start), min(int(frame_end), len(prices)), window_size


class PortfolioTradingEnv(gym.Env):
    """Long-only 0/50/100% allocation with causal observations and explicit fees.

    An observation is made at close[t]. The selected allocation is established
    at that close, fees are charged on turnover, and the allocation earns the
    close[t] -> close[t+1] price move.
    """

    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(
        self,
        env_id: str,
        *,
        segment: str = "train",
        train_fraction: float = 0.70,
        validation_fraction: float = 0.15,
        transaction_cost_bps: float = 10.0,
        initial_cash: float = 100_000.0,
        reward_mode: str = "log_return",
        drawdown_penalty: float = 0.5,
    ):
        super().__init__()
        if not 0 < train_fraction < 1:
            raise ValueError("train_fraction must be between 0 and 1")
        if not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
            raise ValueError("train_fraction + validation_fraction must be less than 1")
        if (
            not np.isfinite(transaction_cost_bps)
            or transaction_cost_bps < 0
            or transaction_cost_bps >= 10_000
        ):
            raise ValueError("transaction_cost_bps must be in [0, 10000)")
        if not np.isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if reward_mode not in ("simple_return", "log_return", "drawdown_adjusted"):
            raise ValueError(f"unknown reward_mode {reward_mode!r}")
        if not np.isfinite(drawdown_penalty) or drawdown_penalty < 0:
            raise ValueError("drawdown_penalty must be non-negative")

        self.env_id = env_id
        frame, self.prices, frame_start, frame_end, self.window_size = market_data(env_id)
        self.timestamps = frame.index
        self.frame = frame
        train_end = frame_start + int((frame_end - frame_start) * train_fraction)
        validation_end = train_end + int((frame_end - frame_start) * validation_fraction)
        bounds = {
            "train": (frame_start, train_end),
            "validation": (train_end, validation_end),
            "test": (validation_end, frame_end),
        }
        if segment not in bounds:
            raise ValueError(f"segment must be one of {tuple(bounds)}, got {segment!r}")
        self.segment = segment
        self.start_tick, self.end_tick = bounds[segment]
        if self.end_tick - self.start_tick < 2:
            raise ValueError(f"{env_id} {segment} segment needs at least two price points")

        self.initial_cash = float(initial_cash)
        self.transaction_cost_rate = float(transaction_cost_bps) / 10_000
        self.reward_mode = reward_mode
        self.drawdown_penalty = float(drawdown_penalty)
        self.action_space = spaces.Discrete(len(ACTION_EXPOSURES))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(OBSERVATION_FEATURES),), dtype=np.float32
        )
        self.current_tick = self.start_tick
        self.position = 0.0
        self.portfolio_value = self.initial_cash
        self.peak_value = self.initial_cash

    def _observation(self) -> np.ndarray:
        tick = self.current_tick
        prices = self.prices
        return_1 = prices[tick] / prices[tick - 1] - 1
        momentum_5 = prices[tick] / prices[max(0, tick - 5)] - 1
        momentum_20 = prices[tick] / prices[max(0, tick - 20)] - 1
        volatility_history = prices[max(0, tick - 20):tick + 1]
        returns = np.diff(np.log(volatility_history))
        volatility_20 = float(np.std(returns)) if len(returns) else 0.0
        history = prices[max(0, tick - 19):tick + 1]
        price_vs_sma_20 = prices[tick] / float(np.mean(history)) - 1

        rsi_returns = np.diff(np.log(prices[max(0, tick - 14):tick + 1]))
        gains = np.maximum(rsi_returns, 0)
        losses = np.maximum(-rsi_returns, 0)
        average_gain = float(np.mean(gains)) if len(gains) else 0.0
        average_loss = float(np.mean(losses)) if len(losses) else 0.0
        denominator = average_gain + average_loss
        rsi_14 = average_gain / denominator if denominator else 0.5

        return np.asarray(
            (
                return_1,
                momentum_5,
                momentum_20,
                volatility_20,
                price_vs_sma_20,
                rsi_14,
                self.position,
            ),
            dtype=np.float32,
        )

    def _info(
        self, *, transaction_cost: float = 0.0, market_return: float = 0.0
    ) -> dict[str, str | int | float]:
        return {
            "tick": self.current_tick,
            "timestamp": str(self.timestamps[self.current_tick]),
            "price": float(self.prices[self.current_tick]),
            "position": self.position,
            "portfolio_value": self.portfolio_value,
            "total_profit": self.portfolio_value / self.initial_cash,
            "transaction_cost": transaction_cost,
            "market_return": market_return,
        }

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict[str, str | int | float]]:
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        self.current_tick = self.start_tick
        self.position = 0.0
        self.portfolio_value = self.initial_cash
        self.peak_value = self.initial_cash
        return self._observation(), self._info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, str | int | float]]:
        if not self.action_space.contains(action):
            raise ValueError(f"action must be an integer in [0, 2], got {action!r}")
        if self.current_tick >= self.end_tick - 1:
            raise RuntimeError("step() called after the trading segment ended; call reset()")

        previous_value = self.portfolio_value
        previous_drawdown = previous_value / self.peak_value - 1
        new_position = ACTION_EXPOSURES[int(action)]
        turnover = abs(new_position - self.position)
        transaction_cost = previous_value * turnover * self.transaction_cost_rate
        market_return = self.prices[self.current_tick + 1] / self.prices[self.current_tick] - 1

        net_growth = (1 - turnover * self.transaction_cost_rate) * (
            1 + new_position * market_return
        )
        self.portfolio_value = previous_value * net_growth
        self.position = new_position
        self.current_tick += 1
        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = self.portfolio_value / self.peak_value - 1

        if self.reward_mode == "simple_return":
            reward = net_growth - 1
        else:
            reward = float(np.log(net_growth))
            if self.reward_mode == "drawdown_adjusted":
                reward -= self.drawdown_penalty * max(0.0, previous_drawdown - drawdown)

        truncated = self.current_tick >= self.end_tick - 1
        info = self._info(transaction_cost=transaction_cost, market_return=market_return)
        info["drawdown"] = drawdown
        info["turnover"] = turnover
        return self._observation(), float(reward), False, truncated, info


def analyze_market_data(env_id: str, output_dir: str | Path) -> Path:
    """Save a reproducible price, returns, volatility, and indicator EDA chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame, prices, _, _, _ = market_data(env_id)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    returns = prices[1:] / prices[:-1] - 1
    rolling_volatility = np.full(len(prices), np.nan)
    for index in range(20, len(prices)):
        rolling_volatility[index] = np.std(returns[index - 20:index])
    moving_average = np.full(len(prices), np.nan)
    for index in range(19, len(prices)):
        moving_average[index] = np.mean(prices[index - 19:index + 1])
    relative_equity = prices / prices[0]
    drawdown = relative_equity / np.maximum.accumulate(relative_equity) - 1
    finite_volatility = rolling_volatility[np.isfinite(rolling_volatility)]
    summary = {
        "environment": env_id,
        "observations": len(prices),
        "start": str(frame.index[0]),
        "end": str(frame.index[-1]),
        "buy_hold_return": relative_equity[-1] - 1,
        "maximum_price_drawdown": float(np.min(drawdown)),
        "mean_period_return": float(np.mean(returns)),
        "period_return_std": float(np.std(returns)),
        "minimum_period_return": float(np.min(returns)),
        "maximum_period_return": float(np.max(returns)),
        "rolling_20_period_volatility_min": (
            float(np.min(finite_volatility)) if len(finite_volatility) else math.nan
        ),
        "rolling_20_period_volatility_max": (
            float(np.max(finite_volatility)) if len(finite_volatility) else math.nan
        ),
        "volume_available": "Volume" in frame,
    }
    with (output_dir / "market_statistics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=tuple(summary))
        writer.writeheader()
        writer.writerow(summary)

    figure, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)
    axes[0].plot(frame.index, prices, label="Close")
    axes[0].plot(frame.index, moving_average, label="SMA 20")
    axes[0].set_title(f"{env_id}: historical price and 20-period moving average")
    axes[0].legend()
    axes[1].hist(returns[np.isfinite(returns)], bins=50)
    axes[1].set_title("One-period return distribution")
    axes[2].plot(frame.index, rolling_volatility)
    axes[2].set_title("Rolling 20-period return volatility")
    axes[3].plot(frame.index, drawdown)
    axes[3].set_title("Price drawdown from the previous peak")
    if "Volume" in frame:
        axes[4].plot(frame.index, frame["Volume"].to_numpy())
        axes[4].set_title("Volume")
    else:
        axes[4].plot(frame.index, prices / moving_average - 1)
        axes[4].set_title("Close relative to SMA 20")
    figure.tight_layout()
    path = output_dir / "market_eda.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _run_strategy(
    env: gym.Env, action_for_observation: Callable[[np.ndarray], int], seed: int
) -> dict[str, Any]:
    observation, _ = env.reset(seed=seed)
    initial_cash = env.unwrapped.initial_cash
    rows = [{
        "timestamp": str(env.unwrapped.timestamps[env.unwrapped.current_tick]),
        "portfolio_value": initial_cash,
        "position": 0.0,
        "transaction_cost": 0.0,
        "turnover": 0.0,
    }]
    done = False
    while not done:
        action = action_for_observation(observation)
        observation, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rows.append({
            "timestamp": info["timestamp"],
            "portfolio_value": info["portfolio_value"],
            "position": info["position"],
            "transaction_cost": info["transaction_cost"],
            "turnover": info["turnover"],
        })
    env.close()

    equity = np.asarray([row["portfolio_value"] for row in rows], dtype=np.float64)
    drawdowns = equity / np.maximum.accumulate(equity) - 1
    return {
        "cumulative_return": equity[-1] / equity[0] - 1,
        "maximum_drawdown": float(np.min(drawdowns)),
        "transaction_costs": float(sum(row["transaction_cost"] for row in rows)),
        "trades": sum(row["turnover"] > 0 for row in rows),
        "average_exposure": float(np.mean([row["position"] for row in rows[1:]])),
        "curve": rows,
    }


def backtest(
    model: Any,
    env_id: str,
    output_dir: str | Path,
    *,
    env_kwargs: dict | None = None,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Compare a deterministic DQN policy with buy-and-hold and random policies."""
    if env_id not in TRADING_ENV_IDS:
        raise ValueError(f"unsupported trading environment {env_id!r}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from rl_games_sb import envs

    options = dict(env_kwargs or {})
    options["segment"] = "test"
    strategies = {
        "DQN": lambda observation, env: int(
            np.asarray(model.predict(observation, deterministic=True)[0]).item()
        ),
        "Buy and hold": lambda observation, env: 2,
        "Random": lambda observation, env: int(env.action_space.sample()),
    }
    results = {}
    for name, policy in strategies.items():
        env = envs.make(env_id, **options)
        results[name] = _run_strategy(
            env, lambda observation, policy=policy, env=env: policy(observation, env), seed
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_fields = (
        "strategy", "cumulative_return", "maximum_drawdown", "transaction_costs",
        "trades", "average_exposure",
    )
    with (output_dir / "strategy_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        for name, result in results.items():
            writer.writerow({"strategy": name, **{key: result[key] for key in summary_fields[1:]}})

    with (output_dir / "equity_curves.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow((
            "strategy", "timestamp", "portfolio_value", "position", "transaction_cost",
            "turnover",
        ))
        for name, result in results.items():
            for row in result["curve"]:
                writer.writerow((name, row["timestamp"], row["portfolio_value"],
                                 row["position"], row["transaction_cost"], row["turnover"]))

    figure, (equity_axis, position_axis) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for name, result in results.items():
        curve = result["curve"]
        timestamps = [row["timestamp"] for row in curve]
        equity_axis.plot(timestamps, [row["portfolio_value"] for row in curve], label=name)
        position_axis.step(timestamps, [row["position"] for row in curve],
                           where="post", label=name)
    equity_axis.set_title(f"{env_id}: held-out equity curves")
    equity_axis.set_ylabel("Portfolio value")
    equity_axis.legend()
    position_axis.set_title("Invested fraction")
    position_axis.set_ylabel("Exposure")
    position_axis.set_ylim(-0.05, 1.05)
    position_axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "backtest.png", dpi=150)
    plt.close(figure)
    return results
