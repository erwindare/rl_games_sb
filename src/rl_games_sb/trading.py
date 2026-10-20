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


def _observation_from_prices(
    prices: np.ndarray, tick: int, position: float
) -> np.ndarray:
    """Build the causal market state at a close, including the current exposure."""
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
            position,
        ),
        dtype=np.float32,
    )


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
        return _observation_from_prices(self.prices, self.current_tick, self.position)

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
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

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

    figure, axes = plt.subplots(5, 1, figsize=(12, 16))
    plot_dates = frame.index
    if len(frame.index) and isinstance(frame.index[0], str):
        plot_dates = np.arange(len(frame.index))
        date_ticks = np.linspace(0, len(frame.index) - 1, 6, dtype=int)
        axes[-1].set_xticks(date_ticks)
        axes[-1].set_xticklabels(
            [frame.index[tick][:16] for tick in date_ticks],
            rotation=20,
            ha="right",
        )
    else:
        date_locator = AutoDateLocator()
        axes[-1].xaxis.set_major_locator(date_locator)
        axes[-1].xaxis.set_major_formatter(ConciseDateFormatter(date_locator))
    axes[0].plot(plot_dates, prices, label="Close")
    axes[0].plot(plot_dates, moving_average, label="SMA 20")
    axes[0].set_title(f"{env_id}: historical price and 20-period moving average")
    axes[0].legend()
    axes[1].hist(returns[np.isfinite(returns)], bins=50)
    axes[1].set_title("One-period return distribution")
    axes[2].plot(plot_dates, rolling_volatility)
    axes[2].set_title("Rolling 20-period return volatility")
    axes[3].plot(plot_dates, drawdown)
    axes[3].set_title("Price drawdown from the previous peak")
    if "Volume" in frame:
        axes[4].plot(plot_dates, frame["Volume"].to_numpy())
        axes[4].set_title("Volume")
    else:
        axes[4].plot(plot_dates, prices / moving_average - 1)
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
        "market_price": float(env.unwrapped.prices[env.unwrapped.current_tick]),
        "market_return": 0.0,
        "reward": 0.0,
    }]
    done = False
    while not done:
        action = action_for_observation(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rows.append({
            "timestamp": info["timestamp"],
            "portfolio_value": info["portfolio_value"],
            "position": info["position"],
            "transaction_cost": info["transaction_cost"],
            "turnover": info["turnover"],
            "market_price": info["price"],
            "market_return": info["market_return"],
            "reward": float(reward),
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


def plot_trading_explanations(
    env_id: str, dqn_curve: list[dict[str, Any]], output_dir: str | Path
) -> tuple[Path, Path]:
    """Save a worked DQN transition and its causal observation windows."""
    if len(dqn_curve) < 2:
        raise ValueError("a DQN curve needs at least one transition")

    frame, prices, _, _, _ = market_data(env_id)
    tick_by_timestamp = {str(timestamp): tick for tick, timestamp in enumerate(frame.index)}
    transition_index = next(
        (
            index for index, row in enumerate(dqn_curve)
            if index > 0 and float(row["turnover"]) > 0
        ),
        1,
    )
    previous = dqn_curve[transition_index - 1]
    transition = dqn_curve[transition_index]
    try:
        decision_tick = tick_by_timestamp[previous["timestamp"]]
    except KeyError as error:
        raise ValueError(
            f"timestamp {previous['timestamp']!r} is not present in {env_id} market data"
        ) from error

    observation = _observation_from_prices(
        prices, decision_tick, float(previous["position"])
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_index = max(0, transition_index - 5)
    end_index = min(len(dqn_curve), transition_index + 6)
    curve_slice = dqn_curve[start_index:end_index]
    timestamps = [row["timestamp"] for row in curve_slice]
    x_values = np.arange(len(curve_slice))
    decision_x = transition_index - 1 - start_index
    result_x = transition_index - start_index
    date_by_timestamp = {
        timestamp: frame.index[tick_by_timestamp[timestamp]]
        for timestamp in timestamps
        if timestamp in tick_by_timestamp
    }
    prices_in_slice = [
        float(prices[tick_by_timestamp[timestamp]]) for timestamp in timestamps
    ]
    market_return = float(transition["market_return"])
    fee = float(transition["transaction_cost"])
    reward = float(transition["reward"])
    old_position = float(previous["position"])
    new_position = float(transition["position"])

    figure, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(x_values, prices_in_slice, marker="o", color="#2864a5")
    axes[0].set_ylabel("Close (USD)")
    axes[0].set_title("GOOGL: evolución del precio y decisiones DQN en el test")
    axes[0].annotate(
        f"Close[t] = {prices[decision_tick]:.2f}\n"
        f"Close[t+1] = {prices[decision_tick + 1]:.2f}\n"
        f"retorno = {market_return:+.2%}",
        xy=(result_x, prices[decision_tick + 1]),
        xytext=(0.02, 0.95),
        textcoords="axes fraction",
        va="top",
        arrowprops={"arrowstyle": "->", "color": "#444444"},
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        fontsize=8,
    )

    axes[1].plot(
        x_values,
        [float(row["portfolio_value"]) for row in curve_slice],
        marker="o",
        color="#2e8b57",
    )
    axes[1].set_ylabel("Patrimonio ($)")
    axes[1].annotate(
        f"Comisión = ${fee:.2f}\n"
        f"Recompensa = {reward:+.6f}",
        xy=(result_x, float(transition["portfolio_value"])),
        xytext=(0.02, 0.95),
        textcoords="axes fraction",
        va="top",
        arrowprops={"arrowstyle": "->", "color": "#444444"},
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        fontsize=8,
    )

    axes[2].step(
        x_values,
        [float(row["position"]) for row in curve_slice],
        where="post",
        color="#b14a39",
    )
    axes[2].scatter(
        [decision_x, result_x], [old_position, new_position], color="#b14a39", zorder=3
    )
    axes[2].set_ylabel("Exposición")
    axes[2].set_ylim(-0.08, 1.08)
    axes[2].set_yticks(ACTION_EXPOSURES, labels=("0 %", "50 %", "100 %"))
    axes[2].set_xticks(
        x_values,
        [
            date_by_timestamp[timestamp].strftime("%Y-%m-%d")
            if timestamp in date_by_timestamp
            else timestamp[:10]
            for timestamp in timestamps
        ],
        rotation=30,
        ha="right",
    )
    for axis in axes:
        axis.axvspan(decision_x, result_x, color="#f2c14e", alpha=0.2)
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"DQN test | {previous['timestamp'][:10]}: "
        f"exposición {old_position:.0%} → {new_position:.0%}\n"
        f"Patrimonio antes / después: ${previous['portfolio_value']:,.2f} / "
        f"${transition['portfolio_value']:,.2f}",
        y=1.04,
        fontsize=10,
    )
    figure.tight_layout()
    transition_path = output_dir / "dqn_transition_example.png"
    figure.savefig(transition_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    history_start = max(0, decision_tick - 20)
    history_ticks = np.arange(history_start, decision_tick + 1)
    relative_ticks = history_ticks - decision_tick
    feature_windows = (
        ("return_1", -1, observation[0]),
        ("momentum_5", -5, observation[1]),
        ("momentum_20", -20, observation[2]),
        ("volatility_20", -20, observation[3]),
        ("price_vs_sma_20", -19, observation[4]),
        ("rsi_14", -14, observation[5]),
    )
    figure, (price_axis, windows_axis) = plt.subplots(
        2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": (3, 2)}
    )
    price_axis.plot(
        relative_ticks,
        prices[history_ticks],
        marker="o",
        markersize=3,
        color="#2864a5",
    )
    price_axis.scatter([0], [prices[decision_tick]], color="#c0392b", zorder=3)
    price_axis.axvline(0, color="#c0392b", linestyle="--", alpha=0.7)
    price_axis.set_title(
        f"Observación en Close[t], {frame.index[decision_tick].date()}: "
        "solo información disponible hasta t"
    )
    price_axis.set_ylabel("Close de GOOGL (USD)")
    price_axis.set_xlabel("Períodos antes de t")
    price_axis.grid(alpha=0.2)

    y_positions = np.arange(len(OBSERVATION_FEATURES))
    windows_axis.set_yticks(y_positions, labels=OBSERVATION_FEATURES)
    windows_axis.set_ylim(len(OBSERVATION_FEATURES) - 0.5, -0.5)
    for y_position, (name, start, value) in enumerate(feature_windows):
        windows_axis.barh(
            y_position, -start, left=start, height=0.62, color="#75aadb"
        )
        windows_axis.text(
            0.5,
            y_position,
            f"{value:+.4f}",
            va="center",
            fontsize=8,
        )
    windows_axis.text(
        0.5,
        len(OBSERVATION_FEATURES) - 1,
        f"{float(observation[-1]):.1f}",
        va="center",
        fontsize=8,
    )
    windows_axis.set_xlim(-20.5, 4)
    windows_axis.set_xticks((-20, -15, -10, -5, 0))
    windows_axis.set_xlabel(
        "Inicio del historial (períodos antes de t); todas las ventanas terminan en t"
    )
    windows_axis.grid(axis="x", alpha=0.2)
    windows_axis.set_title(
        "Horizonte causal de los indicadores y valor observado en Sₜ"
    )
    figure.text(
        0.5,
        0.01,
        "Momentum compara los cierres extremos; las demás ventanas usan los "
        "retornos/cierres indicados. current_exposure no usa precios.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    observation_path = output_dir / "dqn_observation_windows.png"
    figure.savefig(observation_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    return transition_path, observation_path


def plot_dataset_splits(output_dir: str | Path) -> Path:
    """Show the chronological train/validation/test proportions for both markets."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(11, 4.5))
    colors = ("#4c78a8", "#f2cf5b", "#e45756")
    for axis, env_id in zip(axes, TRADING_ENV_IDS):
        _, _, start, end, _ = market_data(env_id)
        total = end - start
        train_end = start + int(total * 0.70)
        validation_end = train_end + int(total * 0.15)
        boundaries = (start, train_end, validation_end, end)
        names = ("Entrenamiento\n70 %", "Validación\n15 %", "Test\n15 %")
        for index, (name, color) in enumerate(zip(names, colors)):
            left = (boundaries[index] - start) / total * 100
            width = (boundaries[index + 1] - boundaries[index]) / total * 100
            axis.barh(0, width, left=left, height=0.55, color=color, edgecolor="white")
            axis.text(
                left + width / 2,
                0,
                name,
                ha="center",
                va="center",
                fontsize=8,
                color="#202020",
            )
        axis.set_title(f"{env_id}: {total:,} cierres evaluables", loc="left", fontsize=10)
        axis.set_xlim(0, 100)
        axis.set_yticks([])
        axis.set_xticks((0, 25, 50, 70, 85, 100))
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
    axes[-1].set_xlabel("Orden cronológico del segmento evaluable (%)")
    figure.suptitle("Separación temporal sin barajar observaciones", fontsize=12)
    figure.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "dataset_splits.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_dqn_algorithm_flow(output_dir: str | Path) -> Path:
    """Draw the SB3 DQN interaction and replay-buffer update cycle."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    boxes = (
        (0.02, 0.65, 0.19, 0.22, "Observación $S_t$\n7 características\ncausales"),
        (0.27, 0.65, 0.19, 0.22, "Red online $Q$\nMLP [128, 128]\n3 valores Q"),
        (0.52, 0.65, 0.19, 0.22, "Política $\\epsilon$-greedy\nelige 0 %, 50 % o\n100 %"),
        (0.77, 0.65, 0.21, 0.22, "Entorno\n$r_t, S_{t+1}, done$\ncon costos"),
        (0.70, 0.28, 0.25, 0.21, "Replay buffer\n50.000 transiciones"),
        (0.38, 0.28, 0.25, 0.21, "Minibatch (64)\n$Q_{target}$ forma\ntarget TD"),
        (0.06, 0.28, 0.25, 0.21, "Pérdida TD\nactualiza $Q_{online}$"),
    )
    for x, y, width, height, label in boxes:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.012",
                facecolor="#eaf2f8",
                edgecolor="#2864a5",
                linewidth=1.2,
            )
        )
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=9)

    arrows = (
        ((0.21, 0.76), (0.27, 0.76), ""),
        ((0.46, 0.76), (0.52, 0.76), ""),
        ((0.71, 0.76), (0.77, 0.76), ""),
        ((0.875, 0.65), (0.825, 0.49), "transición"),
        ((0.70, 0.385), (0.63, 0.385), "muestreo"),
        ((0.38, 0.385), (0.31, 0.385), "gradiente"),
        ((0.18, 0.49), (0.365, 0.65), "cada 1.000 pasos: copiar a $Q_{target}$"),
    )
    for start, end, label in arrows:
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "->",
                "color": "#444444",
                "connectionstyle": "arc3,rad=0.15" if "copiar" in label else "arc3,rad=0",
            },
        )
        if label:
            axis.text(
                (start[0] + end[0]) / 2,
                (start[1] + end[1]) / 2 + 0.035,
                label,
                ha="center",
                fontsize=7,
                color="#444444",
            )
    axis.text(
        0.5,
        0.08,
        "Durante entrenamiento se explora y se aprende de transiciones repetidas; "
        "en validación y test la política es determinista.",
        ha="center",
        fontsize=9,
    )
    figure.suptitle("Ciclo de aprendizaje de Deep Q-Network (Stable-Baselines3)", fontsize=13)
    figure.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "dqn_algorithm_flow.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_baseline_metrics(
    results: dict[str, dict[str, Any]], output_dir: str | Path
) -> Path:
    """Compare held-out return, drawdown, and costs on separate readable axes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    names = tuple(results)
    colors = {"DQN": "#2864a5", "Buy and hold": "#f28e2b", "Random": "#59a14f"}
    metrics = (
        ("cumulative_return", "Retorno acumulado", "Porcentaje", True),
        ("maximum_drawdown", "Drawdown máximo", "Porcentaje", True),
        ("transaction_costs", "Costos de transacción", "USD", False),
    )
    figure, axes = plt.subplots(3, 1, figsize=(9, 7))
    for axis, (key, title, unit, include_zero) in zip(axes, metrics):
        values = [float(results[name][key]) for name in names]
        bars = axis.barh(names, values, color=[colors.get(name, "#888888") for name in names])
        if include_zero:
            axis.axvline(0, color="#444444", linewidth=0.8)
        axis.set_title(title, loc="left", fontsize=10)
        axis.set_xlabel(unit)
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
        if include_zero:
            axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            axis.set_xlim(
                min(0.0, min(values) * 1.35),
                max(0.0, max(values) * 1.10),
            )
        for bar, value in zip(bars, values):
            label = f"{value:+.2%}" if unit == "Porcentaje" else f"${value:,.0f}"
            if value < 0:
                axis.text(
                    value / 2,
                    bar.get_y() + bar.get_height() / 2,
                    label,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )
            else:
                axis.annotate(
                    label,
                    (value, bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=8,
                )
    figure.suptitle("Comparación en el segmento de test", fontsize=12)
    figure.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "baseline_comparison.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_training_reward_summary(output_dir: str | Path) -> Path:
    """Plot documented training/evaluation aggregates without implying a learning curve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_path = Path(__file__).resolve().parents[2] / "reports" / "training_summary.csv"
    with summary_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"training summary is empty: {summary_path}")

    labels = [row["market"] for row in rows]
    training_rewards = [float(row["training_reward_mean"]) for row in rows]
    validation_rewards = [float(row["validation_reward"]) for row in rows]
    figure, axes = plt.subplots(2, 1, figsize=(9, 6))
    for axis, values, title, color in (
        (
            axes[0],
            training_rewards,
            "Entrenamiento: media móvil de recompensa por episodio",
            "#2864a5",
        ),
        (
            axes[1],
            validation_rewards,
            "Validación: recompensa de un episodio determinista",
            "#e45756",
        ),
    ):
        bars = axis.bar(labels, values, color=color)
        axis.axhline(0, color="#444444", linewidth=0.8)
        axis.set_title(title, loc="left", fontsize=10)
        axis.set_ylabel("Recompensa")
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.margins(y=0.25)
        for bar, value in zip(bars, values):
            x_position = bar.get_x() + bar.get_width() / 2
            if value < 0:
                axis.text(
                    x_position,
                    value / 2,
                    f"{value:+.6f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )
            else:
                axis.annotate(
                    f"{value:+.6f}",
                    (x_position, value),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
    figure.suptitle("Recompensas agregadas reportadas para los checkpoints DQN", fontsize=12)
    figure.text(
        0.5,
        0.01,
        "Agregaciones y condiciones distintas; no son comparables entre paneles "
        "ni una curva temporal de aprendizaje.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "training_reward_summary.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


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
    plot_trading_explanations(env_id, results["DQN"]["curve"], output_dir)
    plot_dataset_splits(output_dir)
    plot_dqn_algorithm_flow(output_dir)
    plot_baseline_metrics(results, output_dir)
    plot_training_reward_summary(output_dir)
    return results
