"""Experiment configs: one YAML file per experiment, under configs/.

Contents:
  - Config        : a parsed experiment config
  - load()        : read a YAML config file
  - sample()      : draw hyperparameters from a config's search space (Optuna)

A config file looks like:

    experiment: breakout          # MLflow experiment name (default: the env id)
    env: ALE/Breakout-v5
    algo: dqn
    timesteps: 1_000_000
    eval_episodes: 10             # greedy evaluation after training, 0 to skip
    seed: 42
    env_kwargs:                       # optional environment-specific settings
      transaction_cost_bps: 10
    hyperparams:                  # passed straight to the SB3 constructor
      learning_rate: 1.0e-4
    tune:                         # used by `rlgames-sb tune` only
      trials: 20
      timesteps: 100_000          # per trial
      hyperparams:                # overrides for the short trial runs
        learning_starts: 10_000
      search_space:
        learning_rate: {type: float, low: 1.0e-5, high: 1.0e-3, log: true}
        batch_size: {type: categorical, choices: [32, 64, 128]}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from rl_games_sb.registry import ALGO_CHOICES

if TYPE_CHECKING:  # only for the annotation: importing optuna is slow and noisy
    import optuna


@dataclass
class Config:
    env: str
    algo: str
    timesteps: int
    experiment: str
    eval_episodes: int = 10
    hyperparams: dict[str, Any] = field(default_factory=dict)
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    tune: dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    path: Path | None = None


def load(path: str | Path) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    missing = {"env", "algo", "timesteps"} - raw.keys()
    if missing:
        raise ValueError(f"{path}: missing required keys {sorted(missing)}")
    if raw["algo"] not in ALGO_CHOICES:
        raise ValueError(f"{path}: algo must be one of {ALGO_CHOICES}, got {raw['algo']!r}")
    env_kwargs = raw.get("env_kwargs", {})
    if env_kwargs is None:
        env_kwargs = {}
    if not isinstance(env_kwargs, dict):
        raise TypeError(f"{path}: env_kwargs must be a mapping")

    return Config(
        env=raw["env"],
        algo=raw["algo"],
        timesteps=int(raw["timesteps"]),
        experiment=raw.get("experiment", raw["env"]),
        eval_episodes=int(raw.get("eval_episodes", 10)),
        hyperparams=raw.get("hyperparams") or {},
        env_kwargs=env_kwargs,
        tune=raw.get("tune") or {},
        seed=int(raw.get("seed", 42)),
        path=path,
    )


def sample(trial: optuna.Trial, search_space: dict[str, dict]) -> dict[str, Any]:
    """One value per search-space entry. Types: float, int, categorical."""
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        kind = spec["type"]
        if kind == "float":
            # float() because YAML reads `1e-5` (no dot) as a string
            params[name] = trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]), log=spec.get("log", False)
            )
        elif kind == "int":
            params[name] = trial.suggest_int(
                name, int(spec["low"]), int(spec["high"]), log=spec.get("log", False)
            )
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"search_space.{name}: unknown type {kind!r}")
    return params
