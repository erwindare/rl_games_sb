"""Algorithm lookup, save paths, and construction.

Contents:
  - SAVE_DIR        : directory holding saved models
  - ALGOS           : algorithm key -> Stable-Baselines3 class
  - ALGO_CHOICES    : the valid algorithm keys
  - save_path()     : save file for an (algorithm, env) pair
  - create()        : build a fresh, untrained model
  - load()          : load a saved model
  - load_or_create(): load if a save exists, else create
  - describe_device(): a device as printed by the CLI, e.g. "cuda (RTX 5070 Ti)"
"""
from pathlib import Path

import torch
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.utils import get_device

from rl_games_sb import envs

SAVE_DIR = Path("saves")

# Adding an algorithm = a row.
ALGOS: dict[str, type[BaseAlgorithm]] = {"dqn": DQN, "ppo": PPO, "a2c": A2C}

ALGO_CHOICES = tuple(ALGOS)

# Overrides for image (Atari) envs. SB3's default DQN replay buffer holds 1M
# transitions, each two 4x84x84 frame stacks: ~56 GB. 50k is ~2.8 GB.
IMAGE_KWARGS: dict[str, dict] = {"dqn": {"buffer_size": 50_000}}


def save_path(algo: str, env_id: str) -> Path:
    """One save file per (algorithm, env) pair."""
    return SAVE_DIR / f"{algo}_{env_id.replace('/', '_')}.zip"


def create(
    algo: str, env_id: str, *, env_kwargs: dict | None = None, **hyperparams
) -> BaseAlgorithm:
    """Build a fresh, untrained model: a CNN policy for images, an MLP otherwise.

    `hyperparams` go to the SB3 constructor and override the defaults here.
    """
    env = envs.make(env_id, **(env_kwargs or {}))
    if len(env.observation_space.shape) == 3:
        kwargs = {"verbose": 1, **IMAGE_KWARGS.get(algo, {}), **hyperparams}
        return ALGOS[algo]("CnnPolicy", env, **kwargs)
    return ALGOS[algo]("MlpPolicy", env, **{"verbose": 1, **hyperparams})


def load(
    algo: str,
    env_id: str,
    device: str = "auto",
    *,
    env_kwargs: dict | None = None,
) -> BaseAlgorithm:
    """Load the saved model for this (algorithm, env) pair.

    "auto" means the GPU if there is one, whatever the model trained on.
    """
    return ALGOS[algo].load(
        save_path(algo, env_id),
        env=envs.make(env_id, **(env_kwargs or {})),
        device=device,
    )


def load_or_create(
    algo: str, env_id: str, *, env_kwargs: dict | None = None, **hyperparams
) -> BaseAlgorithm:
    """Resume from a save if there is one, otherwise start fresh.

    A resumed model keeps the hyperparameters it was saved with, except
    `device`, which applies either way.
    """
    if save_path(algo, env_id).exists():
        return load(
            algo, env_id, device=hyperparams.get("device", "auto"), env_kwargs=env_kwargs
        )
    return create(algo, env_id, env_kwargs=env_kwargs, **hyperparams)


def describe_device(device: torch.device | str) -> str:
    """'cuda (NVIDIA GeForce RTX 5070 Ti)' or 'cpu'. Resolves "auto"."""
    device = get_device(device)
    if device.type == "cuda":
        return f"{device} ({torch.cuda.get_device_name(device)})"
    return str(device)
