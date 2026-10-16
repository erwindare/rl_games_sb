"""MLflow experiment tracking.

Contents:
  - MLflowOutputFormat : SB3 logger output that writes metrics to the active
                         MLflow run
  - attach()           : route a model's training logs to MLflow (and stdout)
  - model_params()     : the hyperparameters worth recording for a model

Runs go to MLflow's default tracking store, ./mlflow.db, unless
MLFLOW_TRACKING_URI says otherwise. Browse them with `mlflow ui`.
"""
from __future__ import annotations

import sys
from typing import Any

import mlflow
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger

# Recorded when the model has them; the rest of SB3's attributes are internals.
_PARAM_NAMES = (
    "learning_rate", "gamma", "seed", "batch_size", "buffer_size", "learning_starts",
    "train_freq", "gradient_steps", "target_update_interval", "exploration_fraction",
    "exploration_initial_eps", "exploration_final_eps", "n_steps", "n_epochs",
    "gae_lambda", "clip_range",
    "ent_coef", "vf_coef", "max_grad_norm", "policy_kwargs",
)


class MLflowOutputFormat(KVWriter):
    """Numeric values SB3 logs (rewards, losses, fps, ...) become MLflow metrics."""

    def write(self, key_values: dict[str, Any], key_excluded: dict, step: int = 0) -> None:
        metrics = {
            key: float(value)
            for key, value in key_values.items()
            if isinstance(value, (int, float, np.number)) and not isinstance(value, bool)
        }
        if metrics:
            mlflow.log_metrics(metrics, step=step)


def attach(model: BaseAlgorithm, *, stdout: bool = True) -> None:
    formats: list[KVWriter] = [MLflowOutputFormat()]
    if stdout:
        formats.append(HumanOutputFormat(sys.stdout))
    model.set_logger(Logger(folder=None, output_formats=formats))


def model_params(model: BaseAlgorithm) -> dict[str, Any]:
    params = {name: getattr(model, name) for name in _PARAM_NAMES if hasattr(model, name)}
    params["policy"] = type(model.policy).__name__
    params["device"] = str(model.device)
    for name, value in params.items():
        if callable(value):  # a schedule: record its starting value
            params[name] = value(1.0)
        elif not isinstance(value, (int, float, str)):
            params[name] = repr(value)
    return params
