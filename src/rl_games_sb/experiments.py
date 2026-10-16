"""Tracked training and hyperparameter tuning.

Contents:
  - train() : train (or resume) an agent as one MLflow run
  - tune()  : an Optuna search over a config's search space; one parent MLflow
              run, one nested run per trial

Both take a config.Config, built from a YAML file or from CLI arguments.
"""
from __future__ import annotations

from pathlib import Path

import mlflow
import optuna
from stable_baselines3.common.evaluation import evaluate_policy

from rl_games_sb import config, envs, registry, tracking


def train(cfg: config.Config) -> Path:
    """Train for cfg.timesteps, save, evaluate, and log it all to MLflow."""
    path = registry.save_path(cfg.algo, cfg.env)
    resumed = path.exists()
    if resumed and cfg.hyperparams.keys() - {"device"}:
        print(
            f"Resuming {path}: it keeps the hyperparameters it was saved with, "
            f"so the config's are ignored. Run 'rlgames-sb delete {cfg.algo} "
            f"--env {cfg.env}' to start fresh."
        )
    model = registry.load_or_create(
        cfg.algo,
        cfg.env,
        env_kwargs={**cfg.env_kwargs, "segment": "train"}
        if cfg.env in envs.TRADING_ENV_IDS else cfg.env_kwargs,
        **{"seed": cfg.seed, **cfg.hyperparams},
    )
    print(f"Device: {registry.describe_device(model.device)}")

    mlflow.set_experiment(cfg.experiment)
    with mlflow.start_run(run_name=f"{cfg.algo}-{cfg.env}"):
        mlflow.set_tags({"algo": cfg.algo, "env": cfg.env, "resumed": resumed})
        mlflow.log_params({
            "algo": cfg.algo,
            "env": cfg.env,
            "timesteps": cfg.timesteps,
            "start_timesteps": model.num_timesteps,
            "seed": cfg.seed,
            **{f"env.{key}": value for key, value in cfg.env_kwargs.items()},
            **tracking.model_params(model),
        })
        if cfg.path:
            mlflow.log_artifact(str(cfg.path))

        tracking.attach(model)
        model.learn(total_timesteps=cfg.timesteps, reset_num_timesteps=False)
        model.save(path)
        mlflow.log_artifact(str(path))

        if cfg.eval_episodes:
            eval_env_kwargs = (
                {**cfg.env_kwargs, "segment": "validation"}
                if cfg.env in envs.TRADING_ENV_IDS else cfg.env_kwargs
            )
            mean, std = evaluate_policy(
                model,
                envs.make(cfg.env, **eval_env_kwargs),
                n_eval_episodes=cfg.eval_episodes,
            )
            mlflow.log_metrics(
                {"eval/mean_reward": mean, "eval/std_reward": std}, step=model.num_timesteps
            )
            print(f"Eval ({cfg.eval_episodes} episodes): {mean:.2f} +/- {std:.2f}")

    return path


def tune(
    cfg: config.Config, n_trials: int | None = None, timesteps: int | None = None
) -> optuna.Study:
    """Search cfg.tune.search_space, maximizing mean evaluation reward.

    Each trial trains a fresh model for `timesteps` (default: tune.timesteps).
    Its hyperparameters are cfg.hyperparams, overridden by tune.hyperparams
    (settings for short trial runs), overridden by the sampled values. Trial
    models are not saved: put the best values in the config and run `train`.
    """
    space = cfg.tune.get("search_space")
    if not space:
        raise ValueError(f"{cfg.path}: no tune.search_space to search")
    n_trials = n_trials or int(cfg.tune.get("trials", 20))
    timesteps = timesteps or int(cfg.tune.get("timesteps", cfg.timesteps))
    eval_episodes = cfg.eval_episodes or 5
    base = {**cfg.hyperparams, **(cfg.tune.get("hyperparams") or {})}
    print(f"Device: {registry.describe_device(base.get('device', 'auto'))}")

    mlflow.set_experiment(cfg.experiment)
    with mlflow.start_run(run_name=f"tune-{cfg.algo}-{cfg.env}"):
        mlflow.set_tags({"algo": cfg.algo, "env": cfg.env, "kind": "tune"})
        mlflow.log_params({
            "algo": cfg.algo,
            "env": cfg.env,
            "trials": n_trials,
            "timesteps_per_trial": timesteps,
            "eval_episodes": eval_episodes,
            "seed": cfg.seed,
            **{f"env.{key}": value for key, value in cfg.env_kwargs.items()},
        })
        if cfg.path:
            mlflow.log_artifact(str(cfg.path))

        def objective(trial: optuna.Trial) -> float:
            sampled = config.sample(trial, space)
            with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
                mlflow.set_tags({"algo": cfg.algo, "env": cfg.env, "kind": "trial"})
                trial_hyperparams = {
                    **base, **sampled, "verbose": 0, "seed": cfg.seed + trial.number
                }
                model = registry.create(
                    cfg.algo,
                    cfg.env,
                    env_kwargs={**cfg.env_kwargs, "segment": "train"}
                    if cfg.env in envs.TRADING_ENV_IDS else cfg.env_kwargs,
                    **trial_hyperparams,
                )
                mlflow.log_params(tracking.model_params(model))
                tracking.attach(model, stdout=False)
                model.learn(total_timesteps=timesteps)
                eval_env_kwargs = (
                    {**cfg.env_kwargs, "segment": "validation"}
                    if cfg.env in envs.TRADING_ENV_IDS else cfg.env_kwargs
                )
                mean, std = evaluate_policy(
                    model,
                    envs.make(cfg.env, **eval_env_kwargs),
                    n_eval_episodes=eval_episodes,
                )
                mlflow.log_metrics(
                    {"eval/mean_reward": mean, "eval/std_reward": std},
                    step=model.num_timesteps,
                )
            return mean

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        mlflow.log_params({f"best.{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_mean_reward", study.best_value)

    return study
