"""Environment construction.

Contents:
  - DEFAULT_ENV_ID  : the env used when none is given
  - TRADING_ENV_IDS : the gym-anytrading envs
  - FireOnNewLife   : wrapper that presses FIRE at the start of each life
  - make()          : construct standard, Atari-preprocessed, or portfolio
                      trading environments
"""
import ale_py
import gym_anytrading
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

# Third-party envs register their ids on import: ALE/... (Atari) and
# stocks-v0 / forex-v0 (trading).
gym.register_envs(ale_py)
gym.register_envs(gym_anytrading)

DEFAULT_ENV_ID = "LunarLander-v3"
TRADING_ENV_IDS = ("stocks-v0", "forex-v0")


class FireOnNewLife(gym.Wrapper):
    """Press FIRE after reset and after each lost life.

    Games like Breakout wait for FIRE to launch the ball. Without this, an
    agent that never presses it idles until the time limit.
    """

    FIRE = 1

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.lives = 0

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(self.FIRE)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        self.lives = info["lives"]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info["lives"] < self.lives and not (terminated or truncated):
            obs, extra, terminated, truncated, info = self.env.step(self.FIRE)
            reward += extra
        self.lives = info["lives"]
        return obs, reward, terminated, truncated, info


def make(
    env_id: str | None = None, *, render_mode: str | None = None, **env_kwargs
) -> gym.Env:
    """Create an environment, defaulting to this project's primary env.

    Atari envs get the standard DQN preprocessing: 4-frame skip, 84x84
    grayscale, and a stack of the last 4 frames -> uint8 obs of shape (4, 84, 84).
    Trading envs use PortfolioTradingEnv over the market data registered by
    gym-anytrading. Other environments retain their existing behavior.
    """
    env_id = env_id or DEFAULT_ENV_ID
    if env_id.startswith("ALE/"):
        # frameskip=1 because AtariPreprocessing does its own frame skipping
        env = gym.make(env_id, render_mode=render_mode, frameskip=1, **env_kwargs)
        env = gym.wrappers.AtariPreprocessing(env)
        if env.unwrapped.get_action_meanings()[FireOnNewLife.FIRE] == "FIRE":
            env = FireOnNewLife(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
    elif env_id in TRADING_ENV_IDS:
        from rl_games_sb.trading import PortfolioTradingEnv

        env = PortfolioTradingEnv(env_id, **env_kwargs)
    else:
        if env_kwargs:
            raise TypeError(f"environment options are not supported for {env_id!r}")
        # gym-anytrading returns overlapping views of a price array that never
        # changes. That is harmless, but Gymnasium's env checker warns about
        # it on every command, so the checker is off for those envs.
        env = gym.make(
            env_id,
            render_mode=render_mode,
            disable_env_checker=env_id in TRADING_ENV_IDS,
            **env_kwargs,
        )
    # Monitor records episode rewards, which SB3's logging and evaluation read.
    return Monitor(env)
