# RL Games (Stable-Baselines3)

The twin of [`rl_games`](https://github.com/emiliomunozai/rl_games): the same
CLI and project layout, with the learning algorithms taken from
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) instead of written
by hand. Use it as a reference baseline: if an SB3 agent solves an env and your
hand-written one does not, the bug is in your agent.

## Agents

| Agent | Algorithm |
|---|---|
| `dqn` | Deep Q-Network |
| `ppo` | Proximal Policy Optimization |
| `a2c` | Advantage Actor-Critic |

The policy network is picked from the observation: `MlpPolicy` for state
vectors (LunarLander, CartPole, ...), `CnnPolicy` for images (Atari).

## Setup

```bash
uv sync
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux / macOS
```

### GPU

On Linux and Windows, `uv sync` installs the CUDA 13.0 build of PyTorch
(configured under `[tool.uv.sources]` in `pyproject.toml`). It needs an NVIDIA
driver that supports CUDA 13.0 or newer, and falls back to the CPU on machines
without a GPU. Check it with:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Every command that builds or loads a model prints its device, e.g.
`Device: cuda (NVIDIA GeForce RTX 5070 Ti)`. By default it is the GPU if there
is one. `--device cpu` or `--device cuda` overrides it, including a config's
`device:`:

```bash
rlgames-sb train --config configs/lunarlander_ppo.yaml --device cuda
rlgames-sb sim   ppo --env LunarLander-v3 --device cpu
```

The GPU pays off for CNN policies (Atari). Small MLP policies usually train
faster on the CPU, so `configs/lunarlander_ppo.yaml` sets `device: cpu`.

## CLI usage

Same commands as `rlgames`, under `rlgames-sb`. The one difference: `train`
takes `--timesteps` instead of `--episodes`, because SB3 trains by timesteps.

```bash
rlgames-sb version
rlgames-sb list
rlgames-sb inspect --env CartPole-v1 --steps 10
rlgames-sb init   ppo
rlgames-sb train  ppo --timesteps 500000
rlgames-sb train  --config configs/lunarlander_ppo.yaml
rlgames-sb tune   --config configs/lunarlander_ppo.yaml --trials 20
rlgames-sb load   ppo --eval
rlgames-sb sim    ppo --episodes 2 --steps 10 --verbose
rlgames-sb render ppo --episodes 3
rlgames-sb delete ppo
```

## Full example: LunarLander from start to finish

PPO solving LunarLander (mean reward 200+) with the bundled config, tracked in
MLflow. The full run takes about 6 minutes on a CPU.

**1. Look at the environment.** 8 numbers in (position, velocity, angle,
angular velocity, leg contacts), 4 actions out (do nothing, left engine,
main engine, right engine):

```bash
rlgames-sb inspect --env LunarLander-v3 --steps 1
```
```
Observation space : Box([...], (8,), float32)
Action space      : Discrete(4)
Max episode steps : 1000
  Initial state: [-0.004  1.399 -0.424 -0.514  0.005  0.096  0.     0.   ]
  step   1 | action=3 | reward=-0.506 | done=False
```

**2. Train with the config.** `configs/lunarlander_ppo.yaml` sets 1M timesteps
and PPO hyperparameters adapted from RL Baselines3 Zoo:

```bash
rlgames-sb train --config configs/lunarlander_ppo.yaml
```

SB3 prints a progress table every rollout. Watch `ep_rew_mean`: it starts
around -200, crosses 0 at about 120k steps, first reaches 200 at about
530k, and stays there from about 900k. Training
ends with a 10-episode evaluation:

```
Eval (10 episodes): 265.92 +/- 21.82
Training complete. Saved to saves\ppo_LunarLander-v3.zip
```

**3. Look at the run in MLflow.**

```bash
mlflow ui          # http://127.0.0.1:5000 -> experiment "lunarlander"
```

The run has the hyperparameters as params, the learning curve as
`rollout/ep_rew_mean`, the losses under `train/`, the final `eval/mean_reward`,
and the config and model as artifacts.

**4. Evaluate and watch the agent.**

```bash
rlgames-sb load   ppo --env LunarLander-v3 --eval
rlgames-sb sim    ppo --env LunarLander-v3 --steps 5
rlgames-sb render ppo --env LunarLander-v3 --episodes 3   # opens a window
```
```
PPO agent for LunarLander-v3
  Timesteps trained : 1,000,448
  Network params    : 9,797
  Mean reward: 249.62 +/- 44.07
...
  Result: TERMINATED | Steps: 337 | Total reward: +243.66
```

**5. Tune it (optional).** 20 trials of 200k timesteps each, about 25 minutes:

```bash
rlgames-sb tune --config configs/lunarlander_ppo.yaml --trials 20
```

In MLflow the tuning run has one nested run per trial; sort them by
`eval/mean_reward` to compare. Paste the printed best values under
`hyperparams:` in the config, then retrain from scratch (a resumed model keeps
its old hyperparameters):

```bash
rlgames-sb delete ppo --env LunarLander-v3
rlgames-sb train  --config configs/lunarlander_ppo.yaml
```

To see what happens inside `train`, step by step, open
`notebooks/lunarlander_ppo.ipynb`.

## Experiment configs

An experiment is a YAML file under `configs/`: the env, the algorithm, the
timesteps and the hyperparameters, which go straight to the SB3 constructor.
The `tune:` section defines a search space for `rlgames-sb tune`.

```yaml
experiment: breakout          # MLflow experiment name
env: ALE/Breakout-v5
algo: dqn
timesteps: 2_000_000
eval_episodes: 10             # greedy evaluation after training
hyperparams:
  learning_rate: 1.0e-4
  buffer_size: 100_000
tune:
  trials: 10
  timesteps: 200_000          # per trial
  hyperparams:                # overrides for the short trial runs
    learning_starts: 10_000
  search_space:
    learning_rate: {type: float, low: 3.0e-5, high: 1.0e-3, log: true}
    batch_size: {type: categorical, choices: [32, 64]}
```

| Config | |
|---|---|
| `configs/lunarlander_ppo.yaml` | PPO on LunarLander. Trains in minutes, a good first experiment |
| `configs/breakout_dqn.yaml` | DQN on Atari Breakout, adapted from RL Baselines3 Zoo |
| `configs/stocks_dqn.yaml` | DQN portfolio agent on GOOGL, with temporal validation/test splits |
| `configs/forex_dqn.yaml` | DQN portfolio agent on EUR/USD, with temporal validation/test splits |

`--timesteps` on the command line overrides the config's. Without `--config`,
`train` uses SB3's defaults, as before. Write floats as `1.0e-4`, not `1e-4`,
because YAML reads the latter as a string.

Training resumes from an existing save, and a resumed model keeps the
hyperparameters it was saved with. To train with new hyperparameters, run
`rlgames-sb delete` first.

## Experiment tracking (MLflow)

Every `train` and `tune` run is logged to [MLflow](https://mlflow.org/):

- **params**: algorithm, env, timesteps and the model's hyperparameters
- **metrics**: everything SB3 logs during training (`rollout/ep_rew_mean`,
  `train/loss`, ...) plus `eval/mean_reward` after training
- **artifacts**: the config file and the saved model

```bash
mlflow ui          # then open http://127.0.0.1:5000
```

Runs are stored in `./mlflow.db` and `./mlruns/` (both gitignored). Set
`MLFLOW_TRACKING_URI` to use another store, e.g. a shared MLflow server.

## Hyperparameter tuning

`tune` runs an [Optuna](https://optuna.org/) search over the config's
`search_space`. Each trial trains a fresh model for `tune.timesteps` and is
scored by its mean evaluation reward. In MLflow, a tuning run is one parent run
with a nested run per trial, so trials can be compared side by side.

```bash
rlgames-sb tune --config configs/lunarlander_ppo.yaml --trials 20
```

Trial models are not saved. At the end, `tune` prints the best values as YAML.
Paste them under `hyperparams:` and run `train --config` for the full run.
Short trials favor hyperparameters that learn fast, which are not always the
ones that end up best.

## Notebooks

Each notebook walks through one training run: the environment, a single
`(s, a, r, s', done)` transition, the network, one update computed by hand
(loss and gradient step), a short training run, and evaluation against a
random policy. Open them in VS Code or Jupyter with the project's `.venv` as
the kernel.

| Notebook | |
|---|---|
| `notebooks/lunarlander_ppo.ipynb` | PPO from a state vector: actor-critic, GAE advantages, the clipped loss. Reaches or nears the 200 "solved" mark in about 3 minutes on a CPU |
| `notebooks/breakout_dqn.ipynb` | DQN from pixels: frame stacking, replay buffer, TD targets. Shows the mechanics; playing well needs millions of steps |

## Atari

Any `ALE/...` env id works; there is no Atari-specific code beyond `envs.make()`,
which applies the standard preprocessing (4-frame skip, 84x84 grayscale, a
stack of the last 4 frames):

```bash
rlgames-sb inspect --env ALE/Pong-v5
rlgames-sb train   dqn --env ALE/Pong-v5 --timesteps 2000000
rlgames-sb render  dqn --env ALE/Pong-v5

rlgames-sb train   dqn --env ALE/Breakout-v5 --timesteps 2000000
rlgames-sb render  dqn --env ALE/Breakout-v5
```

Games that wait for FIRE to launch the ball, like Breakout, also get
`FireOnNewLife`, which presses FIRE after a reset and after each lost life.
Without it, an agent that never presses FIRE idles until the time limit.

Atari needs millions of timesteps; train on a GPU if you have one. The DQN replay
buffer is capped at 50k transitions for image envs (`IMAGE_KWARGS` in
`registry.py`), about 2.8 GB of RAM. For tuned hyperparameters per game, see
[RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo).

## Taller: agente inversor con DQN

Observations and Actions:

| | |
|---|---|
| Observation | the last 30 (stocks) / 24 (forex) steps of `[price, price change]` |
| Actions | `0` = Sell (go short), `1` = Buy (go long) |
| Reward | price change while holding a long position |
| Episode | one pass over the whole dataset |

The trading path reuses the `stocks-v0` GOOGL and `forex-v0` EUR/USD data
registered by `gym-anytrading`; `PortfolioTradingEnv` supplies the portfolio
accounting and causal features that those environments' original two-action
long/short task does not provide. Both experiment files use SB3 DQN:

```text
configs/stocks_dqn.yaml
configs/forex_dqn.yaml
```

The bundled `STOCKS_GOOGL.csv` and `FOREX_EURUSD_1H_ASK.csv` source data is
used directly; the package does not download additional market data. Check
its time range and quality before interpreting a run:

```bash
rlgames-sb analyze --env stocks-v0 --output reports\stocks-eda
rlgames-sb analyze --env forex-v0  --output reports\forex-eda
```

Each command saves `market_eda.png` and `market_statistics.csv`. The chart
shows price and SMA-20, one-period return distribution, rolling 20-period
volatility, price drawdown and volume when supplied by the data (otherwise
close relative to SMA-20). The CSV records date range, count, buy-and-hold
price return, maximum price drawdown, return statistics and the observed
volatility range. Peaks and sustained rises/falls in the price panel show
trend and drawdown periods; clusters in the rolling-volatility panel identify
high- and low-volatility intervals. These summaries are descriptive only and
are not used to fit the policy or transform the held-out segment.

### State, actions, transitions, and reward

At close `t`, the state is a seven-value float vector, in this order:

| Feature | Definition using data available at `t` | Why it can help |
|---|---|---|
| `return_1` | `Close[t] / Close[t-1] - 1` | Most recent price direction and magnitude |
| `momentum_5` | `Close[t] / Close[t-5] - 1` | Short-term trend |
| `momentum_20` | `Close[t] / Close[t-20] - 1` | A slower trend context |
| `volatility_20` | Standard deviation of up to 20 trailing log returns | Recent risk/regime |
| `price_vs_sma_20` | `Close[t] / mean(Close[t-19:t]) - 1` | Whether price is above/below its recent baseline |
| `rsi_14` | Trailing 14-return gain fraction, scaled to `[0,1]` | Recent balance of positive and negative moves |
| `current_exposure` | Current invested fraction: `0`, `0.5`, or `1` | Makes the cost of changing position observable |

The return, momentum, volatility, and moving-average windows are 1, 5, 20,
20, and 20 market periods respectively; RSI uses 14 periods. The volatility
window is up to 20 log returns (21 closes), while the SMA uses 20 closes.
These windows are fixed before training, and no volume feature is required.
At reset, the portfolio is in cash. The action space is `Discrete(3)`:

| DQN action | Target exposure | Economic interpretation |
|---:|---:|---|
| `0` | `0%` | All cash for the next price interval |
| `1` | `50%` | Half of current portfolio equity exposed to the asset |
| `2` | `100%` | Fully invested, long-only |

Actions set a target allocation; they are not buy/sell orders or leverage.
Three levels are a small discrete space suited to DQN and express the
workshop's cash/partial/full exposure. Consequently this agent cannot short,
borrow, choose an arbitrary allocation, select individual shares/contracts,
or trade intraperiod.

The timing is deliberately explicit: observe through close `t`, choose the
target, pay `10 bps * abs(target - previous exposure)` of pre-trade equity,
then apply the close-to-close return `Close[t+1]/Close[t]-1` to the new
exposure. If `V[t]` is equity and `p[t]` is the selected exposure, the
post-cost value is:

```text
cost_fraction = 0.001 * abs(p[t] - p[t-1])
V[t+1] = V[t] * (1 - cost_fraction)
               * (1 + p[t] * (Close[t+1] / Close[t] - 1))
```

The default `log_return` reward is `log(V[t+1] / V[t])`; it incorporates the
trading cost and is additive over time. `simple_return` is the initial
return-based reward alternative. `drawdown_adjusted` subtracts
`drawdown_penalty * increase_in_drawdown` from log return (default penalty
`0.5`); it makes risk aversion explicit but is not itself a financial return.
Never compare reward totals from different reward modes as if they were
portfolio returns: compare their held-out equity and risk metrics instead.

Each feature is built from trailing closes only. The action sees `Close[t]`;
`Close[t+1]` is read only inside `step` to compute the subsequent outcome.
Rolling statistics are not centered, and there is no whole-data
normalization. The chronological playable interval is divided into 70%
training, 15% validation and 15% test. For validation/test, earlier closes may
form the trailing observation at the boundary, but no transition from a later
segment is included in that segment's rewards. Hyperparameter tuning and
training evaluation use validation; `backtest` alone uses the final test
interval.

### DQN and training protocol

The two configs use the same reproducible starting settings: 200,000
environment steps, seed 42, a fully connected `[128, 128]` MLP, learning rate
`1e-4`, batch size 64, replay capacity 50,000, 2,000 learning-start steps,
discount `gamma=0.99`, one gradient step per four environment steps, target
network update every 1,000 steps, and linear epsilon-greedy exploration from
the SB3 initial epsilon `1.0` to `0.05` over the first 20% of training. DQN
uses replay-buffer off-policy updates; each episode is one sequential pass
through the training slice, restarted as needed to reach the configured step
count. `train --config` resumes an existing model, so delete the old save
before changing reward settings or comparing a fresh run. Training stops at
the requested timestep count; no test-set score is used as a stopping rule.

```bash
uv sync
rlgames-sb inspect --env stocks-v0 --steps 3
rlgames-sb train --config configs/stocks_dqn.yaml
rlgames-sb train --config configs/forex_dqn.yaml
rlgames-sb tune --config configs/stocks_dqn.yaml --trials 10
```

Tuning trials train on the training slice and select by validation reward.
Use the saved tuned parameters in a fresh training run, not the trial model.
Each config seed is passed to SB3 and the environment; tuning trials use
`seed + trial_number`. MLflow records the seed, environment options, model
parameters, training metrics, config and model artifact.

### Baselines, held-out metrics, and reward iteration

After choosing the final settings using only training/validation data, compare
the deterministic policy against two strategies on identical held-out dates,
starting cash and transaction costs:

```bash
rlgames-sb backtest --config configs/stocks_dqn.yaml --output reports\stocks-final
rlgames-sb backtest --config configs/forex_dqn.yaml  --output reports\forex-final
```

The `Random` policy samples the three target exposures with a fixed seed;
`Buy and hold` sets 100% exposure from the first held-out close and keeps it.
All strategies pay the same 10 bps turnover fee. The report includes:

* `strategy_summary.csv`: cumulative portfolio return, maximum equity
  drawdown, dollar transaction costs, number of exposure changes and average
  exposure.
* `equity_curves.csv`: dated equity, position and fees for every step.
* `backtest.png`: equity curves and invested-fraction time series.
* `dqn_transition_example.png`: a worked DQN test transition, including the
  market move, target exposure, portfolio change, fee and reward.
* `dqn_observation_windows.png`: the causal close-price history used by each
  state feature at the same decision point.
* `dataset_splits.png`: chronological training, validation and test ranges for
  both bundled markets.
* `dqn_algorithm_flow.png`: the DQN interaction, replay-buffer and target-network
  update cycle.
* `baseline_comparison.png`: held-out return, drawdown and transaction costs
  on separate axes.
* `training_reward_summary.png`: documented aggregate training/validation
  rewards; it is not a per-episode learning curve.

`reports/training_summary.csv` records the aggregate rewards and episode counts
reported for the final financial DQN runs. The per-step/per-episode histories
are not included in the current workspace, so the summary chart must not be
interpreted as evidence of convergence.

The return is `final equity / initial equity - 1`; maximum drawdown is the
minimum of `equity / previous running peak - 1`. Read the equity curve together
with drawdown and fees: a higher ending value can still conceal a larger
interim loss or excessive turnover. The position panel reveals prolonged
cash exposure, rapid switching or persistent full-risk exposure. Repeated
switches whose cost is large relative to gross gains indicate turnover-driven
reward hacking; high reward with poor held-out equity indicates that the
reward/objective or training period did not generalize. These are diagnostic
criteria, not claims about results before running the experiment.

To reproduce a reward-design iteration, train and backtest the `simple_return`
variant first (set `env_kwargs.reward_mode: simple_return` in a copy of the
config), then do the same with the recommended `log_return`, and finally
optionally test `drawdown_adjusted`. Before each fresh training run:

```bash
rlgames-sb delete dqn --env stocks-v0
```

Use a distinct `--output` directory for each run so its held-out CSVs and
plots remain comparable. Keep data split, seed, cost, timesteps, network and
DQN hyperparameters fixed; only the reward mode should differ. Inspect
validation behavior during design, but do not repeatedly select versions by
the test score. For each reward version record: (1) reward equation, (2)
training curve and stopping step, (3) position/equity behavior, (4) suspected
failure or exploit, (5) reward adjustment, and (6) held-out metrics. The
repository provides the executable comparison protocol and generated
artifacts; empirical outcomes and reflections must be filled from those
artifacts after training, not invented in advance.

### Limitations and interpretation

These are historical single-series simulations, not investment advice or
evidence of profitable live trading. The allocation model omits spreads,
slippage, taxes, financing, liquidity, corporate actions, partial fills,
cash interest, and intraperiod prices; a fixed fee cannot represent all
market conditions. A single chronological split and one seed provide weak
statistical evidence. DQN may overfit a small number of repeated transitions,
the reward is only a proxy for risk preference, and the random baseline can
vary materially with its seed. A negative result is still informative:
compare it with buy-and-hold, the random policy, turnover and drawdown, then
report unresolved behaviors rather than claiming unexplained success.

### Reflection questions from the workshop

These answers separate methodological conclusions from findings that require
an actual trained policy:

* **Most surprising result (30):** record the largest difference between
  DQN, buy-and-hold and random in the held-out summary, then check whether
  the equity curve explains it. No measured DQN result is asserted here
  because this repository change does not run a long training job.
* **Most influential methodological decision (31):** the chronological
  train/validation/test boundary and the close-`t`/close-`t+1` transition.
  They determine what the agent can know and prevent a favorable test period
  from influencing model selection. Transaction costs are the next important
  choice because target-allocation changes otherwise appear free.
* **What to change on a repeat (32):** use multiple seeds and rolling-origin
  test windows, retain an untouched final period, and model spread/slippage
  from data appropriate to each market. Increase observations or training
  data before enlarging the network.
* **Unexplained behavior (33):** if the agent holds cash, changes position
  rapidly, or accepts large drawdowns, locate those dates in
  `equity_curves.csv`, inspect the seven causal state values and reward
  components, and compare the behavior across seeds/reward modes. Do not
  describe any of these as observed until the reports show them.
* **RL theory versus this application (34):** textbook MDPs assume a stable
  transition/reward distribution and a Markov state; real prices are
  non-stationary, partly observed, and affected by omitted economic events.
  Replay-buffer samples also come from a short, repeatedly traversed
  historical series, so convergence to a high reward is not evidence of a
  robust investment rule.
* **Main limitations/problems (35):** a small set of historical price paths,
  one fixed split/seed, simplified execution and fees, no shorting or
  continuous sizing, proxy reward, and possible tuning overfit. Report
  these with the actual held-out metrics; do not infer statistical
  significance from one backtest.

## Project structure

```
configs/                # experiment YAMLs: hyperparameters and search spaces
notebooks/              # walkthroughs: lunarlander_ppo, breakout_dqn
src/rl_games_sb/
├── cli.py              # argument parsing and output formatting only
├── registry.py         # which algorithms exist, where they are saved
├── envs.py             # env construction, incl. Atari preprocessing and trading envs
├── config.py           # loading experiment YAMLs, sampling search spaces
├── experiments.py      # tracked training and Optuna tuning
├── trading.py          # portfolio env, causal features, EDA and held-out reports
└── tracking.py         # MLflow: SB3 logger output, recorded hyperparameters
```

Saves are written to `saves/` in the working directory, one file per
(agent, environment) pair, e.g. `saves/dqn_ALE_Pong-v5.zip`. The `.zip` is
SB3's own format: it bundles the network weights, the optimizer state, the
hyperparameters and the timestep count, which is what lets `train` resume
where it left off. The replay buffer is not included.

## Using it as a library

```python
from rl_games_sb import registry

model = registry.load_or_create("ppo", "CartPole-v1")
model.learn(total_timesteps=50_000)
model.save(registry.save_path("ppo", "CartPole-v1"))
```

Or with a config, tracked in MLflow:

```python
from rl_games_sb import config, experiments

experiments.train(config.load("configs/lunarlander_ppo.yaml"))
```
