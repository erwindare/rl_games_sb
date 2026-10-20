import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from rl_games_sb.trading import (
    ACTION_EXPOSURES,
    PortfolioTradingEnv,
    _run_strategy,
    analyze_market_data,
    plot_baseline_metrics,
    plot_dataset_splits,
    plot_dqn_algorithm_flow,
    plot_trading_explanations,
    plot_training_reward_summary,
)


class PortfolioTradingEnvTests(unittest.TestCase):
    def setUp(self):
        self.prices = np.linspace(100.0, 149.0, 50)
        self.frame = pd.DataFrame(
            {"Close": self.prices},
            index=pd.date_range("2020-01-01", periods=len(self.prices)),
        )
        self.market_data = patch(
            "rl_games_sb.trading.market_data",
            return_value=(self.frame, self.prices, 25, 50, 10),
        )
        self.market_data.start()
        self.addCleanup(self.market_data.stop)

    def make_env(self, **kwargs):
        return PortfolioTradingEnv(
            "stocks-v0",
            train_fraction=0.70,
            validation_fraction=0.15,
            **kwargs,
        )

    def test_three_actions_set_target_allocation_and_charge_turnover_fee(self):
        env = self.make_env(transaction_cost_bps=10)
        observation, _ = env.reset(seed=7)
        self.assertEqual(env.action_space.n, 3)
        self.assertEqual(ACTION_EXPOSURES, (0.0, 0.5, 1.0))

        next_observation, reward, terminated, truncated, info = env.step(1)
        expected_growth = (1 - 0.5 * 0.001) * (
            1 + 0.5 * (self.prices[26] / self.prices[25] - 1)
        )
        self.assertAlmostEqual(info["transaction_cost"], 50.0)
        self.assertAlmostEqual(info["portfolio_value"], 100_000 * expected_growth)
        self.assertAlmostEqual(reward, np.log(expected_growth))
        self.assertEqual(next_observation[-1], 0.5)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(observation.shape, (7,))
        env.close()

    def test_validation_and_test_are_chronologically_disjoint(self):
        validation = self.make_env(segment="validation")
        test = self.make_env(segment="test")
        self.assertEqual(validation.start_tick, 42)
        self.assertEqual(validation.end_tick, 45)
        self.assertEqual(test.start_tick, 45)
        self.assertEqual(test.end_tick, 50)
        validation.close()
        test.close()

    def test_initial_observation_does_not_use_future_prices(self):
        original = self.make_env()
        original_observation, _ = original.reset()

        changed_future = self.prices.copy()
        changed_future[26:] *= 10
        changed_frame = self.frame.copy()
        changed_frame["Close"] = changed_future
        with patch(
            "rl_games_sb.trading.market_data",
            return_value=(changed_frame, changed_future, 25, 50, 10),
        ):
            changed = self.make_env()
            changed_observation, _ = changed.reset()

        np.testing.assert_array_equal(original_observation, changed_observation)
        original.close()
        changed.close()

    def test_all_cash_policy_has_zero_return_and_zero_cost(self):
        result = _run_strategy(self.make_env(segment="test"), lambda observation: 0, seed=42)
        self.assertEqual(result["cumulative_return"], 0)
        self.assertEqual(result["maximum_drawdown"], 0)
        self.assertEqual(result["transaction_costs"], 0)
        self.assertEqual(result["trades"], 0)

    def test_eda_writes_chart_and_market_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "rl_games_sb.trading.market_data",
                return_value=(self.frame, self.prices, 25, 50, 10),
            ):
                chart = analyze_market_data("stocks-v0", directory)

            self.assertTrue(chart.is_file())
            with (Path(directory) / "market_statistics.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                statistics = next(csv.DictReader(file))
            self.assertEqual(statistics["environment"], "stocks-v0")
            self.assertEqual(statistics["observations"], "50")

    def test_explanation_charts_use_a_real_curve_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            first_tick, second_tick = 25, 26
            market_return = self.prices[second_tick] / self.prices[first_tick] - 1
            fee = 50.0
            portfolio_after = 100_000 * (1 - 0.5 * 0.001) * (
                1 + 0.5 * market_return
            )
            curve = [
                {
                    "timestamp": str(self.frame.index[first_tick]),
                    "portfolio_value": 100_000.0,
                    "position": 0.0,
                    "transaction_cost": 0.0,
                    "turnover": 0.0,
                    "market_price": float(self.prices[first_tick]),
                    "market_return": 0.0,
                    "reward": 0.0,
                },
                {
                    "timestamp": str(self.frame.index[second_tick]),
                    "portfolio_value": portfolio_after,
                    "position": 0.5,
                    "transaction_cost": fee,
                    "turnover": 0.5,
                    "market_price": float(self.prices[second_tick]),
                    "market_return": market_return,
                    "reward": float(np.log(portfolio_after / 100_000)),
                },
            ]
            with patch(
                "rl_games_sb.trading.market_data",
                return_value=(self.frame, self.prices, 25, 50, 10),
            ):
                charts = plot_trading_explanations("stocks-v0", curve, directory)

            self.assertEqual(len(charts), 2)
            self.assertTrue(all(chart.is_file() and chart.stat().st_size > 0 for chart in charts))

    def test_report_charts_cover_splits_algorithm_and_baselines(self):
        results = {
            "DQN": {
                "cumulative_return": 0.08,
                "maximum_drawdown": -0.07,
                "transaction_costs": 2_800.0,
            },
            "Buy and hold": {
                "cumulative_return": 0.48,
                "maximum_drawdown": -0.15,
                "transaction_costs": 100.0,
            },
            "Random": {
                "cumulative_return": -0.08,
                "maximum_drawdown": -0.23,
                "transaction_costs": 15_000.0,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "rl_games_sb.trading.market_data",
                return_value=(self.frame, self.prices, 25, 50, 10),
            ):
                split_chart = plot_dataset_splits(directory)
            algorithm_chart = plot_dqn_algorithm_flow(directory)
            baseline_chart = plot_baseline_metrics(results, directory)
            training_chart = plot_training_reward_summary(directory)
            self.assertTrue(
                all(path.is_file() and path.stat().st_size > 0 for path in (
                    split_chart, algorithm_chart, baseline_chart, training_chart
                ))
            )


if __name__ == "__main__":
    unittest.main()
