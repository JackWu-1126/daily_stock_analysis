# -*- coding: utf-8 -*-
"""Offline unit tests for the TW chip distribution turnover-decay estimator.

Pure-function tests against synthetic OHLCV data — no network, no yfinance import.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_provider.tw_chip_estimator import (  # noqa: E402
    estimate_tw_chip_distribution,
    MIN_REQUIRED_TRADING_DAYS,
)

_SHARES_OUTSTANDING = 1_000_000_000.0


def _make_history(prices, volumes=None, days=None):
    """Build a minimal standardized OHLCV DataFrame from a list of close prices."""
    n = days or len(prices)
    if len(prices) < n:
        prices = list(prices) + [prices[-1]] * (n - len(prices))
    if volumes is None:
        volumes = [_SHARES_OUTSTANDING * 0.02] * n
    elif not isinstance(volumes, (list, tuple)):
        volumes = [volumes] * n
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": volumes,
        }
    )


class TwChipEstimatorTestCase(unittest.TestCase):
    def test_flat_price_window_produces_tight_distribution_around_that_price(self):
        history = _make_history([100.0] * 60)
        metrics = estimate_tw_chip_distribution(history, _SHARES_OUTSTANDING, current_price=100.0)

        self.assertIsNotNone(metrics)
        self.assertAlmostEqual(metrics["avg_cost"], 100.0, delta=0.5)
        self.assertLess(metrics["concentration_90"], 0.01)
        self.assertLess(metrics["concentration_70"], 0.01)
        self.assertGreater(metrics["profit_ratio"], 0.95)

    def test_monotonic_uptrend_yields_high_profit_ratio(self):
        prices = list(np.linspace(50.0, 150.0, 80))
        history = _make_history(prices)
        metrics = estimate_tw_chip_distribution(history, _SHARES_OUTSTANDING, current_price=150.0)

        self.assertIsNotNone(metrics)
        self.assertGreater(metrics["profit_ratio"], 0.7)

    def test_monotonic_downtrend_yields_low_profit_ratio(self):
        prices = list(np.linspace(150.0, 50.0, 80))
        history = _make_history(prices)
        metrics = estimate_tw_chip_distribution(history, _SHARES_OUTSTANDING, current_price=50.0)

        self.assertIsNotNone(metrics)
        self.assertLess(metrics["profit_ratio"], 0.3)

    def test_insufficient_trading_days_returns_none(self):
        history = _make_history([100.0] * (MIN_REQUIRED_TRADING_DAYS - 1))
        metrics = estimate_tw_chip_distribution(history, _SHARES_OUTSTANDING, current_price=100.0)
        self.assertIsNone(metrics)

    def test_zero_or_negative_shares_outstanding_returns_none(self):
        history = _make_history([100.0] * 60)
        self.assertIsNone(estimate_tw_chip_distribution(history, 0.0, current_price=100.0))
        self.assertIsNone(estimate_tw_chip_distribution(history, -1.0, current_price=100.0))
        self.assertIsNone(estimate_tw_chip_distribution(history, None, current_price=100.0))

    def test_degenerate_zero_width_days_do_not_raise(self):
        # Every day is a "一字板" (limit-locked, open=high=low=close).
        history = _make_history([80.0] * 40)
        metrics = estimate_tw_chip_distribution(history, _SHARES_OUTSTANDING, current_price=80.0)
        self.assertIsNotNone(metrics)
        self.assertTrue(np.isfinite(metrics["avg_cost"]))

    def test_single_day_volume_exceeding_shares_outstanding_is_clipped(self):
        prices = [100.0] * 39 + [200.0]
        volumes = [_SHARES_OUTSTANDING * 0.01] * 39 + [_SHARES_OUTSTANDING * 5]
        history = _make_history(prices, volumes=volumes)
        metrics = estimate_tw_chip_distribution(history, _SHARES_OUTSTANDING, current_price=200.0)

        self.assertIsNotNone(metrics)
        # Turnover-rate clipping to 1.0 on the last day should flush prior chips,
        # so the distribution should collapse almost entirely onto the last price.
        self.assertAlmostEqual(metrics["avg_cost"], 200.0, delta=1.0)
        self.assertGreater(metrics["profit_ratio"], 0.95)

    def test_nan_volume_rows_are_dropped_without_error(self):
        history = _make_history([100.0] * 40)
        history.loc[5, "volume"] = np.nan
        history.loc[10, "volume"] = np.nan
        metrics = estimate_tw_chip_distribution(history, _SHARES_OUTSTANDING, current_price=100.0)
        self.assertIsNotNone(metrics)
        self.assertTrue(np.isfinite(metrics["avg_cost"]))

    def test_output_fields_are_fractions_in_zero_to_one_range(self):
        prices = list(np.linspace(80.0, 120.0, 60))
        history = _make_history(prices)
        metrics = estimate_tw_chip_distribution(history, _SHARES_OUTSTANDING, current_price=110.0)

        self.assertIsNotNone(metrics)
        for key in ("profit_ratio", "concentration_90", "concentration_70"):
            value = metrics[key]
            self.assertGreaterEqual(value, 0.0, msg=key)
            self.assertLessEqual(value, 1.0, msg=key)

    def test_missing_required_columns_returns_none(self):
        history = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=30), "close": [100.0] * 30})
        metrics = estimate_tw_chip_distribution(history, _SHARES_OUTSTANDING, current_price=100.0)
        self.assertIsNone(metrics)

    def test_result_keys_match_chip_distribution_schema(self):
        history = _make_history([100.0] * 60)
        metrics = estimate_tw_chip_distribution(history, _SHARES_OUTSTANDING, current_price=100.0)
        self.assertIsNotNone(metrics)
        expected_keys = {
            "profit_ratio",
            "avg_cost",
            "cost_90_low",
            "cost_90_high",
            "concentration_90",
            "cost_70_low",
            "cost_70_high",
            "concentration_70",
        }
        self.assertEqual(set(metrics.keys()), expected_keys)


if __name__ == "__main__":
    unittest.main()