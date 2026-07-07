# -*- coding: utf-8 -*-
"""Wiring tests for YfinanceFetcher.get_chip_distribution (TW local estimate).

Mocks the yfinance-facing edges (get_daily_data, yf.Ticker(...).fast_info) so no
network is touched. Pins: TW-only gating, ETF exclusion via missing shares,
fail-open on any error, and end-to-end reachability through DataFetcherManager.
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_provider.base import DataFetcherManager
from data_provider.realtime_types import ChipDistribution, get_chip_circuit_breaker
from data_provider.yfinance_fetcher import YfinanceFetcher

_SHARES_OUTSTANDING = 1_000_000_000.0


def _synthetic_daily_df(n=150, base_price=100.0):
    prices = list(np.linspace(base_price * 0.8, base_price, n))
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": [_SHARES_OUTSTANDING * 0.02] * n,
        }
    )


def _fast_info_with_shares(shares):
    info = MagicMock()
    info.shares = shares
    return info


def test_non_tw_code_returns_none_without_touching_network():
    fetcher = YfinanceFetcher()
    with patch.object(fetcher, "get_daily_data") as mock_daily:
        assert fetcher.get_chip_distribution("600519") is None
        assert fetcher.get_chip_distribution("AAPL") is None
        mock_daily.assert_not_called()


def test_tw_code_with_valid_data_returns_estimated_chip_distribution():
    fetcher = YfinanceFetcher()
    df = _synthetic_daily_df()
    fake_ticker = MagicMock()
    fake_ticker.fast_info = _fast_info_with_shares(_SHARES_OUTSTANDING)

    with patch.object(fetcher, "get_daily_data", return_value=df), \
            patch("yfinance.Ticker", return_value=fake_ticker):
        chip = fetcher.get_chip_distribution("2330.TW")

    assert isinstance(chip, ChipDistribution)
    assert chip.source == "yfinance_tw_estimate"
    assert chip.avg_cost > 0
    assert 0.0 <= chip.profit_ratio <= 1.0
    assert 0.0 <= chip.concentration_90 <= 1.0


def test_etf_without_shares_outstanding_returns_none():
    fetcher = YfinanceFetcher()
    df = _synthetic_daily_df()
    fake_ticker = MagicMock()
    fake_ticker.fast_info = _fast_info_with_shares(None)

    with patch.object(fetcher, "get_daily_data", return_value=df), \
            patch("yfinance.Ticker", return_value=fake_ticker):
        assert fetcher.get_chip_distribution("0050.TW") is None


def test_insufficient_history_returns_none():
    fetcher = YfinanceFetcher()
    short_df = _synthetic_daily_df(n=5)

    with patch.object(fetcher, "get_daily_data", return_value=short_df):
        assert fetcher.get_chip_distribution("2330.TW") is None


def test_empty_history_returns_none():
    fetcher = YfinanceFetcher()
    with patch.object(fetcher, "get_daily_data", return_value=pd.DataFrame()):
        assert fetcher.get_chip_distribution("2330.TW") is None


def test_get_daily_data_exception_fails_open_to_none():
    fetcher = YfinanceFetcher()
    with patch.object(fetcher, "get_daily_data", side_effect=RuntimeError("network down")):
        assert fetcher.get_chip_distribution("2330.TW") is None


def test_fast_info_exception_fails_open_to_none():
    fetcher = YfinanceFetcher()
    df = _synthetic_daily_df()
    fake_ticker = MagicMock()
    type(fake_ticker).fast_info = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))

    with patch.object(fetcher, "get_daily_data", return_value=df), \
            patch("yfinance.Ticker", return_value=fake_ticker):
        assert fetcher.get_chip_distribution("2330.TW") is None


def test_reachable_end_to_end_through_data_fetcher_manager():
    get_chip_circuit_breaker().reset()
    fetcher = YfinanceFetcher()
    df = _synthetic_daily_df()
    fake_ticker = MagicMock()
    fake_ticker.fast_info = _fast_info_with_shares(_SHARES_OUTSTANDING)

    manager = DataFetcherManager(fetchers=[fetcher])
    with patch("src.config.get_config", return_value=SimpleNamespace(enable_chip_distribution=True)), \
            patch.object(fetcher, "get_daily_data", return_value=df), \
            patch("yfinance.Ticker", return_value=fake_ticker):
        chip = manager.get_chip_distribution("2330.TW")

    assert isinstance(chip, ChipDistribution)
    assert chip.source == "yfinance_tw_estimate"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))