# -*- coding: utf-8 -*-
"""Tests for StockService.get_history_data DB-first + fetch-on-miss behavior."""

import unittest
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd

from src.services.stock_service import StockService
from src.storage import StockDaily


def _make_bar(day: date, close: float) -> StockDaily:
    return StockDaily(
        code="600519",
        date=day,
        open=close - 1,
        high=close + 1,
        low=close - 2,
        close=close,
        volume=1000.0,
        amount=100000.0,
        pct_chg=0.5,
        ma5=close - 0.1,
        ma10=close - 0.2,
        ma20=close - 0.3,
        data_source="TestSource",
    )


class StockServiceHistoryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.service = StockService()
        self.service.repo = MagicMock()

    def test_rejects_non_daily_period(self) -> None:
        with self.assertRaises(ValueError):
            self.service.get_history_data("600519", period="weekly", days=30)

    @patch("data_provider.base.DataFetcherManager")
    def test_uses_db_when_sufficient_bars_and_skips_live_fetch(self, mock_manager_cls: MagicMock) -> None:
        bars = [_make_bar(date(2026, 1, day), 1800.0 + day) for day in range(1, 11)]
        self.service.repo.get_range.return_value = bars

        result = self.service.get_history_data("600519", days=5)

        mock_manager_cls.return_value.get_daily_data.assert_not_called()
        self.service.repo.save_dataframe.assert_not_called()
        self.assertEqual(len(result["data"]), 5)
        last = result["data"][-1]
        self.assertEqual(last["date"], "2026-01-10")
        self.assertAlmostEqual(last["ma5"], bars[-1].ma5)
        self.assertAlmostEqual(last["ma10"], bars[-1].ma10)
        self.assertAlmostEqual(last["ma20"], bars[-1].ma20)
        self.assertAlmostEqual(last["change_percent"], bars[-1].pct_chg)

    @patch("data_provider.base.DataFetcherManager")
    def test_fetches_and_saves_on_db_miss(self, mock_manager_cls: MagicMock) -> None:
        self.service.repo.get_range.return_value = []
        df = pd.DataFrame([
            {
                "date": date(2026, 1, 1),
                "open": 1800.0,
                "high": 1810.0,
                "low": 1790.0,
                "close": 1805.0,
                "volume": 1000.0,
                "amount": 100000.0,
                "pct_chg": 0.3,
                "ma5": 1802.0,
                "ma10": 1798.0,
                "ma20": 1795.0,
            }
        ])
        mock_manager_cls.return_value.get_daily_data.return_value = (df, "TestSource")
        mock_manager_cls.return_value.get_stock_name.return_value = "贵州茅台"

        result = self.service.get_history_data("600519", days=30)

        mock_manager_cls.return_value.get_daily_data.assert_called_once_with("600519", days=30)
        self.service.repo.save_dataframe.assert_called_once()
        saved_args = self.service.repo.save_dataframe.call_args.args
        self.assertEqual(saved_args[1], "600519")
        self.assertEqual(len(result["data"]), 1)
        point = result["data"][0]
        self.assertEqual(point["date"], "2026-01-01")
        self.assertEqual(point["ma5"], 1802.0)
        self.assertEqual(point["ma10"], 1798.0)
        self.assertEqual(point["ma20"], 1795.0)
        self.assertEqual(result["stock_name"], "贵州茅台")

    @patch("data_provider.base.DataFetcherManager")
    def test_returns_empty_data_when_fetch_returns_nothing(self, mock_manager_cls: MagicMock) -> None:
        self.service.repo.get_range.return_value = []
        mock_manager_cls.return_value.get_daily_data.return_value = (None, None)

        result = self.service.get_history_data("600519", days=30)

        self.service.repo.save_dataframe.assert_not_called()
        self.assertEqual(result["data"], [])


if __name__ == "__main__":
    unittest.main()
