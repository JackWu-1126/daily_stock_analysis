# -*- coding: utf-8 -*-
"""Tests for GET /api/v1/stocks/{stock_code}/history (MA fields + days bounds)."""

import inspect
import unittest
from unittest.mock import MagicMock, patch

from api.v1.endpoints.stocks import get_stock_history


class StockHistoryEndpointTestCase(unittest.TestCase):
    @patch("api.v1.endpoints.stocks.StockService")
    def test_response_includes_ma_fields(self, mock_service_cls: MagicMock) -> None:
        mock_service_cls.return_value.get_history_data.return_value = {
            "stock_code": "600519",
            "stock_name": "贵州茅台",
            "period": "daily",
            "data": [
                {
                    "date": "2026-01-01",
                    "open": 1800.0,
                    "high": 1810.0,
                    "low": 1790.0,
                    "close": 1805.0,
                    "volume": 1000.0,
                    "amount": 100000.0,
                    "change_percent": 0.3,
                    "ma5": 1802.0,
                    "ma10": 1798.0,
                    "ma20": 1795.0,
                }
            ],
        }

        response = get_stock_history(stock_code="600519", period="daily", days=120)

        self.assertEqual(response.stock_code, "600519")
        self.assertEqual(len(response.data), 1)
        point = response.data[0]
        self.assertEqual(point.ma5, 1802.0)
        self.assertEqual(point.ma10, 1798.0)
        self.assertEqual(point.ma20, 1795.0)

    def test_days_query_default_is_120(self) -> None:
        days_param = inspect.signature(get_stock_history).parameters["days"]
        query_info = days_param.default
        self.assertEqual(query_info.default, 120)


if __name__ == "__main__":
    unittest.main()
