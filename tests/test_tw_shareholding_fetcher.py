# -*- coding: utf-8 -*-
"""Offline unit tests for TwShareholdingFetcher (台股集保户股权集中度 data-layer fetcher).

Fixture rows are trimmed from a real TDCC OpenData response (captured 2026-06-26)
so the parser is pinned to the actual field layout, tier numbering and value
formats — no network is touched.
"""

import copy
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_provider.tw_shareholding_fetcher import (  # noqa: E402
    TwShareholdingFetcher,
    _to_int,
)

_HEADER = "資料日期,證券代號,持股分級,人數,股數,占集保庫存數比例%\n"

# --- real TDCC row set for 2330 台積電 @ 20260626 (trimmed to the 17 official tiers) ---
_TSMC_ROWS = [
    (1, 2344648, 270636726, "1.04"),
    (2, 436608, 832879585, "3.21"),
    (3, 50777, 364018815, "1.40"),
    (4, 16824, 206436112, "0.79"),
    (5, 7898, 138937505, "0.53"),
    (6, 7606, 186372616, "0.71"),
    (7, 3561, 123533016, "0.47"),
    (8, 2031, 91648916, "0.35"),
    (9, 4047, 282842631, "1.09"),
    (10, 2008, 281471591, "1.08"),
    (11, 1327, 372898750, "1.43"),
    (12, 552, 270219173, "1.04"),
    (13, 355, 246477946, "0.95"),
    (14, 214, 191121152, "0.73"),
    (15, 1482, 22072875533, "85.11"),
    (16, 0, 0, "0.00"),
    (17, 2879938, 25932370067, "100.00"),
]


def _csv_for(code: str, rows, date: str = "20260626") -> str:
    lines = [_HEADER]
    for tier, holders, shares, pct in rows:
        lines.append(f"{date},{code},{tier},{holders},{shares},{pct}\n")
    return "".join(lines)


def _resp(text: str, encoding: str = "utf-8-sig"):
    resp = MagicMock()
    resp.status_code = 200
    resp.content = text.encode(encoding)
    resp.raise_for_status.return_value = None
    return resp


def _fetcher():
    return TwShareholdingFetcher()


class TestPureHelpers(unittest.TestCase):
    def test_to_int_strips_whitespace_and_preserves_value(self):
        self.assertEqual(_to_int("1482"), 1482)
        self.assertEqual(_to_int(" 0 "), 0)
        for blank in ("", None, "n/a", "abc"):
            self.assertIsNone(_to_int(blank), blank)


class TestTdccParsing(unittest.TestCase):
    def test_tsmc_big_holder_and_retail_breakdown(self):
        csv_text = _csv_for("2330  ", _TSMC_ROWS)  # code padded, as served live
        with patch("data_provider.tw_shareholding_fetcher.requests.get", return_value=_resp(csv_text)):
            rec = _fetcher().get_shareholding_concentration("2330.TW")
        self.assertIsNotNone(rec)
        self.assertEqual(rec["stock_code"], "2330")
        self.assertEqual(rec["date"], "20260626")
        self.assertEqual(rec["source"], "TDCC-OpenData")
        self.assertEqual(rec["unit"], "shares")
        self.assertEqual(rec["total_holders"], 2879938)
        self.assertEqual(rec["total_shares"], 25932370067)
        # tier 15 (1,000,001+ 股) = 千張大戶
        self.assertEqual(rec["big_holder_count"], 1482)
        self.assertAlmostEqual(rec["big_holder_pct"], 85.11, places=1)
        # tiers 1-8 summed (retail, <= 50,000 股)
        expected_retail_holders = sum(h for tier, h, _, _ in _TSMC_ROWS if tier <= 8)
        expected_retail_shares = sum(s for tier, _, s, _ in _TSMC_ROWS if tier <= 8)
        self.assertEqual(rec["retail_holder_count"], expected_retail_holders)
        self.assertAlmostEqual(
            rec["retail_pct"], round(expected_retail_shares / rec["total_shares"] * 100, 2), places=1
        )

    def test_bare_code_lookup_matches_dotted_query(self):
        csv_text = _csv_for("2330", _TSMC_ROWS)
        with patch("data_provider.tw_shareholding_fetcher.requests.get", return_value=_resp(csv_text)):
            f = _fetcher()
            rec_tw = f.get_shareholding_concentration("2330.TW")
            rec_two = f.get_shareholding_concentration("2330.TWO")
            rec_bare = f.get_shareholding_concentration("2330")
        self.assertEqual(rec_tw["stock_code"], "2330")
        self.assertEqual(rec_two["stock_code"], "2330")
        self.assertEqual(rec_bare["stock_code"], "2330")

    def test_unknown_stock_returns_none(self):
        csv_text = _csv_for("2330", _TSMC_ROWS)
        with patch("data_provider.tw_shareholding_fetcher.requests.get", return_value=_resp(csv_text)):
            self.assertIsNone(_fetcher().get_shareholding_concentration("9999.TW"))

    def test_empty_code_returns_none_without_fetching(self):
        with patch("data_provider.tw_shareholding_fetcher.requests.get") as mock_get:
            self.assertIsNone(_fetcher().get_shareholding_concentration(""))
            mock_get.assert_not_called()


class TestMissingTierAndZeroSharesFailOpen(unittest.TestCase):
    def test_missing_total_tier_drops_row(self):
        rows = [r for r in _TSMC_ROWS if r[0] != 17]  # drop tier 17 (合計)
        csv_text = _csv_for("2330", rows)
        with patch("data_provider.tw_shareholding_fetcher.requests.get", return_value=_resp(csv_text)):
            self.assertIsNone(_fetcher().get_shareholding_concentration("2330.TW"))

    def test_missing_big_holder_tier_drops_row(self):
        rows = [r for r in _TSMC_ROWS if r[0] != 15]  # drop tier 15 (千張大戶)
        csv_text = _csv_for("2330", rows)
        with patch("data_provider.tw_shareholding_fetcher.requests.get", return_value=_resp(csv_text)):
            self.assertIsNone(_fetcher().get_shareholding_concentration("2330.TW"))

    def test_missing_retail_tier_drops_row(self):
        rows = [r for r in _TSMC_ROWS if r[0] != 3]  # drop a retail-range tier
        csv_text = _csv_for("2330", rows)
        with patch("data_provider.tw_shareholding_fetcher.requests.get", return_value=_resp(csv_text)):
            self.assertIsNone(_fetcher().get_shareholding_concentration("2330.TW"))

    def test_zero_total_shares_does_not_fabricate_pct(self):
        rows = copy.deepcopy(_TSMC_ROWS)
        rows[-1] = (17, 0, 0, "0.00")  # 合計股數 = 0 (delisted / no custody)
        csv_text = _csv_for("2330", rows)
        with patch("data_provider.tw_shareholding_fetcher.requests.get", return_value=_resp(csv_text)):
            self.assertIsNone(_fetcher().get_shareholding_concentration("2330.TW"))

    def test_missing_header_fails_open(self):
        bad_csv = "not,the,right,columns\n1,2,3,4\n"
        with patch("data_provider.tw_shareholding_fetcher.requests.get", return_value=_resp(bad_csv)):
            self.assertIsNone(_fetcher().get_shareholding_concentration("2330.TW"))

    def test_empty_result_not_cached_and_retried(self):
        with patch(
            "data_provider.tw_shareholding_fetcher.requests.get", return_value=_resp(_HEADER)
        ) as mock_get:
            f = _fetcher()
            self.assertIsNone(f.get_shareholding_concentration("2330.TW"))
            self.assertIsNone(f.get_shareholding_concentration("2330.TW"))
            self.assertEqual(mock_get.call_count, 2)  # empty not cached -> re-fetched


class TestCachingAndHttpError(unittest.TestCase):
    def test_whole_market_cached_single_fetch(self):
        csv_text = _csv_for("2330", _TSMC_ROWS)
        with patch(
            "data_provider.tw_shareholding_fetcher.requests.get", return_value=_resp(csv_text)
        ) as mock_get:
            f = _fetcher()
            f.get_shareholding_concentration("2330.TW")
            f.get_shareholding_concentration("2330.TWO")  # same snapshot -> cache hit
            self.assertEqual(mock_get.call_count, 1)

    def test_network_error_fails_open(self):
        with patch(
            "data_provider.tw_shareholding_fetcher.requests.get", side_effect=ConnectionError("boom")
        ):
            self.assertIsNone(_fetcher().get_shareholding_concentration("2330.TW"))

    def test_http_error_fails_open(self):
        import requests as _rq

        resp = MagicMock()
        resp.raise_for_status.side_effect = _rq.HTTPError("503 Service Unavailable")
        with patch("data_provider.tw_shareholding_fetcher.requests.get", return_value=resp):
            self.assertIsNone(_fetcher().get_shareholding_concentration("2330.TW"))


if __name__ == "__main__":
    unittest.main()