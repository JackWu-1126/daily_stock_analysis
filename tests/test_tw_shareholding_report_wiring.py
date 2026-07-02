# -*- coding: utf-8 -*-
"""Report-wiring tests: tw 集保户股权集中度 (TDCC shareholding-tier concentration) into
the offshore ``shareholding_concentration`` block.

Pins the same contract as tests/test_tw_institution_report_wiring.py:
  - tw with data        -> shareholding_concentration coverage 'ok', raw pct figures surfaced.
  - tw fetch-failed/None -> stays 'not_supported' (fail-open, main flow alive).
  - us/hk/jp/kr          -> stays 'not_supported' AND the tw fetcher is never called
                            (strictly-additive: other markets byte-identical).
  - tw shareholding data carries raw holder/share-count figures only — no derived
    signal / score, and never conflated with the price-based ChipDistribution schema
    (avg_cost / cost_90_low / cost_90_high / concentration_90 / profit_ratio).

Mirrors tests/test_tw_institution_report_wiring.py's offline pattern.
"""

import os
import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_provider.base import DataFetcherManager

_TW_FETCHER_METHOD = (
    "data_provider.tw_shareholding_fetcher.TwShareholdingFetcher.get_shareholding_concentration"
)

# Shape mirrors TwShareholdingFetcher._build_record (real 2330 @ 20260626).
_FAKE_REC = {
    "stock_code": "2330",
    "date": "20260626",
    "source": "TDCC-OpenData",
    "unit": "shares",
    "big_holder_pct": 85.11,
    "big_holder_count": 1482,
    "retail_pct": 9.59,
    "retail_holder_count": 2869953,
    "total_holders": 2879938,
    "total_shares": 25932370067,
}

_OFFSHORE_CFG = SimpleNamespace(
    enable_fundamental_pipeline=True,
    fundamental_cache_ttl_seconds=0,
    fundamental_stage_timeout_seconds=1.5,
    fundamental_fetch_timeout_seconds=0.8,
    fundamental_retry_max=1,
)

_EMPTY_BUNDLE = {
    "status": "not_supported",
    "growth": {},
    "earnings": {},
    "belong_boards": [],
    "source_chain": [],
    "errors": [],
}


class TestTwShareholdingReportWiring(unittest.TestCase):
    def _context(self, code, shareholding_return=None, shareholding_side_effect=None):
        """Run get_fundamental_context(code) offline; returns (ctx, tw_fetcher_mock)."""
        manager = DataFetcherManager(fetchers=[])
        kwargs = {}
        if shareholding_side_effect is not None:
            kwargs["side_effect"] = shareholding_side_effect
        else:
            kwargs["return_value"] = shareholding_return
        with patch("src.config.get_config", return_value=_OFFSHORE_CFG), \
                patch.object(manager, "get_realtime_quote", return_value=None), \
                patch(
                    "data_provider.yfinance_fundamental_adapter.YfinanceFundamentalAdapter.get_fundamental_bundle",
                    return_value=_EMPTY_BUNDLE,
                ), \
                patch(
                    "data_provider.tw_institutional_fetcher.TwInstitutionalFetcher.get_institutional_net",
                    return_value=None,
                ), \
                patch(_TW_FETCHER_METHOD, **kwargs) as tw_mock:
            ctx = manager.get_fundamental_context(code)
        return ctx, tw_mock

    # ---- tw with data: shareholding_concentration surfaces the raw pct figures -------
    def test_tw_shareholding_populated_when_fetcher_has_data(self):
        ctx, tw_mock = self._context("2330.TW", shareholding_return=dict(_FAKE_REC))
        self.assertEqual(ctx["market"], "tw")
        self.assertEqual(ctx["coverage"].get("shareholding_concentration"), "ok")
        data = ctx["shareholding_concentration"]["data"]
        self.assertEqual(data["big_holder_pct"], 85.11)
        self.assertEqual(data["big_holder_count"], 1482)
        self.assertEqual(data["retail_pct"], 9.59)
        self.assertEqual(data["retail_holder_count"], 2869953)
        self.assertEqual(data["total_holders"], 2879938)
        self.assertEqual(data["total_shares"], 25932370067)
        self.assertEqual(data["unit"], "shares")
        self.assertEqual(data["source"], "TDCC-OpenData")
        # other offshore blocks untouched
        for block in ("capital_flow", "dragon_tiger", "boards"):
            self.assertEqual(ctx["coverage"].get(block), "not_supported")
        self.assertNotEqual(ctx["status"], "not_supported")
        tw_mock.assert_called_with("2330.TW")

    # ---- tw fail-open: None -> not_supported, no raise ---------------------------
    def test_tw_shareholding_fail_open_when_fetcher_returns_none(self):
        ctx, _ = self._context("2330.TW", shareholding_return=None)
        self.assertEqual(ctx["coverage"].get("shareholding_concentration"), "not_supported")
        self.assertEqual(ctx["shareholding_concentration"].get("data"), {})

    # ---- tw fail-open: fetcher raises -> not_supported, main flow uninterrupted ---
    def test_tw_shareholding_fail_open_when_fetcher_raises(self):
        ctx, _ = self._context("2330.TW", shareholding_side_effect=RuntimeError("boom"))
        self.assertEqual(ctx["coverage"].get("shareholding_concentration"), "not_supported")
        self.assertEqual(ctx["shareholding_concentration"].get("data"), {})
        self.assertEqual(ctx["market"], "tw")

    # ---- strictly-additive: us is byte-identical AND the tw fetcher is never called --
    def test_us_shareholding_unchanged_and_tw_fetcher_not_called(self):
        ctx, tw_mock = self._context("AAPL", shareholding_return=dict(_FAKE_REC))
        self.assertEqual(ctx["market"], "us")
        self.assertEqual(ctx["coverage"].get("shareholding_concentration"), "not_supported")
        self.assertEqual(ctx["shareholding_concentration"].get("data"), {})
        self.assertEqual(tw_mock.call_count, 0)

    # ---- every other offshore market (hk/jp/kr) untouched + fetcher unused -------
    def test_other_offshore_markets_shareholding_unchanged(self):
        for code, market in (("0700.HK", "hk"), ("7203.T", "jp"), ("005930.KS", "kr")):
            ctx, tw_mock = self._context(code, shareholding_return=dict(_FAKE_REC))
            self.assertEqual(ctx["market"], market, f"{code} routed to {ctx['market']}")
            self.assertEqual(ctx["coverage"].get("shareholding_concentration"), "not_supported")
            self.assertEqual(tw_mock.call_count, 0)

    # ---- fail-open on a fetcher WIRING/init failure (not just a fetch failure) ----
    def test_tw_shareholding_fail_open_when_fetcher_init_raises(self):
        manager = DataFetcherManager(fetchers=[])
        with patch("src.config.get_config", return_value=_OFFSHORE_CFG), \
                patch.object(manager, "get_realtime_quote", return_value=None), \
                patch(
                    "data_provider.yfinance_fundamental_adapter.YfinanceFundamentalAdapter.get_fundamental_bundle",
                    return_value=_EMPTY_BUNDLE,
                ), \
                patch(
                    "data_provider.tw_institutional_fetcher.TwInstitutionalFetcher.get_institutional_net",
                    return_value=None,
                ), \
                patch(
                    "data_provider.tw_shareholding_fetcher.TwShareholdingFetcher",
                    side_effect=RuntimeError("init boom"),
                ):
            ctx = manager.get_fundamental_context("2330.TW")  # must NOT raise
        self.assertEqual(ctx["market"], "tw")
        self.assertEqual(ctx["coverage"].get("shareholding_concentration"), "not_supported")
        self.assertEqual(ctx["shareholding_concentration"].get("data"), {})

    # ---- a record missing a core figure is NOT shown as a clean 'ok' -------------
    def test_tw_shareholding_not_ok_when_core_field_missing(self):
        broken = dict(_FAKE_REC, big_holder_pct=None)
        ctx, _ = self._context("2330.TW", shareholding_return=broken)
        self.assertEqual(ctx["coverage"].get("shareholding_concentration"), "not_supported")
        self.assertEqual(ctx["shareholding_concentration"].get("data"), {})

    # ---- a slow fetch must NOT push the analysis past the fundamental stage budget --
    def test_tw_shareholding_fetch_respects_stage_timeout(self):
        slow_cfg = SimpleNamespace(
            enable_fundamental_pipeline=True,
            fundamental_cache_ttl_seconds=0,
            fundamental_stage_timeout_seconds=0.3,
            fundamental_fetch_timeout_seconds=0.3,
            fundamental_retry_max=1,
        )
        manager = DataFetcherManager(fetchers=[])

        def _slow(_code):
            time.sleep(2.0)
            return dict(_FAKE_REC)

        start = time.time()
        with patch("src.config.get_config", return_value=slow_cfg), \
                patch.object(manager, "get_realtime_quote", return_value=None), \
                patch(
                    "data_provider.yfinance_fundamental_adapter.YfinanceFundamentalAdapter.get_fundamental_bundle",
                    return_value=_EMPTY_BUNDLE,
                ), \
                patch(
                    "data_provider.tw_institutional_fetcher.TwInstitutionalFetcher.get_institutional_net",
                    return_value=None,
                ), \
                patch(_TW_FETCHER_METHOD, side_effect=_slow):
            ctx = manager.get_fundamental_context("2330.TW")
        elapsed = time.time() - start
        self.assertLess(elapsed, 1.5, f"shareholding fetch ignored the stage timeout ({elapsed:.2f}s)")
        self.assertEqual(ctx["coverage"].get("shareholding_concentration"), "not_supported")

    # ---- negative: shareholding data carries raw figures only, no derived signal,
    #      and MUST NOT be conflated with the price-based ChipDistribution schema ---
    def test_tw_shareholding_data_has_no_derived_signal_or_chip_schema_leak(self):
        ctx, _ = self._context("2330.TW", shareholding_return=dict(_FAKE_REC))
        data = ctx["shareholding_concentration"]["data"]
        self.assertEqual(
            set(data.keys()),
            {
                "big_holder_pct", "big_holder_count", "retail_pct", "retail_holder_count",
                "total_holders", "total_shares", "unit", "date", "source",
            },
        )
        forbidden = (
            "signal", "score", "weight", "normalized", "rating",
            # price-based ChipDistribution fields must never leak in here.
            "avg_cost", "cost_90_low", "cost_90_high", "concentration_90",
            "cost_70_low", "cost_70_high", "concentration_70", "profit_ratio",
        )
        for key in forbidden:
            self.assertNotIn(key, data, f"forbidden key '{key}' leaked into shareholding_concentration data")

    # ---- institution and shareholding_concentration are independent blocks ------
    def test_institution_failure_does_not_affect_shareholding_concentration(self):
        """tw_institutional_fetcher failing (mocked to None in _context) must not
        prevent shareholding_concentration from surfacing its own data."""
        ctx, _ = self._context("2330.TW", shareholding_return=dict(_FAKE_REC))
        self.assertEqual(ctx["coverage"].get("institution"), "not_supported")
        self.assertEqual(ctx["coverage"].get("shareholding_concentration"), "ok")


if __name__ == "__main__":
    unittest.main()