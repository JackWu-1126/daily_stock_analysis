# -*- coding: utf-8 -*-
"""TwShareholdingFetcher — Taiwan 集保户股权集中度 (TDCC shareholding-tier concentration).

Data-layer only, ``tw``-only, strictly additive. Self-contained: fetches, parses,
caches and fail-opens. Mirrors ``tw_institutional_fetcher.py``'s contract.

Source (政府開放資料, 政府資料開放授權條款第 1 版 / OGDL v1, commercial-safe, no key):
  - TDCC 集保戶股權分散表 (weekly, both TWSE 上市 and TPEx 上櫃 in one feed)
    https://opendata.tdcc.com.tw/getOD.ashx?id=1-5
    (CSV, Big5/UTF-8, columns: 資料日期,證券代號,持股分級,人數,股數,占集保庫存數比例%)

TDCC 持股分級 (shareholding tier) definitions — fixed, official, 17 rows per stock:
  1: 1-999 股            6: 20,001-30,000 股      11: 200,001-400,000 股
  2: 1,000-5,000 股      7: 30,001-40,000 股      12: 400,001-600,000 股
  3: 5,001-10,000 股     8: 40,001-50,000 股      13: 600,001-800,000 股
  4: 10,001-15,000 股    9: 50,001-100,000 股     14: 800,001-1,000,000 股
  5: 15,001-20,000 股    10: 100,001-200,000 股   15: 1,000,001 股以上 (千張大戶)
  16: 差異數 (reconciliation adjustment; usually 0)
  17: 合計 (grand total; always 100%)

This is a holder/share-COUNT concentration metric (how many shares each
ownership-size bracket holds), NOT a cost-basis distribution — it carries no
price information and is therefore a separate concept from the AkShare/Tushare
``ChipDistribution`` (avg_cost / cost_90_low / cost_90_high / concentration_90 /
profit_ratio, all price-derived). It must never be mapped onto that schema.

Fail-open contract: any network error, rate-limit, empty response, unexpected
shape or missing tier returns ``None`` (no data) — it never raises into the
caller, so the analysis main flow is never interrupted.
"""

from __future__ import annotations

import csv
import io
import logging
import threading
import time
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

_TDCC_URL = "https://opendata.tdcc.com.tw/getOD.ashx?id=1-5"
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

_DATE_COL = "資料日期"
_CODE_COL = "證券代號"
_TIER_COL = "持股分級"
_HOLDERS_COL = "人數"
_SHARES_COL = "股數"

_BIG_HOLDER_TIER = 15  # 1,000,001 股以上 (千張大戶)
_RETAIL_TIERS = range(1, 9)  # tiers 1-8: 1-50,000 股
_TOTAL_TIER = 17  # 合計


def _to_int(value: Any) -> Optional[int]:
    """Parse a TDCC numeric cell to int. Empty / non-numeric -> None (never a fabricated 0)."""
    try:
        text = str(value).strip()
    except (TypeError, ValueError):
        return None
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        try:
            return int(float(text))
        except ValueError:
            return None


class TwShareholdingFetcher:
    """Fetch Taiwan per-stock TDCC shareholding-tier concentration (bare code, no suffix)."""

    name = "TwShareholdingFetcher"

    def __init__(
        self,
        *,
        cache_ttl_seconds: int = 86400,
        timeout: int = 20,
    ) -> None:
        # Whole-market single-snapshot cache; TDCC publishes once per week.
        self._cache: Optional[Dict[str, dict]] = None
        self._cache_at: float = 0.0
        self._cache_ttl = cache_ttl_seconds
        self._timeout = timeout
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ public
    def get_shareholding_concentration(self, stock_code: str) -> Optional[dict]:
        """Return the normalized shareholding-concentration record for one TW stock, or ``None``.

        ``stock_code`` may carry an explicit ``.TW`` / ``.TWO`` suffix (stripped) or be bare;
        a non-TW-shaped code still just looks up the bare base against the TDCC table (which
        covers both 上市/上櫃), so this stays permissive at the fetcher layer — market gating
        happens at the call site, same as ``TwInstitutionalFetcher``. Fail-open: any error
        returns ``None``.
        """
        base = self._base_code(stock_code)
        if not base:
            return None
        try:
            table = self._whole_market()
        except Exception as exc:  # noqa: BLE001 - fail-open by contract
            logger.info("[tw-shareholding] fetch failed code=%s: %s", stock_code, exc)
            return None
        if not table:
            return None
        return table.get(base)

    # ------------------------------------------------------------------ routing
    @staticmethod
    def _base_code(stock_code: Any) -> str:
        upper = str(stock_code or "").strip().upper()
        if upper.endswith(".TWO") or upper.endswith(".TW"):
            return upper.rsplit(".", 1)[0]
        return upper

    # -------------------------------------------------- whole-market cached fetch
    def _whole_market(self) -> Dict[str, dict]:
        """Whole-market table {code: record}, cached for ``cache_ttl_seconds``.

        May raise on network / HTTP errors -- the public method wraps this in a
        fail-open try/except. Only a non-empty result is cached, so a transient
        failure is retried on the next call rather than serving an empty table
        for the whole TTL.
        """
        with self._lock:
            if self._cache is not None and (time.time() - self._cache_at) < self._cache_ttl:
                return self._cache
        table = self._fetch_tdcc()
        if table:
            with self._lock:
                self._cache = table
                self._cache_at = time.time()
        return table

    def _fetch_tdcc(self) -> Dict[str, dict]:
        resp = requests.get(_TDCC_URL, headers={"User-Agent": _UA}, timeout=self._timeout)
        resp.raise_for_status()
        # TDCC serves UTF-8 with a BOM; utf-8-sig strips it transparently.
        text = resp.content.decode("utf-8-sig", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames is None:
            return {}
        required = (_DATE_COL, _CODE_COL, _TIER_COL, _HOLDERS_COL, _SHARES_COL)
        if any(col not in reader.fieldnames for col in required):
            logger.info("[tw-shareholding] TDCC header missing/renamed -> fail-open")
            return {}

        # code -> {tier: (holders, shares)}
        raw: Dict[str, Dict[int, tuple]] = {}
        raw_date: Dict[str, str] = {}
        for row in reader:
            code = str(row.get(_CODE_COL, "")).strip()
            tier = _to_int(row.get(_TIER_COL))
            holders = _to_int(row.get(_HOLDERS_COL))
            shares = _to_int(row.get(_SHARES_COL))
            date = str(row.get(_DATE_COL, "")).strip()
            if not code or tier is None or holders is None or shares is None:
                continue
            raw.setdefault(code, {})[tier] = (holders, shares)
            if date:
                raw_date[code] = date

        table: Dict[str, dict] = {}
        for code, tiers in raw.items():
            record = self._build_record(code, raw_date.get(code), tiers)
            if record is not None:
                table[code] = record
        return table

    # -------------------------------------------------------------- normalize
    @staticmethod
    def _build_record(
        code: str, date: Optional[str], tiers: Dict[int, tuple]
    ) -> Optional[dict]:
        total = tiers.get(_TOTAL_TIER)
        big = tiers.get(_BIG_HOLDER_TIER)
        if total is None or big is None or date is None:
            return None
        total_holders, total_shares = total
        big_holders, big_shares = big
        if not total_shares:  # avoid a divide-by-zero fabricating a 0.0%
            return None

        retail_holders = 0
        retail_shares = 0
        missing_retail_tier = False
        for tier_num in _RETAIL_TIERS:
            tier = tiers.get(tier_num)
            if tier is None:
                missing_retail_tier = True
                break
            retail_holders += tier[0]
            retail_shares += tier[1]
        if missing_retail_tier:
            return None

        return {
            "stock_code": code,
            "date": date,
            "source": "TDCC-OpenData",
            "unit": "shares",
            "big_holder_pct": round(big_shares / total_shares * 100, 2),
            "big_holder_count": big_holders,
            "retail_pct": round(retail_shares / total_shares * 100, 2),
            "retail_holder_count": retail_holders,
            "total_holders": total_holders,
            "total_shares": total_shares,
        }