# -*- coding: utf-8 -*-
"""
===================================
台股筹码分布（本地估算）
===================================

台湾没有类似 akshare `stock_cyq_em` / tushare `cyq_chips` 的官方/第三方筹码分布
接口——那两个接口本质上也只是把对方服务器已经算好的成本分布表拿回来做百分位提取
（见 `tushare_fetcher.py` 的 `compute_cyq_metrics`）。筹码分布本身在任何市场都没有
"官方 ground truth"，都是从历史成交数据反推出来的估算值；本模块实现的是同一类
工具（通达信/东方财富等）内部常用的"换手率衰减"算法，只是数据源换成了台股可拿到的
日线 OHLCV + 流通股数（yfinance）。

纯计算，不依赖网络/yfinance，方便用合成数据做单元测试。
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

LOOKBACK_TRADING_DAYS = 120
MIN_REQUIRED_TRADING_DAYS = 20
PRICE_BINS = 200


def _percentile_price(bin_prices: np.ndarray, cumsum: np.ndarray, target_pct: float) -> float:
    """返回累积占比首次达到 target_pct 处的价格（与 tushare compute_cyq_metrics 同一算法）。"""
    idx = int(np.searchsorted(cumsum, target_pct))
    idx = min(idx, len(bin_prices) - 1)
    return float(bin_prices[idx])


def estimate_tw_chip_distribution(
    history_df: pd.DataFrame,
    shares_outstanding: float,
    current_price: float,
) -> Optional[Dict[str, Any]]:
    """
    基于历史日线 OHLCV 和流通股数，用换手率衰减模型估算筹码分布。

    已知局限：yfinance 的 auto_adjust=True 会调整价格但不会回溯调整历史 Volume，
    若窗口内发生股票分割，较早几天的换手率会被低估——对台股大中型股在半年窗口内
    是低概率场景，v1 不处理，仅记录此局限。

    Args:
        history_df: 标准列 date/open/high/low/close/volume，按时间正序排列
        shares_outstanding: 流通股数
        current_price: 当前价（用于计算获利比例）

    Returns:
        与 ChipDistribution 字段对齐的 dict（profit_ratio/avg_cost/cost_90_low/
        cost_90_high/concentration_90/cost_70_low/cost_70_high/concentration_70），
        数据不足或无效时返回 None。
    """
    if shares_outstanding is None or shares_outstanding <= 0:
        return None

    required_cols = {"open", "high", "low", "close", "volume"}
    if history_df is None or not required_cols.issubset(set(history_df.columns)):
        return None

    df = history_df.dropna(subset=["high", "low", "volume"]).copy()
    if len(df) < MIN_REQUIRED_TRADING_DAYS:
        return None

    lows = df["low"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    volumes = df["volume"].to_numpy(dtype=float)

    price_min = float(np.min(lows))
    price_max = float(np.max(highs))
    if price_max <= price_min:
        # 整个窗口价格没有波动（例如恒定一字价），退化为单点分布。
        # 向下（而非向上）扩展价格区间，确保所有桶中心都 <= 原始平价，
        # 这样当 current_price 恰好等于该平价时，获利比例正确趋近 100%
        # 而不会因为分箱离散化把桶中心推到平价之上而误判为 0。
        eps = max(price_min * 1e-4, 1e-6)
        price_max = price_min
        price_min = price_max - eps

    edges = np.linspace(price_min, price_max, PRICE_BINS + 1)
    bin_prices = (edges[:-1] + edges[1:]) / 2.0
    bins = np.zeros(PRICE_BINS, dtype=float)

    for low, high, volume in zip(lows, highs, volumes):
        turnover_rate = min(max(volume, 0.0) / shares_outstanding, 1.0)
        if turnover_rate <= 0:
            continue

        bins *= (1.0 - turnover_rate)

        day_low = min(low, high)
        day_high = max(low, high)
        mask = (bin_prices >= day_low) & (bin_prices <= day_high)
        if not mask.any():
            # 当日区间落在两个桶之间（极小概率），退化为落到最近的桶
            nearest = int(np.argmin(np.abs(bin_prices - (day_low + day_high) / 2.0)))
            bins[nearest] += turnover_rate
        else:
            bins[mask] += turnover_rate / mask.sum()

    total = bins.sum()
    if total <= 0:
        return None

    norm_percent = bins / total * 100.0
    cumsum = np.cumsum(norm_percent)

    winner_rate = float(norm_percent[bin_prices <= current_price].sum())
    avg_cost = float(np.average(bin_prices, weights=norm_percent))

    cost_90_low = _percentile_price(bin_prices, cumsum, 5.0)
    cost_90_high = _percentile_price(bin_prices, cumsum, 95.0)
    concentration_90 = (
        (cost_90_high - cost_90_low) / (cost_90_high + cost_90_low) * 100.0
        if (cost_90_high + cost_90_low) != 0
        else 0.0
    )

    cost_70_low = _percentile_price(bin_prices, cumsum, 15.0)
    cost_70_high = _percentile_price(bin_prices, cumsum, 85.0)
    concentration_70 = (
        (cost_70_high - cost_70_low) / (cost_70_high + cost_70_low) * 100.0
        if (cost_70_high + cost_70_low) != 0
        else 0.0
    )

    return {
        # profit_ratio/concentration_* 是 0-1 的小数（与 ChipDistribution 既有约定一致，
        # 对齐 tushare compute_cyq_metrics 的 /100 换算），不是 0-100 的百分比。
        "profit_ratio": round(winner_rate / 100.0, 4),
        "avg_cost": round(avg_cost, 4),
        "cost_90_low": round(cost_90_low, 4),
        "cost_90_high": round(cost_90_high, 4),
        "concentration_90": round(concentration_90 / 100.0, 4),
        "cost_70_low": round(cost_70_low, 4),
        "cost_70_high": round(cost_70_high, 4),
        "concentration_70": round(concentration_70 / 100.0, 4),
    }