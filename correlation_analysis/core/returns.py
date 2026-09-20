from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from correlation_analysis.core.panel import ReturnsPanel


def log_returns(prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)

    if prices.ndim != 2:
        raise ValueError(f"prices must be 2D (T, N); got ndim={prices.ndim}")

    if prices.shape[0] < 2:
        raise ValueError("at least 2 price rows are required")

    if not np.isfinite(prices).all():
        raise ValueError("prices contains NaN or inf; clean it before converting")

    if (prices <= 0).any():
        n_bad = int((prices <= 0).sum())
        raise ValueError(
            f"prices contains {n_bad} values <= 0 where the logarithm is undefined; "
            "use simple differences for series that cross zero"
        )

    return np.log(prices[1:] / prices[:-1])


def simple_returns(prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)

    if prices.ndim != 2:
        raise ValueError(f"prices must be 2D (T, N); got ndim={prices.ndim}")

    if prices.shape[0] < 2:
        raise ValueError("at least 2 price rows are required")

    if (prices[:-1] == 0).any():
        raise ValueError("prices contains zeros; cannot divide")

    return prices[1:] / prices[:-1] - 1.0


def panel_from_prices(
    prices: np.ndarray,
    names: Sequence[str],
    freq: str = "1d",
    index: np.ndarray | None = None,
    method: str = "log",
    periods_per_year: float | None = None,
) -> ReturnsPanel:
    if method == "log":
        values = log_returns(prices)
    elif method == "simple":
        values = simple_returns(prices)
    else:
        raise ValueError(f"method must be 'log' or 'simple'; got '{method}'")

    aligned_index = None if index is None else np.asarray(index)[1:]

    return ReturnsPanel(
        values=values,
        names=tuple(names),
        freq=freq,
        kind="returns",
        index=aligned_index,
        periods_per_year=periods_per_year,
    )


def panel_from_pnl(
    pnl: np.ndarray,
    names: Sequence[str],
    freq: str = "1d",
    index: np.ndarray | None = None,
    periods_per_year: float | None = None,
) -> ReturnsPanel:
    return ReturnsPanel(
        values=np.asarray(pnl, dtype=float),
        names=tuple(names),
        freq=freq,
        kind="pnl",
        index=None if index is None else np.asarray(index),
        periods_per_year=periods_per_year,
    )
