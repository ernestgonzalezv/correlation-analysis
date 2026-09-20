from __future__ import annotations

import numpy as np

from quantlab.core.panel import ReturnsPanel

VARIANCE_TOLERANCE = 1e-14


def center(panel: ReturnsPanel) -> np.ndarray:
    values = panel.values
    return values - values.mean(axis=0)


def covariance_matrix(panel: ReturnsPanel, ddof: int = 1) -> np.ndarray:
    if panel.T - ddof <= 0:
        raise ValueError(
            f"not enough degrees of freedom: T={panel.T}, ddof={ddof}"
        )

    centered = center(panel)
    return (centered.T @ centered) / (panel.T - ddof)


def volatilities(
    panel: ReturnsPanel, ddof: int = 1, annualize: bool = False
) -> np.ndarray:
    std = panel.values.std(axis=0, ddof=ddof)

    if annualize:
        std = std * np.sqrt(panel.periods_per_year)

    return std


def correlation_matrix(panel: ReturnsPanel, ddof: int = 1) -> np.ndarray:
    cov = covariance_matrix(panel, ddof=ddof)
    std = np.sqrt(np.diag(cov))

    constant = np.where(std < VARIANCE_TOLERANCE)[0]
    if constant.size:
        names = [panel.names[i] for i in constant]
        raise ValueError(
            f"these series are constant and have no defined correlation: {names}"
        )

    corr = cov / np.outer(std, std)

    np.fill_diagonal(corr, 1.0)
    corr = (corr + corr.T) / 2.0

    return np.clip(corr, -1.0, 1.0)
