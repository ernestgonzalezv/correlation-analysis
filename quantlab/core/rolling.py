from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantlab.core.covariance import correlation_matrix
from quantlab.core.panel import ReturnsPanel


@dataclass(frozen=True)
class RollingCorrelation:
    values: np.ndarray
    window: int
    positions: np.ndarray

    @property
    def mean(self) -> float:
        return float(np.mean(self.values))

    @property
    def minimum(self) -> float:
        return float(np.min(self.values))

    @property
    def maximum(self) -> float:
        return float(np.max(self.values))

    @property
    def spread(self) -> float:
        return self.maximum - self.minimum


def mean_pairwise_correlation(corr: np.ndarray) -> float:
    corr = np.asarray(corr, dtype=float)

    if corr.shape[0] < 2:
        raise ValueError("at least 2 series are required for a pairwise mean")

    upper = np.triu_indices_from(corr, k=1)
    return float(corr[upper].mean())


def rolling_mean_correlation(panel: ReturnsPanel, window: int) -> RollingCorrelation:
    if panel.N < 2:
        raise ValueError("at least 2 series are required")

    if window < 4:
        raise ValueError(f"window must be at least 4; got {window}")

    if window > panel.T:
        raise ValueError(
            f"window {window} exceeds the {panel.T} available observations"
        )

    values = []
    positions = []

    for end in range(window, panel.T + 1):
        mask = np.zeros(panel.T, dtype=bool)
        mask[end - window : end] = True
        window_panel = panel.slice_rows(mask)

        try:
            corr = correlation_matrix(window_panel)
        except ValueError:
            continue

        values.append(mean_pairwise_correlation(corr))
        positions.append(end - 1)

    if not values:
        raise ValueError("no window produced a valid correlation matrix")

    return RollingCorrelation(
        values=np.array(values),
        window=window,
        positions=np.array(positions),
    )


def stress_mask(panel: ReturnsPanel, quantile: float = 0.10) -> np.ndarray:
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must lie in (0, 1); got {quantile}")

    portfolio = panel.values.mean(axis=1)
    threshold = np.quantile(portfolio, quantile)

    return portfolio <= threshold


def stress_correlation(
    panel: ReturnsPanel, quantile: float = 0.10
) -> tuple[np.ndarray, int]:
    mask = stress_mask(panel, quantile)
    n_days = int(mask.sum())

    if n_days < 4:
        raise ValueError(
            f"only {n_days} stress observations at quantile {quantile}; "
            "need at least 4"
        )

    return correlation_matrix(panel.slice_rows(mask)), n_days


def correlation_lift_under_stress(
    panel: ReturnsPanel, quantile: float = 0.10
) -> float:
    calm = mean_pairwise_correlation(correlation_matrix(panel))
    stressed, _ = stress_correlation(panel, quantile)

    return mean_pairwise_correlation(stressed) - calm
