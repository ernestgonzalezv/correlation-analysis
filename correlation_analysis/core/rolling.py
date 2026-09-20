from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from correlation_analysis.core.covariance import VARIANCE_TOLERANCE, correlation_matrix
from correlation_analysis.core.panel import ReturnsPanel

MIN_WINDOW = 4


@dataclass(frozen=True)
class RollingCorrelation:
    values: np.ndarray
    window: int
    positions: np.ndarray
    labels: np.ndarray | None = None

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

    @property
    def n_windows(self) -> int:
        return len(self.values)


def mean_pairwise_correlation(corr: np.ndarray) -> float:
    corr = np.asarray(corr, dtype=float)

    if corr.shape[0] < 2:
        raise ValueError("at least 2 series are required for a pairwise mean")

    return float(corr[np.triu_indices_from(corr, k=1)].mean())


def _window_sums(series: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.concatenate(([0.0], np.cumsum(series)))
    return cumulative[window:] - cumulative[:-window]


def rolling_mean_correlation(panel: ReturnsPanel, window: int) -> RollingCorrelation:
    if panel.N < 2:
        raise ValueError("at least 2 series are required")

    if window < MIN_WINDOW:
        raise ValueError(f"window must be at least {MIN_WINDOW}; got {window}")

    if window > panel.T:
        raise ValueError(
            f"window {window} exceeds the {panel.T} available observations"
        )

    values = panel.values - panel.values.mean(axis=0)
    n_series = panel.N
    n = float(window)

    sums = np.column_stack(
        [_window_sums(values[:, k], window) for k in range(n_series)]
    )
    squares = np.column_stack(
        [_window_sums(values[:, k] ** 2, window) for k in range(n_series)]
    )

    variances = (squares - sums**2 / n) / (n - 1.0)

    degenerate = np.where((variances <= VARIANCE_TOLERANCE).any(axis=0))[0]
    if degenerate.size:
        names = [panel.names[i] for i in degenerate]
        raise ValueError(
            f"these series are constant inside at least one window and have no "
            f"defined correlation there: {names}"
        )

    total = np.zeros(sums.shape[0])
    n_pairs = 0

    for i in range(n_series):
        for j in range(i + 1, n_series):
            cross = _window_sums(values[:, i] * values[:, j], window)
            covariance = (cross - sums[:, i] * sums[:, j] / n) / (n - 1.0)
            total += covariance / np.sqrt(variances[:, i] * variances[:, j])
            n_pairs += 1

    positions = np.arange(window - 1, panel.T)

    return RollingCorrelation(
        values=np.clip(total / n_pairs, -1.0, 1.0),
        window=window,
        positions=positions,
        labels=None if panel.index is None else panel.index[positions],
    )


def stress_mask(panel: ReturnsPanel, quantile: float = 0.10) -> np.ndarray:
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must lie in (0, 1); got {quantile}")

    values = panel.values
    deviations = values.std(axis=0, ddof=1)

    constant = np.where(deviations < VARIANCE_TOLERANCE)[0]
    if constant.size:
        names = [panel.names[i] for i in constant]
        raise ValueError(f"these series are constant and cannot be ranked: {names}")

    standardized = (values - values.mean(axis=0)) / deviations
    portfolio = standardized.mean(axis=1)

    return portfolio <= np.quantile(portfolio, quantile)


def stress_correlation(
    panel: ReturnsPanel, quantile: float = 0.10
) -> tuple[np.ndarray, int]:
    mask = stress_mask(panel, quantile)
    n_days = int(mask.sum())

    if n_days < MIN_WINDOW:
        raise ValueError(
            f"only {n_days} stress observations at quantile {quantile}; "
            f"need at least {MIN_WINDOW}"
        )

    return correlation_matrix(panel.slice_rows(mask)), n_days


def correlation_lift_under_stress(
    panel: ReturnsPanel, quantile: float = 0.10
) -> float:
    calm = mean_pairwise_correlation(correlation_matrix(panel))
    stressed, _ = stress_correlation(panel, quantile)

    return mean_pairwise_correlation(stressed) - calm
