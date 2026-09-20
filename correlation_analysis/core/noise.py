from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

DEFAULT_CONFIDENCE_LEVEL = 0.95
MIN_OBSERVATION_RATIO = 10.0
MAX_ABS_CORRELATION = 1.0 - 1e-9


@dataclass(frozen=True)
class NoiseBounds:
    lower: float
    upper: float
    ratio: float

    def is_signal(self, eigenvalue: float) -> bool:
        return eigenvalue > self.upper


def marchenko_pastur_bounds(n_obs: int, n_series: int) -> NoiseBounds:
    if n_obs <= n_series:
        raise ValueError(
            f"Marchenko-Pastur requires n_obs > n_series; "
            f"got n_obs={n_obs}, n_series={n_series}"
        )

    ratio = n_series / n_obs
    root = np.sqrt(ratio)

    return NoiseBounds(
        lower=float((1.0 - root) ** 2),
        upper=float((1.0 + root) ** 2),
        ratio=ratio,
    )


def signal_mask(eigenvalues: np.ndarray, n_obs: int, n_series: int) -> np.ndarray:
    bounds = marchenko_pastur_bounds(n_obs, n_series)
    return np.asarray(eigenvalues, dtype=float) > bounds.upper


def count_signal_factors(eigenvalues: np.ndarray, n_obs: int, n_series: int) -> int:
    return int(signal_mask(eigenvalues, n_obs, n_series).sum())


def fisher_z_stderr(n_obs: int) -> float:
    if n_obs <= 3:
        raise ValueError(f"n_obs must be greater than 3; got {n_obs}")

    return float(1.0 / np.sqrt(n_obs - 3))


def correlation_stderr(correlation: np.ndarray, n_obs: int) -> np.ndarray:
    if n_obs <= 1:
        raise ValueError(f"n_obs must be greater than 1; got {n_obs}")

    correlation = np.asarray(correlation, dtype=float)
    return (1.0 - correlation**2) / np.sqrt(n_obs - 1)


def two_sided_z_score(level: float) -> float:
    if not 0.0 < level < 1.0:
        raise ValueError(f"confidence level must lie in (0, 1); got {level}")

    return float(norm.ppf(0.5 + level / 2.0))


def _is_square_matrix(array: np.ndarray) -> bool:
    return array.ndim == 2 and array.shape[0] == array.shape[1]


def correlation_interval(
    correlation: np.ndarray,
    n_obs: int,
    level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> tuple[np.ndarray, np.ndarray]:
    margin = two_sided_z_score(level) * fisher_z_stderr(n_obs)

    correlation = np.asarray(correlation, dtype=float)
    bounded = np.clip(correlation, -MAX_ABS_CORRELATION, MAX_ABS_CORRELATION)

    z = np.arctanh(bounded)

    low = np.tanh(z - margin)
    high = np.tanh(z + margin)

    if _is_square_matrix(correlation):
        diagonal = np.diag_indices_from(correlation)
        low[diagonal] = correlation[diagonal]
        high[diagonal] = correlation[diagonal]

    return low, high


def is_estimate_reliable(
    n_obs: int, n_series: int, min_ratio: float = MIN_OBSERVATION_RATIO
) -> bool:
    return (n_obs / n_series) >= min_ratio
