from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from correlation_analysis.core.covariance import correlation_matrix, volatilities
from correlation_analysis.core.noise import (
    DEFAULT_CONFIDENCE_LEVEL,
    MIN_OBSERVATION_RATIO,
    NoiseBounds,
    correlation_interval,
    count_signal_factors,
    is_estimate_reliable,
    marchenko_pastur_bounds,
)
from correlation_analysis.core.panel import ReturnsPanel
from correlation_analysis.core.rolling import (
    MIN_WINDOW,
    RollingCorrelation,
    mean_pairwise_correlation,
    rolling_mean_correlation,
    stress_correlation,
)
from correlation_analysis.core.spectral import SpectralResult, decompose

DEFAULT_STRESS_QUANTILE = 0.10
DEFAULT_WINDOW_DIVISOR = 6
DEFAULT_MIN_ROLLING_WINDOW = 20
DEFAULT_CONCENTRATION_THRESHOLD = 0.50
DEFAULT_STRESS_LIFT_THRESHOLD = 0.15


@dataclass(frozen=True)
class AnalysisConfig:
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
    stress_quantile: float = DEFAULT_STRESS_QUANTILE
    rolling_window: int | None = None
    window_divisor: int = DEFAULT_WINDOW_DIVISOR
    min_rolling_window: int = DEFAULT_MIN_ROLLING_WINDOW
    min_observation_ratio: float = MIN_OBSERVATION_RATIO
    concentration_threshold: float = DEFAULT_CONCENTRATION_THRESHOLD
    stress_lift_threshold: float = DEFAULT_STRESS_LIFT_THRESHOLD

    def __post_init__(self) -> None:
        if self.window_divisor < 1:
            raise ValueError(
                f"window_divisor must be at least 1; got {self.window_divisor}"
            )

        if self.min_rolling_window < MIN_WINDOW:
            raise ValueError(
                f"min_rolling_window must be at least {MIN_WINDOW}; "
                f"got {self.min_rolling_window}"
            )

        if self.rolling_window is not None and self.rolling_window < MIN_WINDOW:
            raise ValueError(
                f"rolling_window must be at least {MIN_WINDOW}; "
                f"got {self.rolling_window}"
            )

        if not 0.0 < self.concentration_threshold <= 1.0:
            raise ValueError(
                f"concentration_threshold must lie in (0, 1]; "
                f"got {self.concentration_threshold}"
            )

        if self.min_observation_ratio <= 0:
            raise ValueError(
                f"min_observation_ratio must be positive; "
                f"got {self.min_observation_ratio}"
            )

    def resolve_window(self, n_obs: int) -> int | None:
        if self.rolling_window is not None:
            return self.rolling_window if self.rolling_window <= n_obs else None

        window = max(self.min_rolling_window, n_obs // self.window_divisor)
        return window if window <= n_obs else None


@dataclass(frozen=True)
class DiversificationReport:
    panel: ReturnsPanel
    config: AnalysisConfig
    correlation: np.ndarray
    correlation_low: np.ndarray
    correlation_high: np.ndarray
    mean_correlation: float
    annualized_volatility: np.ndarray
    spectral: SpectralResult
    noise_bounds: NoiseBounds | None
    signal_factors: int | None
    estimate_is_reliable: bool
    rolling: RollingCorrelation | None
    rolling_skipped: str | None
    stress_correlation: np.ndarray | None
    stress_observations: int | None
    stress_lift: float | None
    stress_skipped: str | None

    @property
    def effective_bets(self) -> float:
        return self.spectral.effective_bets

    @property
    def diversification_ratio(self) -> float:
        return self.effective_bets / self.panel.N

    @property
    def warnings(self) -> list[str]:
        messages = []

        if not self.estimate_is_reliable:
            messages.append(
                f"only {self.panel.T} observations for {self.panel.N} series "
                f"(ratio {self.panel.observation_ratio:.1f}); estimation error "
                f"dominates below a ratio of {self.config.min_observation_ratio:.0f}"
            )

        if self.signal_factors == 0 and not self.estimate_is_reliable:
            messages.append(
                "no eigenvalue exceeds the Marchenko-Pastur bound at this "
                "sample size; genuine independence and insufficient data are "
                "not distinguishable here"
            )

        if self.diversification_ratio < self.config.concentration_threshold:
            messages.append(
                f"{self.panel.N} series behave like {self.effective_bets:.2f} "
                "independent bets; exposure is more concentrated than the "
                "position count suggests"
            )

        if self.stress_skipped is not None:
            messages.append(f"stress analysis skipped: {self.stress_skipped}")

        if self.rolling_skipped is not None:
            messages.append(f"rolling analysis skipped: {self.rolling_skipped}")

        if (
            self.stress_lift is not None
            and self.stress_lift > self.config.stress_lift_threshold
        ):
            messages.append(
                f"mean correlation rises by {self.stress_lift:+.2f} on the worst "
                "days; diversification weakens exactly when it is needed"
            )

        return messages


def analyze(
    panel: ReturnsPanel, config: AnalysisConfig | None = None
) -> DiversificationReport:
    config = config or AnalysisConfig()

    correlation = correlation_matrix(panel)
    low, high = correlation_interval(
        correlation, panel.T, level=config.confidence_level
    )
    spectral = decompose(correlation)

    if panel.T > panel.N:
        bounds = marchenko_pastur_bounds(panel.T, panel.N)
        signal_factors = count_signal_factors(spectral.eigenvalues, panel.T, panel.N)
    else:
        bounds = None
        signal_factors = None

    rolling = None
    rolling_skipped = None
    window = config.resolve_window(panel.T)

    if panel.N < 2:
        rolling_skipped = "a rolling correlation needs at least 2 series"
    elif window is None:
        rolling_skipped = f"{panel.T} observations are too few for any window"
    else:
        try:
            rolling = rolling_mean_correlation(panel, window)
        except ValueError as exc:
            rolling_skipped = str(exc)

    stress_corr = None
    stress_observations = None
    stress_lift = None
    stress_skipped = None

    if panel.N < 2:
        stress_skipped = "a stress correlation needs at least 2 series"
    else:
        try:
            stress_corr, stress_observations = stress_correlation(
                panel, config.stress_quantile
            )
            stress_lift = mean_pairwise_correlation(
                stress_corr
            ) - mean_pairwise_correlation(correlation)
        except ValueError as exc:
            stress_skipped = str(exc)

    return DiversificationReport(
        panel=panel,
        config=config,
        correlation=correlation,
        correlation_low=low,
        correlation_high=high,
        mean_correlation=mean_pairwise_correlation(correlation),
        annualized_volatility=volatilities(panel, annualize=True),
        spectral=spectral,
        noise_bounds=bounds,
        signal_factors=signal_factors,
        estimate_is_reliable=is_estimate_reliable(
            panel.T, panel.N, config.min_observation_ratio
        ),
        rolling=rolling,
        rolling_skipped=rolling_skipped,
        stress_correlation=stress_corr,
        stress_observations=stress_observations,
        stress_lift=stress_lift,
        stress_skipped=stress_skipped,
    )
