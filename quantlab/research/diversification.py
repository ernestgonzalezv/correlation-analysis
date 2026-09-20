from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantlab.core.covariance import correlation_matrix, volatilities
from quantlab.core.noise import (
    NoiseBounds,
    correlation_interval,
    count_signal_factors,
    is_estimate_reliable,
    marchenko_pastur_bounds,
)
from quantlab.core.panel import ReturnsPanel
from quantlab.core.rolling import (
    RollingCorrelation,
    mean_pairwise_correlation,
    rolling_mean_correlation,
    stress_correlation,
)
from quantlab.core.spectral import SpectralResult, decompose

DEFAULT_STRESS_QUANTILE = 0.10
MIN_ROLLING_WINDOW = 20


@dataclass(frozen=True)
class DiversificationReport:
    panel: ReturnsPanel
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
    stress_correlation: np.ndarray | None
    stress_observations: int | None
    stress_lift: float | None

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
                f"(ratio {self.panel.observation_ratio:.1f}); the estimate is "
                "dominated by noise below a ratio of 10"
            )

        if self.signal_factors == 0 and not self.estimate_is_reliable:
            messages.append(
                "no eigenvalue exceeds the Marchenko-Pastur bound at this "
                "sample size; genuine independence and insufficient data are "
                "not distinguishable here"
            )

        if self.diversification_ratio < 0.5:
            messages.append(
                f"{self.panel.N} series behave like {self.effective_bets:.2f} "
                "independent bets; exposure is more concentrated than it looks"
            )

        if self.stress_lift is not None and self.stress_lift > 0.15:
            messages.append(
                f"mean correlation rises by {self.stress_lift:+.2f} on the worst "
                "days; diversification weakens exactly when it is needed"
            )

        return messages


def _choose_window(n_obs: int) -> int | None:
    window = max(MIN_ROLLING_WINDOW, n_obs // 6)
    return window if window <= n_obs else None


def analyze(
    panel: ReturnsPanel,
    window: int | None = None,
    stress_quantile: float = DEFAULT_STRESS_QUANTILE,
    confidence: float = 0.95,
) -> DiversificationReport:
    correlation = correlation_matrix(panel)
    low, high = correlation_interval(correlation, panel.T, level=confidence)
    spectral = decompose(correlation)

    if panel.T > panel.N:
        bounds = marchenko_pastur_bounds(panel.T, panel.N)
        signal_factors = count_signal_factors(
            spectral.eigenvalues, panel.T, panel.N
        )
    else:
        bounds = None
        signal_factors = None

    resolved_window = window if window is not None else _choose_window(panel.T)

    rolling = None
    if resolved_window is not None and panel.N >= 2:
        try:
            rolling = rolling_mean_correlation(panel, resolved_window)
        except ValueError:
            rolling = None

    stress_corr = None
    stress_observations = None
    stress_lift = None

    if panel.N >= 2:
        try:
            stress_corr, stress_observations = stress_correlation(
                panel, stress_quantile
            )
            stress_lift = mean_pairwise_correlation(
                stress_corr
            ) - mean_pairwise_correlation(correlation)
        except ValueError:
            pass

    return DiversificationReport(
        panel=panel,
        correlation=correlation,
        correlation_low=low,
        correlation_high=high,
        mean_correlation=mean_pairwise_correlation(correlation),
        annualized_volatility=volatilities(panel, annualize=True),
        spectral=spectral,
        noise_bounds=bounds,
        signal_factors=signal_factors,
        estimate_is_reliable=is_estimate_reliable(panel.T, panel.N),
        rolling=rolling,
        stress_correlation=stress_corr,
        stress_observations=stress_observations,
        stress_lift=stress_lift,
    )
