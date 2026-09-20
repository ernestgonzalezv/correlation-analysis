import numpy as np
import pytest

from quantlab.core.covariance import correlation_matrix
from quantlab.core.panel import ReturnsPanel
from quantlab.core.rolling import (
    correlation_lift_under_stress,
    mean_pairwise_correlation,
    rolling_mean_correlation,
    stress_correlation,
    stress_mask,
)
from quantlab.data.synthetic import equicorrelation_matrix, synthetic_panel


@pytest.fixture
def panel():
    return synthetic_panel(equicorrelation_matrix(4, rho=0.4), n_obs=600, seed=55)


def test_mean_pairwise_ignores_the_diagonal():
    corr = np.array([[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]])
    assert mean_pairwise_correlation(corr) == pytest.approx(0.5)


def test_mean_pairwise_of_identity_is_zero():
    assert mean_pairwise_correlation(np.eye(5)) == pytest.approx(0.0)


def test_mean_pairwise_requires_two_series():
    with pytest.raises(ValueError, match="at least 2 series"):
        mean_pairwise_correlation(np.array([[1.0]]))


def test_rolling_produces_expected_number_of_windows(panel):
    result = rolling_mean_correlation(panel, window=100)
    assert len(result.values) == panel.T - 100 + 1


def test_rolling_positions_align_with_window_ends(panel):
    result = rolling_mean_correlation(panel, window=100)
    assert result.positions[0] == 99
    assert result.positions[-1] == panel.T - 1


def test_rolling_mean_approximates_full_sample_correlation(panel):
    result = rolling_mean_correlation(panel, window=200)
    full = mean_pairwise_correlation(correlation_matrix(panel))
    assert result.mean == pytest.approx(full, abs=0.1)


def test_rolling_correlation_varies_over_time(panel):
    result = rolling_mean_correlation(panel, window=60)
    assert result.spread > 0.0
    assert result.minimum <= result.mean <= result.maximum


def test_short_windows_are_noisier_than_long_ones(panel):
    short = rolling_mean_correlation(panel, window=30)
    long = rolling_mean_correlation(panel, window=250)
    assert short.values.std() > long.values.std()


def test_rolling_rejects_window_larger_than_sample(panel):
    with pytest.raises(ValueError, match="exceeds"):
        rolling_mean_correlation(panel, window=panel.T + 1)


def test_rolling_rejects_tiny_window(panel):
    with pytest.raises(ValueError, match="window must be at least 4"):
        rolling_mean_correlation(panel, window=3)


def test_rolling_requires_two_series():
    single = synthetic_panel(np.eye(1), n_obs=100, seed=1)
    with pytest.raises(ValueError, match="at least 2 series"):
        rolling_mean_correlation(single, window=20)


def test_stress_mask_selects_the_requested_fraction(panel):
    mask = stress_mask(panel, quantile=0.10)
    assert mask.sum() == pytest.approx(panel.T * 0.10, abs=2)


def test_stress_mask_selects_the_worst_days(panel):
    mask = stress_mask(panel, quantile=0.10)
    portfolio = panel.values.mean(axis=1)
    assert portfolio[mask].max() <= portfolio[~mask].min()


def test_stress_mask_rejects_invalid_quantile(panel):
    with pytest.raises(ValueError, match="quantile must lie"):
        stress_mask(panel, quantile=1.5)


def test_stress_correlation_reports_day_count(panel):
    _, n_days = stress_correlation(panel, quantile=0.20)
    assert n_days == pytest.approx(panel.T * 0.20, abs=2)


def test_stress_correlation_is_a_valid_matrix(panel):
    corr, _ = stress_correlation(panel, quantile=0.20)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(corr, corr.T)


def test_stress_correlation_rejects_too_few_days():
    small = synthetic_panel(equicorrelation_matrix(3, rho=0.3), n_obs=20, seed=2)
    with pytest.raises(ValueError, match="stress observations"):
        stress_correlation(small, quantile=0.05)


def test_stress_lift_is_positive_when_a_crash_is_injected():
    rng = np.random.default_rng(77)
    calm = rng.normal(scale=0.01, size=(500, 4))
    crash = rng.normal(scale=0.01, size=(40, 1)) - 0.05
    stressed_block = np.repeat(crash, 4, axis=1) + rng.normal(scale=0.002, size=(40, 4))
    values = np.vstack([calm, stressed_block])

    panel = ReturnsPanel(values, ("A", "B", "C", "D"))

    assert correlation_lift_under_stress(panel, quantile=0.07) > 0.2
