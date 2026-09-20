import numpy as np
import pytest

from quantlab.core.covariance import (
    center,
    correlation_matrix,
    covariance_matrix,
    volatilities,
)
from quantlab.core.panel import ReturnsPanel
from quantlab.data.synthetic import equicorrelation_matrix, synthetic_panel


@pytest.fixture
def panel():
    rng = np.random.default_rng(7)
    return ReturnsPanel(rng.normal(size=(500, 4)), ("A", "B", "C", "D"))


def test_centering_removes_the_mean(panel):
    assert np.allclose(center(panel).mean(axis=0), 0.0, atol=1e-12)


def test_centering_preserves_shape(panel):
    assert center(panel).shape == panel.values.shape


def test_covariance_matches_numpy(panel):
    assert np.allclose(covariance_matrix(panel), np.cov(panel.values, rowvar=False))


def test_covariance_equals_manual_matrix_product(panel):
    centered = panel.values - panel.values.mean(axis=0)
    expected = (centered.T @ centered) / (panel.T - 1)
    assert np.allclose(covariance_matrix(panel), expected)


def test_covariance_is_symmetric(panel):
    cov = covariance_matrix(panel)
    assert np.allclose(cov, cov.T)


def test_covariance_diagonal_is_variance(panel):
    cov = covariance_matrix(panel)
    assert np.allclose(np.diag(cov), panel.values.var(axis=0, ddof=1))


def test_covariance_rejects_insufficient_degrees_of_freedom():
    panel = ReturnsPanel(np.zeros((2, 2)), ("A", "B"))
    with pytest.raises(ValueError, match="degrees of freedom"):
        covariance_matrix(panel, ddof=2)


def test_correlation_matches_numpy(panel):
    assert np.allclose(correlation_matrix(panel), np.corrcoef(panel.values, rowvar=False))


def test_correlation_diagonal_is_exactly_one(panel):
    assert np.array_equal(np.diag(correlation_matrix(panel)), np.ones(panel.N))


def test_correlation_is_exactly_symmetric(panel):
    corr = correlation_matrix(panel)
    assert np.array_equal(corr, corr.T)


def test_correlation_stays_within_bounds(panel):
    corr = correlation_matrix(panel)
    assert corr.min() >= -1.0 and corr.max() <= 1.0


def test_correlation_is_scale_invariant(panel):
    scaled = panel.values.copy()
    scaled[:, 0] *= 1000.0
    scaled[:, 2] *= 0.001
    rescaled = ReturnsPanel(scaled, panel.names)
    assert np.allclose(correlation_matrix(panel), correlation_matrix(rescaled))


def test_correlation_is_translation_invariant(panel):
    shifted = ReturnsPanel(panel.values + 17.0, panel.names)
    assert np.allclose(correlation_matrix(panel), correlation_matrix(shifted))


def test_duplicated_series_correlate_perfectly():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(200, 1))
    panel = ReturnsPanel(np.hstack([base, base * 2.0]), ("A", "A_doubled"))
    assert correlation_matrix(panel)[0, 1] == pytest.approx(1.0)


def test_mirrored_series_correlate_negatively():
    rng = np.random.default_rng(2)
    base = rng.normal(size=(200, 1))
    panel = ReturnsPanel(np.hstack([base, -base]), ("A", "A_mirror"))
    assert correlation_matrix(panel)[0, 1] == pytest.approx(-1.0)


def test_correlation_equals_cosine_of_angle(panel):
    centered = center(panel)
    a, b = centered[:, 0], centered[:, 1]
    cosine = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    assert correlation_matrix(panel)[0, 1] == pytest.approx(cosine)


def test_zero_correlation_means_orthogonal_vectors():
    values = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    panel = ReturnsPanel(values, ("A", "B"))
    centered = center(panel)
    assert centered[:, 0] @ centered[:, 1] == pytest.approx(0.0)
    assert correlation_matrix(panel)[0, 1] == pytest.approx(0.0)


def test_constant_series_is_rejected():
    values = np.column_stack([np.random.default_rng(3).normal(size=100), np.ones(100)])
    panel = ReturnsPanel(values, ("A", "FLAT"))
    with pytest.raises(ValueError, match="constant"):
        correlation_matrix(panel)


def test_correlation_recovers_known_target():
    target = equicorrelation_matrix(4, rho=0.6)
    panel = synthetic_panel(target, n_obs=200_000, seed=11)
    assert np.allclose(correlation_matrix(panel), target, atol=0.01)


def test_volatility_matches_numpy(panel):
    assert np.allclose(volatilities(panel), panel.values.std(axis=0, ddof=1))


def test_annualized_volatility_scales_by_sqrt_periods():
    rng = np.random.default_rng(5)
    panel = ReturnsPanel(rng.normal(size=(300, 2)), ("A", "B"), freq="1d")
    ratio = volatilities(panel, annualize=True) / volatilities(panel)
    assert np.allclose(ratio, np.sqrt(252))


def test_annualization_depends_on_declared_freq():
    rng = np.random.default_rng(6)
    values = rng.normal(size=(300, 1))
    daily = volatilities(ReturnsPanel(values, ("A",), freq="1d"), annualize=True)
    m15 = volatilities(ReturnsPanel(values, ("A",), freq="15m"), annualize=True)
    assert m15 / daily == pytest.approx(np.sqrt(96))
