import numpy as np
import pytest

from quantlab.core.covariance import correlation_matrix
from quantlab.core.spectral import decompose, effective_bets
from quantlab.data.synthetic import (
    block_correlation_matrix,
    equicorrelation_matrix,
    synthetic_panel,
)


@pytest.fixture
def result():
    panel = synthetic_panel(equicorrelation_matrix(5, rho=0.5), n_obs=5_000, seed=21)
    return decompose(correlation_matrix(panel))


def test_eigenvalues_are_sorted_descending(result):
    assert np.all(np.diff(result.eigenvalues) <= 1e-12)


def test_eigenvalues_are_non_negative(result):
    assert result.eigenvalues.min() >= 0.0


def test_eigenvalues_sum_to_number_of_series(result):
    assert result.eigenvalues.sum() == pytest.approx(5.0)


def test_eigenvalue_sum_equals_trace():
    corr = correlation_matrix(
        synthetic_panel(equicorrelation_matrix(6, rho=0.3), n_obs=2_000, seed=22)
    )
    assert decompose(corr).eigenvalues.sum() == pytest.approx(np.trace(corr))


def test_eigen_equation_holds(result):
    corr = correlation_matrix(
        synthetic_panel(equicorrelation_matrix(5, rho=0.5), n_obs=5_000, seed=21)
    )
    for i in range(result.n_factors):
        v = result.loadings(i)
        assert np.allclose(corr @ v, result.eigenvalues[i] * v)


def test_eigenvectors_are_orthonormal(result):
    assert np.allclose(result.eigenvectors.T @ result.eigenvectors, np.eye(5), atol=1e-10)


def test_explained_variance_sums_to_one(result):
    assert result.explained_variance.sum() == pytest.approx(1.0)


def test_explained_variance_is_descending(result):
    assert np.all(np.diff(result.explained_variance) <= 1e-12)


def test_cumulative_variance_reaches_one(result):
    assert result.cumulative_variance()[-1] == pytest.approx(1.0)


def test_identity_matrix_has_unit_eigenvalues():
    result = decompose(np.eye(8))
    assert np.allclose(result.eigenvalues, 1.0)


def test_independent_series_give_maximum_effective_bets():
    assert decompose(np.eye(8)).effective_bets == pytest.approx(8.0)


def test_perfectly_correlated_series_give_one_effective_bet():
    result = decompose(np.ones((6, 6)))
    assert result.effective_bets == pytest.approx(1.0)
    assert result.eigenvalues[0] == pytest.approx(6.0)


def test_effective_bets_lies_between_one_and_n(result):
    assert 1.0 <= result.effective_bets <= 5.0


def test_effective_bets_decreases_with_correlation():
    values = []
    for rho in (0.0, 0.3, 0.6, 0.9):
        corr = equicorrelation_matrix(6, rho=rho)
        values.append(decompose(corr).effective_bets)
    assert values == sorted(values, reverse=True)


def test_block_structure_is_detected():
    corr = block_correlation_matrix([3, 3], within=0.9, between=0.05)
    result = decompose(corr)
    assert result.effective_bets == pytest.approx(2.0, abs=0.3)


def test_dominant_factor_loads_all_series_with_same_sign():
    corr = equicorrelation_matrix(5, rho=0.8)
    loadings = decompose(corr).loadings(0)
    assert np.all(loadings > 0) or np.all(loadings < 0)


def test_factors_for_variance_threshold():
    corr = block_correlation_matrix([4, 4], within=0.95, between=0.0)
    assert decompose(corr).factors_for_variance(0.90) <= 4


def test_factors_for_variance_rejects_bad_threshold(result):
    with pytest.raises(ValueError, match="threshold must be"):
        result.factors_for_variance(1.5)


def test_loadings_reject_out_of_range_factor(result):
    with pytest.raises(IndexError, match="out of range"):
        result.loadings(99)


def test_decompose_rejects_non_square():
    with pytest.raises(ValueError, match="square"):
        decompose(np.zeros((3, 4)))


def test_decompose_rejects_asymmetric():
    matrix = np.array([[1.0, 0.5], [0.2, 1.0]])
    with pytest.raises(ValueError, match="symmetric"):
        decompose(matrix)


def test_decompose_rejects_nan():
    matrix = np.array([[1.0, np.nan], [np.nan, 1.0]])
    with pytest.raises(ValueError, match="NaN or inf"):
        decompose(matrix)


def test_effective_bets_rejects_non_positive_sum():
    with pytest.raises(ValueError, match="positive"):
        effective_bets(np.zeros(4))
