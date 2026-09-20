import numpy as np
import pytest

from eigenrisk.core.covariance import correlation_matrix
from eigenrisk.data.synthetic import (
    block_correlation_matrix,
    correlated_returns,
    equicorrelation_matrix,
    independent_panel,
    synthetic_panel,
)


def test_equicorrelation_shape_and_diagonal():
    matrix = equicorrelation_matrix(5, rho=0.4)
    assert matrix.shape == (5, 5)
    assert np.allclose(np.diag(matrix), 1.0)


def test_equicorrelation_is_positive_definite():
    matrix = equicorrelation_matrix(10, rho=0.5)
    assert np.linalg.eigvalsh(matrix).min() > 0


def test_equicorrelation_rejects_rho_below_stability_bound():
    with pytest.raises(ValueError, match="positive definite"):
        equicorrelation_matrix(4, rho=-0.9)


def test_equicorrelation_rejects_rho_of_one():
    with pytest.raises(ValueError, match="positive definite"):
        equicorrelation_matrix(4, rho=1.0)


def test_block_matrix_layout():
    matrix = block_correlation_matrix([2, 2], within=0.8, between=0.1)
    assert matrix[0, 1] == pytest.approx(0.8)
    assert matrix[2, 3] == pytest.approx(0.8)
    assert matrix[0, 3] == pytest.approx(0.1)
    assert np.allclose(np.diag(matrix), 1.0)


def test_generator_recovers_the_target_correlation():
    target = equicorrelation_matrix(5, rho=0.7)
    panel = synthetic_panel(target, n_obs=400_000, seed=101)
    assert np.allclose(correlation_matrix(panel), target, atol=0.01)


def test_generator_recovers_a_block_target():
    target = block_correlation_matrix([3, 2], within=0.85, between=0.1)
    panel = synthetic_panel(target, n_obs=400_000, seed=102)
    assert np.allclose(correlation_matrix(panel), target, atol=0.01)


def test_generator_recovers_negative_correlation():
    target = np.array([[1.0, -0.6], [-0.6, 1.0]])
    panel = synthetic_panel(target, n_obs=400_000, seed=103)
    assert correlation_matrix(panel)[0, 1] == pytest.approx(-0.6, abs=0.01)


def test_same_seed_reproduces_identical_data():
    target = equicorrelation_matrix(3, rho=0.3)
    first = correlated_returns(target, n_obs=500, seed=7)
    second = correlated_returns(target, n_obs=500, seed=7)
    assert np.array_equal(first, second)


def test_different_seeds_produce_different_data():
    target = equicorrelation_matrix(3, rho=0.3)
    first = correlated_returns(target, n_obs=500, seed=7)
    second = correlated_returns(target, n_obs=500, seed=8)
    assert not np.array_equal(first, second)


def test_volatilities_are_applied_without_changing_correlation():
    target = equicorrelation_matrix(3, rho=0.5)
    vols = np.array([0.01, 1.0, 100.0])
    panel = synthetic_panel(target, n_obs=200_000, volatilities=vols, seed=104)
    assert np.allclose(panel.values.std(axis=0), vols, rtol=0.02)
    assert np.allclose(correlation_matrix(panel), target, atol=0.015)


def test_generator_rejects_non_positive_definite_target():
    bad = np.array([[1.0, 0.99, -0.99], [0.99, 1.0, 0.99], [-0.99, 0.99, 1.0]])
    with pytest.raises(ValueError, match="positive definite"):
        correlated_returns(bad, n_obs=100)


def test_generator_rejects_non_square_target():
    with pytest.raises(ValueError, match="square"):
        correlated_returns(np.zeros((2, 3)), n_obs=100)


def test_independent_panel_is_near_identity():
    panel = independent_panel(n_obs=200_000, n_series=4, seed=105)
    assert np.allclose(correlation_matrix(panel), np.eye(4), atol=0.01)


def test_default_names_are_generated():
    panel = synthetic_panel(np.eye(3), n_obs=100, seed=1)
    assert panel.names == ("S0", "S1", "S2")


def test_block_matrix_rejects_between_stronger_than_within():
    with pytest.raises(ValueError, match="not positive definite"):
        block_correlation_matrix([2, 2], within=0.2, between=0.9)


def test_block_matrix_rejects_negative_within_with_positive_between():
    with pytest.raises(ValueError, match="not positive definite"):
        block_correlation_matrix([2, 2, 2], within=-0.4, between=0.8)


def test_block_matrix_accepts_a_valid_request():
    matrix = block_correlation_matrix([3, 3], within=0.9, between=0.2)
    assert np.linalg.eigvalsh(matrix).min() > 0
