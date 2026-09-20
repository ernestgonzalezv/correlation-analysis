import numpy as np
import pytest

from correlation_analysis.core.covariance import correlation_matrix
from correlation_analysis.core.noise import (
    correlation_interval,
    correlation_stderr,
    count_signal_factors,
    fisher_z_stderr,
    is_estimate_reliable,
    marchenko_pastur_bounds,
    signal_mask,
    two_sided_z_score,
)
from correlation_analysis.core.spectral import decompose
from correlation_analysis.data.synthetic import (
    equicorrelation_matrix,
    independent_panel,
    synthetic_panel,
)


def test_bounds_known_value():
    bounds = marchenko_pastur_bounds(n_obs=500, n_series=8)
    assert bounds.upper == pytest.approx((1 + np.sqrt(8 / 500)) ** 2, abs=1e-9)
    assert bounds.upper == pytest.approx(1.2693, abs=1e-3)


def test_bounds_tighten_with_more_observations():
    few = marchenko_pastur_bounds(n_obs=100, n_series=10)
    many = marchenko_pastur_bounds(n_obs=10_000, n_series=10)
    assert many.upper < few.upper
    assert many.upper == pytest.approx(1.0, abs=0.1)


def test_bounds_bracket_one():
    bounds = marchenko_pastur_bounds(n_obs=1_000, n_series=20)
    assert bounds.lower < 1.0 < bounds.upper


def test_bounds_require_more_observations_than_series():
    with pytest.raises(ValueError, match="requires n_obs > n_series"):
        marchenko_pastur_bounds(n_obs=4, n_series=10)


def test_chatgpt_style_tiny_sample_is_entirely_noise():
    bounds = marchenko_pastur_bounds(n_obs=4, n_series=3)
    assert bounds.upper > 3.0


def test_independent_data_yields_almost_no_signal_factors():
    panel = independent_panel(n_obs=4_000, n_series=10, seed=31)
    eigenvalues = decompose(correlation_matrix(panel)).eigenvalues
    assert count_signal_factors(eigenvalues, panel.T, panel.N) <= 1


def test_correlated_data_yields_a_signal_factor():
    panel = synthetic_panel(equicorrelation_matrix(10, rho=0.6), n_obs=4_000, seed=32)
    eigenvalues = decompose(correlation_matrix(panel)).eigenvalues
    assert count_signal_factors(eigenvalues, panel.T, panel.N) >= 1


def test_signal_mask_flags_the_dominant_factor():
    panel = synthetic_panel(equicorrelation_matrix(8, rho=0.7), n_obs=3_000, seed=33)
    eigenvalues = decompose(correlation_matrix(panel)).eigenvalues
    assert signal_mask(eigenvalues, panel.T, panel.N)[0]


def test_stderr_shrinks_with_sample_size():
    r = np.array(0.5)
    assert correlation_stderr(r, 1_000) < correlation_stderr(r, 50)


def test_stderr_depends_on_the_correlation_value():
    weak = correlation_stderr(np.array(0.1), 500)
    strong = correlation_stderr(np.array(0.9), 500)
    assert strong < weak


def test_stderr_vanishes_at_perfect_correlation():
    assert correlation_stderr(np.array(1.0), 500) == pytest.approx(0.0)


def test_stderr_rejects_degenerate_sample():
    with pytest.raises(ValueError, match="greater than 1"):
        correlation_stderr(np.array(0.5), 1)


def test_fisher_stderr_known_value():
    assert fisher_z_stderr(103) == pytest.approx(0.1)


def test_fisher_stderr_rejects_tiny_samples():
    with pytest.raises(ValueError, match="greater than 3"):
        fisher_z_stderr(3)


def test_fisher_and_pearson_stderr_are_distinct_scales():
    assert fisher_z_stderr(200) != pytest.approx(correlation_stderr(np.array(0.5), 200))


def test_interval_brackets_the_off_diagonal_estimate():
    r = np.array([[1.0, 0.4], [0.4, 1.0]])
    low, high = correlation_interval(r, n_obs=200)
    assert low[0, 1] <= 0.4 <= high[0, 1]


def test_diagonal_is_returned_exactly():
    r = np.array([[1.0, 0.4], [0.4, 1.0]])
    low, high = correlation_interval(r, n_obs=200)
    assert np.array_equal(np.diag(low), np.ones(2))
    assert np.array_equal(np.diag(high), np.ones(2))


def test_scalar_input_has_no_diagonal_handling():
    low, high = correlation_interval(np.array(0.4), n_obs=200)
    assert low < 0.4 < high


def test_interval_narrows_with_more_observations():
    r = np.array(0.3)
    narrow = np.subtract(*reversed(correlation_interval(r, n_obs=5_000)))
    wide = np.subtract(*reversed(correlation_interval(r, n_obs=60)))
    assert narrow < wide


def test_short_window_interval_is_dangerously_wide():
    r = np.array(0.15)
    low, high = correlation_interval(r, n_obs=60)
    assert high - low > 0.4


def test_interval_covers_true_correlation():
    target = equicorrelation_matrix(3, rho=0.5)
    covered = 0
    trials = 40
    for seed in range(trials):
        panel = synthetic_panel(target, n_obs=250, seed=seed)
        corr = correlation_matrix(panel)
        low, high = correlation_interval(corr, panel.T, level=0.95)
        if low[0, 1] <= 0.5 <= high[0, 1]:
            covered += 1
    assert covered >= int(0.85 * trials)


def test_interval_accepts_any_level_in_the_open_unit_interval():
    low, high = correlation_interval(np.array(0.5), n_obs=100, level=0.77)
    assert low < 0.5 < high


def test_interval_rejects_a_level_outside_the_unit_interval():
    with pytest.raises(ValueError, match="must lie in"):
        correlation_interval(np.array(0.5), n_obs=100, level=1.0)


def test_z_score_matches_textbook_values():
    assert two_sided_z_score(0.95) == pytest.approx(1.959964, abs=1e-6)
    assert two_sided_z_score(0.99) == pytest.approx(2.575829, abs=1e-6)
    assert two_sided_z_score(0.90) == pytest.approx(1.644854, abs=1e-6)


def test_z_score_increases_with_confidence():
    levels = [0.80, 0.90, 0.95, 0.99, 0.999]
    scores = [two_sided_z_score(level) for level in levels]
    assert scores == sorted(scores)


def test_z_score_rejects_degenerate_levels():
    for level in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError, match="must lie in"):
            two_sided_z_score(level)


def test_higher_confidence_gives_wider_interval():
    r = np.array(0.4)
    low90, high90 = correlation_interval(r, 500, level=0.90)
    low99, high99 = correlation_interval(r, 500, level=0.99)
    assert (high99 - low99) > (high90 - low90)


def test_reliability_rule_of_thumb():
    assert is_estimate_reliable(n_obs=1_000, n_series=8)
    assert not is_estimate_reliable(n_obs=40, n_series=8)
