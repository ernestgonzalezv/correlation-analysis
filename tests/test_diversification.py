import numpy as np
import pytest

from correlation_analysis import analyze, panel_from_pnl
from correlation_analysis.core.panel import ReturnsPanel
from correlation_analysis.data.synthetic import (
    block_correlation_matrix,
    equicorrelation_matrix,
    independent_panel,
    synthetic_panel,
)
from correlation_analysis.report.console import render


@pytest.fixture
def report():
    panel = synthetic_panel(
        block_correlation_matrix([3, 2], within=0.88, between=0.05),
        n_obs=1_200,
        names=["EURUSD", "GBPUSD", "AUDUSD", "XAUUSD", "US500"],
        seed=9,
    )
    return analyze(panel)


def test_report_exposes_the_panel(report):
    assert report.panel.N == 5
    assert report.panel.T == 1_200


def test_correlation_matrix_shape(report):
    assert report.correlation.shape == (5, 5)


def test_confidence_bands_bracket_the_estimates(report):
    upper = np.triu_indices(5, k=1)
    assert np.all(report.correlation_low[upper] <= report.correlation[upper])
    assert np.all(report.correlation[upper] <= report.correlation_high[upper])


def test_block_structure_yields_two_signal_factors(report):
    assert report.signal_factors == 2


def test_block_structure_yields_two_effective_bets(report):
    assert report.effective_bets == pytest.approx(2.0, abs=0.4)


def test_diversification_ratio_is_bets_over_series(report):
    assert report.diversification_ratio == pytest.approx(report.effective_bets / 5)


def test_estimate_is_flagged_reliable_with_enough_data(report):
    assert report.estimate_is_reliable


def test_concentration_warning_is_raised(report):
    assert any("independent bets" in w for w in report.warnings)


def test_independent_series_give_no_concentration_warning():
    report = analyze(independent_panel(n_obs=3_000, n_series=5, seed=41))
    assert not any("independent bets" in w for w in report.warnings)


def test_independent_series_approach_full_diversification():
    report = analyze(independent_panel(n_obs=20_000, n_series=6, seed=42))
    assert report.effective_bets == pytest.approx(6.0, abs=0.3)


def test_short_sample_is_flagged_unreliable():
    panel = synthetic_panel(equicorrelation_matrix(8, rho=0.3), n_obs=40, seed=43)
    report = analyze(panel)
    assert not report.estimate_is_reliable
    assert any("dominated by noise" in w for w in report.warnings)


def test_noise_bounds_are_skipped_when_series_outnumber_observations():
    panel = synthetic_panel(equicorrelation_matrix(12, rho=0.2), n_obs=10, seed=44)
    report = analyze(panel)
    assert report.noise_bounds is None
    assert report.signal_factors is None


def test_pure_noise_reports_no_signal_factors():
    report = analyze(independent_panel(n_obs=3_000, n_series=12, seed=45))
    assert report.signal_factors == 0


def test_reliable_independence_is_a_finding_not_a_warning():
    report = analyze(independent_panel(n_obs=3_000, n_series=12, seed=45))
    assert report.estimate_is_reliable
    assert report.warnings == []


def test_unreliable_sample_without_factors_is_flagged_as_ambiguous():
    report = analyze(independent_panel(n_obs=60, n_series=10, seed=50))
    assert report.signal_factors == 0
    assert any("not distinguishable" in w for w in report.warnings)


def test_rolling_window_is_chosen_automatically(report):
    assert report.rolling is not None
    assert report.rolling.window == 200


def test_explicit_rolling_window_is_respected():
    panel = synthetic_panel(equicorrelation_matrix(3, rho=0.4), n_obs=600, seed=46)
    assert analyze(panel, window=90).rolling.window == 90


def test_stress_section_is_populated(report):
    assert report.stress_correlation is not None
    assert report.stress_observations == pytest.approx(120, abs=3)


def test_stress_lift_is_detected_when_a_crash_is_injected():
    rng = np.random.default_rng(47)
    calm = rng.normal(scale=0.01, size=(600, 4))
    shock = rng.normal(scale=0.01, size=(60, 1)) - 0.04
    crash = np.repeat(shock, 4, axis=1) + rng.normal(scale=0.002, size=(60, 4))

    report = analyze(ReturnsPanel(np.vstack([calm, crash]), ("A", "B", "C", "D")))

    assert report.stress_lift > 0.2
    assert any("worst days" in w for w in report.warnings)


def test_same_engine_accepts_strategy_pnl():
    rng = np.random.default_rng(48)
    pnl = rng.normal(loc=40.0, scale=300.0, size=(800, 4))
    panel = panel_from_pnl(pnl, ["MBR", "MeanRev", "Swing", "Calendar"])

    report = analyze(panel)

    assert report.panel.kind == "pnl"
    assert report.effective_bets == pytest.approx(4.0, abs=0.4)


def test_analysis_is_deterministic(report):
    repeated = analyze(report.panel)
    assert np.array_equal(repeated.correlation, report.correlation)
    assert repeated.effective_bets == report.effective_bets


def test_render_produces_every_section(report):
    text = render(report)
    for section in (
        "PANEL",
        "ANNUALIZED VOLATILITY",
        "CORRELATION",
        "CORRELATION UNCERTAINTY",
        "FACTOR STRUCTURE",
        "DIVERSIFICATION",
        "CORRELATION THROUGH TIME",
        "CORRELATION UNDER STRESS",
        "WARNINGS",
    ):
        assert section in text


def test_render_lists_every_series_name(report):
    text = render(report)
    assert all(name in text for name in report.panel.names)


def test_render_handles_a_report_without_warnings():
    text = render(analyze(independent_panel(n_obs=5_000, n_series=4, seed=49)))
    assert "WARNINGS" not in text


def test_skips_are_reported_not_swallowed():
    panel = synthetic_panel(equicorrelation_matrix(4, rho=0.3), n_obs=25, seed=51)
    report = analyze(panel, stress_quantile=0.05)

    assert report.stress_correlation is None
    assert report.stress_skipped is not None
    assert any("stress analysis skipped" in w for w in report.warnings)


def test_skip_reason_reaches_the_rendered_report():
    panel = synthetic_panel(equicorrelation_matrix(4, rho=0.3), n_obs=25, seed=52)
    text = render(analyze(panel, stress_quantile=0.05))
    assert "skipped:" in text


def test_successful_analysis_records_no_skips(report):
    assert report.rolling_skipped is None
    assert report.stress_skipped is None


def test_marchenko_pastur_caveat_is_shown(report):
    assert "independent, identically distributed" in render(report)


def test_volatility_units_follow_the_panel_kind():
    rng = np.random.default_rng(53)
    pnl = panel_from_pnl(rng.normal(loc=10.0, scale=200.0, size=(600, 3)), ["A", "B", "C"])
    assert "CURRENCY UNITS" in render(analyze(pnl))
    assert "RETURN UNITS" in render(analyze(independent_panel(600, 3, seed=54)))
