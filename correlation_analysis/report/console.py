from __future__ import annotations

import numpy as np

from correlation_analysis.research.diversification import DiversificationReport

DEFAULT_WIDTH = 78
DEFAULT_NAME_WIDTH = 12


def _rule(width: int, char: str = "=") -> str:
    return char * width


def _header(title: str, width: int) -> list[str]:
    return [_rule(width), title.upper(), _rule(width)]


def _verdict(eigenvalue: float, bound: float | None) -> str:
    if bound is None:
        return "unknown"

    return "signal" if eigenvalue > bound else "noise"


def _matrix_block(
    matrix: np.ndarray, names: tuple[str, ...], name_width: int
) -> list[str]:
    header = " " * name_width + "".join(f"{n[:9]:>10}" for n in names)
    lines = [header]

    for i, name in enumerate(names):
        row = f"{name[:name_width - 1]:<{name_width}}"
        row += "".join(f"{matrix[i, j]:>10.3f}" for j in range(len(names)))
        lines.append(row)

    return lines


def render(
    report: DiversificationReport,
    width: int = DEFAULT_WIDTH,
    name_width: int = DEFAULT_NAME_WIDTH,
) -> str:
    panel = report.panel
    lines: list[str] = []

    lines += _header("panel", width)
    lines.append(f"observations   {panel.T}")
    lines.append(f"series         {panel.N}")
    lines.append(f"frequency      {panel.freq}")
    lines.append(f"kind           {panel.kind}")
    lines.append(f"obs per series {panel.observation_ratio:.1f}")
    lines.append("")

    volatility_units = "currency" if panel.kind == "pnl" else "return"
    lines += _header(f"annualized volatility ({volatility_units} units)", width)
    for name, vol in zip(panel.names, report.annualized_volatility, strict=True):
        lines.append(f"{name[:name_width - 1]:<{name_width}} {vol:>10.4f}")
    lines.append("")

    lines += _header("correlation", width)
    lines += _matrix_block(report.correlation, panel.names, name_width)
    lines.append("")
    lines.append(f"mean pairwise correlation   {report.mean_correlation:+.4f}")
    lines.append("")

    lines += _header("correlation uncertainty", width)
    lines.append("95% interval per pair")
    lines.append("")
    for i in range(panel.N):
        for j in range(i + 1, panel.N):
            pair = f"{panel.names[i][:9]} / {panel.names[j][:9]}"
            lines.append(
                f"{pair:<24} {report.correlation[i, j]:+.3f}   "
                f"[{report.correlation_low[i, j]:+.3f}, "
                f"{report.correlation_high[i, j]:+.3f}]"
            )
    lines.append("")

    lines += _header("factor structure", width)
    spectral = report.spectral
    bound = report.noise_bounds.upper if report.noise_bounds else None

    lines.append(
        f"{'factor':<10}{'eigenvalue':>14}{'explained':>12}"
        f"{'cumulative':>13}{'verdict':>14}"
    )
    cumulative = spectral.cumulative_variance()
    for i, value in enumerate(spectral.eigenvalues):
        verdict = _verdict(value, bound)
        lines.append(
            f"{i + 1:<10}{value:>14.4f}{spectral.explained_variance[i]:>11.1%}"
            f"{cumulative[i]:>13.1%}{verdict:>14}"
        )
    lines.append("")

    if report.noise_bounds is not None:
        lines.append(
            f"Marchenko-Pastur noise ceiling   {report.noise_bounds.upper:.4f}"
        )
        lines.append(f"factors above the ceiling        {report.signal_factors}")
        lines.append("")
        lines.append(
            "the bound assumes independent, identically distributed observations;"
        )
        lines.append(
            "fat tails and volatility clustering widen the true band, so treat a"
        )
        lines.append("verdict near the ceiling as undecided rather than settled")
        lines.append("")

    lines += _header("diversification", width)
    lines.append(f"series held              {panel.N}")
    lines.append(f"effective bets           {report.effective_bets:.3f}")
    lines.append(f"diversification ratio    {report.diversification_ratio:.1%}")
    lines.append("")

    if report.rolling is not None:
        lines += _header("correlation through time", width)
        rolling = report.rolling
        lines.append(f"window            {rolling.window}")
        lines.append(f"mean              {rolling.mean:+.4f}")
        lines.append(f"minimum           {rolling.minimum:+.4f}")
        lines.append(f"maximum           {rolling.maximum:+.4f}")
        lines.append(f"range             {rolling.spread:.4f}")
        lines.append("")

    if report.rolling is None and report.rolling_skipped is not None:
        lines += _header("correlation through time", width)
        lines.append(f"skipped: {report.rolling_skipped}")
        lines.append("")

    if report.stress_correlation is not None:
        lines += _header("correlation under stress", width)
        lines.append(f"worst observations used   {report.stress_observations}")
        lines.append(
            f"mean correlation there    "
            f"{np.mean(report.stress_correlation[np.triu_indices(panel.N, k=1)]):+.4f}"
        )
        lines.append(f"lift versus full sample   {report.stress_lift:+.4f}")
        lines.append("")

    if report.stress_correlation is None and report.stress_skipped is not None:
        lines += _header("correlation under stress", width)
        lines.append(f"skipped: {report.stress_skipped}")
        lines.append("")

    if report.warnings:
        lines += _header("warnings", width)
        for message in report.warnings:
            lines.append(f"- {message}")
        lines.append("")

    return "\n".join(lines)


def print_report(
    report: DiversificationReport,
    width: int = DEFAULT_WIDTH,
    name_width: int = DEFAULT_NAME_WIDTH,
) -> None:
    print(render(report, width=width, name_width=name_width))
