from __future__ import annotations

import numpy as np

from eigenrisk.research.diversification import DiversificationReport

WIDTH = 78
NAME_WIDTH = 12


def _rule(char: str = "=") -> str:
    return char * WIDTH


def _header(title: str) -> list[str]:
    return [_rule(), title.upper(), _rule()]


def _matrix_block(matrix: np.ndarray, names: tuple[str, ...]) -> list[str]:
    header = " " * NAME_WIDTH + "".join(f"{n[:9]:>10}" for n in names)
    lines = [header]

    for i, name in enumerate(names):
        row = f"{name[:NAME_WIDTH - 1]:<{NAME_WIDTH}}"
        row += "".join(f"{matrix[i, j]:>10.3f}" for j in range(len(names)))
        lines.append(row)

    return lines


def render(report: DiversificationReport) -> str:
    panel = report.panel
    lines: list[str] = []

    lines += _header("panel")
    lines.append(f"observations   {panel.T}")
    lines.append(f"series         {panel.N}")
    lines.append(f"frequency      {panel.freq}")
    lines.append(f"kind           {panel.kind}")
    lines.append(f"obs per series {panel.observation_ratio:.1f}")
    lines.append("")

    lines += _header("annualized volatility")
    for name, vol in zip(panel.names, report.annualized_volatility):
        lines.append(f"{name[:NAME_WIDTH - 1]:<{NAME_WIDTH}} {vol:>10.4f}")
    lines.append("")

    lines += _header("correlation")
    lines += _matrix_block(report.correlation, panel.names)
    lines.append("")
    lines.append(f"mean pairwise correlation   {report.mean_correlation:+.4f}")
    lines.append("")

    lines += _header("correlation uncertainty")
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

    lines += _header("factor structure")
    spectral = report.spectral
    bound = report.noise_bounds.upper if report.noise_bounds else None

    lines.append(
        f"{'factor':<10}{'eigenvalue':>14}{'explained':>12}"
        f"{'cumulative':>13}{'verdict':>14}"
    )
    cumulative = spectral.cumulative_variance()
    for i, value in enumerate(spectral.eigenvalues):
        if bound is None:
            verdict = "unknown"
        else:
            verdict = "signal" if value > bound else "noise"
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

    lines += _header("diversification")
    lines.append(f"series held              {panel.N}")
    lines.append(f"effective bets           {report.effective_bets:.3f}")
    lines.append(f"diversification ratio    {report.diversification_ratio:.1%}")
    lines.append("")

    if report.rolling is not None:
        lines += _header("correlation through time")
        rolling = report.rolling
        lines.append(f"window            {rolling.window}")
        lines.append(f"mean              {rolling.mean:+.4f}")
        lines.append(f"minimum           {rolling.minimum:+.4f}")
        lines.append(f"maximum           {rolling.maximum:+.4f}")
        lines.append(f"range             {rolling.spread:.4f}")
        lines.append("")

    if report.stress_correlation is not None:
        lines += _header("correlation under stress")
        lines.append(f"worst observations used   {report.stress_observations}")
        lines.append(
            f"mean correlation there    "
            f"{np.mean(report.stress_correlation[np.triu_indices(panel.N, k=1)]):+.4f}"
        )
        lines.append(f"lift versus full sample   {report.stress_lift:+.4f}")
        lines.append("")

    if report.warnings:
        lines += _header("warnings")
        for message in report.warnings:
            lines.append(f"- {message}")
        lines.append("")

    return "\n".join(lines)


def print_report(report: DiversificationReport) -> None:
    print(render(report))
