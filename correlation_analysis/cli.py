from __future__ import annotations

import argparse
import sys
from pathlib import Path

from correlation_analysis.core.panel import PERIODS_PER_YEAR
from correlation_analysis.data.loaders import LoadResult, load_panel
from correlation_analysis.report.console import DEFAULT_WIDTH, print_report
from correlation_analysis.research.diversification import AnalysisConfig, analyze


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="correlation-analysis",
        description=(
            "Estimate the correlation structure of a panel of series and report "
            "how many independent bets it represents."
        ),
    )

    parser.add_argument("path", type=Path, help="CSV or Parquet file to analyze")
    parser.add_argument(
        "--freq",
        default="1d",
        help=f"bar frequency, one of {sorted(PERIODS_PER_YEAR)} (default: 1d)",
    )
    parser.add_argument(
        "--kind",
        default="returns",
        choices=("returns", "pnl"),
        help="'returns' converts prices to log returns; 'pnl' wraps values as-is",
    )
    parser.add_argument(
        "--method",
        default="log",
        choices=("log", "simple"),
        help="return definition when --kind returns (default: log)",
    )
    parser.add_argument(
        "--index-column",
        default=None,
        help="column to use as the time index instead of the default index",
    )
    parser.add_argument(
        "--periods-per-year",
        type=float,
        default=None,
        help="override the annualization factor for a custom calendar",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="rolling window length; chosen automatically when omitted",
    )
    parser.add_argument(
        "--stress-quantile",
        type=float,
        default=AnalysisConfig.stress_quantile,
        help="tail fraction treated as stress observations (default: 0.10)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=AnalysisConfig.confidence_level,
        help="confidence level for correlation intervals (default: 0.95)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"console width (default: {DEFAULT_WIDTH})",
    )

    return parser


def describe_load(result: LoadResult) -> str:
    lines = [
        f"rows read      {result.rows_read}",
        f"rows dropped   {result.rows_dropped} (incomplete)",
        f"completeness   {result.completeness:.1%}",
    ]

    if result.columns_dropped:
        lines.append(f"non-numeric    {', '.join(result.columns_dropped)}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        result = load_panel(
            args.path,
            freq=args.freq,
            kind=args.kind,
            method=args.method,
            index_column=args.index_column,
            periods_per_year=args.periods_per_year,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(describe_load(result))
    print()

    try:
        config = AnalysisConfig(
            confidence_level=args.confidence,
            stress_quantile=args.stress_quantile,
            rolling_window=args.window,
        )
        report = analyze(result.panel, config)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print_report(report, width=args.width)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
