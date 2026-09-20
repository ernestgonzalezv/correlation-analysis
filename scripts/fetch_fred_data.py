from __future__ import annotations

import argparse
from pathlib import Path

from correlation_analysis.data.fred import (
    CROSS_ASSET,
    DEFAULT_UNIVERSE,
    FX_MAJORS,
    describe,
    fetch_frame,
)

UNIVERSES = {
    "default": DEFAULT_UNIVERSE,
    "fx": FX_MAJORS,
    "cross-asset": CROSS_ASSET,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fetch-fred-data",
        description=(
            "Download daily series from the St. Louis Fed (FRED) and write an "
            "aligned price file. No API key is required."
        ),
    )

    parser.add_argument("output", type=Path, help="destination .csv or .parquet")
    parser.add_argument(
        "--universe", default="default", choices=sorted(UNIVERSES), help="series set"
    )
    parser.add_argument("--start", default="2015-01-01", help="first observation date")
    parser.add_argument("--end", default=None, help="last observation date")
    parser.add_argument(
        "--list", action="store_true", help="print the universe and exit"
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    universe = UNIVERSES[args.universe]

    if args.list:
        print(describe(universe))
        return 0

    frame = fetch_frame(universe, start=args.start, end=args.end)
    complete = frame.dropna(axis=0, how="any")

    if args.output.suffix.lower() == ".parquet":
        complete.to_parquet(args.output)
    else:
        complete.to_csv(args.output)

    print(describe(universe))
    print()
    print(f"downloaded   {len(frame)} dates")
    print(f"complete     {len(complete)} dates ({len(complete) / len(frame):.1%})")
    print(f"range        {complete.index.min().date()} to {complete.index.max().date()}")
    print(f"written      {args.output}")
    print()
    print("analyze it with:")
    print(f"    correlation-analysis {args.output} --index-column date")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
