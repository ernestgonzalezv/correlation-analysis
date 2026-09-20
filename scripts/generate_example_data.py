from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from correlation_analysis.data.synthetic import (
    block_correlation_matrix,
    correlated_returns,
)

DEFAULT_NAMES = ("EURUSD", "GBPUSD", "AUDUSD", "XAUUSD", "US500")
DEFAULT_BLOCKS = (3, 2)
DEFAULT_START_PRICE = 100.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate-example-data",
        description=(
            "Write a synthetic price file with a known block correlation "
            "structure, for exercising the analysis pipeline end to end."
        ),
    )

    parser.add_argument("output", type=Path, help="destination .csv or .parquet")
    parser.add_argument("--rows", type=int, default=1_200, help="number of bars")
    parser.add_argument(
        "--within", type=float, default=0.88, help="correlation inside each block"
    )
    parser.add_argument(
        "--between", type=float, default=0.05, help="correlation across blocks"
    )
    parser.add_argument(
        "--daily-volatility", type=float, default=0.01, help="per-bar volatility"
    )
    parser.add_argument("--seed", type=int, default=9, help="random seed")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    target = block_correlation_matrix(
        list(DEFAULT_BLOCKS), within=args.within, between=args.between
    )

    returns = correlated_returns(
        target,
        n_obs=args.rows,
        volatilities=np.full(len(DEFAULT_NAMES), args.daily_volatility),
        seed=args.seed,
    )

    prices = DEFAULT_START_PRICE * np.exp(np.cumsum(returns, axis=0))
    index = pd.date_range("2020-01-01", periods=args.rows, freq="B", name="date")
    frame = pd.DataFrame(prices, columns=list(DEFAULT_NAMES), index=index)

    suffix = args.output.suffix.lower()
    if suffix == ".parquet":
        frame.to_parquet(args.output)
    else:
        frame.to_csv(args.output)

    print(f"wrote {len(frame)} rows x {frame.shape[1]} columns to {args.output}")
    print(f"target structure: blocks {DEFAULT_BLOCKS}, within {args.within}, "
          f"between {args.between}")
    print()
    print("analyze it with:")
    print(f"    correlation-analysis {args.output} --index-column date")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
