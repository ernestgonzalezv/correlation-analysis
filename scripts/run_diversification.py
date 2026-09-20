from __future__ import annotations

import numpy as np

from eigenrisk import analyze, panel_from_pnl, print_report
from eigenrisk.data.synthetic import block_correlation_matrix, synthetic_panel


def instrument_example() -> None:
    target = block_correlation_matrix([3, 2], within=0.88, between=0.05)

    panel = synthetic_panel(
        target,
        n_obs=1_200,
        names=["EURUSD", "GBPUSD", "AUDUSD", "XAUUSD", "US500"],
        freq="1d",
        seed=9,
    )

    print("\n>>> INSTRUMENTS\n")
    print_report(analyze(panel))


def strategy_example() -> None:
    rng = np.random.default_rng(2024)

    breakout = rng.normal(loc=35.0, scale=280.0, size=1_000)
    mean_reversion = -0.65 * breakout + rng.normal(loc=30.0, scale=210.0, size=1_000)
    swing = 0.80 * breakout + rng.normal(loc=25.0, scale=120.0, size=1_000)
    calendar = rng.normal(loc=20.0, scale=150.0, size=1_000)

    panel = panel_from_pnl(
        np.column_stack([breakout, mean_reversion, swing, calendar]),
        names=["MBR", "MeanRev", "Swing", "Calendar"],
        freq="1d",
    )

    print("\n>>> STRATEGIES\n")
    print_report(analyze(panel))


if __name__ == "__main__":
    instrument_example()
    strategy_example()
