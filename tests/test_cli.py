import numpy as np
import pandas as pd
import pytest

from correlation_analysis.cli import build_parser, main


@pytest.fixture
def price_file(tmp_path):
    rng = np.random.default_rng(90)
    steps = rng.normal(loc=0.0002, scale=0.01, size=(400, 4))
    prices = 100.0 * np.exp(np.cumsum(steps, axis=0))

    frame = pd.DataFrame(prices, columns=["EURUSD", "GBPUSD", "XAUUSD", "US500"])
    path = tmp_path / "prices.csv"
    frame.to_csv(path, index=False)
    return path


def test_parser_defaults():
    args = build_parser().parse_args(["prices.csv"])
    assert args.freq == "1d"
    assert args.kind == "returns"
    assert args.method == "log"
    assert args.window is None


def test_parser_accepts_overrides():
    args = build_parser().parse_args(
        ["p.csv", "--freq", "15m", "--kind", "pnl", "--window", "120"]
    )
    assert args.freq == "15m"
    assert args.kind == "pnl"
    assert args.window == 120


def test_parser_rejects_an_unknown_kind():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["p.csv", "--kind", "levels"])


def test_run_succeeds_and_prints_a_report(price_file, capsys):
    assert main([str(price_file)]) == 0

    output = capsys.readouterr().out
    assert "CORRELATION" in output
    assert "DIVERSIFICATION" in output
    assert "rows read      400" in output


def test_run_reports_a_missing_file(tmp_path, capsys):
    assert main([str(tmp_path / "absent.csv")]) == 1
    assert "no such file" in capsys.readouterr().err


def test_run_reports_an_invalid_frequency(price_file, capsys):
    assert main([str(price_file), "--freq", "3s"]) == 1
    assert "unknown freq" in capsys.readouterr().err


def test_run_reports_an_invalid_window(price_file, capsys):
    assert main([str(price_file), "--window", "2"]) == 1
    assert "rolling_window must be at least" in capsys.readouterr().err


def test_width_flag_changes_the_layout(price_file, capsys):
    main([str(price_file), "--width", "44"])
    assert "=" * 44 in capsys.readouterr().out


def test_pnl_mode_skips_the_price_conversion(price_file, capsys):
    assert main([str(price_file), "--kind", "pnl"]) == 0
    assert "CURRENCY UNITS" in capsys.readouterr().out
