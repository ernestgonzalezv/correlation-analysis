# correlation-analysis

Estimation of the correlation structure of a panel of financial time series,
with random-matrix noise filtering and a summary statistic for the number of
statistically independent bets the panel represents.

The same engine accepts instrument returns and strategy PnL, since both reduce
to a panel of time-indexed series.

---

## Motivation

A book of five positions is not five bets. If the positions load on a common
factor, the realised risk is that of a concentrated position taken at five times
the nominal size, and the portfolio's apparent diversification is an artefact of
counting instruments rather than counting factors.

The same failure applies one level up. A portfolio of systematic strategies that
share an underlying signal family — several trend-following variants, say — will
draw down together, and the correlation of their PnL is the quantity that
governs the aggregate Sharpe ratio:

```math
S_{\text{portfolio}} \;=\; S \sqrt{\frac{N}{1 + (N-1)\,\rho}}
```

| $N$ | $S$ | $\rho$ | $S_{\text{portfolio}}$ |
|---|---|---|---|
| 5 | 1.0 | 0.30 | 1.51 |
| 5 | 1.0 | 0.10 | 1.89 |
| 10 | 1.0 | 0.10 | 2.29 |

The binding constraint in that expression is $\rho$, not $N$. This package
measures $\rho$, decomposes where it comes from, and reports how much of the
estimate survives a noise filter.

---

## Installation

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Requires Python 3.11 or later. Runtime dependencies are NumPy, pandas, SciPy and
PyArrow.

---

## Usage

### Command line

```bash
correlation-analysis prices.csv --index-column date --freq 1d
correlation-analysis strategy_pnl.parquet --kind pnl --confidence 0.99
correlation-analysis bars.csv --freq 15m --window 500 --stress-quantile 0.05
```

Generate a synthetic file with a known correlation structure to exercise the
pipeline:

```bash
python scripts/generate_example_data.py example.csv --rows 1200
correlation-analysis example.csv --index-column date
```

### Library

```python
from correlation_analysis import AnalysisConfig, analyze, print_report
from correlation_analysis.data.loaders import load_panel

result = load_panel("prices.csv", index_column="date", freq="1d")
report = analyze(result.panel, AnalysisConfig(confidence_level=0.99))

print_report(report)

report.effective_bets          # float
report.signal_factors          # eigenvalues above the noise bound
report.correlation             # (N, N) ndarray
report.stress_lift             # correlation change on the worst days
report.warnings                # threshold breaches, as text
```

Strategy PnL enters through the same path:

```python
from correlation_analysis import analyze, panel_from_pnl

panel = panel_from_pnl(daily_pnl, names=["trend", "carry", "meanrev", "calendar"])
report = analyze(panel)
```

---

## Data

Three ingestion paths, all producing the same `ReturnsPanel`.

### Any CSV or Parquet file

One column per series, one row per bar. Non-numeric columns are dropped and
reported; rows with missing values are removed on a complete-case basis and the
count is printed, rather than being filled silently.

```bash
correlation-analysis prices.csv --index-column date --freq 1d
```

### St. Louis Fed (FRED)

Daily series, no API key, history back to 1999 for the major currencies.

```bash
python scripts/fetch_fred_data.py fred_daily.csv --start 2015-01-01
python scripts/fetch_fred_data.py --list --universe fx
correlation-analysis fred_daily.csv --index-column date
```

Quote conventions are normalised on the way in, so every FX column reads as long
the foreign currency against the dollar. Without this, `USDJPY` and `EURUSD`
appear negatively correlated purely because the dollar sits on opposite sides of
the quote, and the loadings of the dominant factor become unreadable.

### A directory of per-symbol bar files

The layout most terminal exports produce: one file per instrument, each with a
date column and OHLC columns.

```python
from correlation_analysis.data.bars import coverage, load_symbol_frame

for row in coverage("bars/d1"):
    print(row.symbol, row.rows, row.first.date(), row.last.date())

frame = load_symbol_frame("bars/d1", symbols=["EURUSD", "GBPUSD"], start="2018-01-01")
```

Run `coverage` before building a panel. Symbols whose history ends early will
silently truncate a complete-case join: in one 57-symbol archive, two delisted
crosses reduced a 3,808-day sample to 437 days, pushing the observation ratio
below the reliability threshold.

---

## Worked example

A panel is constructed with a known factor structure, so every reported
quantity has a value it is required to reproduce.

### Closed form at N = 2

For two series the correlation matrix is

```math
C = \begin{bmatrix} 1 & \rho \\ \rho & 1 \end{bmatrix}
```

whose eigenvalues are exactly $\lambda_{1,2} = 1 \pm \rho$. Substituting the
resulting proportions $p_{1,2} = (1 \pm \rho)/2$ into the effective bet count
gives a closed form:

```math
N_{\text{eff}} \;=\; \frac{1}{\sum_i p_i^{2}}
\;=\; \frac{4}{(1+\rho)^{2} + (1-\rho)^{2}}
\;=\; \frac{2}{1 + \rho^{2}}
```

| $\rho$ | $\lambda_1$ | $\lambda_2$ | $N_{\text{eff}}$ |
|---|---|---|---|
| 0.00 | 1.0000 | 1.0000 | 2.0000 |
| 0.25 | 1.2500 | 0.7500 | 1.8824 |
| 0.50 | 1.5000 | 0.5000 | 1.6000 |
| 0.75 | 1.7500 | 0.2500 | 1.2800 |
| 0.90 | 1.9000 | 0.1000 | 1.1050 |
| 0.99 | 1.9900 | 0.0100 | 1.0100 |

Two series are worth two bets only at $\rho = 0$, and the count degrades
quadratically rather than linearly: at $\rho = 0.5$ the panel still carries
1.60 independent bets, while at $\rho = 0.9$ it carries 1.11.

### Recovery of a block structure at N = 5

Take a target with two blocks, correlation $0.85$ within and $0.10$ across:

```math
C = \begin{bmatrix}
1.00 & 0.85 & 0.85 & 0.10 & 0.10 \\
0.85 & 1.00 & 0.85 & 0.10 & 0.10 \\
0.85 & 0.85 & 1.00 & 0.10 & 0.10 \\
0.10 & 0.10 & 0.10 & 1.00 & 0.85 \\
0.10 & 0.10 & 0.10 & 0.85 & 1.00
\end{bmatrix}
\qquad
\begin{aligned}
\lambda &= 2.7655,\; 1.7845,\; 0.15,\; 0.15,\; 0.15 \\
N_{\text{eff}} &= 2.2936
\end{aligned}
```

The three repeated eigenvalues at $1 - 0.85 = 0.15$ are the degenerate
directions inside the blocks. Sampling $T = 2000$ observations from this matrix
by Cholesky factorisation and estimating from the sample alone:

| Quantity | Exact | Estimated from $T = 2000$ |
|---|---|---|
| $\lambda_1$ | 2.7655 | 2.7516 |
| $\lambda_2$ | 1.7845 | 1.7855 |
| $\lambda_{3,4,5}$ | 0.1500 | 0.1622, 0.1564, 0.1443 |
| $N_{\text{eff}}$ | 2.2936 | 2.3082 |
| $\max_{ij} \lvert \hat{C}_{ij} - C_{ij} \rvert$ | — | 0.0132 |

The Marchenko-Pastur ceiling at $N = 5$, $T = 2000$ is $1.1025$, and exactly two
eigenvalues clear it, matching the rank of the block construction. The three
noise eigenvalues sit two orders of magnitude below the ceiling and are
correctly discarded.

This is the test the suite runs: a structure is specified, data is generated
from it, and the estimator must return the structure it was given.

---

## Design

All producers emit a single canonical structure, and all consumers read it:

```
  yfinance / MT5 / CSV / Parquet        covariance and correlation
  backtest PnL                 ──>      spectral decomposition
  live PnL                   ReturnsPanel      noise filtering
  synthetic generator          ──>      rolling and stress structure
                                        effective bet count
```

The panel carries `values`, `names`, `index`, `kind` and the annualisation
factor. Frequency is resolved to periods per year at construction, so the
annualisation convention is fixed once rather than at each call site; a custom
calendar is supplied with `periods_per_year`.

```
data/       adapters and generators
  loaders.py    generic CSV and Parquet
  fred.py       St. Louis Fed daily series
  bars.py       per-symbol OHLC directories
  synthetic.py  known-answer generator used by the tests
core/       pure mathematics, no I/O, deterministic
  panel.py      the canonical structure and its validation
  returns.py    price to return conversion
  covariance.py centring, covariance, correlation
  spectral.py   eigendecomposition and effective bets
  noise.py      Marchenko-Pastur bounds and Fisher intervals
  rolling.py    rolling and stress correlation
research/   orchestration into a report
report/     rendering
cli.py      command line entry point
```

`core/` imports nothing from `data/`. It takes NumPy arrays and returns numbers.
That constraint is what allows the mathematics to be validated against
synthetic data with a known answer, and what allows the same engine to run on
instrument returns and on strategy PnL without modification.

---

## Method

### Returns

Prices are converted to log returns, $r_t = \ln(P_t / P_{t-1})$. Correlating
price levels produces spurious results because two trending series correlate
through their common drift. Log returns are additive across time, which makes
aggregation and annualisation exact rather than approximate.

### Covariance and correlation

With $R_c$ the column-centred return matrix:

```math
S = \frac{R_c^{\top} R_c}{T - 1}
\qquad
C = D^{-1} S D^{-1}, \quad D = \operatorname{diag}(\sigma)
```

Bessel's correction is applied because the mean is estimated from the same
sample. The correlation matrix is symmetrised and its diagonal set exactly to
unity, so that downstream eigen-decomposition does not encounter small negative
eigenvalues from floating point asymmetry.

Geometrically, each centred column is a vector in $\mathbb{R}^{T}$ and

```math
\operatorname{corr}(a, b) = \frac{a \cdot b}{\lVert a \rVert \, \lVert b \rVert} = \cos\theta
```

so zero correlation is orthogonality, and the search for uncorrelated strategies
is a search for mutually perpendicular return vectors.

### Spectral decomposition

`C = V L V'` via `numpy.linalg.eigh`, which exploits symmetry and returns an
orthonormal basis. Eigenvalues are sorted in descending order and validated to
be non-negative beyond floating point tolerance; a genuinely indefinite input
raises rather than being silently clipped.

For a correlation matrix, $\sum_i \lambda_i = \operatorname{tr}(C) = N$. The
explained-variance proportions $p_i = \lambda_i / N$ describe how concentrated
the panel's variance is along its principal directions.

```math
C = V \Lambda V^{\top}, \qquad \lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_N \geq 0
```

### Effective number of bets

```math
N_{\text{eff}} = \frac{1}{\sum_{i=1}^{N} p_i^{2}}, \qquad 1 \leq N_{\text{eff}} \leq N
```

The inverse Herfindahl index of the explained-variance distribution. It equals
$N$ when all eigenvalues are equal, and $1$ when a single factor explains the
entire panel. Meucci (2009) develops an entropy-based variant of the same
construction.

### Noise filtering

For `T` independent observations of `N` uncorrelated series, the eigenvalues of
the sample correlation matrix converge to the Marchenko-Pastur distribution,
supported on

```math
\lambda_{\pm} = \left(1 \pm \sqrt{\tfrac{N}{T}}\,\right)^{2}
```

Eigenvalues below $\lambda_{+}$ are indistinguishable from sampling noise. This is
the filter applied in Laloux et al. (1999) and Plerou et al. (2002) to financial
correlation matrices, where the typical finding is that only a small number of
eigenvalues carry information.

| $N$ | $T$ | $\lambda_{+}$ | Note |
|---|---|---|---|
| 8 | 500 | 1.269 | eigenvalues above this carry information |
| 3 | 4 | 3.478 | exceeds the total eigenvalue mass of 3 |

The second case is the common tutorial configuration of a handful of
hand-entered prices. Every eigenvalue it produces sits beneath the noise floor.

### Interval estimation

Pairwise correlations are reported with Fisher z-transformed confidence
intervals. The sampling distribution of $r$ is skewed near the boundaries, while
$\operatorname{arctanh}(r)$ is approximately normal with variance $1/(T-3)$:

```math
\text{CI} = \tanh\!\left( \operatorname{arctanh}(r) \;\pm\; \frac{z_{1-\alpha/2}}{\sqrt{T - 3}} \right)
```

At $r = 0.15$ and $T = 60$ the 95% interval is $[-0.11,\, +0.39]$.

Two standard errors are exposed, because they are routinely conflated:

```math
\operatorname{SE}_{z}(T) = \frac{1}{\sqrt{T-3}}
\qquad
\operatorname{SE}_{r}(r, T) = \frac{1 - r^{2}}{\sqrt{T-1}}
```

exposed as `fisher_z_stderr(T)` and `correlation_stderr(r, T)` respectively.

Reporting a correlation of $0.15000$ from sixty observations without an interval
is false precision; the estimate is consistent with anything from mild negative
dependence to a materially concentrated book.

### Time variation and stress

Correlation is not stationary, and it rises in drawdowns. The report includes a
rolling mean pairwise correlation and a **stress correlation** computed on the
lower tail of the sample, ranked on a standardised equal-weight composite so
that the largest-scale series does not determine which observations count as
adverse. The lift between the two is the quantity of interest: diversification
that disappears under stress was never risk reduction.

The rolling statistic is computed from running sums in $O(T N^{2})$ rather than by
re-estimating a correlation matrix per window, which keeps intraday sample sizes
tractable. The test suite pins the vectorised implementation against a naive
per-window reference.

---

## Validation

```bash
pytest
ruff check .
```

A defect in quantitative code does not raise an exception. It returns a
plausible number. The suite is organised accordingly:

| Class | Approach |
|---|---|
| Known-answer | Cholesky generator produces returns with a specified correlation matrix; the estimator must recover it |
| Identities | $\sum_i \lambda_i = \operatorname{tr}(C)$, $Cv = \lambda v$, orthonormal eigenvectors, $\operatorname{corr} = \cos\theta$ |
| Invariance | Correlation unchanged under rescaling and translation of any column |
| Degenerate | Duplicated series correlate at exactly 1, mirrored at exactly -1, constant series rejected |
| Rejection | NaN, misaligned indices, unknown frequencies, indefinite matrices, oversized windows |
| Coverage | The nominal 95% interval contains the true parameter in at least 85% of repeated trials |
| Equivalence | The vectorised rolling estimator matches a naive reference to $10^{-10}$ |

---

## Limitations

These are properties of the method rather than deferred work.

**The Marchenko-Pastur bound assumes i.i.d. observations.** Financial returns are
heavy-tailed and exhibit volatility clustering, so the empirical noise band is
wider than the asymptotic bound. An eigenvalue marginally above $\lambda_{+}$
should be treated as undetermined. Bouchaud and Potters (2011) survey the corrections.

**The estimator is unshrunk.** For $T/N$ near unity the sample covariance matrix
is poorly conditioned. Ledoit-Wolf shrinkage is the standard remedy and is not
yet implemented; the report flags the ratio instead.

**Correlation measures linear dependence.** Series can be strongly dependent and
measure near zero under a non-monotonic relationship. A low reading is evidence
against a linear relationship only.

**Rolling windows overlap**, so successive readings are autocorrelated. The
reported range is informative; the standard deviation of the rolling series is
not a valid independent-sample statistic.

**The stress window is small by construction.** A 10% tail of 1,200 observations
leaves 120 rows, so the stress correlation carries wide error bars of its own.

**Intraday correlation is biased downward** by non-synchronous observation — the
Epps (1979) effect. Correlation structure should be estimated on daily bars even
when signals are generated intraday.

**Every quantity here is an in-sample description.** It constrains position
sizing; it does not forecast.

---

## Roadmap

```
core/
  returns.py        log and simple returns                      complete
  covariance.py     centring, covariance, correlation           complete
  spectral.py       eigendecomposition, effective bets          complete
  noise.py          Marchenko-Pastur, Fisher intervals          complete
  rolling.py        rolling and stress correlation              complete
  shrinkage.py      Ledoit-Wolf covariance estimation
  stationarity.py   ADF, Engle-Granger cointegration
  timeseries.py     ARMA/GARCH, Ornstein-Uhlenbeck
  sizing.py         Kelly, volatility targeting, risk budgeting
```
