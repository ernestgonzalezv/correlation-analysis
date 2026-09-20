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

```
S_portfolio = S * sqrt( N / (1 + (N - 1) * rho) )

  N = 5,  S = 1.0,  rho = 0.30  ->  1.51
  N = 5,  S = 1.0,  rho = 0.10  ->  1.89
  N = 10, S = 1.0,  rho = 0.10  ->  2.29
```

The binding constraint in that expression is `rho`, not `N`. This package
measures `rho`, decomposes where it comes from, and reports how much of the
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
data/       adapters, I/O, synthetic generators
core/       pure mathematics, no I/O, deterministic
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

Prices are converted to log returns, `r_t = ln(P_t / P_{t-1})`. Correlating
price levels produces spurious results because two trending series correlate
through their common drift. Log returns are additive across time, which makes
aggregation and annualisation exact rather than approximate.

### Covariance and correlation

With `Rc` the column-centred return matrix:

```
S = Rc' Rc / (T - 1)
C = D^-1 S D^-1,      D = diag(sigma)
```

Bessel's correction is applied because the mean is estimated from the same
sample. The correlation matrix is symmetrised and its diagonal set exactly to
unity, so that downstream eigen-decomposition does not encounter small negative
eigenvalues from floating point asymmetry.

Geometrically, each centred column is a vector in `R^T` and

```
corr(a, b) = (a . b) / (|a| |b|) = cos(theta)
```

so zero correlation is orthogonality, and the search for uncorrelated strategies
is a search for mutually perpendicular return vectors.

### Spectral decomposition

`C = V L V'` via `numpy.linalg.eigh`, which exploits symmetry and returns an
orthonormal basis. Eigenvalues are sorted in descending order and validated to
be non-negative beyond floating point tolerance; a genuinely indefinite input
raises rather than being silently clipped.

For a correlation matrix, `sum(lambda_i) = trace(C) = N`. The explained-variance
proportions `p_i = lambda_i / N` describe how concentrated the panel's variance
is along its principal directions.

### Effective number of bets

```
N_eff = 1 / sum(p_i^2)
```

The inverse Herfindahl index of the explained-variance distribution. It equals
`N` when all eigenvalues are equal, and 1 when a single factor explains the
entire panel. Meucci (2009) develops an entropy-based variant of the same
construction.

### Noise filtering

For `T` independent observations of `N` uncorrelated series, the eigenvalues of
the sample correlation matrix converge to the Marchenko-Pastur distribution,
supported on

```
lambda_+- = (1 +- sqrt(N / T))^2
```

Eigenvalues below `lambda_+` are indistinguishable from sampling noise. This is
the filter applied in Laloux et al. (1999) and Plerou et al. (2002) to financial
correlation matrices, where the typical finding is that only a small number of
eigenvalues carry information.

```
N = 8,  T = 500  ->  lambda_+ = 1.269
N = 3,  T = 4    ->  lambda_+ = 3.478, exceeding the total eigenvalue mass of 3
```

The second case is the common tutorial configuration of a handful of
hand-entered prices. Every eigenvalue it produces sits beneath the noise floor.

### Interval estimation

Pairwise correlations are reported with Fisher z-transformed confidence
intervals. The sampling distribution of `r` is skewed near the boundaries, while
`arctanh(r)` is approximately normal with variance `1 / (T - 3)`:

```
CI = tanh( arctanh(r)  +-  z_(1-a/2) / sqrt(T - 3) )

r = 0.15, T = 60   ->  95% CI  [-0.11, +0.39]
```

Two standard errors are exposed, because they are routinely conflated:

```
fisher_z_stderr(T)        = 1 / sqrt(T - 3)           scale of arctanh(r)
correlation_stderr(r, T)  = (1 - r^2) / sqrt(T - 1)   scale of r
```

Reporting a correlation of `0.15000` from sixty observations without an interval
is false precision; the estimate is consistent with anything from mild negative
dependence to a materially concentrated book.

### Time variation and stress

Correlation is not stationary, and it rises in drawdowns. The report includes a
rolling mean pairwise correlation and a **stress correlation** computed on the
lower tail of the sample, ranked on a standardised equal-weight composite so
that the largest-scale series does not determine which observations count as
adverse. The lift between the two is the quantity of interest: diversification
that disappears under stress was never risk reduction.

The rolling statistic is computed from running sums in `O(T N^2)` rather than by
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
| Identities | `sum(lambda) = trace`, `Cv = lambda v`, orthonormal eigenvectors, `corr = cos(theta)` |
| Invariance | Correlation unchanged under rescaling and translation of any column |
| Degenerate | Duplicated series correlate at exactly 1, mirrored at exactly -1, constant series rejected |
| Rejection | NaN, misaligned indices, unknown frequencies, indefinite matrices, oversized windows |
| Coverage | The nominal 95% interval contains the true parameter in at least 85% of repeated trials |
| Equivalence | The vectorised rolling estimator matches a naive reference to 1e-10 |

---

## Limitations

These are properties of the method rather than deferred work.

**The Marchenko-Pastur bound assumes i.i.d. observations.** Financial returns are
heavy-tailed and exhibit volatility clustering, so the empirical noise band is
wider than the asymptotic bound. An eigenvalue marginally above `lambda_+` should
be treated as undetermined. Bouchaud and Potters (2011) survey the corrections.

**The estimator is unshrunk.** For `T/N` near unity the sample covariance matrix
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

---

## References

Bouchaud, J.-P. and Potters, M. (2011). *Financial applications of random matrix
theory: a short review.* In The Oxford Handbook of Random Matrix Theory.

Epps, T. W. (1979). *Comovements in stock prices in the very short run.* Journal
of the American Statistical Association 74, 291-298.

Fisher, R. A. (1921). *On the probable error of a coefficient of correlation
deduced from a small sample.* Metron 1, 3-32.

Laloux, L., Cizeau, P., Bouchaud, J.-P. and Potters, M. (1999). *Noise dressing
of financial correlation matrices.* Physical Review Letters 83, 1467-1470.

Ledoit, O. and Wolf, M. (2004). *A well-conditioned estimator for
large-dimensional covariance matrices.* Journal of Multivariate Analysis 88,
365-411.

Marchenko, V. A. and Pastur, L. A. (1967). *Distribution of eigenvalues for some
sets of random matrices.* Matematicheskii Sbornik 72, 507-536.

Meucci, A. (2009). *Managing diversification.* Risk 22(5), 74-79.

Plerou, V., Gopikrishnan, P., Rosenow, B., Amaral, L. A. N., Guhr, T. and
Stanley, H. E. (2002). *Random matrix approach to cross correlations in
financial data.* Physical Review E 65, 066126.

---

## Development

GitFlow. `main` carries tagged releases, `develop` is the integration branch,
work happens on `feature/*` and merges back with `--no-ff`.

CI runs `ruff` and `pytest` against Python 3.11, 3.12 and 3.13.
