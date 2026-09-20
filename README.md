# eigenrisk

A research engine that answers one question:

> **How many independent bets am I actually taking?**

Feed it any panel of time series — instrument prices or strategy PnL, at any
timeframe — and it returns the risk structure hiding inside: correlation,
latent factors, how much of it is noise, and the single number that matters,
the **effective number of independent bets**.

---

## The problem it solves

You open EUR/USD long and GBP/USD long. Two pairs, two positions, feels
diversified. The dollar strengthens and both lose together.

That was never two bets. It was **one bet (short dollar) taken twice at double
size**. The same trap applies to strategies: five systems that lose on the same
day are not five systems.

`eigenrisk` measures that, with a number.

---

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python scripts/run_diversification.py
```

```python
import numpy as np
from eigenrisk import analyze, panel_from_prices, panel_from_pnl, print_report

panel = panel_from_prices(prices, names=["EURUSD", "GBPUSD", "XAUUSD"], freq="1d")
print_report(analyze(panel))

panel = panel_from_pnl(daily_pnl, names=["MBR", "MeanRev", "Swing", "Calendar"])
print_report(analyze(panel))
```

Same engine, same call. That is the whole design.

---

## The central abstraction

Everything flows through one structure, `ReturnsPanel`:

```
  PRODUCERS                  CANONICAL FORM               CONSUMERS

  yfinance     ─┐                                    ┌─  covariance
  MT5           │                                    │   eigen / PCA
  CSV / parquet ├──────>   ReturnsPanel    ──────────┤   noise filtering
  backtest PnL  │          (T x N matrix              │   rolling structure
  live PnL      │           + names                   │   stress analysis
  synthetic    ─┘           + frequency               └─  effective bets
                            + kind)
```

The engine does not know or care whether a column is EUR/USD or a breakout bot.
Both are columns of numbers indexed by time.

The practical consequence: **7 producers + 8 analyses is 15 pieces of code, not
56.** Add a data source and every analysis works with it for free. Add an
analysis and it works with every source for free.

The panel carries its own `freq`, which is the only place the annualization
factor is ever decided. Annualizing a Sharpe ratio with the wrong `sqrt()` is
the most common silent bug in retail quant work; here it is impossible to do by
hand.

---

## Architecture

```
adapters (eigenrisk/data)     I/O, network, disk. Fails, retries, caches.
        │  produces ReturnsPanel
        v
core (eigenrisk/core)         Pure mathematics. Zero I/O. Deterministic.
        │  produces numbers
        v
research (eigenrisk/research) Orchestrates core to answer a question.
        │  produces a report
        v
report (eigenrisk/report)     Presentation only. Never computes.
```

**The rule that holds it together: `core/` imports nothing from `data/`.**

It receives NumPy arrays and returns numbers. If a file in `core/` ever mentions
`yfinance` or `parquet`, the design is broken.

This is not aesthetics. It is what makes the mathematics testable against
synthetic data with a known answer, and what lets the same engine run on prices
today and on strategy PnL tomorrow.

Dependency direction is strictly one way:

```
core  <──  research  <──  report
core  <──  data
```

---

## The mathematics

### 1. Prices to returns

Prices are never correlated directly. Two series that both trend upward always
show high correlation, whether or not they are related — that is spurious
correlation from a shared trend, not a relationship.

```
r_t = ln( P_t / P_(t-1) )
```

Log returns are used because they are **additive over time**: the weekly return
is the sum of the five daily returns. They are also approximately stationary,
which is what every statistical method downstream assumes.

### 2. Covariance is a matrix product

Each column of the centered panel is a **vector in T-dimensional space** — one
dimension per observation. Its direction is that series' movement history.

The dot product of two such vectors measures whether they point the same way:

```
both up on the same day    ->  (+)(+) = positive
both down on the same day  ->  (-)(-) = positive
one up, one down           ->  (+)(-) = negative
unrelated                  ->  signs cancel -> near zero
```

The whole covariance matrix is a single matrix multiplication, not a loop over
pairs:

```
S = Rc.T @ Rc / (T - 1)
```

Entry `[i, j]` is the dot product of column `i` with column `j`. Bessel's
correction `(T - 1)` is used because the mean is estimated from the same sample.

### 3. Correlation is the cosine of an angle

```
corr(A, B) = (a . b) / (|a| |b|) = cos(theta)
```

This is not an analogy. Correlation **is** cosine similarity between centered
return vectors:

```
corr = +1.00  ->    0 degrees  ->  parallel, the same bet
corr = +0.86  ->   31 degrees
corr =  0.00  ->   90 degrees  ->  ORTHOGONAL, genuinely independent
corr = -1.00  ->  180 degrees  ->  exact opposites
```

Searching for uncorrelated strategies is, literally, searching for perpendicular
vectors.

### 4. Eigenvectors find the hidden forces

A marionette has eight strings but two hands. You see eight movements; there are
two.

Eigenvectors of the correlation matrix are the hands. Feed in eight FX pairs and
the decomposition discovers — without being told what a dollar is — that they
are driven by a small number of latent factors, because the loading signs align:

```
Factor 1  (eigenvalue 2.48):
   EURUSD  +0.52   GBPUSD  +0.49
   AUDUSD  +0.47   USDJPY  -0.38      <- the dollar factor

Factor 2  (eigenvalue 0.91):
   XAUUSD  +0.71   US500   -0.55      <- risk-off
```

Eigenvalues measure how much of the total movement lives along each direction.
Because a correlation matrix is symmetric and positive semi-definite, the
spectral theorem guarantees real eigenvalues and orthogonal eigenvectors. A
useful identity that the test suite enforces:

```
sum of eigenvalues = trace = N
```

### 5. Effective number of bets

```
p_i   = lambda_i / sum(lambda)
N_eff = 1 / sum(p_i^2)
```

```
8 instruments, all independent   ->  N_eff = 8.0
8 instruments, one hidden factor ->  N_eff = 2.3
```

Diversification is not counting positions. It is counting factors.

### 6. Separating signal from noise

This is what separates a chart from a decision.

A correlation matrix estimated from a short sample produces eigenvalues that
look like structure but are pure sampling noise. Marchenko-Pastur gives the
ceiling below which an eigenvalue is indistinguishable from randomness:

```
lambda_max = (1 + sqrt(N / T))^2

N = 8,  T = 500  ->  1.269    only eigenvalues above this are signal
N = 3,  T = 4    ->  3.478    total eigenvalue mass is 3 -> all of it is noise
```

That second row is the standard tutorial example: five hand-typed prices for
three instruments. Every eigenvalue it produces is beneath the noise floor.

Correlations carry the same problem, so every estimate is reported with a Fisher
z-transformed confidence interval:

```
z  = arctanh(r)
CI = tanh( z  +/-  z_crit / sqrt(T - 3) )

r = 0.15 over 60 observations  ->  95% CI  [-0.11, +0.39]
```

A correlation of 0.15 measured over three months could easily be 0.39. Sizing
two strategies as independent on that basis is how accounts die. Reporting
`0.15000` with five decimals and no interval is false precision.

### 7. Correlation is a film, not a number

Correlation moves, and it rises exactly when it hurts. Two strategies averaging
0.1 can run at 0.8 during a bad month — and that month is the one that ends the
account.

The engine therefore reports a rolling window and a **stress correlation**
computed only over the worst days of the sample, plus the lift between them.

---

## Reading the output

```
FACTOR STRUCTURE
factor        eigenvalue   explained   cumulative       verdict
1                 2.7937      55.9%        55.9%        signal
2                 1.8441      36.9%        92.8%        signal
3                 0.1283       2.6%        95.3%         noise

Marchenko-Pastur noise ceiling   1.1333
factors above the ceiling        2

DIVERSIFICATION
series held              5
effective bets           2.222
diversification ratio    44.4%
```

Five instruments behaving like 2.2 independent bets. Two real factors; the rest
is sampling noise.

The report ends with warnings that fire only when they mean something: an
unreliable sample size, a concentration that contradicts the position count, or
correlation that spikes under stress.

---

## Testing

```bash
.venv/bin/python -m pytest
```

In quantitative work a mathematical bug does not raise an exception. It quietly
returns a wrong number and costs money. The suite is built around that:

- **Known-answer oracles.** `data/synthetic.py` generates returns with a
  correlation matrix you choose, via Cholesky decomposition. If the engine
  recovers the matrix you asked for, the engine is correct.
- **Mathematical identities.** Eigenvalues sum to the trace. `Cv = lambda v`
  holds for every eigenvector. Eigenvectors are orthonormal. Correlation equals
  the cosine of the angle between centered vectors.
- **Invariances.** Correlation is unchanged by rescaling a column by 1000 or
  shifting it by a constant.
- **Degenerate cases.** Duplicated series correlate at exactly 1. Mirrored series
  at exactly -1. Constant series are rejected rather than returning NaN.
- **Boundary rejection.** Most tests assert what the code refuses: NaN, misaligned
  indices, unknown frequencies, windows longer than the sample.
- **Statistical coverage.** The 95% confidence interval is verified to contain
  the true correlation in at least 85% of repeated trials.

---

## Roadmap

Each module lands in the layer it belongs to; nothing gets rewritten.

```
core/
  returns.py        prices -> log returns                 done
  covariance.py     centering, covariance, correlation    done
  spectral.py       eigendecomposition, effective bets    done
  noise.py          Marchenko-Pastur, Fisher intervals    done
  rolling.py        rolling and stress correlation        done
  stationarity.py   ADF, cointegration
  timeseries.py     ARMA/GARCH, Ornstein-Uhlenbeck
  sizing.py         Kelly, volatility targeting

data/               live adapters: MT5, CSV, crypto
strategies/         one signal definition per file
backtest/           event-driven engine, walk-forward, purged CV
portfolio/          combining strategies into one book
execution/          broker interface, state machine, circuit breakers
monitor/            live drift detection
```

A strategy definition lives in exactly one file, imported by both the backtest
engine and the live executor. That is deliberate: when research and production
hold separate implementations of the same signal, live results stop matching the
backtest and the cause becomes unfindable.

---

## Development

GitFlow.

```
main        tagged releases
develop     integration
feature/*   branched from develop, merged back with --no-ff
release/*   release preparation
hotfix/*    branched from main
```
