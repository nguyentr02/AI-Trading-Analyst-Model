"""Risk-adjusted performance metrics, with how much to trust them.

See docs/research/sharpe-ratio.md. All take per-period returns `r` (a pandas Series) and `q`, the number of
periods per year (2,190 for 4h candles, 365 for daily: crypto trades every day). The risk-free rate is 0.
"""
import numpy as np
from scipy.stats import kurtosis, norm, skew

EULER = 0.5772156649


def sharpe(r, q):
    """Annualised Sharpe ratio: average return divided by its volatility."""
    return r.mean() / r.std() * np.sqrt(q) if r.std() > 0 else np.nan


def sharpe_se(r, q):
    """Standard error of the annualised Sharpe (Lo 2002, i.i.d. returns): about how far it could be off."""
    sr = r.mean() / r.std() if r.std() > 0 else 0.0
    return np.sqrt((1 + sr ** 2 / 2) / len(r)) * np.sqrt(q)


def psr(r, q, benchmark_sharpe=0.0):
    """Probabilistic Sharpe ratio (Bailey & López de Prado 2012): chance the true Sharpe beats the benchmark.

    Accounts for the sample length and for fat tails and skew in the returns.
    """
    if r.std() == 0 or len(r) < 3:
        return np.nan
    sr, n = r.mean() / r.std(), len(r)
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    threshold = benchmark_sharpe / np.sqrt(q)
    denom = 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2
    return float(norm.cdf((sr - threshold) * np.sqrt(n - 1) / np.sqrt(max(denom, 1e-12))))


def deflated_sharpe(r, q, n_trials):
    """Deflated Sharpe ratio (Bailey & López de Prado 2014): PSR against the best Sharpe you would expect
    from `n_trials` strategies with no real edge. Uses the conservative trial variance 1/len(r)."""
    if n_trials < 2:
        return psr(r, q)
    sr0 = np.sqrt(1 / len(r)) * ((1 - EULER) * norm.ppf(1 - 1 / n_trials)
                                 + EULER * norm.ppf(1 - 1 / (n_trials * np.e)))
    return psr(r, q, sr0 * np.sqrt(q))


def sortino(r, q):
    """Like Sharpe, but only counts downside volatility (falls), not upside swings."""
    downside = np.sqrt((np.minimum(r, 0) ** 2).mean())
    return r.mean() / downside * np.sqrt(q) if downside > 0 else np.nan


def max_drawdown(equity):
    return float((equity / equity.cummax() - 1).min())


def calmar(equity, q):
    """Yearly growth rate divided by the worst drop (max drawdown)."""
    years = len(equity) / q
    if years <= 0 or equity.iloc[0] <= 0:
        return np.nan
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    dd = abs(max_drawdown(equity))
    return cagr / dd if dd > 0 else np.nan


def info_ratio(r, r_bench, q):
    """Sharpe of the strategy's returns minus the benchmark's: does it beat buy & hold, per unit of difference?"""
    return sharpe(r - r_bench, q)


def summary(r, q, n_trials=None, bench=None):
    """All of the above for one return series; `bench` = buy & hold returns on the same periods."""
    out = {
        "sharpe": sharpe(r, q),
        "sharpe_se": sharpe_se(r, q),
        "psr": psr(r, q),
        "sortino": sortino(r, q),
        "calmar": calmar((1 + r).cumprod(), q),
    }
    if n_trials:
        out["deflated_sharpe"] = deflated_sharpe(r, q, n_trials)
    if bench is not None:
        out["psr_vs_hold"] = psr(r, q, sharpe(bench, q))
        out["info_ratio"] = info_ratio(r, bench, q)
    return out
