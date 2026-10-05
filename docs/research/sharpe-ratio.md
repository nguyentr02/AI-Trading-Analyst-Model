# Sharpe ratios: what they measure, how far to trust them, and what our backtest really shows

*Research note, 2026-10-05. Written for `cryptoai/` (4h candles, BTC/ETH/BNB/SOL vs USDT on Binance spot, long-or-flat, 0.1% fee per side). Citations are numbered; see [Sources](#sources). Figures marked "(this note's check)" were computed for this note from the repo's walk-forward predictions (`models/oos_4h.csv`) with `cryptoai.backtest`, without changing any repo file.*

## Bottom line

- **The Sharpe ratio is return per unit of wobble.** It is the average return above a risk-free rate, divided by the standard deviation of returns [1][2]. Higher is better, but one number cannot capture crash risk, skew or luck.
- **Our calculation is correct for what it claims.** It uses 6 × 365 periods a year for 4h candles (365 for daily, not 252), includes flat periods as zero return, deducts the fee on every switch, and assumes a 0% risk-free rate. Serial correlation is negligible, so √(periods) annualisation is fine here [3] (this note's check).
- **Five years of data cannot pin a Sharpe ratio down closely.** The standard error is about **±0.44 per coin** whether you use 4h or daily data, because it depends on the number of *years*, not candles [3] (this note's check). A Sharpe of 0.51 is roughly 1.2 standard errors from zero.
- **Is the strategy's Sharpe above 0?** Probably, but not convincingly per coin. The Probabilistic Sharpe Ratio (PSR) [5] gives 81–87% for BTC, ETH and BNB and 99.9% for SOL. An equal-weight basket of the four strategies gives **0.91, PSR 98%**. After deflating for the roughly 10–50 variants we tried [6], BTC, ETH and BNB fall to roughly 10–70% depending on assumptions. SOL and the basket mostly survive (this note's check).
- **Does it beat buy-and-hold?** Not shown, except for SOL. For BTC, ETH and BNB, the Sharpe gap over buy-and-hold is 0.03–0.11 with a bootstrap standard error of about 0.30. For the basket the gap is +0.39 ± 0.27, and its 90% interval includes zero (this note's check). The strategy wins mainly by being out of the market about half the time, which cuts volatility and drawdown; it does not earn more per unit of risk taken.
- **Fees are the biggest lever.** About 470 round trips per coin cost about 18–19% a year. The Sharpe ratios are 0.76–1.61 with no fees, 0.40–1.34 at 0.1% per side, and 0.03–1.07 at 0.2% per side (this note's check).
- **Recommendation:** in the dashboard, show Sharpe **± its standard error**, PSR, a deflated Sharpe with the trial count, Sortino, max drawdown/Calmar, and Sharpe of the *difference* from buy-and-hold. Give the simulator a Sharpe at all (it has none today). Code sketches are in §7.

---

## 1. Definition and origin

William Sharpe introduced the measure in 1966 as the **"reward-to-variability ratio"** for comparing mutual funds [1]. Other people started calling it the Sharpe ratio, and Sharpe adopted that name in his 1994 restatement [2].

In the 1994 form, let *D* be the **differential return**: the fund's return minus a benchmark's return, originally a riskless asset [2].

- **Ex-ante (forward-looking):** S = expected D / predicted standard deviation of D. This is the one you would use to decide what to hold.
- **Ex-post (historical):** Sh = average D / standard deviation of D over the period. This is what a backtest reports.

Every number in this note is ex-post. An ex-post Sharpe is an *estimate* of the ex-ante one, and the estimate is noisy (§2.3).

Sharpe also notes that the benchmark need not be cash. Any "long the fund, short the benchmark" difference works [2]. Using buy-and-hold as the benchmark turns the Sharpe ratio into an information ratio (§5).

## 2. Computing it in practice

### 2.1 Per-period returns and annualisation

1. Take the per-period return series *r_t*, one value per candle, including candles where you were flat (return 0). Fees go in the period they are paid.
2. Subtract the per-period risk-free rate *r_f*.
3. Compute SR_period = mean(r − r_f) / std(r − r_f).
4. Annualise: SR_annual = SR_period × √q, where q is the number of periods per year.

Step 4 is valid only if returns are **uncorrelated over time** [2][3]. If returns are serially correlated, Lo shows the right factor is

η(q) = q / √( q + 2 Σ_{k=1}^{q−1} (q − k) ρ_k ),

where ρ_k is the lag-k autocorrelation. Using √q instead can be badly wrong. In his hedge-fund sample, annual Sharpe ratios were **overstated by as much as 65%** because monthly returns were positively autocorrelated [3]. Smoothed or illiquid returns cause exactly this.

**Crypto trades 24/7, so q differs from stocks:**

| Candle | Periods per year q | √q |
|---|---|---|
| 4h | 6 × 365 = 2,190 | 46.8 |
| 1d | 365 (not 252) | 19.1 |
| 1h | 24 × 365 = 8,760 | 93.6 |

Using the equity-market 252 for daily crypto understates the Sharpe by a factor √(252/365) ≈ 0.83. For example, BTC buy-and-hold on Binance daily data (2019-01 to 2026-10) gives 0.97 with √365 but 0.80 with √252 (this note's check).

### 2.2 Which risk-free rate for a USDT trader?

The textbook choice is a short T-bill rate [2]. For someone holding USDT, the honest benchmark is what the idle USDT could earn. That is 0% if it sits on the exchange, or a stablecoin savings or T-bill-like yield if it is parked. I could not verify a stablecoin yield history from a primary source, so none is quoted.

For scale, the 1-month T-bill rate in the Kenneth French data library averaged **3.6% a year from 2021-09 to 2026-08** [13] (this note's check). Re-running our backtest with that monthly series (this note's check):

| Coin | r_f = 0 (as reported) | Excess return; idle cash earns T-bill | Excess return; idle cash earns 0 | Buy & hold: r_f = 0 → excess |
|---|---|---|---|---|
| BTC | 0.51 | 0.46 | 0.41 | 0.47 → 0.40 |
| ETH | 0.40 | 0.37 | 0.32 | 0.29 → 0.23 |
| BNB | 0.51 | 0.48 | 0.42 | 0.45 → 0.39 |
| SOL | 1.34 | 1.32 | 1.29 | 0.59 → 0.55 |

So **the risk-free rate shifts every Sharpe by only about 0.03–0.10**, because crypto volatility (38–96% a year here) is far larger than 3–5% interest. Ranking and conclusions do not change. A 0% rate is defensible for USDT left on the exchange. State the assumption next to the number.

### 2.3 How precise is a Sharpe ratio? (Lo's standard error)

If returns are i.i.d. and normal, the estimated Sharpe ratio is approximately normal with [3]

**SE(SR̂) ≈ √( (1 + SR²/2) / T )**, where SR and T are in the same units (per period, number of periods).

In annual units with T_years years of data, this is roughly SE ≈ √((1 + SR²/2) / T_years) when measured yearly. With high-frequency data, the per-period SR is tiny, and the annualised SE becomes ≈ √q · √(1/T) = **1 / √(years)**.

**Practical consequence:** sampling more often (4h instead of daily) does *not* make a Sharpe ratio more precise. Only more calendar time does. With 5.1 years of test data, SE ≈ 1/√5.1 ≈ **0.44**.

| Years of data | SE of annual Sharpe (approx.) | Sharpe needed for ~95% one-sided confidence > 0 (1.645 × SE) |
|---|---|---|
| 1 | 1.0 | 1.6 |
| 3 | 0.58 | 0.95 |
| 5 | 0.45 | 0.74 |
| 10 | 0.32 | 0.52 |

*(Computed from Lo's formula with small SR; this note's check.)*

Lo's i.i.d. formula implicitly uses the normal distribution's variance of σ̂². Mertens (2002) generalised it to non-normal returns, adding skewness (γ₃) and kurtosis (γ₄) terms [4] (cited via [6]; I did not open the original):

Var(SR̂) ≈ (1 + SR²/2 − γ₃·SR + (γ₄ − 3)/4 · SR²) / T.

This formula is the basis of PSR (§4.4).

## 3. How to read values

**Practitioner rule of thumb** (not a tested result): 1.0–2.0 "good", 2.0–3.0 "very good", 3.0+ "excellent" [14]. These bands come from fund marketing. They ignore sample length, and they are far above what diversified long-only assets achieve.

**Reference points that can be checked:**

| Asset, period | Annualised Sharpe | Source |
|---|---|---|
| US stock market excess return (Fama-French Mkt-RF), 1926-07 to 2026-08, monthly | 0.45 | [13] (this note's check) |
| Same, 1976-01 to 2026-08 | 0.56 | [13] (this note's check) |
| Same, 2021-09 to 2026-08 (≈ our test window) | 0.55 | [13] (this note's check) |
| BTC buy & hold, Binance daily, 2019-01 to 2026-10, r_f = 0 | 0.97 | repo data (this note's check) |
| ETH buy & hold, same | 0.88 | repo data (this note's check) |
| BTC buy & hold, 4h, 2021-08 to 2026-10 (our test window) | 0.47 | repo data (this note's check) |

BTC's Sharpe depends heavily on the start date: 2019 started near a bear-market low, while 2021-08 is close to a cycle top. I did not find a primary source that publishes a long-run BTC Sharpe ratio I could verify. Borri, Liu, Tsyvinski & Wu describe crypto's risk-adjusted performance as "broadly comparable" to traditional markets, without a number in the abstract (source [3] of `reading-crypto-charts-for-day-trading.md`).

**Takeaway for us:** a strategy Sharpe of 0.4–0.6 on one coin over 5 years is in the same range as simply holding the stock market or BTC. It is not, by itself, evidence of skill.

## 4. Pitfalls

### 4.1 Fat tails and skew are invisible

The Sharpe ratio uses only mean and variance. Two strategies with the same Sharpe can have very different crash risk. Harvey & Liu warn that high Sharpe ratios can come from "an option-like strategy with high ex ante negative skew" [7].

Our strategy returns at 4h have **excess kurtosis of 17–24** (normal = 0), versus 7.5–10 for buy-and-hold. Skew ranges from −0.43 (BNB) to +0.77 (SOL) (this note's check). Long-or-flat switching creates many zeros and a few big moves, which is why kurtosis is high.

### 4.2 It penalises upside volatility

A big up-candle raises the standard deviation just as much as a big down-candle. Strategies that cut losers and keep winners (positive skew) are therefore understated. The Sortino ratio (§5) addresses this.

### 4.3 It can be gamed

Ingersoll, Spiegel, Goetzmann & Welch show that a manager can raise a measured Sharpe ratio without skill, for example by selling options or otherwise reshaping the return distribution, and they propose manipulation-proof measures [8]. Smoothing or illiquid pricing creates positive autocorrelation, which lowers measured volatility and inflates √q-annualised Sharpe ratios [3]. Our backtest uses exchange closes, so smoothing is not an issue. A strategy that always holds a small, steady carry position could still look good on Sharpe while hiding tail risk.

### 4.4 Short samples and many tries: PSR and DSR

**Probabilistic Sharpe Ratio** (Bailey & López de Prado 2012) [5]. PSR answers: *given this track record's length, skew and fat tails, what is the probability that the true Sharpe exceeds a threshold SR\*?*

PSR(SR\*) = Φ( (SR̂ − SR\*) · √(T − 1) / √(1 − γ₃·SR̂ + (γ₄ − 1)/4 · SR̂²) )

- SR̂ is the **per-period** (not annualised) Sharpe.
- T is the number of periods.
- γ₃ is skewness and γ₄ is kurtosis (3 for normal).
- Φ is the standard normal CDF.

PSR = 0.95 means "95% confident the true Sharpe is above SR\*".

**Deflated Sharpe Ratio** (Bailey & López de Prado 2014) [6]. If you try N strategies and keep the best, its Sharpe is inflated even when none has skill. DSR is a PSR whose threshold is the Sharpe you would *expect the best of N skill-less trials to show*:

SR₀ = √V · ( (1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ),   DSR = PSR(SR₀)

- V is the variance of the Sharpe ratios across the N trials.
- γ ≈ 0.5772 is the Euler–Mascheroni constant.
- e ≈ 2.718.

N should count *effective independent* trials. Highly correlated variants count as fewer than N, and the paper's Appendix 3 shows how to estimate this [6].

**Haircut Sharpe ratios** (Harvey & Liu 2015) [7]. Convert the Sharpe to a t-statistic, adjust its p-value for the number of tests (Bonferroni, Holm or BHY), and convert back. They find:

- The common "cut it by 50%" rule is wrong in both directions.
- Haircuts are "almost always more than and sometimes much larger than 50%" when the annual Sharpe is below 0.4.
- Haircuts are "at most 25%" when the Sharpe is above 1.0 [7].

Marginal strategies are punished hardest because they are most likely false discoveries.

## 5. Alternatives and complements

| Metric | Formula (plain) | What it adds | Use when |
|---|---|---|---|
| **Sortino** [9] | (mean return − MAR) / downside deviation, where downside deviation = √(mean of min(r − MAR, 0)²) | Only losses below a minimum acceptable return (MAR) count as risk | Returns are skewed; you care about losses, not upside swings |
| **Max drawdown** | Worst peak-to-trough fall of the equity curve | The pain you would actually live through | Always; it is what makes people quit a strategy |
| **Calmar** [10] *(trade-journal origin)* | Annual return / \|max drawdown\| (originally over 36 months) | Return per unit of worst loss | Comparing trend/timing systems; long samples (a single drawdown is one noisy observation) |
| **Information ratio** [12] | mean(r − r_bench) / std(r − r_bench) | Skill *relative to a benchmark* (here: buy-and-hold) | Answering "is this better than just holding?" |
| **Omega** [11] | Probability-weighted gains above a threshold ÷ probability-weighted losses below it | Uses the whole return distribution, all moments | Strongly non-normal returns; comparing at several thresholds |

Our strategies versus buy-and-hold on these measures (4h, 2021-08 to 2026-10, MAR = 0) (this note's check):

| Coin | Sharpe strat / B&H | Sortino strat / B&H | Max DD strat / B&H | CAGR strat / B&H | Calmar strat / B&H | IR vs B&H |
|---|---|---|---|---|---|---|
| BTC | 0.51 / 0.47 | 0.72 / 0.67 | −60% / −77% | 12.8% / 11.8% | 0.21 / 0.15 | −0.14 |
| ETH | 0.40 / 0.29 | 0.55 / 0.40 | −68% / −81% | 7.4% / −3.2% | 0.11 / −0.04 | 0.01 |
| BNB | 0.51 / 0.45 | 0.71 / 0.63 | −53% / −72% | 13.4% / 9.8% | 0.25 / 0.14 | −0.12 |
| SOL | 1.34 / 0.59 | 2.03 / 0.86 | −78% / −97% | 101% / 11.0% | 1.29 / 0.11 | 0.59 |

The strategy has smaller drawdowns everywhere. However, the information ratio is about zero or negative for BTC, ETH and BNB. Its edge over holding comes from **holding less risk**, not from extra return per unit of risk.

## 6. Applying this to our project

### 6.1 Is our Sharpe calculation correct?

Reviewed `cryptoai/backtest.py` (`run`, `_stats`) and `cryptoai/simulate.py`:

| Check | Finding |
|---|---|
| Periods per year | `PERIODS_PER_YEAR = {"4h": 6*365, "1d": 365}`. **Correct** for 24/7 markets. |
| Formula | `r.mean() / r.std() * sqrt(ppy)` on simple per-candle returns, sample std. **Correct** ex-post Sharpe [2]. |
| Risk-free rate | Implicitly 0. Acceptable for USDT on exchange; costs 0.03–0.10 if a 3.6% T-bill is the benchmark (§2.2). Should be stated in the UI. |
| Flat periods | Included as 0 return (`pos * fwd_ret_1`). **Correct**: dropping them would inflate the Sharpe by ignoring time out of the market. |
| Fees | `trades * fee` with `trades = |Δpos|`, so 0.1% on every entry and every exit, booked in the switch candle. **Correct**. No spread or slippage is modelled. |
| Buy & hold | No entry/exit fee in `backtest.py` (the simulator does charge it). Effect is negligible (0.2% once over 5 years). |
| Annualisation validity | Lag-1 autocorrelation of strategy returns is −0.006 to −0.024. Lo's η(6)/√6 is 0.99–1.03, and Sharpe from daily-compounded returns × √365 matches 4h × √2190 within 0.01 (this note's check). **√q is fine.** |
| "Mean Sharpe across 4 coins" | 0.69 is an average of four Sharpe ratios, not the Sharpe of any portfolio. The coins' strategy returns are correlated 0.55–0.74, so the four are not independent evidence. An equal-weight basket has Sharpe **0.91** (this note's check). |
| `simulate.py` | Reports return, worst drop, win rate and fees, but **no Sharpe or Sortino**. Its equity series is sampled at every decision candle, so a Sharpe can be added directly (§7). |

All reported numbers reproduced: strategy 0.51 / 0.51 / 0.40 / 1.34 (BNB/BTC/ETH/SOL), mean 0.69; buy-and-hold BTC 0.47 (this note's check).

### 6.2 How confident can we be?

*All figures in this subsection are this note's check.* T = 11,198 4h candles (5.1 years) per coin. The block bootstrap uses 30-day blocks and 1,000–2,000 resamples.

| | BTC | ETH | BNB | SOL | Equal-weight basket |
|---|---|---|---|---|---|
| Strategy Sharpe | 0.51 | 0.40 | 0.51 | 1.34 | 0.91 |
| SE, Lo i.i.d. / non-normal [3][4] | 0.44 / 0.44 | 0.44 / 0.44 | 0.44 / 0.44 | 0.44 / 0.44 | — |
| SE, block bootstrap | 0.46 | 0.43 | 0.41 | 0.47 | — |
| PSR(SR > 0) [5] | 87% | 81% | 87% | 99.9% | 98% |
| Sharpe − B&H Sharpe | +0.03 | +0.11 | +0.05 | +0.75 | +0.39 |
| SE of that gap (bootstrap) | 0.29 | 0.29 | 0.32 | 0.31 | 0.27 |
| 90% interval of gap | −0.46…+0.50 | −0.38…+0.59 | −0.47…+0.59 | +0.25…+1.30 | −0.06…+0.82 |

Fat tails barely widen the standard error here. In the Mertens/PSR formula the skew and kurtosis terms multiply the *per-period* Sharpe (≈ 0.01), so they matter little at 4h frequency. The bootstrap, which also captures volatility clustering, gives similar values.

**Deflating for the search (DSR [6]).** We tried about 6–8 feature sets, several strategy variants (4h-next, combined, 1d) and hand-picked thresholds, so N ≈ 10–50 is plausible. V is unknown, so two bounds are shown:

- *Conservative:* V = 1/T, the spread expected if every trial were pure noise.
- *Lenient:* V = the observed spread of Sharpe ratios across 34 nearby entry/exit threshold pairs (0.52–0.60 / 0.45–0.50). These are highly correlated, so this V is small.

| N trials | Threshold SR₀ (annual), conservative / lenient | DSR BTC | DSR ETH | DSR BNB | DSR SOL | DSR basket |
|---|---|---|---|---|---|---|
| 8 | 0.65 / 0.25 | 38% / 72% | 29% / 63% | 38% / 72% | 94% / 99% | 72% / 93% |
| 20 | 0.84 / 0.32 | 23% / 66% | 16% / 57% | 23% / 66% | 87% / 99% | 56% / 91% |
| 50 | 1.01 / 0.38 | 13% / 61% | 8% / 51% | 13% / 61% | 78% / 99% | 41% / 88% |
| 100 | 1.12 / 0.43 | 8% / 57% | 5% / 47% | 8% / 57% | 69% / 98% | 32% / 86% |

**Reading it plainly:**

1. **For BTC, ETH and BNB alone, there is no statistically solid evidence** that the strategy's true Sharpe is above zero once the search is counted. It is also not shown to beat buy-and-hold: the gaps are a tenth of a standard error to a third of one.
2. **SOL is the one strong result.** Its Sharpe of 1.34 survives deflation and beats buy-and-hold with z ≈ 2.4. But it is one coin, and SOL's 2021 and 2023 rallies dominate the result (strategy Sharpe 4.76 in 2021-H2 and 3.63 in 2023). Treat it as "promising, possibly coin-specific".
3. **The basket is the fairest single summary:** Sharpe 0.91 and PSR 98%, but DSR is only 41–88% at N = 50, and the advantage over holding the same basket is borderline.
4. **The pattern by year is defensive.** In 2022, BTC strategy vs buy-and-hold was −0.52 vs −1.39. In 2023 it was 1.92 vs 2.45, and in 2024 0.80 vs 1.79. The strategy does better in falling markets and lags in strong bull years, which fits "out of the market about half the time".
5. **Fees decide the verdict.** The no-fee Sharpe is 0.76–1.61 and the 0.2%-per-side Sharpe is 0.03–1.07. Using maker orders or lower-fee tiers would help more than any feature found so far.

## 7. Recommendations: what to show, with code sketches

Show these in the dashboard Backtest page and in the simulator summary:

1. **Sharpe ± SE**, e.g. "0.51 ± 0.44". This stops anyone reading 0.51 vs 0.47 as a difference.
2. **PSR(>0)** and **PSR(> buy-and-hold Sharpe)**, labelled "chance the true Sharpe is above…".
3. **Deflated Sharpe**, with N read from a trial counter. Log every backtest variant you run (feature set, thresholds, model) in `training_log.csv` or similar, so N is counted rather than guessed.
4. **Sortino, max drawdown, Calmar** next to Sharpe.
5. **Information ratio vs buy-and-hold** (Sharpe of strategy − B&H returns), plus the bootstrap interval of the Sharpe gap.
6. **Equal-weight basket Sharpe** instead of, or next to, "mean of per-coin Sharpes".
7. **Per-year Sharpe table**, and a **fee-sensitivity line** (0 / 0.05% / 0.1% / 0.2%).
8. A one-line footnote: "r_f = 0; 4h candles, q = 2,190; fees 0.1% per side, no slippage".

```python
import numpy as np
from scipy.stats import norm, skew, kurtosis

EULER = 0.5772156649

def sharpe(r, q):                      # r: per-period returns (pd.Series), q: periods per year
    return r.mean() / r.std() * np.sqrt(q)

def sharpe_se(r, q):                   # Lo (2002) i.i.d. SE, annualised
    sr = r.mean() / r.std()
    return np.sqrt((1 + sr**2 / 2) / len(r)) * np.sqrt(q)

def psr(r, sr_star_annual=0.0, q=1):   # Bailey & López de Prado (2012)
    sr, T = r.mean() / r.std(), len(r)
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    thr = sr_star_annual / np.sqrt(q)
    return norm.cdf((sr - thr) * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2))

def dsr(r, n_trials, var_trial_sr, q):  # var_trial_sr: variance of PER-PERIOD Sharpes across trials
    sr0 = np.sqrt(var_trial_sr) * ((1 - EULER) * norm.ppf(1 - 1 / n_trials)
                                   + EULER * norm.ppf(1 - 1 / (n_trials * np.e)))
    return psr(r, sr0 * np.sqrt(q), q)  # if unknown, var_trial_sr = 1/len(r) is the conservative choice

def sortino(r, q, mar=0.0):
    downside = np.sqrt((np.minimum(r - mar, 0) ** 2).mean())
    return (r.mean() - mar) / downside * np.sqrt(q)

def calmar(equity, q):
    years = len(equity) / q
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    return cagr / abs((equity / equity.cummax() - 1).min())

def info_ratio(r, r_bench, q):
    return sharpe(r - r_bench, q)
```

For `simulate.py`, compute per-period returns from the balance series (`eq.pct_change().dropna()`). Its index steps every 4h for the "Next 1 day" and "Next 4 hours" strategies and every day for "Next 3 days", so pass the matching q. Keep in mind that simulator windows are often a few months long, and **the SE for a 6-month window is about 1.4**. On short windows, display the Sharpe greyed out or with its interval, not as a headline.

## Caveats

- **Not opened:** Sharpe's 1966 paper [1] and Mertens [4] are cited from reference lists ([2], [6]). The Mertens variance formula is as reproduced in the PSR/DSR literature [5][6].
- **Bootstrap settings:** the bootstrap intervals use one block length (30 days). Other choices would move them a little.
- **DSR inputs are assumptions:** N and V are guesses, so DSR is shown as a range, not a verdict.
- **Not modelled:** spread, slippage, Binance fee tiers and BNB fee discounts. These would mostly lower the reported Sharpe ratios.
- **One test window:** the window (2021-08 to 2026-10) covers one full bear-to-bull cycle. A different start date changes buy-and-hold Sharpe ratios a lot (BTC: 0.97 from 2019, 0.47 from 2021-08).
- **Rules of thumb:** the "1 / 2 / 3" Sharpe bands [14] are a practitioner convention with no statistical basis.
- **No verified crypto benchmark:** I could not find a primary source publishing a long-run BTC Sharpe, or a verifiable stablecoin-yield history, so neither is quoted except as this note's own computation from repo data.

## Sources

1. Sharpe, W. F. (1966). Mutual Fund Performance. *Journal of Business* 39(1), 119–138. (Citation as given in [2] and [6]; original not opened.)
2. Sharpe, W. F. (1994). The Sharpe Ratio. *Journal of Portfolio Management* 21(1), 49–58. Author's copy: https://web.stanford.edu/~wfsharpe/art/sr/sr.htm
3. Lo, A. W. (2002). The Statistics of Sharpe Ratios. *Financial Analysts Journal* 58(4), 36–52. doi:10.2469/faj.v58.n4.2453. https://rpc.cfainstitute.org/research/financial-analysts-journal/2002/the-statistics-of-sharpe-ratios ; https://ideas.repec.org/a/taf/ufajxx/v58y2002i4p36-52.html
4. Mertens, E. (2002). Variance of the IID estimator in Lo (2002). Working paper, University of Basel. (Cited via [6]; not opened.)
5. Bailey, D. H. & López de Prado, M. (2012). The Sharpe Ratio Efficient Frontier. *Journal of Risk* 15(2). https://www.risk.net/journal-risk/2223785/sharpe-ratio-efficient-frontier
6. Bailey, D. H. & López de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality. *Journal of Portfolio Management* 40(5), 94–107. https://papers.ssrn.com/abstract=2460551 ; author PDF: https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf
7. Harvey, C. R. & Liu, Y. (2015). Backtesting. *Journal of Portfolio Management*, Fall 2015 (article begins p. 12). https://people.duke.edu/~charvey/Research/Published_Papers/P120_Backtesting.PDF ; SSRN: https://ssrn.com/abstract=2345489
8. Ingersoll, J., Spiegel, M., Goetzmann, W. & Welch, I. (2007). Portfolio Performance Manipulation and Manipulation-proof Performance Measures. *Review of Financial Studies* 20(5), 1504–1546. (Citation as given in [6].)
9. Sortino, F. A. & Price, L. N. (1994). Performance Measurement in a Downside Risk Framework. *Journal of Investing* 3(3), Fall 1994. (Citation confirmed via R PerformanceAnalytics documentation: https://rdrr.io/github/R-Finance/PerformanceAnalytics/man/SortinoRatio.html ; original not opened.)
10. Young, T. W. (1991). Calmar Ratio: A Smoother Tool. *Futures* 20(1) *(trade magazine, practitioner)*. Via secondary summary: https://en.wikipedia.org/wiki/Calmar_ratio
11. Keating, C. & Shadwick, W. F. (2002). A Universal Performance Measure. *Journal of Performance Measurement* 6(3). https://people.duke.edu/~charvey/Teaching/BA453_2004/Keating_A_universal_performance.pdf
12. Goodwin, T. H. (1998). The Information Ratio. *Financial Analysts Journal* 54(4), 34–43. https://ideas.repec.org/a/taf/ufajxx/v54y1998i4p34-43.html
13. Kenneth R. French Data Library, Fama/French 3 Factors (monthly, file built from CRSP 202608; 1-month T-bill from Ibbotson to 2024-05, ICE BofA thereafter). https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html (downloaded 2026-10-05)
14. MoneySense, "What is the Sharpe ratio?" *(practitioner glossary)*: "A number between 1.0 and 2.0 is considered good, 2.0 to 3.0 very good and 3.0 or higher, excellent." https://moneysense.ca/glossary/what-is-the-sharpe-ratio
