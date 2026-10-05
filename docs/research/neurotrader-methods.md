# What neurotrader teaches, and what is worth applying to `cryptoai/`

*Research note, 2026-10-05. Reviews the YouTube channel "neurotrader" (@neurotrader888) and its 14 GitHub repositories, to decide what `cryptoai/` should adopt. `cryptoai/` predicts P(up) for BTC/ETH/BNB/SOL vs USDT on Binance, with 4h candles and a 6-candle horizon and 1d candles and a 3-candle horizon. It uses a pooled HistGradientBoosting model, walk-forward evaluation with an embargo, a 0.1% fee per side, and trades long-or-flat. Citations are numbered; see [Sources](#sources).*

**How this was researched.** I read every repository's README and main Python files. I took video titles, dates and descriptions from YouTube, and read the auto-generated English captions of 17 videos (downloaded 2026-10-05). The transcripts are his own words, but auto-captions contain transcription errors. I also re-ran two of his strategies on his own data and on newer Binance data, and ran a prototype permutation test on our model. These checks are my own, are not peer reviewed, and are labelled *(this note's check)*.

## Bottom line

- **His most valuable idea is the validation process, not any single indicator.** He uses four gates: in-sample excellence, an in-sample Monte Carlo permutation test (MCPT), a walk-forward test, and a walk-forward MCPT [1]. A strategy must reach p < 1% on the in-sample MCPT. On the walk-forward MCPT he accepts about 5% for one year of data and 1% for two or more years [1]. The method comes from Timothy Masters' books [19][20] and is the same logic as White's Reality Check for data snooping [21].
- **We should adopt the walk-forward MCPT for the model, with three changes:** permute all four coins jointly, use AUC and net-of-fee Sharpe as the test statistics, and repeat the *whole feature-set selection* on every permutation so that the five feature sets already tried are accounted for. It is affordable: one 4h walk-forward takes about 9 s and one 1d walk-forward about 4 s on this machine *(this note's check)*.
- **A prototype of that test on our model gave sobering results** *(this note's check; details in §1.6)*. The test used the period where all four coins have data (2020-08 onward).
  - **4h passes:** AUC 0.5276 vs a null mean of 0.498, p = 0.01 with 100 permutations.
  - **1d does not:** AUC 0.4986, p = 0.45 with 200 permutations.
  - **The null AUC spread is far wider than the 0.003 standard error assumed so far:** sd 0.0085 on 4h and 0.015 on 1d. As a result, the daily acceptance gate `MIN_AUC = 0.505` cannot separate skill from luck.
- **His indicator results almost never include fees, and only some are walk-forward.** Where he does test properly, things often fail. His own optimised Donchian breakout failed the walk-forward MCPT (p = 22%) [1]. His PIP pattern miner worked in 2020 and then went flat in 2021–2022 [16].
- **Re-running two of his strategies after publication shows decay.** His ETH-vs-BTC "intramarket difference" strategy had a gross profit factor (PF) of 1.079 on 2018–2022 hourly data, with MCPT p = 0.005. On 2023-01 to 2026-09 it fell to PF 0.987 (p = 0.68) *(this note's check)*. His volatility-Hawkes strategy fell from PF 1.064 to 1.028 gross on BTC hourly data *(this note's check)*.
- **Features worth one pre-registered test, in order:**
  1. A Hawkes-style volatility state (an exponentially decayed sum of ATR-normalised candle range) plus the "price change since volatility was last low" [8].
  2. Trendline support and resistance slopes and distances, normalised by ATR [7].
  3. Permutation entropy as a regime variable [12].

  Expect gains of a few thousandths of AUC at most. Test the three as **one bundle, once**, with the selection-aware MCPT.
- **Skip for now:** RSI-PCA (the code projects onto the wrong eigenvector axis, and trees do not need PCA), visibility graphs, reversibility indices, the TVL indicator, harmonic patterns, and meta-labelling (we have no primary rule to label). Use the trade-dependence runs test as a cheap diagnostic on our backtest trades, not as a feature.

---

## 1. Validation methods

### 1.1 The four-step process

The steps come from his dedicated video on strategy development (2025-03-03) [1]:

1. **In-sample excellence.** Optimise the strategy on development data, for example a grid search over a look-back. Then ask two questions: "is this excellent?" and "is it obviously overfit?" A near-100% win rate means overfitting or a future leak [1].
2. **In-sample MCPT.** Re-run the *same optimisation* on many permuted copies of the development data. The p-value is the fraction of permutations whose *optimised* objective is at least as good as the real one. The null hypothesis is "the strategy is garbage, and its in-sample performance is pure data-mining bias" [1].
3. **Walk-forward test.** Re-optimise on a rolling training window and trade the next block. This removes data-mining bias from the test results but not *selection* bias. If you walk forward 100 ideas on 2020 and keep the best one, 2020 has become validation data, not out-of-sample data [1].
4. **Walk-forward MCPT.** Permute only the bars *after* the first training fold, re-run the entire walk-forward on each permutation, and compare [1].

He evaluates strategies on **bar-level returns** (position × next-bar log return), not per-trade returns. Bar-level returns give the objective function more data points and more stable estimates. He credits Masters' *Testing and Tuning Market Trading Systems* [20] for this [1]. His usual objective is the profit factor: the sum of positive bar returns divided by the absolute sum of negative ones.

### 1.2 The bar permutation algorithm (`mcpt/bar_permute.py`)

The algorithm works on log prices [1]:
- For each bar t, it computes the high, low and close relative to the bar's open, plus the gap: this bar's open minus the previous close.
- It shuffles the intrabar triples (high, low, close) with one random permutation and the gaps with a second, independent one.
- It rebuilds the price path from the real starting bar.

The permuted path has the same first open and last close as the real one, so the overall drift is kept. The return mean, standard deviation, skew and kurtosis come out nearly identical [1]. Passing a list of markets applies **the same permutation index to every market**. This keeps the contemporaneous correlation between them (he demonstrates this with BTC and ETH) [1]. A `start_index` argument leaves bars before it untouched, which is what the walk-forward version needs.

He states the main weakness himself: **the permutation destroys volatility clustering and long memory.** A strategy that relies on those properties gets an optimistically biased test. He argues the bias is tolerable because failing even a lenient test is informative [1].

### 1.3 Numbers he reports

| Test | Data | Result |
|---|---|---|
| Donchian breakout, look-back optimised over 12–168, in-sample | BTC 1h, 2016–2019 | best look-back 19, PF 1.08; 1,000 permutations, **p = 0.3%** (pass) [1] |
| Deliberately overfit decision tree (`min_samples_leaf=5`, 3 return features, 24h direction target) | same | permutations did "just as good or better" (fail); his code calls it "trash" [1] |
| Donchian, walk-forward (4-year train, refit every 30 days) | BTC 1h, test year 2020 | PF 1.04; 200 permutations, **p = 22%** (fail); he would not trade it [1] |
| Moving average as support/resistance ("bounce %" optimised over MA 24–200) | BTC/USDT 1h, 2018–2022 | best bounce rate 64%; the best of 1,000 permutations reached about 54% [4] |
| PIP chart-pattern miner (cluster price shapes, keep the best clusters by Martin ratio) | BTC 1h, 2018–2022 | passed the in-sample MCPT; walk-forward good in 2020, "essentially flat" in 2021–2022 [16] |

A secondary summary of the strategy-development video reports the same figures (19, 1.08, 0.3%, 1.04, 22%) [2].

Two of his recommendations matter for us. First, use at least 1,000 permutations, with 100 as a hard minimum. Second, do not treat p < 1% as a target: "if you fiddle with your strategy enough you could probably make this test pass on anything" [1].

### 1.4 What the MCPT does and does not prove

- **The null hypothesis is "no temporal structure at all".** Rejecting it shows that the strategy exploits *some* structure, but not necessarily a tradable directional edge. Volatility clustering, the bull and bear regime pattern, and slow-moving trends are all destroyed by the shuffle. A model can therefore "beat the permutations" by learning things that are real but do not pay after fees. His moving-average bounce test [4] has this problem. It shuffles close-to-close changes, so any bounce effect that comes from volatility clustering or short-term mean reversion counts as evidence for "traders watching MAs".
- **Block permutation keeps volatility clustering** and gives a stricter null. The stationary bootstrap of Politis & Romano [27] is one way to do it. Running both nulls is cheap.
- **It accounts for selection only if you repeat the selection.** His in-sample test re-runs the optimiser on each permutation, so the selection of the look-back is covered [1]. Our situation is different. We compared five feature sets on the *same* walk-forward periods, and that choice is not covered unless every permutation also tries all five and keeps the best. This "max over candidates" construction is the core of White's Reality Check [21] and Hansen's SPA test [22]. Aronson tested more than 6,400 technical rules on the S&P 500 with data-mining-bias corrections and found none statistically significant [23].
- **p-values are capped by the number of permutations.** With 200 permutations the smallest possible p-value is 0.005, so "p < 0.01" needs at least 100 permutations and really 1,000 [1].
- **Costs are absent in his MCPT code.** The statistic is gross PF [1]. For a long-or-flat strategy paying 0.2% per round trip, the statistic should be net of fees.
- Related tools for estimating how overfit a backtest is are the Probability of Backtest Overfitting [24] and the Deflated Sharpe Ratio [25]. A t-statistic above 3 is the usual threshold for new factors [26].

### 1.5 Meta-labelling (`TrendlineBreakoutMetaLabel`)

Meta-labelling comes from López de Prado [28]. A *primary* rule decides direction, and a *secondary* classifier decides whether to take each signal. His version works as follows [6]:

- **Primary rule.** Go long when the close breaks above a 72-bar resistance trendline fitted on log closes. The window excludes the current bar, and the line is projected forward one bar. Exits are a take-profit or stop-loss at 3 × ATR(168), or after 12 bars.
- **Features at entry:** resistance slope / ATR; mean distance of prices from the line / ATR; maximum distance / ATR (the most informative feature); volume / 168-bar median; ADX(72).
- **Label.** 1 if the trade's log return is above 0.
- **Model.** Random forest with 1,000 trees and `max_depth=3`. The walk-forward uses a 2-year training window and retrains yearly. Only trades whose *exit* is before the training time enter the training set, which prevents leakage. A trade is taken if P(win) > 0.5.
- **Results.** BTC 1h, 2018–2022, no fees. The unfiltered rule had a win rate of about 50%, PF 1.02 and an average trade of about 0.05%. The filter cut time in market from 30% to 20% and "more than doubled" the average trade [6]. He ran no permutation test and modelled no fees. A doubled 0.05% average trade is still below a 0.2% round-trip cost.

**For us:** our model is already a direct classifier, so there is no primary rule to meta-label. It would only become relevant if we adopted a simple trend rule as the primary signal and used P(up) as the filter.

### 1.6 Trade-dependence runs test (`TradeDependenceRunsTest`)

The test converts the sequence of trade returns into signs (+ and −), counts runs (streaks), and compares the count with its expectation under independence. This is the Wald–Wolfowitz runs test [29]:
- expected runs μ = 2·n₊·n₋/n + 1
- variance σ² = (μ−1)(μ−2)/(n−1)
- z = (runs − μ)/σ

A positive z means winners and losers alternate more often than chance would produce [5].

He applied it to the Donchian breakout (always in the market, close-only channel) on BTC 1h. At a look-back of 24, z = 2.7, and z was mostly positive across look-backs 12–168. A rule of "only trade after a losing signal" raised PF at every look-back, while "only after a winner" was mostly unprofitable [5]. He notes that the Turtle traders used a similar rule and that he has "only seen strong trade dependence on trend following strategies" [5]. There was no out-of-sample test, no fees and no permutation test. The repo also includes a rolling runs-test z-score as an indicator that he says he has "not researched" [5].

**For us:** it costs a few lines to run on each coin's backtest trades from `backtest.run`. A significant |z| would suggest a "skip after win/loss" overlay worth testing; otherwise, ignore it.

### 1.7 Prototype walk-forward MCPT on our model *(this note's check)*

I applied his bar permutation to `cryptoai` without modifying the repo:
- The permutation used a shared index across BTC/ETH/BNB/SOL and permuted OHLCV, with volume moving together with the intrabar triple.
- It was restricted to the period where all four coins exist (from 2020-08-11), because a shared permutation needs a common index.
- Bars were permuted from the first walk-forward test fold onward.
- For each permutation I rebuilt features and targets with `features.build_all` and `features.target`, ran `model.walk_forward` unchanged (8 folds, embargo), and recorded the pooled AUC and the mean long-or-flat Sharpe net of 0.1%/side (`backtest.positions`, 0.55/0.48 thresholds).

| | 4h (100 permutations) | 1d (200 permutations) |
|---|---|---|
| Common period / permuted from | 2020-08-11 → 2026-10-05 / 2022-09-19 | 2020-08-11 → 2026-10-04 / 2023-01-05 |
| Real walk-forward AUC | **0.5276** | **0.4986** |
| Null AUC: mean / sd | 0.4984 / **0.0085** | 0.4971 / **0.0152** |
| Null AUC: 95th / 99th percentile / max | 0.5126 / 0.5169 / 0.5234 | 0.5241 / 0.5318 / 0.5413 |
| AUC p-value | **0.010** (the minimum possible with N = 100) | **0.45** |
| Real mean net Sharpe vs null mean (95th pct) | 0.56 vs 0.15 (0.51); p = 0.05 | 0.63 vs 0.44 (0.95); p = 0.24 |

What this shows:
1. **The 4h model beats a no-structure null.** No permutation reached its AUC. The net Sharpe is only marginal (p = 0.05).
2. **On this restricted period the 1d model is indistinguishable from noise.** The higher production figure (0.527) comes from a longer history and an earlier test window. With a null sd of about 0.015 at 1d, 0.527 is only about 2 sd above chance.
3. **The null AUC sd is about 3× (4h) and 5× (1d) the 0.003 "standard error" quoted in the earlier note [45].** Overlapping labels, pooled coins and refitting all widen it.
4. **The daily gate `MIN_AUC = 0.505` sits inside the noise band**: under 1 null sd on 4h and under 0.5 sd on 1d. It cannot tell a useful retrain from a lucky one.
5. **The null Sharpe is positive** (0.15 and 0.44), because the permutation keeps the overall drift and long-or-flat collects it. Compare Sharpe with the null distribution, not with zero.

---

## 2. Volatility Hawkes process (`VolatilityHawkes`)

**What it computes.** The input is the normalised range (ln H − ln L)/ATR₃₃₆ on log prices. The "Hawkes process" is the recursion yₜ = e^(−κ)·yₜ₋₁ + xₜ, multiplied by κ [8]. This is an exponentially decaying sum, equivalent to an EWMA up to scale, with the exponential kernel of a Hawkes intensity [30]. It is not a fitted Hawkes model: no baseline intensity or branching ratio is estimated. He credits the idea to the "tr8dr" blog's Hawkes buy/sell-imbalance indicator [8][43]. Academic Hawkes work in finance is mostly at tick frequency [31].

**Strategy.** Track the 5% and 95% rolling quantiles (168 bars) of y. When y crosses above q95 after having been below q05, take the direction of the price change since the last time y was below q05. Exit when y drops below q05. Parameters: κ = 0.1 and a quantile look-back of 168 [8].

**Reported results.** BTC 1h, with no stated period split, no fees and no permutation test:
- PF 1.07.
- Long trades: 58% win rate, 2.9% average trade. Short trades: 49% win rate, 1.1% average trade.
- In the market 50% of the time.
- All 25 combinations of κ ∈ {0.5…0.01} and look-back ∈ {24…336} had PF > 1. He presents this robustness grid as the main evidence [8].

He says its main value is as an **exit** for momentum strategies [8].

**Replication** *(this note's check)*:
- **His data.** Gross PF 1.064; net of 0.1%/side 1.057. Average gross return per position change is about 1%, so fees matter little.
- **Post-publication.** BTC 1h from 2023-01 to 2026-09: gross PF 1.028, net 1.016. Yearly net log return: +0.21 (2023), +0.57 (2024), −0.43 (2025), +0.13 (2026 to date).
- **4h, 2023–2026.** With parameters scaled from hourly (ATR 84, quantile look-back 42), net PF was 1.01 (BTC), 1.09 (ETH), 0.96 (BNB) and 1.05 (SOL).

**For us:** the strategy itself is weak after publication. As **features**, two quantities are cheap and partly new:
- `vol_hawkes_pct`, the rolling percentile of y. Today `vol_ratio` is the closest feature we have.
- `move_since_vol_low`, the log return since y was last below its q05. This conditions trend direction on a volatility cycle, which no current feature does.

---

## 3. Intramarket difference (`IntramarketDifference`)

**Indicator.** CMMA = (close − SMA_lb)/(ATR₁₆₈·√lb). The √lb term puts look-backs on a common scale, using random-walk scaling. The signal is the difference CMMA(ETH) − CMMA(BTC). The rule is **momentum**: go long ETH when the difference exceeds +0.25, go short below −0.25, and exit when it returns to 0 [9].

**Reported results.** On BTC/ETH 1h, 2018–2023, the hourly return correlation was 0.84. With look-back 24 and threshold 0.25: PF 1.08, about 4,400 trades, win rates slightly above 50%, **no transaction costs**. Nearly every cell in a look-back × threshold heat map had PF > 1 [9]. Because the parameters and the heat map came from the same data, this is an in-sample result.

**Replication** *(this note's check; his exact code logic and parameters, ATR implemented as Wilder's RMA)*:

| Data | Gross PF | Net PF (0.1%/side) | MCPT p (200 joint BTC+ETH permutations, fixed parameters) |
|---|---|---|---|
| His CSVs, ETH 1h 2018–2022 | 1.079 | 1.052 | 0.005 |
| Binance ETH 1h 2023-01 → 2026-09 | **0.987** | **0.945** | **0.68** |
| ETH / BNB / SOL vs BTC, 4h (look-back 6, ATR 42), 2019–2022 | 1.13 / 1.09 / 1.11 | 1.08 / 1.04 / 1.08 | — |
| same, 2023 → 2026-10 | 1.03 / 1.02 / 1.02 | 0.96 / 0.93 / 0.97 | 0.16 / 0.25 / 0.24 |

The edge passed his own permutation test inside his sample and then disappeared. This fits the decay pattern seen elsewhere in crypto technical trading [45][47].

**For us:** we already feed `rel_btc_6` and `rel_btc_24` (return differences vs BTC). A volatility-normalised CMMA difference is a cleaner version of the same information, but the post-2023 evidence is negative. Low priority.

---

## 4. RSI principal components (`RSI-PCA`)

**Method.** Compute RSI for periods 2–24 (23 series), which are highly correlated (the minimum correlation is 0.43, between RSI-2 and RSI-24). Take the eigenvectors of their covariance, fit OLS of the 6-bar-ahead log return on the first n components, and trade only when the prediction is above its in-sample 99th percentile (long) or below its 1st percentile (short). Position size is the rolling mean of signals over the horizon, so overlapping signals stack. The walk-forward uses a 2-year training window and a 1-year step [10].

**Reported results.** BTC 1h, 3 components, look-ahead 6: PF 1.53, with no fees [10]. A heat map over look-ahead 1–24 × components 1–6 was computed on the same walk-forward output. That turns the out-of-sample period into a selection set, which is the selection bias he warns about in [1]. He notes that look-ahead 1 would not survive fees [10].

**Code issue** *(this note's reading)*. `scipy.linalg.eigh` returns eigenvectors as **columns**. The code sorts the columns but then projects onto `evecs[j]`, which is a **row** (`pca.py`, `model_in_sample.py`, `walkforward.py`) [10]. The "components" are therefore not the principal components. They are fixed linear combinations of the RSIs. The walk-forward PF is still an honest result for *that* linear model, but the PCA interpretation, including his eigenvector plots, does not hold.

**For us:** tree ensembles do not need decorrelated inputs. If anything, collinear RSI and Bollinger features dilute importance scores, which the earlier note already flagged. Low priority.

---

## 5. Volume Spread Analysis indicator (`VSAIndicator`)

**Method.** Compute normalised range = (H − L)/ATR₁₆₈ and normalised volume = V/median₁₆₈(V). In a rolling 168-bar window, regress normalised range on normalised volume. If the slope is ≤ 0 or r < 0.2, output 0. Otherwise output the actual minus the predicted range [11].
- A **negative** reading means a small range on high volume, which practitioners call "absorption".
- A **positive** reading means a large range on low volume.

**Evidence.** He reports a 0.81 correlation between normalised range and volume (BTC 1h, 2018–2023) and shows hand-picked chart examples. He says the indicator is **not directional** and "not magic". There is no backtest [11]. The volume–volatility link itself is well established [32]. The claim that the residual predicts direction is practitioner lore.

**For us:** this is a cheap feature, and the median normalisation is more robust than our `volume_z`. However, the Binance zero-fee BTC period from July 2022 to March 2023 distorts BTC volume, as the earlier note documents. Low to medium priority. Try it only inside the feature bundle, not on its own.

---

## 6. Trendlines (`TrendLineAutomation`, plus the breakout above)

**Algorithm.**
1. Fit an OLS line to the window.
2. Take the pivot: the bar with the largest positive residual (for resistance) or the most negative one (for support).
3. Search the slope through that pivot that minimises the sum of squared distances to all points, subject to the line staying entirely above (resistance) or below (support) the data.

The search starts from the OLS slope, takes a numerical derivative, and halves the step from (max − min)/n down to 10⁻⁴ of that [7]. A high/low variant fits resistance to highs and support to lows [7]. He applies it to log prices and plots rolling 30-day support and resistance slopes on BTC daily [7].

**Evidence.** An always-in-market breakout rule (long above projected resistance, short below support, look-back 72) on BTC 1h had PF 1.035, with no fees. He called a PF spike at look-backs 32–42 "just random luck" [6].

**For us:** rolling support and resistance **slopes ÷ ATR**, and the **distance of the close from each line ÷ ATR**, on 4h (window about 30–72 candles) and 1d (about 30). These describe the shape of the recent price envelope, which `range_pos`, `from_high_100` and `ema50_slope` capture only partly. The computation is an O(window) loop per bar, which is fine at our data size. Medium-low expected value.

---

## 7. Complexity measures: permutation entropy, reversibility, visibility graphs

- **Permutation entropy** [12]. Bandt & Pompe's ordinal-pattern entropy [33], using the pattern encoding of Unakafova & Keller [34]. With embedding dimension d = 3 there are 6 patterns. The window is mult × d! (he uses 28 × 6 = 168 hourly bars), and the entropy is normalised to [0, 1]. He says entropy "has a loose tendency to drop" in trends. He uses it as a **filter**, especially for mean-reversion systems, which "perform better when the measured entropy is low". He gives no numbers [12].
- **Time-series reversibility** [13]. He implements two measures:
  - PTSR: the KL divergence between forward and time-reversed ordinal-pattern distributions, with d = 3 [35].
  - Relative asynchronous index: built from out-degree sequences of the horizontal visibility graph of the series and of its reversal [36].

  He found both useful for filtering mean-reversion trades and in a meta-labelling model. He says they have "almost zero correlation" with his other features, but again gives no numbers [13].
- **Visibility graphs** [14]. A natural visibility graph [37] is built on the last 12 closes and on the negated closes, and the rule compares their average shortest-path lengths. He also mentions horizontal visibility graphs [38]. The trading rule (long if the positive graph's path length is greater, else short) is always in the market, "has a lot of whipsaws", and is shown without fees. He calls it "not a complete trading strategy" [14]. An unfinished VG forecasting function implements Zhan & Xiao [40].

**For us:** these are the only ideas on the channel that are plausibly *uncorrelated* with our existing momentum and volatility features. That is their appeal for a tree model. Permutation entropy is cheap (d = 3; windows of about 42 on 4h and 30 on 1d) and well defined. Reversibility needs long windows to be stable (he suggests at least 60 points), and visibility graphs are slow and noisy. **Only permutation entropy goes in the bundle.**

---

## 8. Market structure and chart-pattern automation (brief)

- **`market-structure`** [15]. Swing points come from an ATR directional-change detector: a swing is confirmed when price retraces 1 × ATR from the pending extreme. Swings are then promoted level by level: a top surrounded by two lower tops at level k becomes a level-k+1 top. Alternating tops and bottoms are enforced, and points never redraw, but confirmation lag grows with the level. He credits the idea to Larry Williams' book [42] (practitioner) and reports no backtest [15].
- **`TechnicalAnalysisAutomation`** [16]. Uses rolling-window extremes, directional change and perceptually important points to detect head-and-shoulders (with "early" detection), flags and pennants, and harmonic patterns. It also includes a market-profile support/resistance tool and the PIP pattern miner from §1.3. For BTC 1h he found H&S performance "inconsistent at varying scales", although the early-detected inverse H&S looked good [16]. **For the algorithms themselves, see `docs/research/swing-points-rolling-window-directional-change-pip.md`.**
- **`TVLIndicator`** [17]. Regresses log ETH close on log DefiLlama TVL over a rolling 7-day window, takes the residual ÷ ATR(30), and lags TVL one day to avoid a future leak. The Spearman correlation with next-day return was negative in each year 2019–2022. It was strong in 2021 (|ρ| > 0.2) and "disappointing" in 2022, and he concludes it is "not convincing enough" to trade on its own [17]. It does not apply to BTC. Caveat (my reasoning, unverified): TVL is mostly priced in ETH, so it is mechanically tied to the price, and historical TVL series can be revised.

---

## 9. Recommendations

| Rank | Idea | What it would add to `cryptoai/` | Effort | Expected value | Evidence quality |
|---|---|---|---|---|---|
| 1 | **Walk-forward MCPT of the whole pipeline** (selection-aware) | An honest p-value for "the model beats noise after fees", and a data-driven `MIN_AUC` | Medium (≈1 day; runtime ≈ N × 12 s on 4h) | **High**: guards every future decision | Strong method [19][21][22]; his null is lenient [1] |
| 2 | **Pre-registered feature bundle:** Hawkes vol percentile + move since vol low; trendline slopes and distances ÷ ATR; permutation entropy | Volatility-cycle, envelope-shape and complexity information not in the current features | Low–medium | Low–medium (a few 0.001 AUC, if any) | Weak: his tests are in-sample or gross, and post-2023 replication is weak *(this note's check)* |
| 3 | Block-permutation null alongside his bar shuffle | Removes the optimistic bias from destroyed volatility clustering | Low (once #1 exists) | Medium | Strong [27] |
| 4 | Runs test on backtest trade signs per coin | Detects trade dependence and whether a skip-after-win/loss overlay is worth testing | Very low | Low | Classical test [29]; his evidence is in-sample only [5] |
| 5 | VSA residual (in the bundle only if #2 passes) | A volume–range anomaly feature with median-normalised volume | Low | Low | Practitioner; no backtest [11] |
| 6 | ATR-normalised CMMA difference vs BTC | A cleaner version of `rel_btc_*` | Very low | Low: decayed after 2023 *(this note's check)* | In-sample only [9] |
| 7 | Meta-labelling | A filter on a primary trend rule | High | Unclear | AFML [28]; his test had no fees [6] |
| — | RSI-PCA, visibility graphs, reversibility indices, TVL, harmonic patterns | — | — | Very low | Code bug [10] / no numbers [13][14] / ETH only [17] |

### 9.1 How to adopt MCPT with our walk-forward (concrete)

1. **Add `cryptoai/mcpt.py`** (or a script) containing:
   - `permute(raw, start, rng)`: his `bar_permute` algorithm applied to all four coins with **one shared permutation index**, with volume, taker volume and trade count carried along with the intrabar triple.
   - `walk_forward_stat(raw)`: calls `features.build_all`, then `features.target`, then `model.walk_forward`, and returns the pooled AUC and the mean net Sharpe from `backtest`.
2. **Handle unequal histories.** SOL starts in 2020-08. Either run on the common index (as in §1.7), or permute each coin over its own span with independent permutations before 2020-08 and a shared one after. Report which.
3. **Choose `start`.** Use the first walk-forward test bar minus the horizon, so the first training fold is real data and everything after is permuted. This matches his `start_index` [1].
4. **Use N = 200 for routine checks** (about 40 minutes on 4h, measured) **and N = 1,000 before any production change.** Run it weekly or when features change, not in the daily job.
5. **Make it selection-aware.** For each permutation, evaluate **every feature set we have tried** (currently five; keep a registry in `config.py`) and record the *maximum* AUC. Compare the real chosen set's AUC with this distribution of maxima [21][22].
6. **Replace `MIN_AUC = 0.505`** with the 95th percentile of the null AUC distribution for the current feature set. In the §1.7 prototype that was about 0.513 on 4h and 0.524 on 1d, against today's 0.505. Today's daily gate re-tests nearly the same data every day, so it mostly measures noise.
7. **Add a second null:** a block permutation with blocks of about 1 week on 4h and about 1 month on 1d [27]. Report both p-values.

### 9.2 Feature test protocol

Implement the three features from rank #2 for both timeframes, with parameters fixed in advance:
- κ ∈ {0.1}; quantile look-back of 1 week on 4h and 4 weeks on 1d.
- Trendline window of 42 bars on 4h and 30 on 1d.
- Permutation entropy with d = 3, window 42 on 4h and 30 on 1d.

Add them **as one bundle**, run the selection-aware MCPT once, and accept only if the bundle's AUC beats the maximum-over-sets null at p < 0.05 on both timeframes. Do not tune parameters on the walk-forward output (his RSI-PCA heat map [10] is the counter-example).

---

## Caveats

- **Most of his evidence is gross of fees, on BTC 1h, on 2018–2022 data, and in-sample.** Exceptions: the strategy-development video (walk-forward + MCPT, which the strategy failed) [1]; the Donchian video, which deducts fees and shows the cliff at short look-backs [3]; and the RSI-PCA and meta-labelling videos (walk-forward, no fees) [6][10].
- **Transcripts are YouTube auto-captions**, so numbers were cross-checked against code where possible. The strategy-development figures are also confirmed by secondary summaries [2]. For the remaining videos, I did not find independent summaries.
- **My replications** use his logic but my own ATR (Wilder RMA, as `pandas_ta` uses by default), my own parameter scaling for 4h, and 200 permutations. They are not peer reviewed. Post-publication 1h data came from the Binance spot klines API [46].
- **Prototype MCPT on our model** used the common 2020-08+ period, so its real AUCs differ from the production walk-forward (0.531 / 0.527 on full history). Only 4h volume and OHLC were permuted. The taker, funding and premium features are excluded in production anyway.
- **Books by Masters [19][20]** are cited as the origin of the method on his statement [1]. I did not read them for this note.
- **I did not verify** the content of the "3 Must-Know Algorithms", flags/pennants and harmonic-pattern videos beyond titles and repo code.

## Sources

1. neurotrader, "How I Develop Trading Strategies | Permutation Tests and Trading Strategy Development with Python", YouTube, 2025-03-03, https://www.youtube.com/watch?v=NLBXgSmRBgU ; code https://github.com/neurotrader888/mcpt (`bar_permute.py`, `insample_donchian_mcpt.py`, `insample_tree_mcpt.py`, `walkforward_donchian_mcpt.py`, `donchian.py`, `tree_strat.py`)
2. *(secondary)* Video summaries of [1]: Lilys.ai, https://lilys.ai/en/notes/quant-investment-strategy-20251214/permutation-tests-trading-strategies-python ; YouTubeSummary, https://youtubesummary.com/summary/NLBXgSmRBgU
3. neurotrader, "Donchian Channel Crypto Trading Strategy That Works at Every Parameter", YouTube, 2023-01-23, https://www.youtube.com/watch?v=ncJKep-6vU8
4. neurotrader, "Do Moving Averages Actually Work as Support and Resistance?", YouTube, 2023-02-07, https://www.youtube.com/watch?v=3zI_l_P-lF8
5. neurotrader, "Using Trade Dependence to Improve the Donchian Breakout Trading Strategy", YouTube, 2023-06-26, https://www.youtube.com/watch?v=BM3KZPg6zic ; code https://github.com/neurotrader888/TradeDependenceRunsTest
6. neurotrader, "Trend Line Breakout Machine Learning Algorithmic Trading Strategy in Python", YouTube, 2023-06-14, https://www.youtube.com/watch?v=jCBnbQ1PUkE ; code https://github.com/neurotrader888/TrendlineBreakoutMetaLabel
7. neurotrader, "Automated Price Trend Lines in Python | Algorithmic Trading Indicator", YouTube, 2023-02-27, https://www.youtube.com/watch?v=wbFoefnidTU ; code https://github.com/neurotrader888/TrendLineAutomation
8. neurotrader, "Self-Exciting Behavior and Detecting the End of Price Trends | Algorithmic Trading Strategy", YouTube, 2023-04-18, https://www.youtube.com/watch?v=wdsiZBIhAFw ; code https://github.com/neurotrader888/VolatilityHawkes
9. neurotrader, "Intramarket Indicator Differences | Algorithmic Crypto Trading Strategy in Python", YouTube, 2023-07-28, https://www.youtube.com/watch?v=n2mY86S01fg ; code https://github.com/neurotrader888/IntramarketDifference
10. neurotrader, "Principal Components of the RSI | Machine Learning Trading Strategy in Python", YouTube, 2023-05-09, https://www.youtube.com/watch?v=mdncZ034Q7k ; code https://github.com/neurotrader888/RSI-PCA
11. neurotrader, "Volume Spread Analysis with Python | Algorithmic Trading Indicator", YouTube, 2023-05-16, https://www.youtube.com/watch?v=FmPThiXtLYc ; code https://github.com/neurotrader888/VSAIndicator
12. neurotrader, "Ordinal Patterns and Permutation Entropy | Algorithmic Trading Indicator", YouTube, 2023-04-04, https://www.youtube.com/watch?v=PsQbKJvpGDU ; code https://github.com/neurotrader888/PermutationEntropy
13. neurotrader, "Time Series Reversibility | Algorithmic Trading Indicators in Python", YouTube, 2023-06-04, https://www.youtube.com/watch?v=W8kfC0NhKEw ; code https://github.com/neurotrader888/TimeSeriesReversibility
14. neurotrader, "Applying Graph Theory to Algorithmic Trading | Time Series Visibility Graphs", YouTube, 2023-05-24, https://www.youtube.com/watch?v=bA1I4Upzxgc ; code https://github.com/neurotrader888/TimeSeriesVisibilityGraphs
15. neurotrader, "Quantifying Market Structure at Multiple Scales for Algorithmic Trading with Python", YouTube, 2025-01-30, https://www.youtube.com/watch?v=EuFakzlBLOA ; code https://github.com/neurotrader888/market-structure
16. neurotrader, TechnicalAnalysisAutomation, https://github.com/neurotrader888/TechnicalAnalysisAutomation ; videos "Automated Head and Shoulders Chart Pattern in Python" (2023-03-21) https://www.youtube.com/watch?v=6iFqjd5BOHw , "Data Mining Novel Chart Patterns With Python" (2023-03-28) https://www.youtube.com/watch?v=P4u5drToePM , "Flag and Pennant Pattern Recognition in Python" https://www.youtube.com/watch?v=Lb5SPCTp4uY , "3 Must-Know Algorithms for Automating Chart Pattern Trading in Python" https://www.youtube.com/watch?v=X31hyMhB-3s
17. neurotrader, "Ethereum TVL Algorithmic Trading Indicator in Python", YouTube, 2023-01-27, https://www.youtube.com/watch?v=9W5mczpSboE ; code https://github.com/neurotrader888/TVLIndicator
18. neurotrader, "Books for Algorithmic Trading I Wish I Had Read Sooner", YouTube, 2023-05-02, https://www.youtube.com/watch?v=ftFptCxm5ZU
19. Masters, T. (2020). *Permutation and Randomization Tests for Trading System Development: Algorithms in C++*. Apress. https://www.goodreads.com/book/show/51518758-permutation-and-randomization-tests-for-trading-system-development
20. Masters, T. (2018). *Testing and Tuning Market Trading Systems: Algorithms in C++*. Apress. ISBN 9781484241721. https://www.managementboek.nl/boek/9781484241721/testing-and-tuning-market-trading-systems-timothy-masters
21. White, H. (2000). A Reality Check for Data Snooping. *Econometrica* 68(5), 1097–1126. https://doi.org/10.1111/1468-0262.00152
22. Hansen, P. R. (2005). A Test for Superior Predictive Ability. *Journal of Business & Economic Statistics* 23(4), 365–380. https://doi.org/10.1198/073500105000000063
23. Aronson, D. (2006). *Evidence-Based Technical Analysis: Applying the Scientific Method and Statistical Inference to Trading Signals*. Wiley. ISBN 9780470008744. https://www.tenlong.com.tw/products/9780470008744 ; summary of the 6,400-rule result: https://www.earnforex.com/guides/book-review-evidence-based-technical-analysis-by-david-aronson/ *(secondary)*
24. Bailey, D., Borwein, J., López de Prado, M. & Zhu, Q. (2017). The Probability of Backtest Overfitting. *Journal of Computational Finance* 20(4). https://papers.ssrn.com/abstract=2326253
25. Bailey, D. & López de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management* 40(5), 94–107. https://papers.ssrn.com/abstract=2460551
26. Harvey, C., Liu, Y. & Zhu, H. (2016). …and the Cross-Section of Expected Returns. *Review of Financial Studies* 29(1), 5–68. https://doi.org/10.1093/rfs/hhv059
27. Politis, D. & Romano, J. (1994). The Stationary Bootstrap. *Journal of the American Statistical Association* 89(428), 1303–1313. https://doi.org/10.1080/01621459.1994.10476870
28. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley, ISBN 9781119482086 (ch. 3 "Labeling": triple-barrier method, meta-labeling). https://oreilly.com/library/view/advances-in-financial/9781119482086
29. Wald, A. & Wolfowitz, J. (1940). On a Test Whether Two Samples are from the Same Population. *Annals of Mathematical Statistics* 11(2), 147–162. https://doi.org/10.1214/aoms/1177731909
30. Hawkes, A. G. (1971). Spectra of some self-exciting and mutually exciting point processes. *Biometrika* 58(1), 83–90. https://doi.org/10.1093/biomet/58.1.83
31. Bacry, E., Mastromatteo, I. & Muzy, J.-F. (2015). Hawkes processes in finance. *Market Microstructure and Liquidity* 1(1). arXiv:1502.04592. https://arxiv.org/abs/1502.04592
32. Karpoff, J. (1987). The Relation between Price Changes and Trading Volume: A Survey. *Journal of Financial and Quantitative Analysis* 22(1), 109–126. https://doi.org/10.2307/2330874
33. Bandt, C. & Pompe, B. (2002). Permutation Entropy: A Natural Complexity Measure for Time Series. *Physical Review Letters* 88, 174102. https://doi.org/10.1103/PhysRevLett.88.174102
34. Unakafova, V. & Keller, K. (2013). Efficiently Measuring Complexity on the Basis of Real-World Data. *Entropy* 15(10), 4392–4415. https://www.mdpi.com/1099-4300/15/10/4392
35. Zanin, M., Rodríguez-González, A., Menasalvas Ruiz, E. & Papo, D. (2018). Assessing Time Series Reversibility through Permutation Patterns. *Entropy* 20(9), 665. https://doi.org/10.3390/e20090665
36. Yang, P. & Shang, P. (2018). Relative asynchronous index: a new measure for time series irreversibility. *Nonlinear Dynamics* 93, 1545–1557. https://doi.org/10.1007/s11071-018-4275-1
37. Lacasa, L., Luque, B., Ballesteros, F., Luque, J. & Nuño, J. C. (2008). From time series to complex networks: The visibility graph. *PNAS* 105(13), 4972–4975. https://doi.org/10.1073/pnas.0709247105
38. Luque, B., Lacasa, L., Ballesteros, F. & Luque, J. (2009). Horizontal visibility graphs: Exact results for random time series. *Physical Review E* 80, 046103. https://doi.org/10.1103/PhysRevE.80.046103
39. Lan, X., Mo, H., Chen, S., Liu, Q. & Deng, Y. (2015). Fast transformation from time series to visibility graphs. *Chaos* 25(8), 083105 (cited in [14]).
40. Zhan, T. & Xiao, F. (2021). A novel weighted approach for time series forecasting based on visibility graph. arXiv:2103.13870. https://arxiv.org/abs/2103.13870
41. Osler, C. (2000). Support for Resistance: Technical Analysis and Intraday Exchange Rates. *FRBNY Economic Policy Review* (the bounce/penetration method used in [4]). https://www.newyorkfed.org/medialibrary/media/research/epr/00v06n2/0007osle.html
42. *(practitioner)* Williams, L. *Long-Term Secrets to Short-Term Trading*. Wiley (market-structure chapter, cited in [15]).
43. *(practitioner blog)* tr8dr, "Hawkes BSI", https://github.com/tr8dr/tseries-patterns/blob/master/docs/HawkesBSI.md (cited in [8]).
44. This repo: `cryptoai/model.py` (`walk_forward`, `train`), `cryptoai/backtest.py`, `cryptoai/features.py`, `cryptoai/config.py` (`MIN_AUC`, `UNUSED_FEATURES`).
45. This repo: `docs/research/reading-crypto-charts-for-day-trading.md` (feature-set comparison and multiple-testing note).
46. Binance Spot API, `GET /api/v3/klines` (1h BTCUSDT/ETHUSDT, 2023-01-01 to 2026-09-30, fetched 2026-10-05). https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints
47. Hudson, R. & Urquhart, A. (2021). Technical trading and cryptocurrencies. *Annals of Operations Research* 297, 191–220. https://doi.org/10.1007/s10479-019-03357-1
