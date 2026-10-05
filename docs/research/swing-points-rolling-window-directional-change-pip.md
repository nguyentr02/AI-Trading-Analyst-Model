# Finding swing highs and lows: rolling window, directional change and PIP

*Research note, 2026-10-05. Written to decide whether swing-point features should go into `cryptoai/`: 4h and 1d candles, BTC/ETH/BNB/SOL vs USDT on Binance spot, P(up) over 6 candles (4h) or 3 candles (1d), long-or-flat, 0.1% fee per side, walk-forward AUC about 0.53. Citations are numbered; see [Sources](#sources). Companion to [reading-crypto-charts-for-day-trading.md](reading-crypto-charts-for-day-trading.md).*

## Bottom line

- **All three methods find turning points only after the fact. The lag is the main thing to manage.**
  - A rolling-window top at bar *i* is known only at bar *i + k*.
  - A directional-change (DC) extreme is known only when price has reversed by θ from it [8]. Measured in bars, that lag varies.
  - The perceptually important points (PIPs) of a trailing window repaint: they move as new bars arrive, and the newest bar is always a PIP.
  - Lo, Mamaysky & Wang build this lag into their kernel-regression pattern detector with an explicit delay parameter *d* = 3 days [18]. Any backtest or training row that uses a swing before its confirmation time leaks the future.
- **The research mostly comes from intraday FX or US stocks, not from 4h/1d crypto.**
  - DC has solid descriptive results in tick FX: overshoots average about θ and take about twice as long as the DC move itself [2].
  - DC trading papers from the Essex/Kent group report strong results. They use 10-minute FX data in month-long datasets [6][7], or stocks [9].
  - I found **no peer-reviewed DC, PIP or swing-pattern trading study on Binance major coins at 4h/1d that is out of sample and after costs.** The only DC study I found with Bitcoin data is about volatility, not direction [10].
- **Chart-pattern evidence is modest.** Automated head-and-shoulders and similar patterns change the conditional return distribution in US stocks [18][19] and FX [20]. A stand-alone head-and-shoulders strategy was not profitable on 1990s US indices [19]. PIP plus pattern matching did not beat the random walk on 18 equity indices. It sometimes did on GBP/USD [17].
- **For this model, most swing features will duplicate what is already there.** `range_pos`, `from_high_100`, `ret_n` and RSI already measure where price sits relative to recent extremes. The candidates with some chance of adding information are listed below. All of them encode *time and structure* rather than level:
  1. Bars since the last confirmed swing.
  2. Higher-high/higher-low structure state.
  3. DC trend age and overshoot in θ units (aTMV, "below max").
  4. Distance to support/resistance levels formed by confirmed swings.
- **Recommendation:** add one small feature group per test, starting with DC state and swing structure. Verify causality with a truncation test (section 4.4). Expect changes in the same ±0.003 AUC band as the order-flow experiment, and adopt nothing that doesn't beat that consistently across folds.

---

## 1. The three algorithms

Notation: bars *t* = 0…T−1, with high *H*, low *L* and close *C*. "Known at *t*" means computable at the close of bar *t* using bars ≤ *t* only.

### 1.1 Rolling-window local extrema

**Definition.** Bar *i* is a swing high if *H<sub>i</sub>* = max(*H<sub>i−k</sub>…H<sub>i+k</sub>*), and a swing low if *L<sub>i</sub>* = min(*L<sub>i−k</sub>…L<sub>i+k</sub>*). The only parameter is the half-width *k*. It sets the scale: larger *k* finds fewer, more significant swings. Practitioner code uses this definition [24] *(practitioner)*. Lo, Mamaysky & Wang use the same sign-change idea on a *smoothed* series (section 1.4).

```
for t in range(2k, T):             # at the close of bar t
    i = t - k                      # the candidate is k bars back
    if H[i] == max(H[t-2k .. t]):  emit swing_high(bar=i, price=H[i], known_at=t)
    if L[i] == min(L[t-2k .. t]):  emit swing_low (bar=i, price=L[i], known_at=t)
```

**Cost.** O(T·k) naively, or O(T) with a monotone deque. `pandas.rolling().max()` is fast enough here.

**Confirmation lag.** Exactly *k* bars. The lag is fixed, which makes this the easiest method to handle without leakage. The standard leaks are `rolling(2k+1, center=True)` and `scipy.signal.argrelextrema`. Both look *k* bars into the future when applied to the whole series. Two smaller issues:
- Ties (equal highs) can mark two adjacent bars.
- With a fixed *k*, the same window means very different price moves in quiet and volatile regimes.

### 1.2 Directional change (DC)

**Idea.** DC replaces clock time with "intrinsic" event time. Fix a threshold θ, such as 3%. A trend ends, and its extreme becomes known, only when price reverses by θ from the running extreme. Guillaume et al. introduced the approach for intraday FX in 1997 [1]. Glattfelder, Dupuis & Olsen then found scaling laws with it [2], and Glattfelder & Olsen give a recent overview [12].

Each trend splits into two parts [2][8]:
- The **DC event** runs from the extreme to the DC confirmation point (DCC), where the reversal reaches θ.
- The **overshoot (OS)** runs from the DCC to the next extreme.

```
state: mode ∈ {up, down}, ext (running extreme price), ext_i, prev_ext, dcc_i
for t in range(T):
    p = C[t]
    if mode == up:
        if p > ext:                 ext, ext_i = p, t                # still in the up-trend
        elif p <= ext * (1 - θ):                                     # down DC confirmed at t
            emit extreme(type=high, bar=ext_i, price=ext, known_at=t)
            prev_ext = ext; mode = down; ext, ext_i = p, t; dcc_i = t
    else:  # mirror image with p < ext and p >= ext * (1 + θ)
```

**Parameters.**
- θ, either fixed or scaled to volatility.
- The price input: closes (simplest), or highs/lows. With highs/lows, a single bar can both set a new extreme and confirm a reversal, so you need a conservative rule for that case.

Several thresholds are often run at once [7][9].

**Cost.** One O(T) pass with constant state. It suits a live system that updates once per candle.

**Confirmation lag.** The lag is variable. Price must retrace θ from the extreme, so the extreme at bar *i* is known at DCC bar *j* > *i*, and *j − i* is random. Tsang, Ma & Chinthalapati state it plainly: "DC can only be confirmed in hindsight" [8]. Their worked example (Figure 1 in [8]) shows an extreme at 01:48 that is confirmed only at 05:08.

The usual leak is to label the bars between the extreme and the DCC as already belonging to the new trend. Plotting libraries do this, and so does any code that post-processes the full extreme list. At bar *t*, the only causal facts are these:
- the last *confirmed* extreme,
- the current mode,
- the running extreme so far,
- how far price is from that running extreme.

**Useful scaling facts.** These come from 13 FX pairs, tick data, Dec 2002 to Dec 2007 [2]:
- On average the overshoot is about as large as θ (fitted coefficient about 1.06, exponent about 1.04), so a full trend is about 2θ.
- The overshoot takes about twice as long as the DC event.
- The number of DCs scales roughly as θ<sup>−2</sup> (mean exponent about −2.03).

The same paper notes that "taking transaction costs into account breaks the scaling law for small thresholds" [2]. Aloud et al. report related "coastline" results for EUR/USD and EUR/CHF tick data from 2006 to 2009 [3].

**DC indicators.** The Essex group measures each trend in units of θ [4][8].
- **aTMV** (absolute total movement) = |P − P<sub>EP</sub>| / (P<sub>EP</sub>·θ). It is measured from the trend's starting extreme EP. It is ≥ 1 at the DC confirmation point [8], but it can fall back below 1 on later pullbacks that don't reach θ.
- **BM** ("below max") = |P<sub>max</sub> − P| / (P<sub>max</sub>·θ). It measures the pullback from the best price so far in the current trend. It lies in [0, 1), and BM = 1 would confirm a reversal.

Both are causal at every tick.
- In EUR/USD tick data from 2009 to 2014 with θ = 0.16%, half of all trends ended before aTMV reached about 1.61. Only 10% went past about 3.13, and the survival probability fell roughly exponentially [8].
- **Time-adjusted return** R = TMV·θ / T is the return per unit time of a completed trend. It is a natural feature in the spirit of these indicators. I did not verify the exact formula in [4], so treat it as this note's definition.

### 1.3 Perceptually important points (PIP)

**Idea.** PIP picks out the few points a human would use to sketch the shape of a price series, and then matches those points against pattern templates or rules. Chung, Fu, Luk & Ng introduced it in 2001 [13]. Fu et al. compared three distance measures and template-based vs rule-based matching [14], and proposed a tree structure for incremental updating [15]. Fu's 2011 survey describes PIP as a shape-preserving dimension-reduction method [16].

```
pips = [0, n-1]                       # both endpoints of the window are always PIPs
while len(pips) < m:
    for each pair of adjacent PIPs (a, b), for each a < x < b:
        line(x) = y[a] + (y[b] - y[a]) * (x - a) / (b - a)
        d(x) = one of
          vertical (VD):       |y[x] - line(x)|
          perpendicular (PD):  |s*(x - a) - (y[x] - y[a])| / sqrt(1 + s^2),  s = slope(a, b)
          Euclidean (ED):      dist((a, y[a]), (x, y[x])) + dist((x, y[x]), (b, y[b]))
    add the x with the largest d to pips
```

**Parameters.** The window length *n*, the number of points *m*, and the distance measure [14]. PD and ED mix time units with price units, so they depend on how *y* is scaled (raw price, log price, or ATR units). VD does not have this problem.

**Cost.** Each insertion scans the window, so one window costs O(n·m). Computing it at every bar costs O(T·n·m). For T ≈ 15,000 4h bars, n = 48 and m = 7, that is about 5M distance evaluations. That is trivial in numpy.

**Confirmation lag and repainting.** Within a trailing window, PIPs are recomputed every bar.
- The last bar is always a PIP, by construction.
- Interior PIPs can move or disappear as the window grows or slides.
- A "head-and-shoulders completed at bar *i*" found on the full history may not have been visible at bar *i*.

The causal approach: at each bar *t*, run PIP on `y[t-n+1 : t+1]` only, and use features from that run alone. Never use PIPs found on the full series as features or as event times.

### 1.4 Comparison: Lo, Mamaysky & Wang's kernel-regression extrema

Lo, Mamaysky & Wang [18] fit a Nadaraya–Watson kernel regression to each rolling window of *l* + *d* = 35 + 3 = 38 trading days. They use a bandwidth of 0.3× the cross-validated bandwidth, a figure chosen "through trial and error" and by polling technical analysts.

The method has three steps:
1. Find local extrema where the derivative of the smoothed curve changes sign.
2. Map each one to the actual price extremum within ±1 day.
3. Test pattern rules (head-and-shoulders, double tops, and so on) on those extrema.

The paper handles lag explicitly. A pattern's last extremum must fall by day *t+l−1*, so detection waits *d* days, and the conditional return is measured *after* the window. The authors say this means there is no "look-ahead" bias.

This approach sits between rolling windows and PIP:
- **Smoothing** reduces noise, like a larger *k*.
- **The fixed delay *d*** is like the *k*-bar confirmation of a rolling window.
- **The window is refit every bar**, so it repaints like PIP.

### 1.5 Summary table

| | Rolling window | Directional change | PIP | Kernel (LMW) |
|---|---|---|---|---|
| Scale parameter | half-width *k* (bars) | θ (% move) | window *n*, points *m* | bandwidth, window *l+d* |
| Adapts to volatility | no, unless *k* varies | yes in price terms; θ can be ATR-scaled | partly (shape only) | partly |
| When an extreme is known | exactly *k* bars later | at the DCC; lag varies | never final; repaints | after *d* bars (by design) |
| Cost per new bar | O(1) amortised | O(1) | O(n·m) | O(window²) naive |
| Main leak | centred window / `argrelextrema` | dating the trend change at the extreme, not the DCC | full-series PIPs | fitting kernel on data past the window |

---

## 2. What the evidence says

### 2.1 Directional change

**Descriptive (stylised facts).**
- Guillaume et al. [1] and Glattfelder et al. [2] use intraday FX.
- Aloud et al. [3] use EUR/USD and EUR/CHF tick data from 2006 to 2009.
- Petrov, Golub & Olsen [10] use DC to measure instantaneous volatility and its weekly seasonality in three FX rates, one Bitcoin rate and a stock index. **This is the only DC paper with Bitcoin data that I could verify, and it is about volatility, not return direction.**

**Trading and prediction.**
- **Adegboye & Kampouridis** (ESWA 2021) [6]: ML classification (will an overshoot occur?) plus regression (how long will it be?) to forecast DC trend reversals. The data are 20 FX pairs at 10-minute frequency over 10 months, split into 1,000 datasets. The paper reports significantly higher profit and lower risk than ten benchmarks. Code is public.
- **Adegboye, Kampouridis & Otero** (AI Review 2023, online 2022) [7]: a genetic algorithm combines multi-threshold DC strategies. Tested on 200 monthly datasets (20 FX pairs, 10-minute data). The paper reports statistically significant outperformance of DC and non-DC benchmarks in return and risk.
- **Salman, Melissourgos & Kampouridis** (AI Review 2025) [9]: the same multi-threshold genetic algorithm on 200 NYSE stocks, reporting high profit at low risk.
- **Golub, Glattfelder & Olsen**, "The Alpha Engine" (SSRN 2017) [5]: a counter-trend FX system that trades at DC/overshoot events. I did not verify its performance figures, so none are quoted.
- **Tsang, Ma & Chinthalapati** (ISAFM 2024) [8]: "nowcast" a reversal before the DCC using aTMV and BM. In the GBP/USD test at θ = 0.32%, the rule fired only **9 times in 747 trends**, with 7 correct. It is a proof of concept, not a strategy.
- **Wu & Han** (arXiv 2023) [11]: add HMM regime detection to DC strategies on FX tick data. Not peer reviewed.

**Assessment.** The DC literature consistently reports in-sample and short out-of-sample wins in high-frequency FX. Three problems weaken it for this repo:
1. Datasets are short (monthly).
2. Strategy grids are large, and they are optimised with genetic algorithms or genetic programming, which carries a data-snooping risk.
3. I could not confirm how each paper models transaction costs.

None of these studies covers 4h/1d crypto. The scaling laws are robust, but descriptive. "Overshoot ≈ θ" holds *on average* and is not a profitable rule by itself: the large spread around that average is the whole problem. Hudson & Urquhart's finding of no out-of-sample predictability for BTC across about 15,000 technical rules [22] is a reasonable prior for any new rule family on BTC.

### 2.2 PIP and pattern matching

- The core PIP papers [13][14][15] are about representation and matching on Hong Kong stock data. They do not test trading profits.
- **Tsinaslanidis & Kugiumtzis** (ESWA 2014) [17] segment prices with PIPs, find similar past segments with dynamic time warping, and forecast from what followed them. They used 18 equity indices and GBP/USD. The equity results were consistent with market efficiency. For the exchange rate, the method "could beat at cases" the last-price (random walk) forecast. That is a weak result.
- I found no peer-reviewed PIP study on crypto.

### 2.3 Chart patterns built on extrema

- **Lo, Mamaysky & Wang** (JF 2000) [18] studied NYSE/AMEX and Nasdaq stocks from 1962 to 1996 in seven 5-year subperiods. The return distribution after patterns differs from the unconditional distribution, and "several technical indicators do provide incremental information". They test distributions, not trading profits, and model no costs.
- **Savin, Weller & Zvingelis** (JFEc 2007) [19] used a modified LMW algorithm on the S&P 500 and Russell 2000 from 1990 to 1999. Head-and-shoulders gave "little or no support" for a stand-alone strategy. Conditioned on the pattern, risk-adjusted excess returns were about 5–7% a year, mainly as an overlay on the market portfolio.
- **Osler & Chang** (FRBNY 1995) [20] detected head-and-shoulders patterns with an objective algorithm in daily FX from 1973 to 1994. They tested profits against 10,000 bootstrapped random-walk series. I did not verify their currency-by-currency results, so none are quoted.
- **Osler** (2000) [21]: support and resistance levels published by FX dealers predicted intraday trend interruptions (see the companion note).

### 2.4 Machine learning with swing or DC features

The clearest ML use of DC features is Adegboye & Kampouridis [6]. They predict whether an overshoot will occur and how long it will last, but in 10-minute FX. I found no peer-reviewed study that adds swing-point or DC state features to a daily or 4h crypto direction classifier and reports out-of-sample results after costs. Treat any gain in this repo as a new, unproven result.

### 2.5 Practitioner implementations

neurotrader888's `TechnicalAnalysisAutomation` repository [24] *(practitioner)* implements all three methods: `rolling_window.py`, `directional_change.py` and `perceptually_important.py`. It also detects patterns on top of them (head-and-shoulders, flags and pennants, trendlines, harmonics). The README does not discuss look-ahead. Check code like this for the leaks in section 1.5 before reusing it. A separate note in this folder covers those repositories.

---

## 3. Recommendations for this model

### 3.1 What the current features already cover

`range_pos` is a 20-bar stochastic. `from_high_100` is the distance from the 100-bar high. `ret_n`, `rsi_*` and `dist_ema*` measure trend and position. A swing feature that says "price is near the top of its recent range" will be redundant. The information these features *don't* have:
1. **Time since the last turn.** None of the features measures how old the current move is.
2. **The sequence of swings**, such as higher highs and higher lows.
3. **Volatility-normalised trend progress**: how far, in θ units, the current leg has run compared with how far legs usually run [2][8].
4. **Specific price levels** where earlier swings turned.

### 3.2 Candidate features, prioritised

All candidates are causal at the candle close. ATR is the absolute 14-bar ATR, `atr_pct * close`.

| # | Feature group | Definition (causal) | 4h params | 1d params | Expected value |
|---|---|---|---|---|---|
| 1 | **DC state** | `dc_dir` (±1); `dc_atmv` = \|C − last confirmed extreme\| / (extreme·θ); `dc_bm` = pullback from the running extreme / θ, in [0,1); `dc_bars_since_dcc`; `dc_bars_since_ext`; `dc_prev_R` = last completed trend's TMV·θ / T | θ ∈ {2.5×, 5×} `atr_pct`, frozen at each DCC; or fixed {3%, 6%} | θ ∈ {1.5×, 3×} `atr_pct`; or fixed {6%, 12%} | **Low to medium. The best candidate**: trend age and survival in θ units [2][8] is information the model doesn't have |
| 2 | **Swing structure** (rolling window) | `sw_bars_since_hi/lo`; `sw_dist_hi/lo_atr` = (C − last confirmed swing price)/ATR; `sw_struct` ∈ {−1, 0, +1} from HH/HL vs LH/LL; `sw_bos` = close above the last swing high or below the last swing low | k ∈ {3, 6} | k ∈ {2, 5} | Low to medium. The distance part duplicates `range_pos`; the time and structure parts are new |
| 3 | **Support/resistance distance** | Distance in ATR to the nearest level above and below, from the last M confirmed swing highs and lows (rolling window or DC extremes); number of swings within ±0.5 ATR of each level ("touches") | k = 6, M = 10 | k = 5, M = 10 | Low. Osler-type evidence [21] is intraday FX with dealer levels, not mechanical swings |
| 4 | **PIP shape** | On the trailing n bars of log close, m PIPs (VD); features are the interior PIP positions (x/n) and heights (y − y<sub>t</sub>)/σ<sub>window</sub> | n = 48, m = 7 (10 features) | n = 30, m = 5 (6 features) | **Low.** Many noisy, correlated inputs; weak forecasting evidence [17]; the last PIP is always "now" |
| 5 | Pattern flags (head-and-shoulders, double top/bottom, flag) on confirmed swings | LMW-style rules [18] on the last 5 confirmed extrema | k = 6 | k = 3 | Very low. Rare events, so little training signal for 4 coins |

**Choosing θ.** The number of DC events scales roughly as θ<sup>−2</sup> [2], so set θ by how many trends it produces. Aim for roughly 1–3 confirmed trends per week on 4h and per month on 1d, and check the count on the training folds. Freeze θ (and the ATR used to set it) when a trend starts, so the reversal rule doesn't change while the trend is running. For the cross-coin pool, an ATR multiple keeps the meaning of θ the same for BTC and SOL. Fixed percentages do not.

**Suggested order.** Test #1 alone, then #1 + #2, then #3. Skip #4 and #5 unless the first three show a consistent gain in walk-forward grouped permutation importance. Add each group to `UNUSED_FEATURES` first, then include it, so the earlier baselines stay reproducible. Log every variant: each is another trial for the multiple-testing correction [23].

### 3.3 Python sketches (causal)

**Rolling-window swings and structure.**

```python
import numpy as np
import pandas as pd

def swing_features(df, k, atr):
    """Row t uses bars <= t. A swing at bar t-k is confirmed at the close of bar t."""
    h, l, c = df["high"], df["low"], df["close"]
    w = 2 * k + 1
    top = h.shift(k).eq(h.rolling(w).max())        # bar t-k is the max of bars t-2k..t
    bot = l.shift(k).eq(l.rolling(w).min())
    top_px, bot_px = h.shift(k).where(top), l.shift(k).where(bot)
    i = pd.Series(np.arange(len(df)), index=df.index, dtype=float)

    last_hi, last_lo = top_px.ffill(), bot_px.ffill()
    prev_hi = top_px.dropna().shift(1).reindex(df.index).ffill()   # the swing high before last_hi
    prev_lo = bot_px.dropna().shift(1).reindex(df.index).ffill()

    f = pd.DataFrame(index=df.index)
    f[f"sw{k}_bars_hi"] = i - (i - k).where(top).ffill()
    f[f"sw{k}_bars_lo"] = i - (i - k).where(bot).ffill()
    f[f"sw{k}_dist_hi"] = (c - last_hi) / atr
    f[f"sw{k}_dist_lo"] = (c - last_lo) / atr
    hh, hl = last_hi > prev_hi, last_lo > prev_lo
    f[f"sw{k}_struct"] = np.select([hh & hl, ~hh & ~hl], [1, -1], 0)
    f.loc[prev_hi.isna() | prev_lo.isna(), f"sw{k}_struct"] = np.nan
    f[f"sw{k}_bos"] = np.sign((c > last_hi).astype(int) - (c < last_lo).astype(int))
    return f
```

**Directional change state (one pass, closes, θ frozen per trend).**

```python
def dc_features(close, theta):
    """theta: Series of fractional thresholds (e.g. 2.5 * atr_pct). All outputs are known at t."""
    p, th_s = close.to_numpy(float), theta.to_numpy(float)
    n = len(p)
    out = {k: np.full(n, np.nan) for k in
           ("dc_dir", "dc_atmv", "dc_bm", "dc_bars_dcc", "dc_bars_ext", "dc_prev_R")}
    up, ext, ext_i, th = True, p[0], 0, th_s[0]
    ep, ep_i, dcc_i, prev_R = np.nan, None, None, np.nan     # ep = last *confirmed* extreme
    for t in range(n):
        if np.isnan(th):
            th = th_s[t]; ext, ext_i = p[t], t; continue      # wait for ATR warm-up
        if up:
            if p[t] > ext: ext, ext_i = p[t], t
            elif p[t] <= ext * (1 - th):                       # down-DC confirmed now
                if ep_i is not None:                           # completed up-trend ep -> ext
                    prev_R = (ext / ep - 1) / max(ext_i - ep_i, 1)
                ep, ep_i, up, dcc_i = ext, ext_i, False, t
                ext, ext_i, th = p[t], t, th_s[t]
        else:
            if p[t] < ext: ext, ext_i = p[t], t
            elif p[t] >= ext * (1 + th):                       # up-DC confirmed now
                if ep_i is not None:
                    prev_R = (ext / ep - 1) / max(ext_i - ep_i, 1)
                ep, ep_i, up, dcc_i = ext, ext_i, True, t
                ext, ext_i, th = p[t], t, th_s[t]
        if dcc_i is None:
            continue                                           # no confirmed trend yet
        out["dc_dir"][t] = 1 if up else -1
        out["dc_atmv"][t] = abs(p[t] - ep) / (ep * th)         # ~1 at the DCC [8]; can dip below on pullbacks
        out["dc_bm"][t] = abs(ext - p[t]) / (ext * th)         # pullback; 1 would confirm reversal
        out["dc_bars_dcc"][t] = t - dcc_i
        out["dc_bars_ext"][t] = t - ext_i
        out["dc_prev_R"][t] = prev_R                           # return per bar of the last full trend
    return pd.DataFrame(out, index=close.index)
```

Notes:
- `ep` is the extreme that *started* the current trend. It was confirmed at `dcc_i ≤ t`, so it is known.
- `ext` is the running extreme of the current, unconfirmed trend. It is fine as a *feature* (it's the best price so far), but it must never be treated as a confirmed turning point.
- `dc_prev_R` updates only at a DCC, which is when the previous trend becomes complete.

**PIP shape of the trailing window.**

```python
def pip_indices(y, m):
    idx = [0, len(y) - 1]
    while len(idx) < m:
        s = sorted(idx); best, best_d = None, -1.0
        for a, b in zip(s[:-1], s[1:]):
            if b - a < 2:
                continue
            x = np.arange(a + 1, b)
            d = np.abs(y[x] - (y[a] + (y[b] - y[a]) * (x - a) / (b - a)))   # vertical distance
            j = int(np.argmax(d))
            if d[j] > best_d:
                best, best_d = int(x[j]), float(d[j])
        if best is None:
            break
        idx.append(best)
    return np.array(sorted(idx))

def pip_features(close, n=48, m=7):
    y_all = np.log(close.to_numpy(float))
    cols = [f"pip_x{j}" for j in range(1, m - 1)] + [f"pip_y{j}" for j in range(m - 1)]
    out = np.full((len(y_all), len(cols)), np.nan)
    for t in range(n - 1, len(y_all)):
        w = y_all[t - n + 1 : t + 1]                       # window ends at t: no future bars
        ix = pip_indices(w, m)
        sd = w.std() or 1e-12
        out[t] = np.r_[ix[1:-1] / (n - 1), (w[ix[:-1]] - w[-1]) / sd]
    return pd.DataFrame(out, index=close.index, columns=cols)
```

### 3.4 Leakage check: the truncation test

A feature is causal only if its value at bar *t* is the same whether you compute it on the full history or on history cut at *t*. Run this test for every new feature function before any walk-forward run:

```python
def assert_causal(feature_fn, df, n_checks=25, warmup=400, seed=0):
    full = feature_fn(df)
    rng = np.random.default_rng(seed)
    for t in rng.choice(np.arange(warmup, len(df)), n_checks, replace=False):
        part = feature_fn(df.iloc[: t + 1]).iloc[-1]
        pd.testing.assert_series_equal(full.iloc[t], part, check_names=False, rtol=1e-9)
```

- It catches all three leaks above: the centred window, extremes dated before their DCC, and full-series PIPs.
- It also catches any normalisation fitted on the full sample.
- It does not catch leakage through the *target*. The existing embargo still handles that.

Swing points also leak in another way. If swing points or "bars until the next DC" are ever used as *labels*, the label horizon is variable, and the embargo must cover the longest confirmation lag, not just the 6- or 3-candle horizon.

---

## Caveats

- Almost all DC and PIP results come from intraday FX tick or 10-minute data, or from stocks [2][3][6][7][8][9][14][17]. Transfer to 4h/1d crypto is untested.
- The DC trading papers [6][7][9] tune strategies with genetic algorithms on many short datasets. I did not verify their cost assumptions or how far their out-of-sample tests extend. Their headline claims are summarised from the abstracts.
- I could not access Tsang (2010), the Essex working paper often cited for DC definitions, so it is not cited. Definitions here follow [2] and [8]. The time-adjusted return R is this note's own definition.
- I did not verify performance figures for the Alpha Engine [5] or the per-currency results of Osler & Chang [20], so none are quoted.
- The suggested θ, *k*, *n* and *m* values are starting points chosen for these timeframes, not tested results. Tuning them across folds adds trials that need multiple-testing correction [23].
- In this repo, the order-flow features changed AUC by less than one standard error (about 0.003), and the funding features hurt (companion note, "Results in this repo"). Swing features are transforms of the same OHLC data the model already sees, so the prior for a gain should be lower, not higher.

## Sources

1. Guillaume, D., Dacorogna, M., Davé, R., Müller, U., Olsen, R. & Pictet, O. (1997). From the bird's eye to the microscope: A survey of new stylized facts of the intra-daily foreign exchange markets. *Finance and Stochastics* 1(2). https://doi.org/10.1007/s007800050018
2. Glattfelder, J. B., Dupuis, A. & Olsen, R. B. (2011). Patterns in high-frequency FX data: discovery of 12 empirical scaling laws. *Quantitative Finance* 11(4), 599–614. https://arxiv.org/abs/0809.1040
3. Aloud, M., Tsang, E., Olsen, R. & Dupuis, A. (2012). A directional-change event approach for studying financial time series. *Economics: The Open-Access, Open-Assessment E-Journal* 6. https://ideas.repec.org/a/zbw/ifweej/201236.html
4. Tsang, E. P. K., Tao, R., Serguieva, A. & Ma, S. (2017). Profiling high-frequency equity price movements in directional changes. *Quantitative Finance* 17(2), 217–225. https://doi.org/10.1080/14697688.2016.1164887
5. Golub, A., Glattfelder, J. B. & Olsen, R. B. (2017). The Alpha Engine: Designing an Automated Trading Algorithm. SSRN 2951348. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2951348
6. Adegboye, A. & Kampouridis, M. (2021). Machine learning classification and regression models for predicting directional changes trend reversal in FX markets. *Expert Systems with Applications* 173, 114645. https://doi.org/10.1016/j.eswa.2021.114645
7. Adegboye, A., Kampouridis, M. & Otero, F. (2023; online 2022). Algorithmic trading with directional changes. *Artificial Intelligence Review* 56(6), 5619–5644. https://doi.org/10.1007/s10462-022-10307-0
8. Tsang, E. P. K., Ma, S. & Chinthalapati, V. L. R. (2024). Nowcasting directional change in high frequency FX markets. *Intelligent Systems in Accounting, Finance and Management* 31, e1552. https://doi.org/10.1002/isaf.1552 (open access: https://research.gold.ac.uk/id/eprint/37843)
9. Salman, O., Melissourgos, T. & Kampouridis, M. (2025). A genetic algorithm for the optimization of multi-threshold trading strategies in the directional changes paradigm. *Artificial Intelligence Review*. https://doi.org/10.1007/s10462-025-11419-z
10. Petrov, V., Golub, A. & Olsen, R. (2019). Instantaneous Volatility Seasonality of High-Frequency Markets in Directional-Change Intrinsic Time. *Journal of Risk and Financial Management* 12(2), 54. https://www.mdpi.com/1911-8074/12/2/54
11. Wu, B. & Han, X. (2023). Intelligent trading strategy based on improved directional change and regime change detection. arXiv:2309.15383. https://arxiv.org/abs/2309.15383
12. Glattfelder, J. B. & Olsen, R. B. (2024). The Theory of Intrinsic Time: A Primer. arXiv:2406.07354. https://arxiv.org/abs/2406.07354
13. Chung, F. L., Fu, T. C., Luk, R. & Ng, V. (2001). Flexible time series pattern matching based on perceptually important points. Workshop on Learning from Temporal and Spatial Data, IJCAI 2001, 1–7. https://research.polyu.edu.hk/en/publications/flexible-time-series-pattern-matching-based-on-perceptually-impor/
14. Fu, T. C., Chung, F. L., Luk, R. & Ng, C. M. (2007). Stock time series pattern matching: Template-based vs. rule-based approaches. *Engineering Applications of Artificial Intelligence* 20(3), 347–364. https://doi.org/10.1016/j.engappai.2006.07.003
15. Fu, T. C., Chung, F. L., Luk, R. & Ng, C. M. (2008). Representing financial time series based on data point importance. *Engineering Applications of Artificial Intelligence* 21(2), 277–300. https://doi.org/10.1016/j.engappai.2007.04.009
16. Fu, T. C. (2011). A review on time series data mining. *Engineering Applications of Artificial Intelligence* 24(1), 164–181. https://repository.vtc.edu.hk/ive-it-sp/1
17. Tsinaslanidis, P. & Kugiumtzis, D. (2014). A prediction scheme using perceptually important points and dynamic time warping. *Expert Systems with Applications* 41(15), 6848–6860. https://doi.org/10.1016/j.eswa.2014.04.028
18. Lo, A., Mamaysky, H. & Wang, J. (2000). Foundations of Technical Analysis: Computational Algorithms, Statistical Inference, and Empirical Implementation. *Journal of Finance* 55, 1705–1765. NBER WP 7613: https://www.nber.org/papers/w7613
19. Savin, G., Weller, P. & Zvingelis, J. (2007). The Predictive Power of "Head-and-Shoulders" Price Patterns in the U.S. Stock Market. *Journal of Financial Econometrics* 5(2), 243–265. https://ideas.repec.org/a/oup/jfinec/v5yi2p243-265.html
20. Osler, C. L. & Chang, P. H. K. (1995). Head and Shoulders: Not Just a Flaky Pattern. Federal Reserve Bank of New York Staff Report 4. https://fraser.stlouisfed.org/title/staff-reports-federal-reserve-bank-new-york-9235/head-shoulders-673652
21. Osler, C. (2000). Support for Resistance: Technical Analysis and Intraday Exchange Rates. *FRBNY Economic Policy Review*, July 2000. https://www.newyorkfed.org/medialibrary/media/research/epr/00v06n2/0007osle.html
22. Hudson, R. & Urquhart, A. (2021). Technical trading and cryptocurrencies. *Annals of Operations Research* 297, 191–220. https://doi.org/10.1007/s10479-019-03357-1
23. Bailey, D. & López de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management* 40(5), 94–107. https://papers.ssrn.com/abstract=2460551
24. *(practitioner)* neurotrader888, TechnicalAnalysisAutomation (GitHub repository: rolling-window, directional-change and PIP extrema; head-and-shoulders, flag, trendline detection). https://github.com/neurotrader888/TechnicalAnalysisAutomation
