# Classic technical-analysis concepts: what can be computed, what has evidence, what to test

*Research note, 2026-10-05. Covers the 22 chapters of a popular YouTube overview of technical analysis, to decide which of them `cryptoai/` should "learn". `cryptoai/` predicts P(up) for BTC/ETH/BNB/SOL vs USDT on Binance with a pooled HistGradientBoosting model: 4h candles (next 4 hours and next 1 day) and 1d candles (next 3 days). It uses walk-forward evaluation and a 0.1% fee per side. Current walk-forward AUC is about 0.546, 0.533 and 0.527. Citations are numbered; see [Sources](#sources). Companion to [reading-crypto-charts-for-day-trading.md](reading-crypto-charts-for-day-trading.md) (indicators, order flow, funding), [swing-points-rolling-window-directional-change-pip.md](swing-points-rolling-window-directional-change-pip.md) (swing detection, confirmation lag, truncation test), [neurotrader-methods.md](neurotrader-methods.md) (trendlines, permutation test) and [chartscanai.md](chartscanai.md) (chart images). Those notes are not repeated here.*

## Bottom line

- **A tree model can only "learn" a concept that has been turned into a number at each candle close.** Concepts that need a human to choose the points (Elliott wave counts, harmonic fits, hand-drawn zones and trend lines) cannot be learned this way without first replacing the human with a fixed rule. Once a fixed rule replaces the human, most of these concepts become swing-point features, which the swing-point note already covers.
- **Of the 22 concepts, about half are already in the model in some form:**
  - momentum and trend indicators, oscillators, and moving averages used as dynamic support and resistance;
  - range breakouts (`range_pos`, `from_high_100`);
  - candle shape (`candle_body`, plus the full-OHLC chart-image bundle that was tested and not adopted);
  - volume (`volume_z`; taker flow was tested and not adopted).
- **Peer-reviewed evidence is weak or absent for most of the rest:**
  - **Fibonacci.** An automated test on three equity markets found that price did not behave differently at Fibonacci and non-Fibonacci zones [13]. Fibonacci ratios between trend legs in the Dow occurred no more often than chance would predict [14].
  - **Candlesticks.** Not profitable on US stocks [1] or Japanese stocks [2]. Positive results exist only in some samples and subsets [3][4][5].
  - **Elliott wave, Gann angles, harmonic patterns, Heikin Ashi, Renko and the "smart money" concepts** (FVG, order blocks, BOS/CHoCH). I found no peer-reviewed tests. The sources are practitioner material only.
  - **Lunar cycle.** There is a real but small effect in stock *indices* over days to weeks [15][16]. The one crypto paper I could verify found none [17].
- **The concepts with a plausible mechanism are:**
  - **Round numbers.** Stop-loss and take-profit orders cluster there (Osler [11]), and Bitcoin prices cluster at round numbers too [12]. However, Urquhart found no return pattern after round numbers in Bitcoin [12].
  - **Market structure.** The time since the last structure break, and whether that break went with or against the trend, is information the model doesn't have.
- **Recommendation: test one bundle of 8 features, once.** The features are Fibonacci position and distance, round-number distance and crossing, BOS/CHoCH state, and RSI divergence on confirmed swings, all on 4h candles. Run the lunar phase separately, once, as a placebo.
  - Heikin Ashi run length and FVG balance were left out. On real data, the existing features explain 74% and 78% of their variance *(this note's check)*.
  - The code passes the truncation test on synthetic data and on real BTC 4h data *(this note's check)*.
  - **Expect at most +0.003 AUC, and most likely nothing.** Fib distance and the moon feature should add nothing.

---

## 1. Concept by concept

**How to read the table.**
- **Causal?** asks whether the feature can be computed at the close of bar *t* from bars ≤ *t*, with no value that changes later.
- **Confirmation lag.** A rolling-window swing with half-width *k* is known only *k* bars after it happens (swing-point note §1.1). Any concept built on swings inherits that lag.
- **Evidence** labels sources as peer-reviewed (P), working paper (W) or practitioner (Pr).

| # | Concept | Standard definition used here | Causal? How | Evidence | In model already? | Verdict |
|---|---|---|---|---|---|---|
| 1 | **Fibonacci retracement / extension** | Take the last swing leg A→B. Levels are B − r·(B − A) for r ∈ {0.236, 0.382, 0.5, 0.618, 0.786}, and extensions r ∈ {1.272, 1.618} beyond A or −0.272/−0.618 beyond B. Choosing A and B is the subjective step | Yes, if A and B are the last *confirmed* rolling-window swings (lag *k*) | Fib zones were no different from non-Fib zones on three equity markets [13] (P). Fib ratios between Dow trend legs were no more common than chance [14] (W) | Partly: the position inside the leg is like `range_pos` | **Test as a feature** (in bundle; expect null) |
| 2 | **Breakout patterns** (range, Donchian, triangle or flag breaks) | Close > highest high of the last *n* bars (trading-range break). Named patterns need fitted swing points | Range break: yes, exactly. Named patterns: yes on confirmed swings, but they are rare | Trading-range breaks worked on the Dow from 1897 to 1986 [8] (P), but the effect did not survive a data-snooping test out of sample [9] (P). Weaker than MA rules on Bitcoin (companion note [6]) | Yes: `range_pos` (20 bars), `from_high_100`, `ret_n` | **Already covered** |
| 3 | **Reversal patterns** (head-and-shoulders, double top/bottom) | Rules on the last 5 confirmed extrema [6] | Yes on confirmed swings; the lag is the extra bars needed to confirm | Patterns change the return distribution of US stocks a little [6] (P). Stand-alone head-and-shoulders was not profitable [7] (P) | No | **No credible evidence / skip** (rare; swing note #5) |
| 4 | **Elliott Wave** | 5-wave impulse plus 3-wave correction, with Fibonacci ratios between waves [26] (Pr) | **No.** Wave counts are chosen by a person and are revised as new bars arrive, which is repainting by design. An automated zig-zag count is no longer Elliott; it is a swing feature | No rigorous test found. The Fibonacci-ratio claim behind it failed on the Dow [14] (W) | — | **Not testable causally** |
| 5 | **Fair Value Gap (FVG)** | Three-candle imbalance. Bullish if low<sub>t</sub> > high<sub>t−2</sub>; bearish if high<sub>t</sub> < low<sub>t−2</sub>. "Filled" when price later trades back through the gap [23] (Pr) | Yes. It is known at the close of the third candle; fill status updates causally | None found (Pr only [23]) | **Largely yes:** the existing features explain 78% of the variance of net open FVGs *(this note's check)*. It mostly encodes a strong 3-bar move (`ret_3`, `dist_ema50`) | **Already covered** |
| 6 | **Candlestick patterns** (engulfing, hammer, doji, …) | Fixed rules on 1–3 candles' OHLC | Yes, at the close, with no lag | Not profitable on US stocks [1] or Japanese stocks from 1975 to 2004 [2] (P). Profitable in an S&P 500 sample from 1992 to 1996 [3] (P). Some bullish patterns were profitable in Taiwan [4] (P). Results depend on liquidity and size in China [5] (P) | Yes: `candle_body`. The chart-image bundle (OHLC of the last 5 candles, scaled to the range) was tested and not adopted (chartscanai note) | **Already covered** |
| 7 | **Heikin Ashi** | HA close = (O+H+L+C)/4; HA open<sub>t</sub> = (HA open<sub>t−1</sub> + HA close<sub>t−1</sub>)/2 [22] (Pr) | Yes. The recursion depends on the first bar, but that effect decays by half each bar | Practitioner only [22] | **Yes:** the existing features explain 74% of the variance of HA run length, mostly through `rsi_7` *(this note's check)* | **Already covered** |
| 8 | **Moon phases / lunar cycle** | Phase = (time − reference new moon) / 29.5306 days, mod 1 | Yes. It depends only on the timestamp | Stock indices: returns around new moons are about double those around full moons in 25 countries [15] (P). 3–5% p.a. difference in 48 countries [16] (P). Bitcoin: no significant effect [17] (P, conference) | No | **Test as a placebo** (separate run; expect null) |
| 9 | **Renko** | Bricks of fixed size *b* (or *b* = ATR); a new brick forms when the close moves *b* beyond the last brick; a reversal needs 2*b* | Yes on closes. Bricks have no fixed time, so features must be sampled at the candle close. Wick-based Renko repaints inside the bar | Practitioner only | Same as directional change with θ ≈ 2*b*/price (swing note §1.2, candidate #1) | **Already covered** (by the DC candidate) |
| 10 | **Harmonic patterns** (Gartley, bat, butterfly, crab) | XABCD swings with ratio tolerances such as AB = 0.618·XA [27] (Pr) | Yes on confirmed swings, but the ratio tolerance is a free choice and patterns are rare | Practitioner tests only (e.g. [27]); none peer-reviewed found. Neurotrader note: skip | — | **No credible evidence / skip** |
| 11 | **Support and resistance** | Horizontal levels at prior swing highs and lows, or round numbers | Swing levels: yes, with lag *k*. Round numbers: yes, exactly | FX dealers' published levels predicted intraday trend interruptions [10] (P). Stop-loss and take-profit orders cluster at and just past round numbers [11] (P). Bitcoin prices cluster at round numbers, but no return pattern follows [12] (P) | Swing levels: partly, via `from_high_100` and `range_pos`. Round numbers: no | **Test as a feature** (round numbers; swing levels are swing note #3) |
| 12 | **Dynamic S/R (moving averages)** | Price relative to EMA 20/50/200 | Yes | Price-to-MA ratio predicts daily BTC returns (companion note [4], P) | **Yes:** `dist_ema20/50/200`, `ema50_slope` | **Already covered** |
| 13 | **Trend lines** | A line through two or more swing lows (support) or highs (resistance) | Yes with an automated fit [neurotrader §6]. Hand-drawn lines are not | Practitioner only | No | **Already proposed** in the neurotrader bundle (slope and distance ÷ ATR) |
| 14 | **Gann fan / angles** | Lines from a pivot at slopes of 1×1, 1×2, 2×1, … price units per time unit | Computable from a confirmed pivot, but the price-per-bar scale is arbitrary. With ATR per bar as the scale, it reduces to "return since swing ÷ (bars since swing · ATR)" | No rigorous test found | Mostly (swing features) | **No credible evidence / skip** |
| 15 | **Momentum and trend indicators** (MA crossover, MACD, ADX, ROC) | Standard | Yes | Time-series momentum across 58 futures [18] (P); crypto momentum and MA rules, but fading and weakest for BTC (companion note [1][4][5]) | **Yes:** `ret_n`, `macd`, `macd_hist`, EMA distances, 15m/1h momentum | **Already covered** |
| 16 | **Oscillators** (RSI, Stochastic, Williams %R, CCI) | Standard | Yes | Weak as stand-alone rules; RSI's value in this model comes from being a transform of recent returns | **Yes:** `rsi_7`, `rsi_14`, `range_pos` (= 20-bar %K; Williams %R = %K − 1), `bb_pctb` (≈ CCI) | **Already covered** |
| 17 | **Divergence** (price vs RSI/MACD) | Bullish: price makes a lower swing low while RSI makes a higher low; bearish is the mirror | Yes on confirmed swings, with lag *k* after the second swing. Divergence on the unconfirmed current bar repaints | Bulkowski, 994 US stocks, 1995–2010: bullish RSI divergence beat the index only in bull markets and "fails more often than it works" [24] (Pr). No peer-reviewed test found | No. It is a *conditional* comparison across two swing points, which trees struggle to build from current-bar features | **Test as a feature** (in bundle) |
| 18 | **Volume indicators** (OBV, A/D, MFI, VWAP, volume profile) | Standard | Yes. OBV and A/D are cumulative and non-stationary, so use their slope or z-score | High-volume return premium in US stocks over the following month [19] (P). Volume Granger-causes extreme returns in 7 coins [21] (P). Volume definitions and factor structure [20] (P) | **Yes:** `volume_z`, 15m volume share. Taker flow was tested and not adopted (companion note) | **Already covered** |
| 19 | **Supply/demand zones, order blocks** | Order block = the last opposite-coloured candle before an impulsive move that breaks structure; its range is the zone [23] (Pr) | Only after the break confirms it, and choosing "impulsive" is subjective. Zones are rare and overlap FVGs | None found | No | **No credible evidence / skip** |
| 20 | **Market structure** (HH/HL vs LH/LL) | Sequence of confirmed swing highs and lows | Yes, with lag *k* | Practitioner; no peer-reviewed test found | No (swing note #2 proposes it) | **Test as a feature** (in bundle, as BOS/CHoCH state) |
| 21 | **Break of structure (BOS)** | Close beyond the last confirmed swing high (low) *in the trend's direction* | Yes. Use only swings already confirmed at *t*. Each level can be broken only once | None found (Pr [23]) | No | **Test as a feature** (in bundle) |
| 22 | **Change of character (CHoCH)** | The first break *against* the current structure trend | Same as BOS | None found (Pr [23]) | No | **Test as a feature** (in bundle) |

### 1.1 Notes on the subjective concepts

- **Elliott wave.** It is not a fixed rule. Analysts keep "alternate counts" and relabel waves as price develops, so any historical labelling is fitted after the fact. An automated version would be a zig-zag with Fibonacci-ratio checks. That is concepts 1 and 20 again, so nothing extra is lost by skipping it.
- **Harmonics and Gann.** Both can be automated on confirmed swings. Harmonic patterns need a ratio tolerance, and Gann angles need a price-per-bar scale. These choices are free parameters, so each setting is another trial for the multiple-testing count (`TRIALS_TESTED`). The patterns are also rare: a few per coin per year on 4h. With 4 coins, that is far too few for a tree to learn from.
- **Smart money concepts (FVG, order blocks, BOS/CHoCH).**
  - These ideas come from the "Inner Circle Trader" teaching material. They are practitioner vocabulary, not tested models. I found no peer-reviewed or preprint test.
  - Public implementations differ on several choices:
    - whether a break needs a close or only a wick;
    - the swing length (practitioners use separate "internal" and "swing" structure);
    - whether an FVG counts as filled when it is touched or when it is fully crossed.

    The open-source `smartmoneyconcepts` package [23] is one reference implementation. Check any such code for look-ahead before reuse. A swing confirmed at bar *i* + *k* must not be used at bar *i*.
- **Lunar effect.** The stock-index result [15][16] is published and replicated across countries, so it is not a crank idea. The effect is small (a few % per year), and it would be invisible in a 4h AUC. It is a good **placebo**: if adding it "helps", that shows how big a pure-noise gain can look.

### 1.2 Overlap measured on real data *(this note's check)*

Data: the 8 candidate features plus FVG and Heikin Ashi, on 4h data for all four coins pooled (63,891 rows). The existing `build()` features exclude the calendar and the unused order-flow and funding features. **The target was not used**, so this check spends none of the test's out-of-sample budget.

| New feature | Max \|Spearman\| with an existing feature | R² explained by all existing features (HGB, 5-fold) |
|---|---|---|
| `fib_retr` | 0.17 (`vol_ratio`) | 0.15 |
| `fib_dist_atr` | 0.27 (`bb_width`) | 0.54 |
| `round_dist_atr` | 0.01 | ≈ 0 |
| `round_cross` | 0.71 (`ret_1`) | 0.51 |
| `ms_trend` | 0.73 (`macd`) | 0.59 |
| `ms_bars_since_break` | 0.27 (`volume_z`) | 0.22 |
| `ms_choch` | 0.12 | 0.17 |
| `rsi_div` | 0.26 (`ema50_slope`) | 0.11 |
| `fvg_bal` | 0.85 (`dist_ema50`) | **0.78** |
| `ha_run` | 0.76 (`rsi_7`) | **0.74** |

Low overlap does not mean useful. `round_dist_atr` is almost independent of the existing features, but it may simply be noise.

---

## 2. Recommended bundle (pre-registered, 4h candles)

**Rules for the test.**
1. Add all 8 features at once to both 4h models (next 4h and next 1 day). Leave the hyperparameters unchanged.
2. Decide on 2022–2024 and confirm on 2025–2026, as usual. Use the team's existing acceptance rule, and do not tune any parameter below after seeing results.
3. Run `moon_cos` **separately, once**, as a placebo with the same procedure. It is not eligible for adoption whatever the result.
4. Raise `TRIALS_TESTED` by 2.
5. If the swing-point note's candidate #2 (swing structure) has not been run yet, this bundle replaces it, because the `ms_*` features cover the same idea. The neurotrader bundle (Hawkes volatility, trendlines, permutation entropy) stays a separate trial.
6. 1d candles: do not use them to decide, because the 1d model already fails the permutation test (neurotrader note). If they are reported anyway, use *k* = 3 for structure, *k* = 2 for divergence, and *k* = 3 for Fibonacci.

| Feature | Definition (known at the close of bar *t*) | Params (4h) | Expectation |
|---|---|---|---|
| `fib_retr` | Position of the close in the last confirmed swing leg. If the last confirmed swing is a high H (leg L→H): (H − C)/(H − L). If it is a low: (C − L)/(H − L). Clipped to [−1, 2] | Rolling-window swings, *k* = 6 (lag 24h) | Low. The leg-position idea is new to the model, but similar in spirit to `range_pos` |
| `fib_dist_atr` | min over levels \|C − level\| / ATR14. Levels are r ∈ {−0.618, −0.272, 0.236, 0.382, 0.5, 0.618, 0.786, 1.272, 1.618} of the same leg | as above | **Null** expected [13][14]. This is the direct test of "Fibonacci levels matter" |
| `round_dist_atr` | (C − nearest multiple of S)/ATR14, with S = 10<sup>⌊log10 C⌋ − 1</sup> (BTC 60k → 1,000; ETH 2.5k → 100; SOL/BNB → 10) | — | Low. Mechanism exists for FX [11]; clustering exists in BTC but with no return pattern [12] |
| `round_cross` | sign(⌊C<sub>t</sub>/S⌋ − ⌊C<sub>t−1</sub>/S⌋): +1 if the bar closed above a new round level, −1 below | — | Low. Osler: trends speed up after crossing a level where stops cluster [11] |
| `ms_trend` | +1 after the last structure break was upward, −1 after downward. A break is a close beyond the last confirmed, not-yet-broken swing high or low | *k* = 5 (lag 20h) | Low. 59% of its variance is explained by existing features |
| `ms_bars_since_break` | Bars since that break | *k* = 5 | Low to medium. The trend's age is new information (swing note §3.1) |
| `ms_choch` | 1 if the last break was a CHoCH (against the previous `ms_trend`), 0 if it was a BOS | *k* = 5 | Low |
| `rsi_div` | +1 for 6 bars after a bullish divergence is confirmed: the newest confirmed swing low is lower than the previous one, but RSI14 at it is higher. −1 for the bearish mirror on swing highs. Otherwise 0 | *k* = 3 (lag 12h), active 6 bars | Low. Practitioner evidence is mixed [24] |

**Placebo (separate run):** `moon_cos` = cos(2π·phase) at candle close time. It is +1 at new moon and −1 at full moon. Phase uses the mean synodic month (29.530588853 d) from the new moon of 2000-01-06 18:14 UTC. That is accurate to within about a day, which is enough for this test.

**Expected to add nothing:** `fib_dist_atr`, `moon_cos` and probably `round_dist_atr`. Heikin Ashi run length and FVG balance were left out because the existing features already explain most of their variance (§1.2). The most plausible gain is from `ms_bars_since_break` and `rsi_div`, and even that is likely to be inside the ±0.003 AUC noise band seen in earlier experiments.

### 2.1 Code sketches

These pass `assert_causal` (swing note §3.4), meaning the value at *t* is unchanged when the data is cut at *t*. They were tested on 3,000 synthetic bars and on BTC/USDT 4h (17,004 bars) *(this note's check)*. ATR is the absolute 14-bar ATR (`atr_pct * close` in `features.py`).

```python
import numpy as np
import pandas as pd
from cryptoai.features import _rsi

def _atr(df, n=14):
    c = df["close"]
    tr = pd.concat([df["high"] - df["low"], (df["high"] - c.shift()).abs(),
                    (df["low"] - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _swings(df, k):
    """Last confirmed swing high/low price and bar, known from bar i + k onwards."""
    h, l, w = df["high"], df["low"], 2 * k + 1
    top = h.shift(k).eq(h.rolling(w).max())
    bot = l.shift(k).eq(l.rolling(w).min())
    i = pd.Series(np.arange(len(df)), index=df.index, dtype=float)
    return (h.shift(k).where(top).ffill(), (i - k).where(top).ffill(),
            l.shift(k).where(bot).ffill(), (i - k).where(bot).ffill())

FIB = np.array([-0.618, -0.272, 0.236, 0.382, 0.5, 0.618, 0.786, 1.272, 1.618])

def fib_features(df, atr, k=6):
    c = df["close"]
    hi, hi_i, lo, lo_i = _swings(df, k)
    leg = (hi - lo).where(hi > lo)
    up = hi_i > lo_i                                   # last confirmed swing is a high: leg lo -> hi
    retr = ((hi - c) / leg).where(up, (c - lo) / leg)
    base, sgn = hi.where(up, lo).to_numpy(), np.where(up, -1.0, 1.0)
    levels = base[:, None] + sgn[:, None] * FIB[None, :] * leg.to_numpy()[:, None]
    dist = np.abs(c.to_numpy()[:, None] - levels).min(axis=1) / atr.to_numpy()
    return pd.DataFrame({"fib_retr": retr.clip(-1, 2), "fib_dist_atr": dist}, index=df.index)

def round_features(df, atr):
    c = df["close"]
    step = 10.0 ** (np.floor(np.log10(c)) - 1)
    return pd.DataFrame({
        "round_dist_atr": (c - (c / step).round() * step) / atr,
        "round_cross": np.sign(np.floor(c / step) - np.floor(c.shift() / step)),
    }, index=df.index)

def structure_features(df, k=5):
    """BOS/CHoCH on close breaks of confirmed, not-yet-broken swing levels."""
    h, l, c = (df[x].to_numpy(float) for x in ("high", "low", "close"))
    n = len(c)
    trend, since, choch = (np.full(n, np.nan) for _ in range(3))
    hi = lo = np.nan
    hi_live = lo_live = False
    tr, last_t, last_choch = 0, None, np.nan
    for t in range(2 * k, n):
        i = t - k                                      # candidate swing, confirmed now
        if h[i] == h[t - 2 * k : t + 1].max(): hi, hi_live = h[i], True
        if l[i] == l[t - 2 * k : t + 1].min(): lo, lo_live = l[i], True
        if hi_live and c[t] > hi:                      # break up: CHoCH if trend was down
            last_choch = float(tr == -1); tr, last_t, hi_live = 1, t, False
        elif lo_live and c[t] < lo:
            last_choch = float(tr == 1); tr, last_t, lo_live = -1, t, False
        trend[t] = tr
        if last_t is not None:
            since[t], choch[t] = t - last_t, last_choch
    return pd.DataFrame({"ms_trend": trend, "ms_bars_since_break": since,
                         "ms_choch": choch}, index=df.index)

def rsi_divergence(df, rsi, k=3, active=6):
    h, l, w = df["high"], df["low"], 2 * k + 1
    bot = l.shift(k).eq(l.rolling(w).min())
    top = h.shift(k).eq(h.rolling(w).max())
    lo_px, lo_rsi = l.shift(k).where(bot), rsi.shift(k).where(bot)
    hi_px, hi_rsi = h.shift(k).where(top), rsi.shift(k).where(top)
    prev = lambda s: s.dropna().shift(1).reindex(s.index)   # previous confirmed swing's value
    bull = bot & (lo_px < prev(lo_px)) & (lo_rsi > prev(lo_rsi))
    bear = top & (hi_px > prev(hi_px)) & (hi_rsi < prev(hi_rsi))
    sig = bull.astype(float) - bear.astype(float)
    i = pd.Series(np.arange(len(df)), index=df.index, dtype=float)
    last_t, last_s = i.where(sig != 0).ffill(), sig.where(sig != 0).ffill()
    return last_s.where(i - last_t < active, 0.0).rename("rsi_div")

SYNODIC = 29.530588853
REF_NEW_MOON = pd.Timestamp("2000-01-06 18:14", tz="UTC")

def moon_cos(index, bar=pd.Timedelta("4h")):          # placebo; index is candle OPEN time (UTC)
    phase = (((index + bar) - REF_NEW_MOON) / pd.Timedelta("1D") / SYNODIC) % 1.0
    return pd.Series(np.cos(2 * np.pi * np.asarray(phase, float)), index=index, name="moon_cos")

def ta_bundle(df):
    atr = _atr(df)
    return pd.concat([fib_features(df, atr), round_features(df, atr), structure_features(df),
                      rsi_divergence(df, _rsi(df["close"]))], axis=1)
```

**Implementation notes.**
- **Break detection is correct without extra checks.** A swing high confirmed at *t* is the maximum high of bars *t* − 2*k* … *t*. No close in that window can already be above it, so `structure_features` never misses a break.
- **Ties.** Equal highs can mark two adjacent swings. That is harmless here.
- **`rsi_div` has no maximum gap** between the two swings. Practitioners usually require them to be within about 60 bars. Decide that *before* running the test, not after.
- **Excluded features, for reference.**
  - Heikin Ashi: the recursion `ha_open[t] = (ha_open[t-1] + ha_close[t-1]) / 2` depends on the first bar loaded. The live service needs about 40 bars of warm-up for the value to match the backtest.
  - FVG fill tracking: a loop over open gaps (min size 0.1·ATR, max age 50 bars). A gap is filled when a later low (high) crosses the gap's far edge.

---

## Results in this repo (2026-10-05)

The bundle in section 2 was tested once, exactly as specified, with the moon-phase placebo run separately.
The test used walk-forward AUC on 4h candles with all four coins, split into the years used for choosing
(2022–2024) and the check years (2025–2026).

| Model | Variant | AUC 2022–2024 | AUC 2025–2026 | AUC all |
|---|---|---|---|---|
| Next 1 day (`4h`) | current features | 0.5368 | 0.5292 | 0.5324 |
| | + TA bundle | 0.5371 | 0.5277 | 0.5321 |
| | + moon placebo | 0.5352 | 0.5343 | 0.5328 |
| Next 4 hours (`4h_next`) | current features | 0.5552 | 0.5248 | 0.5458 |
| | + TA bundle | 0.5561 | 0.5265 | 0.5469 |
| | + moon placebo | 0.5559 | 0.5259 | 0.5467 |

**Not adopted.**

- **Next 1 day:** the bundle changes nothing in 2022–2024 (+0.0003) and is slightly worse in 2025–2026 (−0.0015).
- **Next 4 hours:** the bundle's gain (+0.0009, then +0.0017) is about the same as the moon placebo's (+0.0007, +0.0011). Moon phases cannot predict crypto prices, so gains of this size are noise from retraining with one more column, not information.
- **The placebo's own +0.005** on next-1-day in 2025–2026 shows how large pure noise can look in a single check window.

As the note expected, Fibonacci, round numbers, market structure (BOS/CHoCH) and RSI divergence add nothing measurable beyond the existing trend, range and momentum features. `TRIALS_TESTED` in `cryptoai/config.py` was raised to 14.

## Caveats

- Most evidence comes from equities or FX in decades before crypto existed [1][2][3][8][10][11][13][14][15][16]. None of it tests 4h Binance majors after costs.
- I read abstracts or summaries, not full texts, for [2][4][5][12][13][17][21]. For [13] I could confirm only the authors, journal (Expert Systems with Applications, 2022) and findings, not the volume or pages, so none are given. I found a master's thesis and an unrefereed notebook claiming lunar effects in crypto. I did not cite them, because I could not check their methods.
- I found no peer-reviewed test of Elliott wave, Gann, harmonic patterns, Heikin Ashi, Renko, FVG, order blocks or BOS/CHoCH. "None found" is not proof that none exists. A search summary claimed a 2015 Gann-angle study with specific accuracy figures, but I could not find the source, so it is not cited.
- Parameter values (*k*, the Fibonacci level set, the round-number step, the 6-bar divergence window) are conventional starting points, not tested choices. Changing them after seeing results makes them new trials.
- The overlap check (§1.2) used cross-validation on pooled rows. It describes redundancy only and says nothing about predictive value.

## Sources

1. Marshall, B., Young, M. & Rose, L. (2006). Candlestick technical trading strategies: Can they create value for investors? *Journal of Banking & Finance* 30(8), 2303–2323. https://ideas.repec.org/a/eee/jbfina/v30y2006i8p2303-2323.html
2. Marshall, B., Young, M. & Cahan, R. (2008). Are candlestick technical trading strategies profitable in the Japanese equity market? *Review of Quantitative Finance and Accounting* 31(2), 191–207. https://ideas.repec.org/a/kap/rqfnac/v31y2008i2p191-207.html
3. Caginalp, G. & Laurent, H. (1998). The predictive power of price patterns. *Applied Mathematical Finance* 5(3–4), 181–205. https://ideas.repec.org/a/taf/apmtfi/v5y1998i3-4p181-205.html
4. Lu, T.-H., Shiu, Y.-M. & Liu, T.-C. (2012). Profitable candlestick trading strategies—The evidence from a new perspective. *Review of Financial Economics* 21(2), 63–68. https://ideas.repec.org/a/wly/revfec/v21y2012i2p63-68.html
5. Zhu, M., Atri, S. & Yegen, E. (2016). Are candlestick trading strategies effective in certain stocks with distinct features? *Pacific-Basin Finance Journal* 37, 116–127. https://ideas.repec.org/a/eee/pacfin/v37y2016icp116-127.html
6. Lo, A., Mamaysky, H. & Wang, J. (2000). Foundations of Technical Analysis. *Journal of Finance* 55, 1705–1765. https://www.nber.org/papers/w7613
7. Savin, G., Weller, P. & Zvingelis, J. (2007). The Predictive Power of "Head-and-Shoulders" Price Patterns in the U.S. Stock Market. *Journal of Financial Econometrics* 5(2), 243–265. https://ideas.repec.org/a/oup/jfinec/v5yi2p243-265.html
8. Brock, W., Lakonishok, J. & LeBaron, B. (1992). Simple Technical Trading Rules and the Stochastic Properties of Stock Returns. *Journal of Finance* 47(5), 1731–1764. https://ideas.repec.org/a/bla/jfinan/v47y1992i5p1731-64.html
9. Sullivan, R., Timmermann, A. & White, H. (1999). Data-Snooping, Technical Trading Rule Performance, and the Bootstrap. *Journal of Finance* 54(5), 1647–1691.
10. Osler, C. (2000). Support for Resistance: Technical Analysis and Intraday Exchange Rates. *FRBNY Economic Policy Review*, July 2000. https://www.newyorkfed.org/medialibrary/media/research/epr/00v06n2/0007osle.html
11. Osler, C. (2003). Currency Orders and Exchange Rate Dynamics: An Explanation for the Predictive Success of Technical Analysis. *Journal of Finance* 58(5), 1791–1819. Staff report version: https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr125.pdf
12. Urquhart, A. (2017). Price clustering in Bitcoin. *Economics Letters* 159, 145–148. https://ideas.repec.org/a/eee/ecolet/v159y2017icp145-148.html
13. Tsinaslanidis, P., Guijarro, F. & Voukelatos, N. (2022). Automatic identification and evaluation of Fibonacci retracements: Empirical evidence from three equity markets. *Expert Systems with Applications*.
14. Batchelor, R. & Ramyar, R. (2006). Magic numbers in the Dow. Discussion paper, Cass Business School, City University London. https://openaccess.city.ac.uk/16276/
15. Dichev, I. & Janes, T. (2003). Lunar cycle effects in stock returns. *Journal of Private Equity* 6(4), 8–29.
16. Yuan, K., Zheng, L. & Zhu, Q. (2006). Are investors moonstruck? Lunar phases and stock returns. *Journal of Empirical Finance* 13(1), 1–23. https://eprints.lse.ac.uk/39409
17. Erdogan, Isildar, Erdogan & Akal (2023). Do Lunar Cycles Affect Bitcoin Prices? *ICCIDA 2022*, Lecture Notes in Networks and Systems 643. https://research.hacettepe.edu.tr/en/publications/do-lunar-cycles-affect-bitcoin-prices-2/
18. Moskowitz, T., Ooi, Y. H. & Pedersen, L. (2012). Time series momentum. *Journal of Financial Economics* 104(2), 228–250. https://research.cbs.dk/en/publications/time-series-momentum/
19. Gervais, S., Kaniel, R. & Mingelgrin, D. (2001). The High-Volume Return Premium. *Journal of Finance* 56(3), 877–919. https://ideas.repec.org/a/bla/jfinan/v56y2001i3p877-919.html
20. Lo, A. & Wang, J. (2000). Trading Volume: Definitions, Data Analysis, and Implications of Portfolio Theory. *Review of Financial Studies* 13(2), 257–300. https://ideas.repec.org/a/oup/rfinst/v13y2000i2p257-300.html
21. Bouri, E., Lau, C. K. M., Lucey, B. & Roubaud, D. (2019). Trading volume and the predictability of return and volatility in the cryptocurrency market. *Finance Research Letters* 29, 340–346. https://ideas.repec.org/a/eee/finlet/v29y2019icp340-346.html
22. *(practitioner)* Valcu, D. (2004). Using the Heikin-Ashi Technique. *Technical Analysis of Stocks & Commodities* 22(2). https://store.traders.com/v221usheteby.html
23. *(practitioner)* joshyattridge, smart-money-concepts (Python package: FVG, order blocks, BOS/CHoCH, swing highs/lows). https://github.com/joshyattridge/smart-money-concepts
24. *(practitioner)* Bulkowski, T. Divergence test (RSI divergence, 994 stocks, 1995–2010). https://www.thepatternsite.com/DivergenceTest.html
25. Companion notes in this folder: [reading-crypto-charts-for-day-trading.md](reading-crypto-charts-for-day-trading.md), [swing-points-rolling-window-directional-change-pip.md](swing-points-rolling-window-directional-change-pip.md), [neurotrader-methods.md](neurotrader-methods.md), [chartscanai.md](chartscanai.md).
26. *(practitioner)* Frost, A. J. & Prechter, R. R. *Elliott Wave Principle* (book, first ed. 1978).
27. *(practitioner)* Carney, S. *Harmonic Trading* (book series); practitioner pattern tests such as https://www.liberatedstocktrader.com/harmonic-butterfly-pattern-trading/
