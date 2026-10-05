# Reading crypto charts for day trading: what the evidence says, and what to feed the model

*Research note, 2026-10-05. Written to choose features for `cryptoai/` (4h and 1d candles, BTC/ETH/BNB/SOL vs USDT on Binance spot, P(up) over the next 1 or 3 days, long-or-flat, 0.1% fee per side). Citations are numbered; see [Sources](#sources).*

## Bottom line

- **Most chart reading adds little on top of what the model already has.** The signals with the most support in crypto are trend and momentum: price relative to moving averages [4], time-series momentum at daily-to-weekly horizons [1][13], and technical-rule families [5][6]. The current feature set (returns, EMA distances, RSI, MACD, Bollinger) already covers them. Expect diminishing returns from adding more indicators of the same kind.
- **The evidence is getting weaker over time.** Hudson & Urquhart found no out-of-sample predictability for Bitcoin even though their in-sample results were strong [5]. A 2026 study found that the funding-rate carry trade's Sharpe ratio fell from 6.45 over the full sample to negative in 2025 [3]. AUC ≈ 0.53 is roughly what an honest daily-horizon model on large coins should get. Be suspicious of anything much higher.
- **The best new information is positioning and flow data from futures, not more price transforms.** Ranked by evidence and cost:
  1. **Taker buy/sell imbalance.** It is free in the same Binance kline response, but `ccxt.fetch_ohlcv` discards it.
  2. **Funding rate and perp premium.** Full history is available from 2019–2020.
  3. **Open interest.** History comes only from bulk files; the REST endpoint keeps 30 days.
- **Order-flow imbalance predicts mostly at very short horizons.** The one study that finds predictability at 4–12 hours uses quarter-hour-opening imbalance on Binance perps [16]. Other order-flow papers find contemporaneous effects or effects measured in seconds to minutes [17][18]. Treat it as a moderate-value feature, not a breakthrough.
- **High funding or basis is a crash-risk signal, not a "go long" signal** [21][22]. It should mainly help the model avoid long entries after crowded rallies.
- **Some current features are weak or risky.** These are `dow`/`hour` (calendar effects are inconsistent [24][25]), `xs_rank_24` (cross-sectional momentum evidence comes from hundreds of coins at weekly horizons [2], not from a 4-coin rank), and raw `volume_z` (Binance spot volume was distorted by the zero-fee BTC promotion from July 2022 to March 2023 [41]).
- **Candlestick patterns and hand-drawn support and resistance have weak or no evidence** as standalone predictors [27][29]. The model's `candle_body` and `range_pos` already capture what can be computed.

---

## 1. How practitioners read charts (brief)

Practitioner conventions are described here so the features map to how traders think. Sources marked *(practitioner)* are not evidence of predictive value.

**Candlesticks and patterns.** Each candle shows open, high, low and close. Traders read the body size, the wicks (rejection of a price level), and named patterns such as engulfing, hammer, doji and morning star. The best-known test on large US stocks found candlestick signals "not generally profitable", and one bullish signal reliably predicted *negative* returns [27]. I found no peer-reviewed crypto study showing robust candlestick profits after costs.

**Support/resistance and market structure.** Support and resistance are price levels where a trend is expected to stall or reverse. Market structure means reading swings: a trend is a series of higher highs and higher lows (or the reverse), and a "break of structure" signals that the trend may end. Osler found that support and resistance levels published by FX dealers did predict intraday trend interruptions, with an effect lasting about 5 business days [29]. Trading-range breakout rules on Bitcoin had weaker results than moving-average rules [6]. Lo, Mamaysky & Wang automated chart patterns such as head-and-shoulders and found "incremental information" in US stocks from 1962 to 1996 [28].

**Volume and volume profile.** Traders look for volume confirming a breakout, volume spikes at turning points, and a "volume-by-price" profile to find high-activity price zones. VWAP is the volume-weighted average price, and it resets each session [40e]. Crypto trades 24/7, so the session boundary is a convention, usually 00:00 UTC.

**Indicators.** The model already uses most of these:

| Indicator | Originator / definition | In model? |
|---|---|---|
| RSI(14), 70/30 levels | Wilder, 1978 [40a] | yes (7, 14) |
| ATR, true range | Wilder, 1978 [40b] | yes (`atr_pct`) |
| MACD (12, 26, 9) | Appel, late 1970s [40c] | yes |
| Bollinger Bands (20, 2σ) | Bollinger [40d] | yes (%B, width) |
| Stochastic %K = (C − LL)/(HH − LL) | Lane [40f] | effectively yes: `range_pos` is a 20-period %K |
| VWAP | session VWAP [40e] | no |
| Moving averages / crossovers | — | yes (EMA 20/50/200 distance, slope) |

**Multi-timeframe analysis.** Traders take direction from a higher timeframe and time entries on a lower one, for example a 1d trend with a 4h trigger. The model has no explicit higher-timeframe context for 4h candles beyond long lookbacks (ret_100, dist_ema200).

**Crypto-specific data.** Traders watch several kinds of crypto-only data:
- **Perpetual-futures funding rates.** Longs pay shorts when the perp trades above spot, so a high funding rate means crowded longs [20].
- **Open interest.** This is the number of outstanding contracts. Rising open interest together with rising price is read as new leverage.
- **Liquidations.** These are forced closes, and cascades of them often mark local extremes.
- **Long/short account ratios.**
- **Taker buy/sell volume.** This measures aggressive buying vs selling.
- **Order-book depth and imbalance.**
- **Exchange inflows/outflows.** These are on-chain data and are not available from Binance.

**Risk management.** Traders size each position so that a stop-out loses a fixed fraction of equity (commonly 0.5–2%). They place stops a multiple of ATR beyond entry or beyond a structure level, and they require a reward-to-risk ratio of at least about 2:1 *(practitioner convention; no source cited)*. The ATR definition is Wilder's [40b]. For this repo, the relevant point is that the backtest's fixed 0.55/0.48 probability thresholds ignore volatility. Any edge in P(up) must clear 0.2% round-trip fees.

---

## 2. What the evidence says

### 2.1 Time-series momentum and reversal (the strongest family)

- **Liu & Tsyvinski** (RFS 2021) [1] cover BTC daily data from 2011-01 to 2018-05, with XRP and ETH over shorter windows. A one-standard-deviation BTC daily return raises returns 1, 3, 5 and 6 days ahead by 0.33, 0.17, 0.39 and 0.50 percentage points. Weekly returns predict 1–4 weeks ahead. The effect survives a no-lookahead quintile test, but it is weaker for ETH. The data come from the early, thin-market era, and no costs are modeled.
- **Borgards** (NAJEF 2021) [13] studies 20 coins at daily and intraday frequencies. Crypto momentum periods are longer and larger than in US stocks at all frequencies.
- **Shen, Urquhart & Wang** (Financial Review 2022) [11] find that, for Bitcoin, the first half-hour of the trading "day" predicts the last half-hour. The effect is driven by liquidity provision.
- **Wen, Bouri, Xu & Zhao** (NAJEF 2022) [12] use BTC data from 2013-03 to 2020-05 and find both intraday momentum and reversal. The pattern changes around large jumps and FOMC announcements, and it extends to ETH, LTC and XRP.
- **Kitron & Wengrowicz** (arXiv 2026) [14] study 15-minute candles on 183 Binance pairs. They find pervasive reversal of the previous candle's sign, concentrated after moves driven by aggressive taker flow. The gross edge is about 1.3 bp per trade against a 5 bp round-trip cost, so it is **not tradable**. Out-of-sample AUC gain is +0.011 to +0.031. This is a useful calibration: honest crypto direction AUCs are close to 0.5.

**Implication.** The model's return lags already target this family. For 1d candles, the 6/12/24 lags cover 1–4 weeks. For 4h candles, 1 week is about 42 candles, which falls between ret_24 and ret_50.

### 2.2 Technical trading rules in crypto

- **Detzel, Liu, Strauss, Zhou & Zhu** (Financial Management 2021) [4] find that the ratio of price to its moving average forecasts *daily* Bitcoin returns both in and out of sample, with economically significant alpha vs buy-and-hold. This is the `dist_emaN` feature the model already has.
- **Hudson & Urquhart** (Annals of OR 2021) [5] test about 15,000 rules from five rule classes on two BTC markets and three other coins. They correct for data snooping, and breakeven costs exceed typical crypto costs. However, **they find no out-of-sample predictability for Bitcoin**; predictability remains only in the other coins.
- **Corbet, Eraslan, Lucey & Sensoy** (FRL 2019) [6] use high-frequency BTC data. Moving-average rules work, with the variable-length MA performing best and buy signals beating sell signals. Trading-range breakout rules are tested too.
- **Grobys, Ahmed & Sapkota** (FRL 2020) [7] study 11 coins from 2016 to 2018 with daily data. A 20-day MA rule earns 8.76% p.a. excess return **excluding Bitcoin**, which suggests BTC was already the most efficient coin.
- **Fieberg et al., "CTREND"** (JFQA 2025) [10] combine many technical indicators across horizons, using price and volume, with ML on 3,000+ coins. The signal predicts the cross-section, survives costs, and persists in big, liquid coins. This supports the repo's approach of feeding many indicators into a tree model rather than hand-picking one rule.

**Implication.** Trend-following technicals have real but fading predictive value, and the effect is weakest in BTC. Adding more of them mainly adds collinearity.

### 2.3 Cross-sectional momentum

- **Liu, Tsyvinski & Wu** (JF 2022) [2] study all coins above $1M market cap from 2014 to 2018 at weekly frequency. A three-factor model (market, size, momentum) prices the cross-section. Strategies based on 1–4-week momentum earn about 2.5–4.1% per week long-short, and low-volume coins outperform high-volume coins.
- **Grobys & Sapkota** (Econ Letters 2019) [8] study 143 coins from 2014 to 2018 and find **no significant momentum payoffs**.
- **Dobrynskaya** (J. Alternative Investments; via HSE summary) [9] finds momentum up to about 2–4 weeks and reversal beyond 4–6 weeks for coins from 2014 to 2020.
- **Borri, Liu, Tsyvinski & Wu** (arXiv 2025, rev. 2026) [3] find that cross-sectional momentum stays positive after 2020.
- **Bianchi, Babiak & Dickerson** (JBF 2022) [15] study 2017-03 to 2022-03 and find that short-term reversal returns, a proxy for paying liquidity providers, are concentrated in *less* active, smaller, less liquid pairs. BTC, ETH, BNB and SOL are the opposite of that.

**Implication.** These results come from long-short portfolios over hundreds of coins, driven largely by small coins. A percentile rank across four large coins (`xs_rank_24`) takes only four values per timestamp and has little support in this literature.

### 2.4 Volume, order flow and microstructure

- **Silantyev** (Digital Finance 2019) [17] uses BitMEX XBTUSD data. Trade-flow imbalance explains *contemporaneous* price changes better than order-book flow imbalance. This is explanation, not forecasting.
- **Kim & Hansen** (arXiv 2026) [16] use six Binance perpetuals. **Order imbalance at quarter-hour openings predicts returns over the next 4–12 hours**, and opening returns are predictable out of sample. Effects are weaker at finer clock-time frequencies. This is the most directly relevant order-flow result for a 4h/1d model.
- **Bieganowski & Ślepaczuk** (arXiv 2026) [18] use 1-second Binance futures order books from 2022-01 to 2025-10. Order-flow imbalance is the top feature in a CatBoost model across five coins. The horizon is very short, and a taker backtest is reported.
- **Easley, O'Hara, Yang & Zhang** (SSRN 2024) [19] find that microstructure measures such as Roll and VPIN predict price *dynamics* (volatility and other quantities relevant for market making and hedging) for five coins. The effects include cross-market effects from BTC and ETH, and they are stable through the crypto winter.
- **Liu, Tsyvinski & Wu** [2]: low dollar volume predicts higher returns cross-sectionally (a size or liquidity effect).

**Implication.** Taker imbalance is the best-supported flow variable at the model's horizon, and Binance already ships it inside every kline. Order-book snapshots mostly matter at second-to-minute horizons.

### 2.5 Funding rate, basis and derivatives positioning

- **He, Manela, Ross & von Wachter** (arXiv 2022, v7 2026) [20] explain how funding works and derive no-arbitrage prices. Perp-spot deviations in crypto are larger than in FX, move together across coins, and have shrunk over time.
- **Schmeling, Schrimpf & Todorov** (BIS WP 1087, 2023) [21] find that crypto futures carry (basis) can reach about 60% p.a. and varies with trend-chasing retail demand. **High carry predicts future price crashes.**
- **Christin, Routledge, Soska & Zetlin-Jones** (2022; Management Science) [22] find that derivatives volume is driven by the long side, and the short-perp/long-spot carry trade had in-sample Sharpe ratios of 7–10.
- **Borri, Liu, Tsyvinski & Wu** [3] report that the carry Sharpe ratio was 6.45 over 2020–2025, fell to 4.06 from 2024, and turned **negative in 2025**. This is evidence of arbitrage capital arriving and the premium decaying.
- **Hazelkorn, Moskowitz & Vasudevan** (NBER w26773) [23] find that, in equity-index and FX futures, the basis *negatively predicts* spot and futures returns, consistent with uninformed leverage demand. This is not a crypto result, but it is the same mechanism.

**Implication.** Funding and premium carry information about crowded leverage and downside risk. That is useful for a long-or-flat model deciding when *not* to be long. The evidence for funding as a positive directional signal is thin, and the effect has decayed.

**Open interest, liquidations and long/short ratios.** I found **no peer-reviewed crypto study** in this search that cleanly shows open interest, Binance long/short ratios, or liquidation counts predicting 1–3-day direction out of sample after costs. Treat these as unproven.

### 2.6 Seasonality

- **Caporale & Plastun** (FRL 2019) [24]: only BTC shows a day-of-week effect (higher Monday returns). Simulated profits are mostly indistinguishable from random.
- **Baur, Cahill, Godfrey & Liu** (2019, SSRN) [25] use more than 15M observations from 7 exchanges. Time-of-day and day-of-week effects vary over time, with "no consistent or persistent patterns".
- Kim & Hansen [16] document real periodicity, but at 1-, 5- and 15-minute clock marks. That is invisible at 4h.

### 2.7 Volatility clustering

- **Katsiampa** (Econ Letters 2017) [26]: Bitcoin volatility is best fit by an AR-CGARCH model, with both short-run and long-run variance components. Volatility is persistent and forecastable.
- **Borri et al.** [3]: jumps are frequent; for BTC and ETH a significant jump occurs on about 40% of days before 2020.

**Implication.** Volatility features (`vol_10`, `vol_50`, `atr_pct`) forecast the *size* of moves, not their sign. They are still useful as conditioning variables, because the tree can learn that momentum works differently in high-volatility regimes. They are also useful for deciding whether a move is large enough to clear fees.

---

## 3. Recommendations for this model

### 3.1 New features and data, prioritised

All endpoints are public and need no API key. History limits were checked live on 2026-10-05.

| # | Feature | Evidence | Binance source / endpoint | Effort | Expected value |
|---|---|---|---|---|---|
| 1 | **Taker buy ratio** = taker_buy_base / volume, plus its 6/24-candle rolling sums, z-score, and interaction with `ret_1` (flow-driven moves reverse [14]) | Order imbalance predicts 4–12h returns [16]; top short-horizon feature [18]; explains contemporaneous moves [17] | Spot `GET /api/v3/klines`, field 10 ("Taker buy base asset volume") and field 9 ("Number of trades") [32]; 1000 rows/request, weight 2; full spot history. Bulk: `data.binance.vision/data/spot/monthly/klines/` [39]. **`ccxt.fetch_ohlcv` drops these columns**: call the raw endpoint, or use `ccxt`'s `publicGetKlines` | Low | **Medium**. The cheapest new information available |
| 2 | **Funding rate**: last value, 3-day and 7-day mean, 90-day z-score, cross-coin average | High carry predicts crashes [21]; long-side demand [22]; basis negatively predicts returns in other markets [23]; decaying [3] | `GET /fapi/v1/fundingRate` (max 1000 rows, ascending, 500 requests/5 min/IP shared with fundingInfo) [33]. History starts BTC 2019-09-10, ETH 2019-11-27, BNB 2020-02-10, SOL 2020-09-13. Usually an 8h interval: forward-fill onto candles, using only settlements at or before candle close | Low | **Medium**, mainly for avoiding longs |
| 3 | **Perp premium / basis**, on the same 4h/1d grid as the model: close, 7-day mean, z-score | Same as #2; perp-spot gap is the input funding is computed from [20] | `GET /fapi/v1/premiumIndexKlines` (max 1500 rows) [37]; bulk monthly files from 2020-01 for BTC (`data/futures/um/monthly/premiumIndexKlines/`) [39]. Alternatively, perp close (`/fapi/v1/klines`) ÷ spot close − 1 | Low | Medium. Overlaps #2 but is continuous and finer |
| 4 | **Open interest**: 6- and 24-candle % change, OI change × sign of price change, OI / volume | **Unproven**: no peer-reviewed crypto evidence found | REST `GET /futures/data/openInterestHist` serves **only the last 30 days** (older `startTime` returns error -1130) [34]; the REST long/short ratio endpoints have the same 30-day limit [36]. For history, use the bulk `data/futures/um/daily/metrics/` files (5-minute rows: `sum_open_interest`, `sum_open_interest_value`, top-trader and account long/short ratios, `sum_taker_long_short_vol_ratio`), from 2020-09-01 for BTC and 2021-12-01 for ETH/BNB/SOL [39] | Medium (daily zip files, about 1,800 per coin) | Low to medium. Test it, but don't expect much |
| 5 | **Futures taker buy/sell ratio** (perp market, not spot) | As #1; perps carry most leveraged flow [20][22] | `/fapi/v1/klines` field 10 (full history), or `/futures/data/takerlongshortRatio` (**30 days only**, max 500 rows) [35]; bulk metrics as in #4 | Low | Low to medium. Likely correlated with #1 |
| 6 | **Weekly lookbacks and vol-scaled returns**: for 4h, `ret_42` (1 week) and `ret_84`; for both timeframes, `ret_n / vol_50` | Weekly time-series momentum horizons [1][9]; scaling by volatility is an engineering suggestion, not a tested result | Existing OHLCV | Very low | Low. Mostly a re-expression of existing features |
| 7 | **Higher-timeframe context for 4h**: daily `dist_ema50`, daily RSI, merged as-of the last *closed* daily candle | Multi-timeframe practice; daily MA-ratio evidence [4] | Existing 1d data | Low | Low. Watch for leakage |
| 8 | **Illiquidity / microstructure**: Amihud = \|ret\| / quote volume; Roll spread estimate from close-to-close autocovariance | Liquidity effects [2][15]; Roll measure predicts dynamics [19] | Spot klines field 8 ("Quote asset volume") [32] | Low | Low. Large coins are liquid |
| 9 | **Liquidations** | Unproven; crash dynamics [21] | Public liquidation history is not available. The WebSocket `<symbol>@forceOrder` / `!forceOrder@arr` streams push at most one liquidation per symbol per 1000 ms [38], so they must be recorded forward and undercount | High (needs a collector) | Low for now |
| 10 | **Order-book depth imbalance** | Strong at second-to-minute horizons [18][19]; little evidence at 1 day | `GET /api/v3/depth` gives snapshots only (limit ≤5000; weight 5–250) [32]. Bulk futures `bookDepth` daily files exist from 2023-01 [39] | High | Low at this horizon |

**Suggested order:** do #1 and #2 together, because both are cheap and #1 needs no new data source. Add #3, then try #4 only if #1–#3 show gains in walk-forward permutation importance. Assets missing early history work without imputation, because `HistGradientBoostingClassifier` handles NaN natively. Still compare training from 2021 onward to check for regime effects.

### 3.2 Current features to question

| Feature | Concern | Suggestion |
|---|---|---|
| `dow`, `hour` | Calendar effects are inconsistent [24][25]. `hour` takes only 6 values on 4h and is constant on 1d. These are easy to overfit | Drop, or keep only if walk-forward permutation importance is consistently positive |
| `xs_rank_24` | Cross-sectional evidence comes from hundreds of coins at weekly horizons [2][8]. A 4-coin rank takes only 4 values, and `rel_btc_*` already carries most of the information | Low priority; check importance |
| `volume_z` (raw spot volume) | Binance ran zero-fee BTC spot pairs (including BTC/USDT) from July 2022 to March 2023. BTC spot volume dropped more than 65% when fees returned [41] *(news source)*. This breaks the volume regime for BTC/USDT | Use taker *ratio* (#1) instead of raw level, or volume z-scored against a longer window. Flag 2022-07 to 2023-03 in diagnostics |
| `range_pos` vs Stochastic | They are the same calculation [40f] | Fine. Just don't add a separate Stochastic |
| `rsi_7`, `rsi_14`, `bb_pctb`, `range_pos`, `dist_ema20` | Highly collinear transforms of recent returns. Harmless for trees, but they make importance scores misleading | Use grouped permutation importance |
| `candle_body` | Candlestick evidence is weak [27] | Low value; keep or drop |

### 3.3 Evaluation hygiene (affects whether any new feature "works")

- **Overlapping labels.** On 4h candles, the target looks 6 candles ahead, so consecutive labels overlap. Walk-forward folds need a gap of at least the horizon between train and test, or test AUC is inflated.
- **Multiple testing.** Each feature you try is a hypothesis. Harvey, Liu & Zhu argue for t > 3.0 on new factors [30], and Bailey & López de Prado's deflated Sharpe ratio corrects backtests for the number of trials [31]. Keep a log of every feature set tried.
- **Decay.** Re-run results on 2024 onward separately. Both BTC technical predictability [5] and carry [3] have weakened recently.

---


## Results in this repo (2026-10-05)

The top two recommendations were implemented and tested with the existing walk-forward setup
(8 folds, embargo equal to the horizon, 0.1% fee per side). Mean Sharpe is the long-or-flat strategy
averaged over the four coins.

| Feature set | 4h AUC | 4h AUC since 2024 | 4h mean Sharpe | 1d AUC | 1d AUC since 2024 | 1d mean Sharpe |
|---|---|---|---|---|---|---|
| Baseline (chart + market context) | 0.5310 | 0.5266 | 0.56 | **0.5272** | **0.5310** | **0.78** |
| + order flow (taker ratio, trade count) | **0.5326** | **0.5275** | **0.59** | 0.5259 | 0.5257 | 0.68 |
| + order flow + funding/premium (levels and z-scores) | 0.5267 | 0.5227 | 0.34 | 0.5169 | 0.5234 | 0.47 |
| + order flow + funding/premium z-scores only | 0.5276 | 0.5230 | 0.35 | 0.5246 | 0.5285 | 0.69 |
| Everything, minus `dow`/`hour`/`xs_rank_24` | 0.5255 | 0.5187 | 0.37 | 0.5142 | 0.5166 | 0.55 |

Conclusions:

- **Funding and premium lowered results on both timeframes**, in both raw and normalised form. The
  evidence above says they mainly predict crashes, which are rare, and the funding regime shifted a lot
  between 2020 and 2025. With shallow, heavily regularised trees, the extra noisy inputs seem to crowd out
  the trend features.
- **Order flow helped 4h by +0.0016 AUC**, well within noise (one standard error is about 0.003 at
  44,800 test rows), **and hurt 1d**. Not adopted.
- **Removing calendar and rank features hurt**, so they stay despite the weak literature.
- All new features are still computed (`cryptoai/features.py`) and their data is still downloaded
  (`cryptoai/data.py`). They are excluded through `UNUSED_FEATURES` in `cryptoai/config.py`, so they can be
  re-tested later, for example with more data or a different model.
- Five feature sets were compared, so the best result is optimistic by some amount (see the multiple-testing
  caveat in 3.3).

## Caveats

- Most peer-reviewed crypto evidence uses data from before 2020, often aggregated from CoinMarketCap or CoinDesk, and covers many small coins. Results for four large Binance pairs in 2021–2026 may be much weaker.
- Several of the most relevant papers are 2026 arXiv preprints that have not been peer reviewed [14][16][18], or SSRN working papers [19].
- The Dobrynskaya finding [9] is cited from the author's institution's summary, not the paper itself.
- Some Binance doc URLs now redirect to a JavaScript-rendered catalog page. Endpoint names and limits were confirmed from the docs via search snippets and by live calls on 2026-10-05: funding history start dates, the 30-day OI limit returning error -1130, and the metrics file columns and start dates. Binance can change these limits.
- Futures data comes from Binance USDⓈ-M perps, while the model trades spot. The basis between them is itself one of the proposed features.
- I could not verify sample periods for Detzel et al. [4] and Hudson & Urquhart [5] from their abstracts, so none are stated.
- Risk-management conventions in §1 are practitioner norms, not tested results.

## Sources

1. Liu, Y. & Tsyvinski, A. (2021). Risks and Returns of Cryptocurrency. *Review of Financial Studies* 34(6), 2689–2727. NBER WP 24877: https://www.nber.org/papers/w24877
2. Liu, Y., Tsyvinski, A. & Wu, X. (2022). Common Risk Factors in Cryptocurrency. *Journal of Finance* 77, 1133–1177. https://papers.ssrn.com/abstract=3379131
3. Borri, N., Liu, Y., Tsyvinski, A. & Wu, X. (2025, rev. 2026). Cryptocurrency as an Investable Asset Class: Coming of Age. arXiv:2510.14435. https://arxiv.org/abs/2510.14435
4. Detzel, A., Liu, H., Strauss, J., Zhou, G. & Zhu, Y. (2021). Learning and predictability via technical analysis: Evidence from bitcoin and stocks with hard-to-value fundamentals. *Financial Management* 50(1), 107–137. https://ideas.repec.org/a/bla/finmgt/v50y2021i1p107-137.html
5. Hudson, R. & Urquhart, A. (2021). Technical trading and cryptocurrencies. *Annals of Operations Research* 297, 191–220. https://doi.org/10.1007/s10479-019-03357-1
6. Corbet, S., Eraslan, V., Lucey, B. & Sensoy, A. (2019). The effectiveness of technical trading rules in cryptocurrency markets. *Finance Research Letters* 31, 32–37. https://ideas.repec.org/a/eee/finlet/v31y2019icp32-37.html
7. Grobys, K., Ahmed, S. & Sapkota, N. (2020). Technical trading rules in the cryptocurrency market. *Finance Research Letters* 32. https://ideas.repec.org/a/eee/finlet/v32y2020ics1544612319308852.html
8. Grobys, K. & Sapkota, N. (2019). Cryptocurrencies and momentum. *Economics Letters* 180, 6–10. https://ideas.repec.org/a/eee/ecolet/v180y2019icp6-10.html
9. Dobrynskaya, V. Cryptocurrency Momentum and Reversal. *Journal of Alternative Investments* (forthcoming at time of summary). Summary: https://www.hse.ru/en/news/research/559533243.html
10. Fieberg, C., Liedtke, G., Poddig, T., Walker, T. & Zaremba, A. (2025). A Trend Factor for the Cross Section of Cryptocurrency Returns. *JFQA* 60(7), 3116–3153. https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/trend-factor-for-the-cross-section-of-cryptocurrency-returns/4C1509ACBA33D5DCAF0AC24379148178
11. Shen, D., Urquhart, A. & Wang, P. (2022). Bitcoin intraday time series momentum. *Financial Review* 57(2), 319–344. https://research.birmingham.ac.uk/en/publications/bitcoin-intraday-time-series-momentum/
12. Wen, Z., Bouri, E., Xu, Y. & Zhao, Y. (2022). Intraday return predictability in the cryptocurrency markets: Momentum, reversal, or both. *North American Journal of Economics and Finance* 62. https://ideas.repec.org/a/eee/ecofin/v62y2022ics1062940822000833.html
13. Borgards, O. (2021). Dynamic time series momentum of cryptocurrencies. *North American Journal of Economics and Finance* 57. https://ideas.repec.org/a/eee/ecofin/v57y2021ics1062940821000590.html
14. Kitron, N. A. & Wengrowicz, J. M. (2026). Short-horizon mean reversion in cryptocurrency markets: a matched cross-market measurement. arXiv:2608.21888. https://arxiv.org/abs/2608.21888
15. Bianchi, D., Babiak, M. & Dickerson, A. (2022). Trading volume and liquidity provision in cryptocurrency markets. *Journal of Banking & Finance* 142, 106547. https://ideas.repec.org/a/eee/jbfina/v142y2022ics0378426622001418.html
16. Kim, C. & Hansen, P. R. (2026). The Quarter-Hour Effect: Periodic Algorithmic Trading and Return Predictability in Cryptocurrency Futures. arXiv:2607.09426. https://arxiv.org/abs/2607.09426
17. Silantyev, E. (2019). Order flow analysis of cryptocurrency markets. *Digital Finance* 1(1), 191–218. https://ideas.repec.org/a/spr/digfin/v1y2019i1d10.1007_s42521-019-00007-w.html
18. Bieganowski, B. & Ślepaczuk, R. (2026). Explainable Patterns in Cryptocurrency Microstructure. arXiv:2602.00776. https://arxiv.org/abs/2602.00776
19. Easley, D., O'Hara, M., Yang, S. & Zhang, Z. (2024). Microstructure and Market Dynamics in Crypto Markets. SSRN 4814346. https://ssrn.com/abstract=4814346
20. He, S., Manela, A., Ross, O. & von Wachter, V. (2022, v7 2026). Fundamentals of Perpetual Futures. arXiv:2212.06888. https://arxiv.org/abs/2212.06888
21. Schmeling, M., Schrimpf, A. & Todorov, K. (2023). Crypto carry. BIS Working Paper 1087. https://www.bis.org/publ/work1087.htm
22. Christin, N., Routledge, B., Soska, K. & Zetlin-Jones, A. (2022). The Crypto Carry Trade. https://www.andrew.cmu.edu/user/azj/files/CarryTrade.v1.0.pdf (published as "Crypto Carry", *Management Science*, doi:10.1287/mnsc.2024.05069)
23. Hazelkorn, T., Moskowitz, T. & Vasudevan, K. (2020). Beyond Basis Basics: Leverage Demand and Deviations from the Law of One Price. NBER WP 26773. https://www.nber.org/papers/w26773
24. Caporale, G. M. & Plastun, A. (2019). The day of the week effect in the cryptocurrency market. *Finance Research Letters*. https://bura.brunel.ac.uk/handle/2438/17208
25. Baur, D., Cahill, D., Godfrey, K. & Liu, Z. Bitcoin time-of-day, day-of-week and month-of-year effects in returns and trading volume. SSRN 3088472. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3088472
26. Katsiampa, P. (2017). Volatility estimation for Bitcoin: A comparison of GARCH models. *Economics Letters* 158, 3–6. https://shura.shu.ac.uk/16526/
27. Marshall, B., Young, M. & Rose, L. (2006). Candlestick technical trading strategies: Can they create value for investors? *Journal of Banking & Finance* 30(8), 2303–2323. https://ideas.repec.org/a/eee/jbfina/v30y2006i8p2303-2323.html
28. Lo, A., Mamaysky, H. & Wang, J. (2000). Foundations of Technical Analysis. *Journal of Finance* 55, 1705–1765. https://www.nber.org/papers/w7613
29. Osler, C. (2000). Support for Resistance: Technical Analysis and Intraday Exchange Rates. *FRBNY Economic Policy Review*, July 2000. https://www.newyorkfed.org/medialibrary/media/research/epr/00v06n2/0007osle.html
30. Harvey, C., Liu, Y. & Zhu, H. (2016). …and the Cross-Section of Expected Returns. *Review of Financial Studies*. https://papers.ssrn.com/abstract=2513152
31. Bailey, D. & López de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management* 40(5), 94–107. https://papers.ssrn.com/abstract=2460551
32. Binance Spot API: Market data endpoints (`/api/v3/klines`, `/api/v3/depth`). https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints
33. Binance USDⓈ-M Futures: Get Funding Rate History (`/fapi/v1/fundingRate`). https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History
34. Binance USDⓈ-M Futures: Open Interest Statistics (`/futures/data/openInterestHist`). https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest-Statistics
35. Binance USDⓈ-M Futures: Taker Buy/Sell Volume (`/futures/data/takerlongshortRatio`). https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Taker-BuySell-Volume
36. Binance USDⓈ-M Futures: Long/Short Ratio (`/futures/data/globalLongShortAccountRatio`, 30 days only). https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Long-Short-Ratio
37. Binance USDⓈ-M Futures: Premium Index Kline Data (`/fapi/v1/premiumIndexKlines`). https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Premium-Index-Kline-Data
38. Binance USDⓈ-M Futures: Liquidation Order Streams. https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/Liquidation-Order-Streams
39. Binance public data (bulk files): https://github.com/binance/binance-public-data and https://data.binance.vision/ (paths `data/spot/monthly/klines/`, `data/futures/um/daily/metrics/`, `data/futures/um/daily/bookDepth/`, `data/futures/um/monthly/premiumIndexKlines/`; start dates checked 2026-10-05)
40. StockCharts ChartSchool *(practitioner)*: (a) RSI https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/relative-strength-index-rsi ; (b) ATR https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-true-range-atr ; (c) MACD https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/macd-moving-average-convergence-divergence-oscillator ; (d) Bollinger Bands https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/bollinger-bands ; (e) VWAP https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/volume-weighted-average-price-vwap ; (f) Stochastic https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/stochastic-oscillator-fast-slow-and-full
41. *(news)* The Block: Binance zero-fee BTC pairs from July 2022 https://www.theblock.co/post/156121/binance-set-to-drop-bitcoin-trading-fees-to-zero-on-select-pairs-globally ; BTC spot volume drop after fees returned https://www.theblock.co/post/229071/bitcoin-spot-trading-volume-binance
