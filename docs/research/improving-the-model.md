# Improving the model and the trading strategy: research and tests, 2026-10-06

*Research note, 2026-10-06. Summarises a literature review of what improves crypto prediction and trading systems, and the experiments run on `cryptoai/` the same day to test the most promising ideas. Every experiment fixed its variants and pass rule before results were seen, chose on 2022-2024 and checked once on 2025-01-01 to 2026-10-06, on a basket of BTC/ETH/BNB/SOL, after fees. Companions: [binance-trading-bots.md](binance-trading-bots.md), [other-ai-trading-models.md](other-ai-trading-models.md).*

## Bottom line

- **The AI's direction signal is real and is not a copy of the trend.** Its 1-day predictions have an information coefficient (IC) of +0.059 (t 3.8). After removing everything trend features explain, +0.049 (t 3.3), with similar values in 2022-2024 and 2025-2026. A placebo (predictions shifted by 1-2 years) gave t between -0.45 and +0.13.
- **The edge is lost to trading costs.** Sizing positions smoothly by the AI's confidence has a 2022-2026 Sharpe of 1.54-1.58 *before fees*, against 1.03 for the 50-day rule, but it turns positions over 187-340 times a year. At 0.1% per side that costs 19-34% a year, and net results are poor.
- **Trading the AI slowly mostly turns it into a risk reducer.** Holding a coin only while both the 50-day rule and the AI (smoothed, with hysteresis) agree halved the worst fall in both periods (-23% vs -39%, -12.5% vs -28%). Its return was lower in 2022-2024 (+35% vs +155%) and higher on a Sharpe basis in 2025-2026 (1.24 vs 0.96).
- **The 50-day rule stays the core.** Volatility sizing, the drop brake, more coins, split entries, calibrated sizing and new volatility features for the drop warning all failed to beat it, or the current models, out of sample.
- **The research agrees:** direction AUC of 0.53-0.55 is normal; trend following has the strongest evidence in crypto; gradient-boosted trees beat deep nets on this kind of data; volatility and risk are more predictable than direction; most data sources sold as predictive (ETF flows, on-chain, sentiment) have weak or no out-of-sample support.

## Literature summary

| Idea | Evidence | Verdict |
|---|---|---|
| Trend following across lookbacks, volatility-sized | Strong: Man AHL "In Crypto We Trend" (2024); Zarattini, Pagani & Barbon (2025); Detzel et al. (2021) | Our 50-day rule already captures it |
| Volatility targeting | Strong for smaller drawdowns, weak for Sharpe gains (Harvey et al. 2018; Cederburg et al. 2020) | Confirmed here |
| HAR volatility forecasting; implied vol (Deribit DVOL) | Strong for forecasting volatility; weak for direction | No gain for the drop warning here |
| Order-book imbalance | Works for seconds to minutes (Cont, Kukanov & Stoikov 2014) | Irrelevant at 4h+ |
| On-chain, ETF flows, Google Trends, stablecoin mints | Mostly vendor claims, levels regressions or in-sample results | Not pursued |
| Deep learning (LSTM/Transformer) | Trees beat deep nets on tabular data (Grinsztajn et al. 2022); Kaggle G-Research crypto winners used LightGBM | Not pursued |
| Target engineering, purged CV, deflated Sharpe | Strong methodological consensus (Lopez de Prado) | Already used: triple-barrier drop label, embargoes, TRIALS_TESTED |
| Evaluate by IC per period and neutralise against known factors (Numerai) | Practitioner standard | Used here; per-month IC turned out biased for slow signals (see below) |
| Wider coin universe | Moderate (Man AHL: Sharpe peaks at 10-15 coins) | Failed here: volume-ranked alts were hype-driven |

## Results in this repo

| Experiment | Question | Result |
|---|---|---|
| `experiments/ai_vs_trend.py` | Does the AI know anything trend doesn't? | Yes: IC +0.049 after removing trend, t 3.3; trend score itself has IC +0.003 at 1 day |
| `experiments/trend_core.py` | 50-day rule + volatility sizing and/or drop brake | Not adopted: chosen brake variant Sharpe 0.85 vs 0.92 |
| `experiments/trend_universe.py` | 50-day rule on top 8/12/16 coins by volume | Not adopted: 2025-2026 Sharpe 0.28 vs 0.93 |
| `experiments/split_entries.py` | Enter over 3/5/10 days | Not adopted: 0.79 vs 0.93 |
| `experiments/calibrated_sizing.py` | Size by calibrated P(up); 50-day rule x AI | Not adopted: 0.27 vs 0.96; costs of 187-340x turnover a year |
| `experiments/low_turnover_ai.py` | Same signal, traded slowly | Chosen `sma50_ai_hyst` passed the check (1.24 vs 0.96, worst fall -12.5% vs -28%) but lost clearly on 2022-2024 (0.62 vs 0.99). Treated as a lower-risk variant, not a better strategy |
| `experiments/drop_features.py` | HAR realised-vol and DVOL features for the drop warning | Not adopted: HAR +0.000 / -0.002 AUC; DVOL -0.014 / -0.053 (implied vol in 2025-2026 far below the training range) |

**A measurement trap found on the way.** An IC computed inside each month is strongly biased for slow-moving signals: on a simulated random walk the 50-day trend score averaged -0.25. The diagnostic therefore uses one IC over the whole period with a block bootstrap over months, checked with placebos.

**A lesson about pass rules.** The low-turnover rule required a win only on the check period. A better rule requires not losing clearly on the choice period as well. Future experiments should state both.

## What to try next

1. **Trade the AI signal with lower costs rather than less often.** Maker-only limit orders (0.075% with BNB, no spread) would cut the fee drag of the AI's turnover by about a quarter. That is still not enough at 187x a year, so it is only useful combined with slower trading.
2. **A slower AI target.** The 1-day signal decays fast; a model built to predict 3-7 day moves with the same features would need far fewer trades. The current 3-day model is weak (AUC 0.527), so this means new features or a triple-barrier direction label, tested as one bundle.
3. **Offer `sma50_ai_hyst` as an optional low-risk mode** after the 4-week paper trial ends, and paper-trade it before any adoption.
