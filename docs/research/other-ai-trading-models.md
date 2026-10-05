# Other AI trading models: what has been built, what beats simple rules, and what to adopt

*Research note, 2026-10-06. Surveys open-source frameworks, deep-learning studies, time-series foundation models, LLM trading agents and industry practice, to decide what `cryptoai/` should learn from them. `cryptoai/` trades BTC/ETH/BNB/SOL vs USDT on Binance, long-or-flat, with pooled HistGradientBoosting models on 40–57 tabular features (4h candles: next 4h and next 1 day; 1d candles: next 3 days), walk-forward retraining on an expanding window, and 0.1% fees. Citations are numbered; see [Sources](#sources). Companion to [neurotrader-methods.md](neurotrader-methods.md) (permutation tests, meta-labelling), [sharpe-ratio.md](sharpe-ratio.md) (PSR, deflated Sharpe), [reading-crypto-charts-for-day-trading.md](reading-crypto-charts-for-day-trading.md) (momentum and technical-rule evidence in crypto) and [technical-analysis-concepts.md](technical-analysis-concepts.md). Those notes are not repeated here [54].*

**The standard used throughout.** Is there credible evidence that the system beats **buy-and-hold** and a **50-day moving-average trend rule** *after costs*, on data it never saw? Our own baseline test sets the bar. On the one-time holdout (2025-01 to 2026-10, daily, 0.15% per side), the SMA50 rule made +54.0% (Sharpe 0.94), buy-and-hold −9.4%, and a LightGBM on daily price/volume features −29.3% (AUC 0.491) [52]. Our main AI beat the trend rule in only about 50–53% of the 2022–2026 half-year runs [52].

**How this was researched.** I read the GitHub READMEs and documentation of each framework, and the arXiv abstracts or HTML full texts of the papers. Where I relied on an abstract, a search snippet or a news report, the citation says so. One small calculation from repo data is labelled *(this note's check)*.

## Bottom line

- **Almost nothing published clears our bar.** I found no open-source framework, foundation model or LLM agent with credible evidence of beating buy-and-hold *and* a simple trend rule on crypto, after costs, on unseen data. The frameworks make no performance claims [1][4][10][11]. The papers with strong claims use test windows of 30 minutes to 3 months [5][6][34], ignore costs [34][35], or never compare against a trend rule.
- **On engineered tabular features, gradient-boosted trees are as good as anything.** In Microsoft Qlib's benchmark (Chinese stocks, costs included), LightGBM beats LSTM, GRU, Transformer, TFT and TCN on Alpha158 features. The only model that beats it is DoubleEnsemble, which is itself an ensemble of boosted trees [8][9]. Our choice of model family is not the problem.
- **Simple trend following with volatility sizing has the best crypto evidence.** A 2025 working paper reports a net-of-fee Sharpe above 1.5 for an ensemble of Donchian trend rules with volatility-based sizing on the top 20 coins [49]. Volatility targeting reduces tail events in every asset class and raises Sharpe for risk assets [48]. That is the same family as the SMA50 rule that beat our models [52].
- **Off-the-shelf time-series foundation models do not forecast returns usefully.** On daily returns in 94 countries, zero-shot and fine-tuned generic models "perform poorly" [30]. A 2026 test on US stocks found accuracy gains over a random walk of order 0.1%, significant in only 2 of 10 cases [31]. Kronos, a model built for candlesticks, reports large gains in rank correlation, but there is no independent test and no backtest with costs [28][29].
- **LLM traders are entertainment so far.** In nof1's Alpha Arena season 1 (real money, crypto perpetuals, 17 days), four of six models lost 42–59%. The six together lost about 30%, while BTC was about flat over the same days [39] *(this note's check)*. In academic tests, LLM advantages "deteriorate significantly" over 20 years and 100+ symbols [38].
- **Deep RL results in crypto are not evidence.** The FinRL contest's Bitcoin task tested on about two days of 2021 order-book data [6]. The best published ensemble made 0.66% [5].
- **The most valuable lesson is to make the trend rule the base strategy and use ML only to adjust it.** Concretely, in order of value per effort:
  1. trend rule plus volatility targeting;
  2. an ensemble of trend look-backs;
  3. ML probability used as a position-size overlay on the trend rule, not as the entry signal;
  4. a FreqAI-style sliding training window;
  5. one small zero-shot test of Chronos-Bolt or Kronos on CPU.
- **Avoid:** LLM agents as traders, RL without trend-rule baselines, deep sequence models on our small dataset, order-book deep learning, and any bought "AI bot" (§7).

---

## Comparison table

"OOS & costs" means the best evidence is out of sample *and* net of realistic costs. "Beats simple rules" means against buy-and-hold **and** a trend rule, on crypto unless stated.

| System / approach | What it is | Best evidence | OOS & after costs? | Beats simple rules? | Relevance to us |
|---|---|---|---|---|---|
| **Freqtrade + FreqAI** | Open-source crypto bot with adaptive ML retraining on sliding windows [1][2] | JOSS software paper [3]; the docs make no performance claim [1] | Its backtester emulates retraining, but feature look-ahead is left to the user [2] | No evidence either way | **Medium:** the sliding-window retraining idea (§5 item 4) |
| **FinRL** (+ contests) | Deep-RL library (A2C, DDPG, PPO, SAC, TD3) for stocks and crypto [4] | Contest Bitcoin task: about a 2-day test on second-level 2021 order-book data, 0.1% cost [6] | Costs yes; test far too short | Not shown. The best ensemble made +0.66% vs BTC −7.35% over the test window [5][6] | **Low:** a cautionary example |
| **Qlib** (Microsoft) | Quant research platform and model zoo for stock ranking [8] | 20-seed benchmark on CSI300, test 2017-01 to 2020-08, buy 0.05% / sell 0.15% costs [8][9] | Yes (stocks, cross-sectional) | Different task: a long top-50 portfolio vs an index, not timing | **High** as evidence that GBDT ≥ deep nets on engineered features |
| **Jesse** | Python crypto backtesting framework with an ML pipeline and Monte Carlo tests [10] | None (no claims; "educational purposes only") [10] | n/a | n/a | Low |
| **Hummingbot** | Market-making and arbitrage bot framework, 50+ exchange connectors [11] | Usage volume only ($34B a year) [11] | n/a (it is not a predictor) | n/a | None (a different business) |
| **LSTM/GRU crypto studies** | Recurrent nets on daily coin returns | Ensembles beat buy-and-hold in a bear test window after 0.5% costs [15]; long-short Sharpe 3.23 across 100 coins [16] | Partly | Not against a trend rule; strongest on small coins | Low–medium |
| **TFT, TCN, Transformer** | Attention or convolution forecasters [12] | Qlib: worse than LightGBM on Alpha158 [8] | Yes (stocks) | No | Low |
| **DeepLOB and LOB deep learning** | CNN-LSTM on limit-order-book snapshots [13] | All 15 models drop significantly on new data [14] | Classification metrics only, no costs | n/a (horizon in seconds) | None (we use 4h/1d candles) |
| **Deep momentum networks** | LSTM that sizes trend positions to maximise Sharpe [20] | More than 2× the Sharpe of classic methods without costs; holds only up to 2–3 bp of costs [20]. A 2026 benchmark: specialised sequence models beat linear ones on futures [19] | Yes (futures) | Beats classic time-series momentum on futures; not tested on crypto | **Medium:** "ML sizes a trend position" (§5 item 3) |
| **Chronos/Bolt, TimesFM, Moirai, TimeGPT** | General zero-shot forecasting models [23]–[27] | Finance tests: zero-shot "perform poorly" [30]; gains over a random walk of order 10⁻³ [31] | No backtest with costs | No | Low (one cheap test, §5 item 5) |
| **Kronos** | Foundation model for OHLCV candles; 12B candles from 45 exchanges [28][29] | Authors: +93% RankIC over the best generic model [29] | No costs; no independent test | Unknown | Low–medium (one cheap test) |
| **FinGPT, BloombergGPT** | Financial LLMs for NLP tasks [32][33] | Sentiment F1 and other NLP benchmarks | No trading evaluation [32][33] | n/a | None for price timing |
| **TradingAgents, FinMem** | Multi-agent and memory-based LLM traders [34][35] | 3-month (TradingAgents) and 6-month (FinMem) tests on a handful of US stocks; no costs mentioned [34][35] | No | Not credibly | None |
| **LLM news sentiment** | An LLM scores headlines; trade the scores [36] | Positive return predictability in 2021–22; returns decline as LLM adoption rises [36] | Stocks only; look-ahead and "distraction" problems [37] | Not for crypto | Low |
| **Alpha Arena (nof1)** | Six LLMs trading crypto perpetuals with $10k of real money each [39] | One 17-day season | Live, real fees, but n = 1 | Combined −30% vs BTC ≈ flat [39] *(this note's check)* | None, except as a warning |
| **Numerai** | Crowd-sourced models on obfuscated data, a staked meta-model, a market-neutral equity fund [41][42] | Self-reported 2024: +25.45% net, Sharpe 2.75 [42] | Live fund (self-reported) | Equities, market-neutral: not comparable | Medium as a design idea (ensembling many weak models) |
| **Trend + volatility sizing** (benchmark family) | Ensemble of Donchian trend rules with volatility-based position sizing [49] | Sharpe > 1.5 net of fees on the top 20 coins, survivorship-free [49] | Yes (working paper) | *Is* the simple rule, improved | **High** (§5 items 1–2) |

---

## 1. Open-source frameworks

**Freqtrade + FreqAI.**
- **What it is.** FreqAI adds "self-adaptive retraining" to the Freqtrade crypto bot [1]. Supported models include LightGBM, XGBoost, CatBoost, PyTorch models and reinforcement-learning agents. It offers outlier removal (dissimilarity index, SVM, DBSCAN) [1]. It is described in a JOSS software paper [3].
- **How it avoids look-ahead.** Backtests emulate live retraining with a **sliding window**. `train_period_days` sets the training length and `backtest_period_days` the length of each test block. In the docs' example, each model trains on the previous 30 days and trades the next 7, then the window slides forward [2]. Targets are recomputed once per window "without look ahead bias". However, `feature_engineering_*()` runs once on the whole range, so **feature look-ahead is the user's responsibility** [2]. Live settings include `live_retrain_hours`, and `expiration_hours`, which blocks entries when the model is stale [2].
- **Evidence.** The docs make no performance claim. They call the example strategy one "for showcasing/testing" [1].
- **For us.** We already retrain at every candle close, but on an **expanding** window (all earlier data) [53]. The one transferable idea is to test a *sliding* window, which forgets old regimes (§5 item 4).

**FinRL.**
- **What it is.** A deep-RL library with A2C, DDPG, PPO, SAC and TD3 agents, environments for stocks and crypto, and data feeds including Binance [4]. Its README makes no trading claims and disclaims financial advice [4].
- **Contest results.** The FinRL contests ran from 2023 to 2025 [6]. In the 2024 crypto task, the training data was second-level Bitcoin order-book data from 2021-04-07 to 04-17, and **the test ran from 2021-04-17 00:38 to 2021-04-19 09:54**, with "a cost of 0.1% for each action" [6].
  - The 2024 winner made +0.23% (Sharpe 0.0037). The 2025 winner made −0.05%. The BTC price over the window fell 7.35% [6].
  - The organisers' ensemble paper reports +0.66% and Sharpe 0.28, against BTC +0.74% in its own (shorter) evaluation window [5].
  - Beating a falling BTC over two days says nothing about a long-or-flat timing strategy. "Stay flat" would have beaten BTC too.
- **Backtest overfitting.** A related FinRL paper tests DRL crypto agents for backtest overfitting with a hypothesis test and rejects overfit agents. The surviving agents beat the market benchmark in May–June 2022 [7]. That was a crash, when any agent that holds cash beats the market.

**Qlib (Microsoft).**
- **Setup.** The model zoo is benchmarked on CSI300 stocks. Training runs 2008–2014, validation 2015–2016 and **test 2017-01 to 2020-08**. The strategy holds the top 50 stocks and drops 5 per day, with costs of **0.05% to buy and 0.15% to sell** [9]. Results are means over 20 random seeds [8].
- **Results on Alpha158** (engineered features, like ours) [8]:

| Model | IC | Annualised return | Information ratio |
|---|---|---|---|
| **DoubleEnsemble** | 0.052 | 11.6% | **1.34** |
| **LightGBM** | 0.045 | 9.0% | **1.02** |
| TFT | 0.036 | 8.5% | 0.81 |
| LSTM | 0.032 | 3.8% | 0.56 |
| GRU | 0.032 | 3.4% | 0.52 |
| TCN | 0.028 | 2.6% | 0.41 |
| Transformer | 0.026 | 2.7% | 0.40 |

- **Results on Alpha360** (raw price and volume) [8]. GRU (information ratio 0.97) and LSTM (0.90) edge out LightGBM (0.76). The Transformer loses money (−0.34).
- **Lesson.** On hand-made features, boosted trees beat deep sequence models. Deep nets only catch up when they must build their own features from raw prices, and even then by a small margin. Ensembling trees adds the most.
- **Caveat.** This is cross-sectional stock ranking, which has many assets per day to learn from. Single-asset timing on four coins has far fewer independent samples.

**Jesse** is a Python crypto backtester. It offers multi-timeframe data "without look-ahead bias", Optuna optimisation, an ML pipeline built on scikit-learn, and Monte Carlo and "rule significance" tests. It is "for educational purposes only" and makes no performance claims [10].

**Hummingbot** runs market-making, arbitrage, grid and TWAP strategies over 50+ exchange connectors [11]. It earns the spread rather than predicting direction, so it is a different business. It needs low fees, inventory risk management and uptime, not forecasts.

## 2. Deep learning for price prediction

**Crypto studies with costs.**
- **Sebastião & Godinho (2021).** Trained on 2015–2019 daily data, with a bear-market test window. An ensemble of five models applied to ETH and LTC earned an annualised 9.62% and 5.73% after 0.5% proportional costs. Individual models' hit rates were 46–60% [15]. No trend-rule benchmark.
- **Jaquart, Köpke & Weinhardt (2022).** Predict daily relative moves of the 100 largest coins. Accuracy is 52.9–54.1%, rising to about 58–60% on the 10% most confident predictions [16]. A long-short LSTM/GRU portfolio reports an out-of-sample Sharpe of 3.23 and 3.12 after costs, against 1.33 for buy-and-hold [16] (from the abstract via search; full text not read). This is cross-sectional across 100 coins, which is where crypto ML has the most published success. It is not timing four large coins.
- **Bysik & Ślepaczuk (2026).** Hourly BTC-USDT, 2018–2026, 27-fold walk-forward. XGBoost, LSTM and iTransformer "naive sign-based strategies fail once transaction costs of ten basis points are imposed". A cost-aware filter that skips low-confidence trades restored profit in some configurations, and XGBoost was descriptively best [18].
- **A practitioner repo that mirrors our setup** (unreviewed). LightGBM on 4h BTC/ETH/SOL/BNB from 2020 to 2026, with 0.30% round-trip costs and pre-registered go/no-go rules. It "finds real, statistically detectable signal", but "the edge is smaller than the 0.30% round-trip cost". All four coins failed [22]. This is our experience exactly.
- **Studies without costs.** Many published crypto deep-learning papers report only error metrics. For example, a 2024 comparison of LSTM variants, CNN and Transformer on BTC/ETH/DOGE/LTC reports RMSE only, with no trading test [17].

**Non-crypto benchmarks worth knowing.**
- **Temporal Fusion Transformer** [12]. An interpretable multi-horizon forecaster. In Qlib it ranks below LightGBM [8].
- **Deep momentum networks** [20]. An LSTM directly outputs a trend-following position, trained to maximise Sharpe. It more than doubles the Sharpe of classic time-series momentum without costs, and still outperforms at costs of 2–3 bp. That is a key limit: crypto spot costs are 10+ bp per side.
- **2026 futures benchmark** [19]. Daily futures from 2010 to 2025. Specialised temporal models (a variable-selection network with LSTM or xLSTM) "consistently outperform linear benchmarks". xLSTM has the largest breakeven-cost buffer.

**DeepLOB and order-book models.** DeepLOB (CNN + LSTM on order-book snapshots) reported F1 of about 78% on the FI-2010 benchmark and stable accuracy on LSE stocks [13]. LOBCAST re-ran 15 such models, and **all "exhibit a significant performance drop when exposed to new data"**. Most overfit FI-2010 [14]. The horizons are in ticks, and we have no order-book history. Our earlier order-flow tests also failed (companion note).

**How often studies report out-of-sample, after-cost results.** I could not find a review that counts this for crypto deep learning. "None found" is not proof that none exists.
- A systematic survey of deep learning in stock markets narrowed more than 10,000 search hits to 35 papers that showed "some indication of consideration of backtesting". It does not count how many included costs or a buy-and-hold benchmark [21].
- Across all of ML-based science, Kapoor & Narayanan found data leakage in 294 papers in 17 fields. When the leakage was fixed, claimed ML superiority often vanished [47].

## 3. Time-series foundation models used zero-shot

**The models.**
- **Chronos** (Amazon) tokenises scaled values and trains T5-style language models on them [24].
- **Chronos-Bolt** comes in 9M–205M parameter sizes. It is "up to 250 times faster" than the original Chronos, so it is practical on CPU [23]. **Chronos-2** (120M) adds covariates and multivariate inputs [23].
- **TimesFM** (Google) is decoder-only [25]. **Moirai** (Salesforce) is a universal masked-encoder model [26]. **TimeGPT** (Nixtla) is a commercial API [27].
- All are trained mostly on non-financial series and report general benchmarks (GIFT-Eval, fev-bench). None claims direction or return skill on prices [23]–[27].

**Independent finance tests.**
- **Rahimikia, Ni & Wang (2025).** 34 years of daily excess returns in 94 countries. "Off-the-shelf pre-trained TSFMs perform poorly in zero-shot and fine-tuning settings". Only models **pre-trained from scratch on financial data** gave "substantial forecasting and economic improvements" [30].
- **Noguer i Alonso & Franklin (2026).** Five US stocks, 2024–2026, 20-day horizon. Gains over a random walk are "small and sparse". Improvements are "of order 10⁻³", significant only for Chronos on AMZN and Moirai-2.0 on GOOG. The authors estimate the information in past returns at about 0.005 nats per forecast [31]. There is no trading backtest.

**Kronos (2025).**
- **What it is.** A family of foundation models for candlesticks. A tokenizer turns OHLCV into hierarchical tokens, and an autoregressive Transformer is pre-trained on **12 billion K-line records from 45 exchanges** [29]. It was accepted at AAAI 2026 [28].
- **Open models.** Kronos-mini (4.1M parameters, 2,048-token context), Kronos-small (24.7M, 512) and Kronos-base (102.3M, 512) are open under the MIT licence. Kronos-large (499M) is not released [28]. The small sizes should run on CPU, but the README does not state CPU speed.
- **Claims.** Price-forecast RankIC is 93% higher than "the leading TSFM" and 87% higher than the best non-pretrained baseline. Volatility MAE is 9% lower [29]. These are the authors' own benchmarks.
- **What is missing.** The paper reports no backtest with costs [29]. The repo's backtest demo (Chinese A-shares, top-K) is "not a production-ready quantitative trading system" and says a real backtest "should meticulously model transaction costs" [28]. I found no independent test of Kronos on crypto direction. The live BTC/USDT demo on its site is a forecast display, not evidence.
- **A specific risk for us.** Kronos's pre-training corpus probably includes the crypto history we would test on, and the paper does not state the data's end date. A fair zero-shot test must use only candles **after** the model's release (2025-08 onward).

## 4. LLM trading agents

**Financial LLMs.**
- **BloombergGPT** is a 50B-parameter model trained on 363B financial and 345B general tokens. It is evaluated on NLP tasks (sentiment, named entities, question answering), **not trading** [33].
- **FinGPT** fine-tunes open LLMs cheaply. It claims sentiment F1 of 0.88 on Financial PhraseBank vs BloombergGPT's 0.51. Its "Forecaster" predicts Dow 30 moves from news, but the repo gives **no trading evaluation** [32].

**LLM agents.**
- **TradingAgents** [34]. A "trading firm" of LLM agents: analysts, bull and bear researchers who debate, a trader, risk managers. It reports cumulative returns of +26.6% on AAPL, +24.4% on GOOGL and +23.2% on AMZN, with **Sharpe ratios of 5.6–8.2**, over **2024-01-01 to 2024-03-29** [34]. That is three stocks over three months, with no costs mentioned. A Sharpe of 8 over one quarter is a red flag, not a result. The repo itself gives no performance numbers [34].
- **FinMem** [35]. A memory-based LLM agent tested on TSLA, NFLX, AMZN, MSFT and COIN from 2022-10 to 2023-04. It reports, for example, TSLA +61.8% vs buy-and-hold −18.6%. The test is six months long and transaction costs are not discussed [35].
- **FINSABER (KDD 2026)** [38]. Re-tested published LLM strategies over 20 years and 100+ symbols, with corrections for survivorship, look-ahead and data snooping. Reported LLM advantages "deteriorate significantly" and "often vanish". The LLM strategies were too conservative in bull markets and lost disproportionately in bear markets.

**News sentiment.** Lopez-Lira & Tang found that ChatGPT headline scores predict next-day stock returns (2021-10 to 2022-12). In the latest version, strategy returns **decline as LLM adoption rises** [36]. Glasserman & Lin show that backtests inside an LLM's training window are biased. The bigger problem turned out to be "distraction" by the model's knowledge of the company, which anonymising the headlines fixes [37]. Any LLM backtest on 2022–2024 crypto news would sit inside current models' training data.

**Alpha Arena (nof1), live.**
- **Setup.** Season 1 ran from **2025-10-18 to 2025-11-03**. Six LLMs each got **$10,000 of real money** to trade crypto perpetuals (BTC, ETH, SOL, BNB, DOGE, XRP) on Hyperliquid, with no human intervention [39].
- **Final balances** (as reported by news outlets [39]):

| Model | Return | Final balance |
|---|---|---|
| Qwen 3 Max | +22.3% | $12,231 |
| DeepSeek V3.1 | +4.9% | $10,489 |
| Claude Sonnet 4.5 | −42.0% | $5,799 |
| Gemini 2.5 Pro | −45.6% | $5,445 |
| Grok 4 | −57.9% | $4,208 |
| GPT-5 | −58.7% | $4,126 |

- **In aggregate:** $42,298 left of $60,000, or **−29.5%**.
- **Buy-and-hold over the same days** *(this note's check, Binance daily open on 10-18 to close on 11-03)*: BTC +0.1%, ETH −5.9%, BNB −7.2%, SOL −8.7%. Holding any of these lost far less than the average LLM.
- **Organisers' caveats.** The organisers acknowledged "prompt bias, limited sample sizes / lack of statistical rigor, and shortness of evaluation period" (quoted via a secondary report [39]; nof1's own blog page did not load for me).
- **Season 1.5** moved to US stocks. Reports say only Grok 4.2 was profitable at the end, at about +12% [40].
- **Verdict.** One 17-day run with leverage cannot rank models. It does show what LLMs do with live leverage: their failures came from over-leverage and weak risk control [39].

## 5. Industry evidence

**Numerai** [41][42].
- **How it works.** Thousands of participants train models on **obfuscated** features. Only *staked* predictions enter the **meta-model**, which drives a market-neutral global equity fund. Payouts depend on correlation (CORR) and meta-model contribution (MMC), and bad predictions have their stake burned [41]. Numerai also runs Signals and a Crypto tournament [41].
- **Performance (self-reported).** The fund reports 2024 net **+25.45%, Sharpe 2.75**, growth from $60M to $450M under management, and a JPMorgan commitment of up to $500M (2025-08) [42]. The docs themselves give no performance figures [41].
- **Lessons for us.** Combine many weakly correlated models. Reward each model's *marginal* contribution to the ensemble, not its standalone score. Keep the target market-neutral, which removes the beta that a long-only crypto timer cannot avoid.

**Crypto quant funds.** I found no primary source that publishes performance for an ML-driven crypto timing fund. That is not proof none exist, but marketing claims about such funds cannot be verified. The closest academic evidence for systematic crypto returns is trend following [49] and cross-sectional momentum (companion note).

**Failures and hype.**
- The US CFTC warned in 2024 that "AI technology can't predict the future". It cited AI-bot schemes promising guaranteed returns, including Mirror Trading International, which took over $1.7B in bitcoin from at least 23,000 people [43].
- López de Prado lists ten reasons most ML funds fail. They include research silos, backtest overfitting, wrong labelling and non-stationary features [44].

## 6. Methodological pitfalls and how often they appear

| Pitfall | What it does | Seen in this survey |
|---|---|---|
| **Tiny test sets** | Any result is luck | FinRL crypto: about 2 days [6]; TradingAgents: 3 months [34]; Alpha Arena: 17 days [39] |
| **No costs** | Turns a 51% hit rate into "profit" | TradingAgents, FinMem [34][35]; hourly ML fails at 10 bp [18]; deep momentum holds only to 2–3 bp [20] |
| **Weak benchmark** | Beating a falling market by being flat | FinRL vs a −7.35% BTC [6]; DRL agents in the 2022 crash [7]; never compared to a trend rule |
| **Look-ahead / leakage** | Future data in features or in model weights | FreqAI leaves features to the user [2]; LLM training-window overlap [37]; leakage in 294 papers in 17 fields [47] |
| **Backtest overfitting** | The best of N configurations looks good by chance | With 5 years of data, no more than 45 independent configurations should be tried [45]; PBO [46] and deflated Sharpe (neurotrader note) |
| **Survivorship / selection** | Testing only coins or stocks that survived or were hand-picked | FINSABER's motivation [38]; the trend paper uses survivorship-free data [49] |
| **Error metrics only** | Low RMSE does not mean tradable | LSTM/Transformer crypto comparison [17]; foundation-model benchmarks [23]–[27] |
| **Dataset overfitting** | Models tuned to one benchmark | LOB models drop on new data [14] |

Bailey et al. prove that high backtest Sharpe ratios are easy to reach after trying a modest number of configurations, and they give a minimum backtest length that grows with the number of trials [45]. With `TRIALS_TESTED = 14` [53] and about 4–5 years of test data, we are already near the regime where a "winner" is expected by chance (sharpe-ratio note).

---

## 7. What we could learn or adopt

**Common test protocol for every item.**
- Fix all parameters in a script before running.
- Choose on **2022–2024** and check once on **2025–2026**.
- Compare against **the SMA50 trend rule and buy-and-hold**, using the 0.1% fee + 0.05% slippage per side from the baseline test [52].
- Report the basket and each coin, Sharpe ± standard error, maximum drawdown and turnover.
- Raise `TRIALS_TESTED` for every variant [53].
- **Accept only if** the variant beats the SMA50 rule's Sharpe on 2022–2024 *and* does not lose to it on 2025–2026.

Ranked by expected value per unit of effort:

| Rank | Idea | Why (evidence) | Effort | Expected value | Fair test |
|---|---|---|---|---|---|
| 1 | **Trend rule + volatility targeting** as the base strategy | The SMA50 rule beat every model we built [52]. Volatility targeting cuts left tails in every asset class and raises Sharpe for risk assets [48][50]. Trend + volatility sizing is the best-documented crypto result [49] | Very low (a few lines in `simulate.py`) | **High**: it improves the strategy that already wins | Position = 1 if close > SMA50, scaled by min(1, target / realised vol). Fix in advance: 30-day realised vol and a target equal to the median 2022–2024 vol of each coin; no leverage (spot). Compare with plain SMA50 on both windows |
| 2 | **Ensemble of trend look-backs** (e.g. SMA 20/50/100/200, averaged into a 0–1 exposure) | Averaging look-backs avoids picking one lucky parameter [49] and smooths turnover | Very low | Medium–high | Fix the look-back set in advance, so it counts as one trial. Run with and without #1 |
| 3 | **ML as an overlay on the trend rule, not as the signal** | Our models avoid crashes but lag rallies [52]. A trend rule lags too, but stays in during rallies. Deep momentum networks let ML size a trend position [20]. This is not the failed meta-label filter, which filtered our own "Smart" buy signals [53] | Low (the model already outputs P(up)) | Medium (the model's AUC of 0.53–0.55 limits it) | Exposure = trend position × f(P(up)), e.g. 1.0 if P ≥ 0.5, else 0.5. Fix f in advance; at most 2 variants. Must beat #1 alone, not just plain SMA50 |
| 4 | **Sliding vs expanding training window** (FreqAI style) | Crypto regimes change, and FreqAI's design assumes recency matters [1][2]. Published technical edges decay (Hudson & Urquhart [51]; neurotrader note) | Low (one parameter in `model.walk_forward`) | Low–medium | Three variants: expanding (current), 2-year sliding, 1-year sliding. Compare AUC and strategy Sharpe; choose on 2022–2024 |
| 5 | **Zero-shot foundation-model check** (Chronos-Bolt-small, and Kronos-small or Kronos-mini, on CPU) | Cheap to try; Kronos claims candle-specific skill [29]. Independent evidence says generic models fail on returns [30][31] | Low–medium (pip install, about 1 hour of CPU for daily data) | Low | Daily closes (OHLCV for Kronos), context of 512 candles, forecast 1 and 3 days ahead. Convert to P(up) from the forecast samples or quantiles. Score AUC and an SMA50-gated strategy **only on candles after each model's release** (Kronos: 2025-08 onward; Chronos-Bolt: 2024-12 onward) to avoid pre-training leakage. Use it as a feature only if AUC beats our model's on the same window |
| 6 | **Ensembling** (HGB + LightGBM + logistic regression, or several seeds or windows) | Ensembles top Qlib [8] and helped in crypto [15]. Numerai's whole design is an ensemble [41] | Low | Low (our variance is mostly the data, not the model) | One pre-fixed equal-weight average; compare AUC and the strategy from #3 |
| 7 | **Cost-aware no-trade band** | Hourly ML fails at 10 bp unless low-confidence trades are skipped [18] | Very low (thresholds already exist) | Low | Only if #3 is adopted: test one wider band, fixed in advance |

**What to avoid.**
- **LLM agents as traders.** There is no credible evidence they beat simple rules [34][35][38]. Live, they lost money with leverage [39]. They cannot be backtested honestly on data inside their training window [37], and they are slow and expensive per decision. Using an LLM as a news *feature* would need text data we do not have and a post-cutoff test window.
- **Reinforcement learning without strong baselines.** Published crypto RL is tested on days, not years [5][6], or in a crash [7]. RL adds a reward-design and overfitting surface with no evidence of beating a trend rule.
- **Deep sequence models (LSTM/Transformer/TFT) on our data.** In the largest fair comparison on engineered features they lose to boosted trees [8]. Four coins of 4h/1d candles is a small dataset.
- **Order-book deep learning (DeepLOB and similar).** It does not generalise [14], it needs tick data, and its horizons are too short to beat 0.3% round-trip costs.
- **Fine-tuning or pre-training a foundation model ourselves.** The only finance setting that helped was pre-training from scratch on very large financial panels [30]. That is out of scope for a CPU project.
- **Buying "AI bots" or signals.** See [43].

---

## Results in this repo: their training methods on our models (2026-10-06)

`experiments/training_practices.py` tested how other systems train, using our features, folds and embargo.
The variant was chosen on 2022–2024 by walk-forward AUC, averaged over the two 4h models, then checked once
on 2025–2026. Adoption needed a gain above 0.003 in both phases, the size of gain a meaningless moon-phase
feature produced.

| Training method (source) | AUC 2022–2024 | AUC 2025–2026 | Gain vs current |
|---|---|---|---|
| Expanding window, all history equally weighted (current) | 0.5460 | 0.5270 | – |
| Ensemble: HistGradientBoosting + LightGBM (Qlib DoubleEnsemble, Numerai) | 0.5459 | 0.5292 | −0.0001 / +0.0022 |
| Sliding window, last 3 years (FreqAI) | 0.5408 | 0.5202 | −0.0052 / −0.0067 |
| Recency weights, halving each year (Qlib sample reweighting) | 0.5360 | 0.5156 | −0.0101 / −0.0113 |
| Sliding window, last 2 years (FreqAI) | 0.5289 | 0.5177 | −0.0171 / −0.0093 |
| Sliding window, last 1 year (FreqAI) | 0.5246 | 0.5031 | −0.0214 / −0.0238 |

**Nothing adopted.**

- **Shorter windows and recency weights hurt, consistently in both phases.** Whatever weak signal the
  models find is stable across years, so more history beats "forgetting old regimes". The shorter the
  window, the worse.
- **The ensemble is a wash:** within the noise in both phases.
- **Our training already matches the best practice** found in this survey: gradient-boosted trees on
  engineered features, retrained continuously on an expanding window with an embargo.
- **The remaining ideas here are strategy-level, not training:** trend rule + volatility targeting, an
  ensemble of trend look-backs, and the AI's P(up) as a position sizer on top of the trend rule
  (section 7, items 1–3).

`TRIALS_TESTED` was raised to 25.

## Results in this repo: trend rules, volatility targeting and a learned combiner (2026-10-06)

`experiments/trend_ai_strategies.py` tested section 7's items 1–3, plus a **learned combiner**. The combiner
is a model retrained each year on all earlier years that learns *when* the trend signals work, from trend
flags, distances from the averages, volatility, momentum and the AI's own P(up). The test used daily
decisions, an equal-weight 4-coin basket and 0.1% fee + 0.05% slippage per side. The strategy was chosen
by Sharpe on 2022–2024 and checked on 2025–2026.

| Strategy | 2022–2024 return / Sharpe / worst drop | 2025–2026 return / Sharpe / worst drop |
|---|---|---|
| SMA50 trend rule (benchmark) | +145.8% / 0.98 / −39.5% | **+50.9% / 0.91** / −28.4% |
| Trend ensemble (20/50/100/200) × AI sizer (chosen) | +81.8% / **1.03** / −20.9% | +25.3% / 0.83 / **−15.1%** |
| Trend ensemble + 40% volatility target | +80.0% / 0.98 / −22.9% | +26.2% / 0.74 / −21.5% |
| Trend ensemble + 60% volatility target | +92.2% / 0.89 / −30.5% | +27.2% / 0.66 / −26.9% |
| Trend ensemble | +100.9% / 0.85 / −34.9% | +25.0% / 0.59 / −29.8% |
| Learned combiner + 40% volatility target | +29.2% / 0.63 / −20.4% | +14.3% / 0.66 / −14.9% |
| Learned combiner | +35.5% / 0.55 / −29.7% | +12.3% / 0.49 / −19.1% |
| Buy & hold | +54.6% / 0.55 / −75.3% | −11.0% / 0.15 / −61.6% |

**Not adopted.**

- **The chosen strategy fell short on the check.** Trend ensemble × AI sizer had the best 2022–2024 Sharpe,
  but on the check it trailed SMA50 (0.83 vs 0.91, uncertainty ±0.76).
- **Its reliable effect is risk, not return.** It halves the worst drop, at the cost of about half the
  return.
- **The learned combiner was the worst active strategy in both phases.** Teaching a model when to trust the
  trend signals made it worse than following them, the same pattern as every other attempt here to have ML
  out-time the market.
- **Plain SMA50 had the highest return in both phases.**

`TRIALS_TESTED` was raised to 31.

## Caveats

- **Abstracts and secondary sources.** Several papers were read via arXiv abstract or HTML pages through a summarising tool, not in full: [15][16][18][19][30][31][36][38]. The Jaquart et al. Sharpe of 3.23 [16] comes from a search-result abstract; the publisher page was blocked.
- **Alpha Arena results** come from news reports [39][40]. nof1's own blog page returned "not found" when fetched, so the official recap was not read. The buy-and-hold comparison uses daily candles, and the arena's exact start time may differ by hours.
- **Two different BTC baselines for the FinRL contest.** The ensemble paper reports BTC +0.74% [5] and the contest paper −7.35% [6]. They appear to use different evaluation windows. Both windows are a few days at most.
- **Numerai performance** is self-reported in a company blog post [42], and its documentation states no returns [41].
- **Qlib.** The benchmark README does not restate the costs. They are taken from the model configs [9], and I checked only the LightGBM Alpha158 config.
- **The crypto trend paper** [49] is a working paper. I read only its abstract page and could not verify BTC-only numbers or its cost assumptions.
- **No counted survey.** I found no review that counts how many crypto ML papers report out-of-sample, after-cost results. The statements in §6 are examples, not prevalence rates.
- **Kronos.** Its README does not state CPU speed, and the paper does not state when its training data ends. Both should be checked before the test in §7 item 5.

## Sources

1. Freqtrade, "FreqAI" documentation (stable). https://www.freqtrade.io/en/stable/freqai/
2. Freqtrade, "Running FreqAI" (backtesting, sliding window, `live_retrain_hours`, `expiration_hours`). https://www.freqtrade.io/en/stable/freqai-running/
3. Caulk, R. et al. (2022). FreqAI: generalizing adaptive modeling for chaotic time-series market forecasts. *Journal of Open Source Software* 7(80), 4864. https://doi.org/10.21105/joss.04864
4. AI4Finance Foundation, FinRL repository. https://github.com/AI4Finance-Foundation/FinRL
5. "Revisiting Ensemble Methods for Stock Trading and Crypto Trading Tasks at ACM ICAIF FinRL Contest 2023–2024", arXiv:2501.10709 (2025). https://arxiv.org/abs/2501.10709
6. Wang, K. et al. "FinRL Contests: Benchmarking Data-driven Financial Reinforcement Learning Agents", arXiv:2504.02281 (v4). https://arxiv.org/abs/2504.02281
7. Gort, B., Liu, X.-Y., Sun, X., Gao, J., Chen, S. & Wang, C. "Deep Reinforcement Learning for Cryptocurrency Trading: Practical Approach to Address Backtest Overfitting", arXiv:2209.05559. https://arxiv.org/abs/2209.05559
8. Microsoft Qlib, benchmarks README (CSI300, Alpha158/Alpha360, 20 seeds). https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md
9. Microsoft Qlib, `workflow_config_lightgbm_Alpha158.yaml` (segments, TopkDropout, costs). https://github.com/microsoft/qlib/blob/main/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
10. Jesse repository. https://github.com/jesse-ai/jesse
11. Hummingbot repository. https://github.com/hummingbot/hummingbot
12. Lim, B., Arık, S. Ö., Loeff, N. & Pfister, T. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting* 37(4). arXiv:1912.09363. https://arxiv.org/abs/1912.09363
13. Zhang, Z., Zohren, S. & Roberts, S. (2019). DeepLOB: Deep Convolutional Neural Networks for Limit Order Books. *IEEE Transactions on Signal Processing*. arXiv:1808.03668. https://arxiv.org/abs/1808.03668
14. Prata, M. et al. (2024). LOB-based deep learning models for stock price trend prediction: a benchmark study. *Artificial Intelligence Review* 57(5). https://link.springer.com/article/10.1007/s10462-024-10715-4 ; arXiv:2308.01915
15. Sebastião, H. & Godinho, P. (2021). Forecasting and trading cryptocurrencies with machine learning under changing market conditions. *Financial Innovation* 7, 3. https://pmc.ncbi.nlm.nih.gov/articles/PMC7785332/
16. Jaquart, P., Köpke, S. & Weinhardt, C. (2022). Machine learning for cryptocurrency market prediction and trading. *Journal of Finance and Data Science* 8. https://www.sciencedirect.com/science/article/pii/S2405918822000174 ; DOAJ record https://doaj.org/article/cfc09801314441198e3e1c92a3914e54
17. Wu, J., Zhang, X., Huang, F., Zhou, H. & Chandra, R. (2024). Review of deep learning models for crypto price prediction: implementation and evaluation. arXiv:2405.11431. https://arxiv.org/abs/2405.11431
18. Bysik, A. & Ślepaczuk, R. (2026). Machine Learning-Based Bitcoin Trading Under Transaction Costs: Evidence From Walk-Forward Forecasting. arXiv:2606.00060. https://arxiv.org/abs/2606.00060
19. Saly-Kaufmann, A., Wood, K., Calliess, J.-P. & Zohren, S. (2026). Deep Learning for Financial Time Series: A Large-Scale Benchmark of Risk-Adjusted Performance. arXiv:2603.01820. https://arxiv.org/abs/2603.01820
20. Lim, B., Zohren, S. & Roberts, S. (2019). Enhancing Time Series Momentum Strategies Using Deep Neural Networks. *Journal of Financial Data Science*. arXiv:1904.04912. https://arxiv.org/abs/1904.04912
21. Olorunnimbe, K. & Viktor, H. (2023). Deep learning in the stock market—a systematic survey of practice, backtesting, and applications. *Artificial Intelligence Review*. https://pmc.ncbi.nlm.nih.gov/articles/PMC9245389/
22. *(practitioner, unreviewed)* Hassnat07, crypto-ml-backtest: "ML pipeline testing whether gradient-boosted trees beat transaction costs on 4h crypto bars. Pre-registered result: no." https://github.com/Hassnat07/crypto-ml-backtest
23. Amazon Science, chronos-forecasting repository (Chronos, Chronos-Bolt, Chronos-2). https://github.com/amazon-science/chronos-forecasting
24. Ansari, A. F. et al. (2024). Chronos: Learning the Language of Time Series. arXiv:2403.07815. https://arxiv.org/abs/2403.07815
25. Das, A., Kong, W., Sen, R. & Zhou, Y. (2024). A decoder-only foundation model for time-series forecasting (TimesFM). ICML 2024. arXiv:2310.10688. https://arxiv.org/abs/2310.10688
26. Woo, G. et al. (2024). Unified Training of Universal Time Series Forecasting Transformers (Moirai). ICML 2024. arXiv:2402.02592. https://arxiv.org/abs/2402.02592
27. Garza, A. & Mergenthaler-Canseco, M. (2023). TimeGPT-1. arXiv:2310.03589. https://arxiv.org/abs/2310.03589
28. shiyu-coder, Kronos repository (models, context lengths, finetuning and backtest disclaimers). https://github.com/shiyu-coder/Kronos ; AAAI 2026 version: https://ojs.aaai.org/index.php/AAAI/article/view/39730
29. Shi, Y. et al. (2025). Kronos: A Foundation Model for the Language of Financial Markets. arXiv:2508.02739. https://arxiv.org/abs/2508.02739
30. Rahimikia, E., Ni, H. & Wang, W. (2025). Re(Visiting) Time Series Foundation Models in Finance. arXiv:2511.18578. https://arxiv.org/abs/2511.18578
31. Noguer i Alonso, M. & Franklin, R. P. (2026). Pretrained Time-Series Foundation Models for Financial Return Forecasting. arXiv:2606.27100. https://arxiv.org/abs/2606.27100
32. AI4Finance Foundation, FinGPT repository. https://github.com/AI4Finance-Foundation/FinGPT
33. Wu, S. et al. (2023). BloombergGPT: A Large Language Model for Finance. arXiv:2303.17564. https://arxiv.org/abs/2303.17564
34. Xiao, Y., Sun, E., Luo, D. & Wang, W. (2024/2025). TradingAgents: Multi-Agents LLM Financial Trading Framework. arXiv:2412.20138 (Table 1 in v4: https://arxiv.org/html/2412.20138v4). Repository: https://github.com/TauricResearch/TradingAgents
35. Yu, Y. et al. (2023). FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design. arXiv:2311.13743. https://arxiv.org/abs/2311.13743
36. Lopez-Lira, A. & Tang, Y. Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models. arXiv:2304.07619. https://arxiv.org/abs/2304.07619
37. Glasserman, P. & Lin, C. (2023). Assessing Look-Ahead Bias in Stock Return Predictions Generated by GPT Sentiment Analysis. arXiv:2309.17322. https://arxiv.org/abs/2309.17322
38. Li, W. W., Kim, H., Cucuringu, M. & Ma, T. (2025, rev. 2026). Can LLM-based Financial Investing Strategies Outperform the Market in Long Run? (FINSABER; KDD 2026). arXiv:2505.07078. https://arxiv.org/abs/2505.07078
39. nof1, Alpha Arena (https://nof1.ai, https://alphaarena.ai). Season 1 results via *(secondary)* ForkLog, "Four Out of Six AI Models Suffer Losses in Trading Tournament", https://forklog.com/en/four-out-of-six-ai-models-suffer-losses-in-trading-tournament/ ; *(secondary)* iWeaver, "Alpha Arena Season 1 Results: Final Ranking and Lessons", https://www.iweaver.ai/blog/alpha-arena-ai-trading-season-1-results/ . Buy-and-hold comparison: this repo's `data/*_USDT_1d.csv` (Binance).
40. *(secondary)* ForkLog, "AI model Grok 4.2 triumphs in trading tournament" (Alpha Arena Season 1.5). https://forklog.com/en/ai-model-grok-4-2-triumphs-in-trading-tournament/
41. Numerai documentation (tournament, meta model, staking, CORR/MMC, Signals, Crypto). https://docs.numer.ai/
42. Numerai blog, "JPMorgan Secures $500m Capacity in Numerai Following Breakthrough Year", 2025-08-26 (self-reported performance). https://blog.numer.ai/jpmorgan-secures-500m-capacity/
43. US CFTC, Customer Advisory: "AI Won't Turn Trading Bots into Money Machines", 2024-01-25. https://www.cftc.gov/LearnAndProtect/AdvisoriesAndArticles/AITradingBots.html
44. López de Prado, M. (2018). The 10 Reasons Most Machine Learning Funds Fail. *Journal of Portfolio Management* 44(6). https://www.garp.org/white-paper/the-10-reasons-most-machine-learning-funds-fail
45. Bailey, D., Borwein, J., López de Prado, M. & Zhu, Q. (2014). Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance. *Notices of the AMS* 61(5), 458–471. https://scholarworks.wmich.edu/math_pubs/40/
46. Bailey, D., Borwein, J., López de Prado, M. & Zhu, Q. (2017). The Probability of Backtest Overfitting. *Journal of Computational Finance* 20(4). https://papers.ssrn.com/abstract=2326253
47. Kapoor, S. & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns* 4(9). https://doi.org/10.1016/j.patter.2023.100804
48. Harvey, C., Hoyle, E., Korgaonkar, R., Rattray, S., Sargaison, M. & Van Hemert, O. (2018). The Impact of Volatility Targeting. *Journal of Portfolio Management* 45(1). https://scholars.duke.edu/publication/1370354
49. Zarattini, C., Pagani, A. & Barbon, A. (2025). Catching Crypto Trends: A Tactical Approach for Bitcoin and Altcoins. Swiss Finance Institute Research Paper 25-80. https://ideas.repec.org/p/chf/rpseri/rp2580.html ; https://abarbon.com/papers/catching-crypto-trends
50. Moreira, A. & Muir, T. (2017). Volatility-Managed Portfolios. *Journal of Finance* 72(4), 1611–1644. https://doi.org/10.1111/jofi.12513
51. Hudson, R. & Urquhart, A. (2021). Technical trading and cryptocurrencies. *Annals of Operations Research* 297, 191–220. https://doi.org/10.1007/s10479-019-03357-1
52. This repo: `docs/backTestResult/Baseline_LightGBM_vs_trivial_rules_holdout.html` (one-time holdout; SMA50 +54.0% vs LightGBM −29.3%; main AI beat the trend rule in 50–53% of 2022–2026 runs), `docs/backTestResult/Multi_period_test_2022-2026_all_coins.html`, `experiments/baseline_lightgbm.py`.
53. This repo: `cryptoai/model.py` (`walk_forward`: expanding training window with embargo), `cryptoai/config.py` (`MIN_AUC`, `TRIALS_TESTED = 14`, `FEE`), `cryptoai/simulate.py` (`meta_min` hook); commit 240f230 (meta-label filter on "Smart" buy signals, not adopted).
54. Companion notes in this folder: [neurotrader-methods.md](neurotrader-methods.md), [sharpe-ratio.md](sharpe-ratio.md), [reading-crypto-charts-for-day-trading.md](reading-crypto-charts-for-day-trading.md), [technical-analysis-concepts.md](technical-analysis-concepts.md), [swing-points-rolling-window-directional-change-pip.md](swing-points-rolling-window-directional-change-pip.md), [chartscanai.md](chartscanai.md).
