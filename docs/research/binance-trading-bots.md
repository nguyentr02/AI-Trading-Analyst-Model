# Binance trading bots: how they work, what the evidence says, what `cryptoai/` could use

*Research note, 2026-10-06. Covers Binance's built-in bots (Spot Grid, Futures Grid, Spot/Futures DCA, Rebalancing, Funding-rate Arbitrage, TWAP/VP algo orders, Auto-Invest, Copy Trading, AI Pro), the independent evidence on grid and DCA strategies, and which ideas are worth testing in `cryptoai/` (spot only, BTC/ETH/BNB/SOL, 50-day trend rule as the core, ML models with AUC 0.53-0.57). Mechanics and fees come from Binance's own documentation; performance claims are labelled by source. Companion to [other-ai-trading-models.md](other-ai-trading-models.md).*

## Bottom line

- **Binance's bots are rule-based order schedulers, not forecasters.** The "AI parameters" for grids are a documented heuristic: range = moving average ± 3 standard deviations of daily closes over 7, 30 or 180 days; grid count = 1.3 × range / ATR (futures) [2][6].
- **Spot bots cost nothing extra** beyond normal trading fees (0.1% per side, 25% off when paying in BNB). Copy trading takes 10% of follower profit on spot and up to 30% on futures [12][13].
- **There is no independent evidence that grids or DCA give an edge.** A classic grid on a random walk has an expected return of exactly zero before fees: grid profits are offset by losses on the coins held when price leaves the range [20]. Lump-sum investing beats DCA in about 68% of periods (Vanguard, equities 1976-2022) [21].
- **Leaderboard ROI is not evidence.** It is shown for short runtimes, inflated by leverage on futures (ROI is on margin), excludes stopped or liquidated bots, and "grid profit" ignores losses on the inventory held [11].
- **No public API creates grid, DCA, rebalancing or arbitrage bots** [19]. Only TWAP/VP algo orders have endpoints [17][18]. Every bot is simple to rebuild in Python with ordinary orders, which keeps backtest and live behaviour identical.
- **For `cryptoai/`:** the only Binance-style ideas worth an honest test are splitting entries when the 50-day rule turns long (DCA vs lump sum), and possibly a small, capped grid while the trend rule is in cash. Skip futures grids, copy trading, leaderboard copying, martingale-style DCA and funding arbitrage.

## 1. The bots

| Bot | What it does | Fees | Notes |
|---|---|---|---|
| **Spot Grid** | Buy and sell limit orders at 2-500 levels between a lower and upper price, arithmetic (equal steps) or geometric (equal ratio r = (U/L)^(1/N)). Each filled buy places a sell one level up, and vice versa. Optional stop-loss / take-profit, "sell all on stop", Trailing Up (moves the grid up when price breaks above it) [2][3][4][5] | Spot fees | Profit per geometric grid = (1-c)·r - 1 - c with fee c. With 0.1% fees a round trip costs 0.2%, so steps under ~0.4% are mostly fee |
| **Futures Grid** | Neutral, long or short grid on USDⓈ-M perpetuals with leverage; expires when the risk ratio (margin balance / occupied margin) drops below 1, which is a liquidation [6][7] | Futures fees | Trailing up and down; at most 169 orders around the price |
| **Spot DCA** | Base order, then up to N extra buys each time price falls by a set deviation; deviation and size multipliers (e.g. 1000, 2000, 4000 USDT); take profit from the average price (fixed or trailing); stop-loss ends the strategy [8] | Spot fees | A martingale: the largest position sits at the worst price |
| **Futures DCA** | Same on perpetuals (reported launch, secondary source) [24] | Futures fees | |
| **Rebalancing** | 2-10 coins at target weights, rebalanced when a weight drifts 0.5-50% or on a 30-minute to 28-day schedule [9] | Spot fees | |
| **Funding-rate arbitrage** | Long spot + short perpetual when funding is positive (reverse when negative); 10% kept as margin buffer [10] | 4 fee legs | Funding can flip; short leg can be liquidated |
| **TWAP / VP algo orders** | Split a large order over 5 minutes to 24 hours (TWAP) or by share of volume (VP, futures) [17][18] | Standard fees | API available |
| **Auto-Invest** | Recurring buys, daily to monthly, from 1 USDT [14][15] | Shown per buy; zero-fee promotions | |
| **Copy trading** | Mirror a lead trader. Spot: 10% profit share + 10% of fees, 0.3-0.5% slippage guard (copies can fail). Futures: up to 30% profit share, weekly high-water mark [12][13] | Profit share | |
| **Binance AI Pro** | LLM assistant subscription (beta 2026-03-25, $9.99-$29.99/month); Binance says it gives no advice, the user sets the rules [25] | Subscription | No published edge |

## 2. Evidence on performance

**Grid trading.** Chen, Chen & Jang (2025) prove a classic grid has zero expected value on a symmetric random walk without fees [20]. Their re-centring "dynamic grid" reports 60-70% IRR on 1-minute BTC/ETH data, January 2021 to July 2024, with 0.08% fees. That result is unreplicated, with parameters chosen on the same period and a strongly rising sample; the authors offer no theory for why it should work. Economically a grid is short volatility: it earns small amounts while price ranges and gives them back, with interest, when price trends.

**DCA vs lump sum.** Lump sum wins about two thirds of the time when the expected drift is positive; DCA mainly reduces regret and the spread of outcomes [21].

**Copy trading.** The only multi-exchange figures (a "YieldFund 2025" 90-day study, relayed by competitor KuCoin) say about 48.5% of followers were profitable, while almost all lead traders were. The figures are internally inconsistent and the method is unpublished [22]. No peer-reviewed study of Binance follower returns was found. The leader-follower gap fits entry lag, slippage, fees and the 10-30% profit share.

**Failure modes.**
1. A breakout leaves a grid all in cash (missing the rally) or all in coin (holding the fall).
2. Futures grids are liquidated when the risk ratio drops below 1.
3. Fees swallow narrow grid steps.
4. DCA safety orders put the most money in at the worst price, and capital runs out in a long fall.
5. Funding arbitrage loses when funding turns and pays four fee legs.

**Marketing vs verified.**

| Claim | Status |
|---|---|
| Mechanics, formulas, fees | Binance documentation; reliable as description |
| "AI" parameters | Marketing label for a Bollinger/ATR rule |
| Leaderboard ROI | Selection-, survivorship- and leverage-biased; unverified |
| Copy-trading statistics | Third-party study via a competitor; unverified |
| Grid zero expectation | Independent proof [20] |
| Dynamic grid profits | Independent but unreplicated, in-sample |
| DCA vs lump sum | Independent (equities); same logic in crypto, not directly tested |

## 3. What `cryptoai/` could test

Backtest rules for any of these: 1-minute bars; a limit order fills only if price trades *through* its level; never count a buy and the sell above it in the same bar; 0.1% fee per side plus slippage on market orders; Binance minimum order sizes; choose on 2022-2024, check once on 2025-2026; compare with buy & hold, the 50-day rule and cash.

1. **Split entries when the 50-day rule turns long** (worth testing, cheap). Buy all at once vs over 3, 5 or 10 days, and the same for exits. Lump sum should win in real trends; the question is whether splitting cuts enough whipsaw losses. The 50-day rule loses about 4 trades in 5 (average loss -3% to -5%), so the whipsaw cost is real.
2. **Rebalancing between the 4 coins** (expect it to fail). Rebalancing earns when relative prices mean-revert; crypto has shown momentum between coins, so it tends to sell the winners.
3. **A grid only while the trend rule is in cash** (small and sceptical). When the rule says cash, price is often falling, the exact case where a grid fills all one side. If tested at all: geometric steps of at least 0.8-1%, a range of about ±2-3 ATR, sell everything on a close below the range, and judge it against cash.
4. **Execution** (low priority). At our size, impact is negligible. Binance's spot TWAP sends market slices that pay taker fees; posting limit orders at the best price for a few minutes before falling back to market is cheaper and can be measured in the paper trial.
5. **Skip:** futures grid and DCA (leverage, out of scope), copy trading, copying leaderboard parameters, martingale DCA, funding arbitrage.

## Results in this repo

**Split entries (idea 1), tested 2026-10-06 in `experiments/split_entries.py`: not adopted.** Basket of BTC/ETH/BNB/SOL, 50-day rule, 0.15% cost per side. Buying over 3, 5 or 10 days instead of at once barely changed 2022-2024 (Sharpe 0.98-1.01 vs 0.98), and split5 was chosen. On 2025-2026 it lost to lump sum: Sharpe 0.79 vs 0.93, return +38.5% vs +52.8%, worst drop about the same (-28.6% vs -28.4%). Splitting exits as well was worst in both periods (2025-2026 Sharpe 0.68). The rule's profits come from the first days of big trends; entering slowly gives up more of those than it saves on whipsaws. This matches the lump-sum evidence [21].

Not tested: rebalancing (idea 2), grid while in cash (idea 3), execution (idea 4).

Related: the 50-day rule's trades on 2022-2026 win only 21-26% of the time, but average wins are 5.6-9.7 times average losses (`experiments/top_traders.py` comparison), the opposite profile to a grid (many small wins, rare large losses).

## Sources

1. Binance, Trading Bots landing page. https://www.binance.com/en/trading-bots
2. Binance FAQ, Spot Grid trading parameters. https://www.binance.com/en/support/faq/binance-spot-grid-trading-parameters-688ff6ff08734848915de76a07b953dd
3. Binance FAQ, Spot Grid auto parameters. https://www.binance.com/en/support/faq/how-to-use-spot-grid-trading-auto-parameters-76bd4effa3c4456c971a1c6835762742
4. Binance FAQ, What is Spot Grid trading. https://www.binance.com/en/support/faq/what-is-spot-grid-trading-and-how-does-it-work-d5f441e8ab544a5b98241e00efb3a4ab
5. Binance FAQ, Trailing Up. https://www.binance.com/en/support/faq/how-to-use-the-trailing-up-function-in-spot-grid-trading-3d987afd7906495cb4d997eccb8515bf
6. Binance FAQ, Futures Grid AI parameters. https://www.binance.com/en-TR/support/faq/binance-futures-grid-trading-ai-parameters-guide-647b0dba72d145219688b04aa51405fc
7. Binance FAQ, What is Futures Grid trading. https://www.binance.com/en/support/faq/what-is-futures-grid-trading-f4c453bab89648beb722aa26634120c3
8. Binance FAQ, Spot DCA. https://www.binance.com/en/support/faq/what-is-spot-dca-and-how-does-it-work-27713d3ddb3c406da52f36b9aaaa1360
9. Binance FAQ, Rebalancing Bot. https://www.binance.com/en/support/faq/what-is-rebalancing-bot-and-frequently-asked-questions-29bbbd2e7fc24085be7a8a7d02779457
10. Binance FAQ, Funding-rate Arbitrage Bot. https://www.binance.com/en/support/faq/what-is-the-binance-funding-rate-arbitrage-bot-and-frequently-asked-questions-f330e17d6fc04679b9b21d6f9350e787
11. Binance FAQ, Trading bots landing page and ROI. https://www.binance.com/en/support/faq/how-to-use-the-binance-trading-bots-landing-page-f0c2bd5bc16c40b9998d22549e91cd1c
12. Binance FAQ, Spot copy trading. https://www.binance.com/en/support/faq/frequently-asked-questions-on-binance-spot-copy-trading-1826c3b426a149949851554bdde227d3
13. Binance FAQ, Futures copy trading. https://www.binance.com/en/support/faq/frequently-asked-questions-on-binance-futures-copy-trading-6ed0995daf0b42d5816beaf1e31ca09d
14. Binance FAQ, Auto-Invest. https://www.binance.com/en/support/faq/what-is-auto-invest-and-how-to-use-it-3dd41bc1d4ea4879863ffbf2211a17fe
15. Binance announcement, Auto-Invest zero fees. https://www.binance.com/en/support/announcement/binance-auto-invest-offers-zero-fees-9227e6c8660d4a88a00a173ad284c0d5
16. Binance FAQ, Smart Arbitrage. https://www.binance.com/en/support/faq/what-is-binance-smart-arbitrage-and-how-to-get-started-2c65b90111e14be6b0156d32e0ff94d9
17. Binance API docs, Spot TWAP new order. https://developers.binance.com/docs/algo/spot-algo/Time-Weighted-Average-Price-New-Order
18. Binance API docs, Futures TWAP new order. https://developers.binance.com/docs/algo/future-algo/Time-Weighted-Average-Price-New-Order
19. Binance developer forum, strategy/grid API. https://dev.binance.vision/t/binance-strategy-futures-grid-api/15124
20. Chen, Chen & Jang (2025), grid trading, arXiv 2506.11921. https://arxiv.org/abs/2506.11921v1
21. Vanguard lump sum vs DCA research, via SmartAsset. https://smartasset.com/investing/lump-sum-investing-research
22. KuCoin blog, Is crypto copy trading profitable in 2026. https://www.kucoin.com/blog/is-crypto-copytrading-profitable-in-2026
23. The Block, Binance spot copy trading. https://www.theblock.co/post/290345/binance-spot-copy-trading
24. Blockchain.news, Binance Futures DCA bot. https://blockchain.news/flashnews/binance-launches-futures-dca-bot-for-customizable-trading
25. Unlock, Binance AI Pro beta. https://www.unlock-bc.com/en/binance-rolls-out-ai-pro-beta-for-ai-powered-trading
