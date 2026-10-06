# Whale tracking: data sources, evidence, and what `cryptoai/` does with it

*Research note, 2026-10-06. Reviews APIs that track or alert on large crypto holders ("whales"), the evidence that whale activity predicts prices, and the whale watch and feature test added to `cryptoai/` the same day. Companions: [improving-the-model.md](improving-the-model.md), [binance-trading-bots.md](binance-trading-bots.md).*

## Bottom line

- **Whale and flow data mostly predicts volatility, not direction.** The best-supported link is that BTC moving onto exchanges goes with higher volatility and somewhat lower returns over days, partly because falling prices cause the inflows rather than the other way round. No study found shows an after-fee trading edge out of sample.
- **Free Binance data covers the most useful signals.** It gives large spot trades, futures liquidations, open interest and long/short ratios. The live REST endpoints keep only 30 days, but data.binance.vision has 5-minute futures metrics from 2020-09 (BTC) and 2021-12 (ETH, BNB, SOL).
- **Paid on-chain feeds are mainly for alerts.** Whale Alert is $29.95/month for personal use; full history needs $699+/month plans. CryptoQuant exchange flows with history start at about $99/month.
- **In this repo:**
  - a whale watch now alerts on whale trade bursts, liquidation cascades and open-interest jumps, as information only;
  - in a pre-registered test, whale positioning features did not help the direction model;
  - they raised the drop warning's AUC in both periods (+0.010, +0.012), but not reliably enough to pass (80% of bootstrap resamples, needed 95%).

## Data sources

| Source | What it gives | Cost | History |
|---|---|---|---|
| Binance spot aggTrade stream | Every trade; large ones show big players | Free | data.binance.vision aggTrades (large files) |
| Binance futures `!forceOrder@arr` | Liquidations (at most one per coin per second, so undercounted) | Free | None; record it yourself |
| Binance futures open interest, top-trader and account long/short ratios | Leverage and positioning | Free | REST 30 days; data.binance.vision metrics from 2020-09 / 2021-12 |
| Whale Alert | $100k+ on-chain transfers labelled by exchange (BTC, ETH, SOL, Tron and others; BNB chain unclear) | $29.95/month personal; $699+ for history | 30-90 days on paid plans |
| CryptoQuant | Exchange inflow, outflow, reserves, Exchange Whale Ratio | ~$99/month with API | From 2016 |
| Glassnode | Exchange flows, including point-in-time values (no label look-ahead) | Quote | Long |
| Santiment | Whale transaction counts | Free tier with 30-day lag | 1-2 years |
| CoinGlass | Liquidations, Hyperliquid whale positions | $29-699/month | Limited at low tiers |
| Hyperliquid Info API | Every account's positions | Free | Record it yourself |
| Arkham, Nansen, Lookonchain | Entity labels, "smart money", posts on X | Quote / $49+ / none | Not suited to BTC/ETH spot or backtests |

Order-book "walls" are not used: one study classed 31% of large limit orders as potential spoofs, and there is no free history.

## Evidence

- **Exchange reserves.** Hoang & Baur (SSRN 3902504; daily data 2016-2021) found reserve changes negatively related to current and future BTC returns, with large changes going with higher volatility. This is in-sample, with reverse causality.
- **Whale Alert transfers.** Herremans & Low (arXiv 2211.08281) found net exchange flows correlate 0.47 with daily volatility, and predicted next-day volatility spikes with F1 0.46, on a single 2020-2021 test window.
- **Order flow.** Order flow predicts minutes to hours ahead and decays fast. Our own taker buy/sell features lowered the 4h/1d AUC (2026-10-05).
- **Smart money.** The evidence comes from the vendors themselves and covers small-cap tokens.
- **Liquidation cascades.** After the 10 Oct 2025 Hyperliquid cascade, prices recovered within a day while order-book depth stayed thin for weeks. That makes cascades a risk signal, not a direction signal.

## Results in this repo

**Whale watch (`cryptoai/whales.py`, started by the live service).** Alerts go out through Windows and Zalo and are never traded on:
- a burst of 3+ same-side spot trades of at least $2M (BTC), $1M (ETH) or $300k (BNB, SOL) within 60 s, or one trade 5 times that size;
- more than $10M of longs or shorts liquidated within 5 minutes on our 4 coins;
- a 1-hour open-interest change of at least 3% and 3 standard deviations of recent 1-hour changes.

Every event is written to `logs/whale_events.csv`, which builds our own history (liquidations have no free history). The Market page shows the last hour and recent events. Turn the alerts off with `"whales": false` in notify.json.

**Feature test (`experiments/whale_features.py`, pre-registered).** Nine features, as known at each 4h close: open-interest change (4h, 24h) and its 30-day z-score, top traders' long/short ratio by position and by account, its 24h change, the crowd's ratio, top traders minus the crowd, and spot average trade size vs its last 30 days. Both versions trained only on rows from 2021-12 (when all four coins have data), retrained monthly.

| Model | AUC gain 2022-07 to 2024 | AUC gain 2025-2026 | Positive in resamples | Verdict |
|---|---|---|---|---|
| Direction (next 1 day) | +0.0034 | -0.0005 | 47% | Not adopted |
| Drop warning | +0.0096 | +0.0119 | 80% | Not adopted (needed 95%) |

Data source: `experiments/download_futures_metrics.py`.

**Caveat on the drop-warning baseline.** Training from 2021-12 only, the drop warning without whale features reached AUC 0.554 on 2025-2026, below the live model trained from 2019 (0.568). The whale version (0.566) did not beat the live model either.

**Next test (one trial):** the full-history drop warning with whale features left blank before they exist, against the live model, with the same pass rule.
