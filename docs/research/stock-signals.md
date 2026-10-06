# What predicts US stock returns, and what we can use for the stock AI

*Research note, 2026-10-06. Literature and free-data review for improving the stock AI (`cryptoai/stockai.py`), which uses only daily price and volume and is weak (AUC 0.48-0.53). Companion: [improving-the-model.md](improving-the-model.md).*

## Bottom line

- **Weak results are what the literature predicts for 15 mega-caps at 1-week horizons.** McLean & Pontiff (2016): 97 published predictors earn 26% less out of sample and 58% less after publication; the decay is largest in big, liquid stocks.
- **Statistical power is the main problem.** With 15 stocks and about 84 month-ends, the standard error of a rank IC is about 0.028, while realistic large-cap ICs are 0.01-0.04. With 500 stocks it falls to about 0.005. Slow fundamental features must be learnt on a wide universe (S&P 100-500 with historical membership) and then applied to our stocks.
- **The 15-stock list was chosen with hindsight** (PLTR, HOOD, COIN, MSTR), which flatters momentum-like features.
- **What our own tests found:** the monthly 200-day buy-for-hold list works (adopted); ranking stocks by potential, a one-stock pick, the 5-day AI, a shared crypto+stock AI, limit-order entries, crash exits and market-stress filters did not beat it.

## Evidence by signal (large caps, after 2010)

| Signal | Evidence | For mega-caps |
|---|---|---|
| Post-earnings drift (SUE) | Martineau (2022): gone for large stocks since about 2006; 2025 "revival" disappears without microcaps | Essentially dead |
| Earnings-announcement return (EAR) | Brandt et al. (2008); prices and dates only | Small; maybe a sell-side veto |
| Analyst revisions / rating changes | Mostly priced on the day; drift in low-coverage stocks | Weak (40+ analysts each); no free point-in-time consensus history |
| Gross profitability / quality | Novy-Marx (2013); robust in Jensen, Kelly & Pedersen (2023), works in large caps | Modest, months-to-a-year horizon |
| Value (P/E, B/M) | HML fell ~55% 2007-2020; book value misses intangibles | Low priority for tech names |
| Accruals | Gone after the mid-2000s | Skip |
| Net issuance / buybacks | Pontiff & Woodgate (2008); durable, works in large caps | Plausible; our list spans buyers (AAPL, META) and issuers (MSTR, COIN, PLTR) |
| Insider buying (Form 4) | Cohen, Malloy & Pomorski: opportunistic buys +82 bp/month (to 2007) | Almost never happens at mega-caps |
| Short interest | Stock level weak in large caps; aggregate short interest timed the market (Rapach et al., to 2014) | Market-level timer worth checking |
| Options skew / put-call parity | Xing-Zhang-Zhao (2010), Cremers-Weinbaum (2010) | No free history; collect forward only |
| LLM news sentiment | Lopez-Lira & Tang: next-day, mostly small caps, fading; backtests leak hindsight | Collect forward only |
| Macro (VIX, curve, credit) | Goyal, Welch & Zafirov (2024): half of the predictors fail out of sample; better as risk filters | Tested here: did not help (below) |

## Free data with point-in-time history

- **SEC EDGAR** (no key; a User-Agent header is required; 10 requests/s). Ticker to CIK: `sec.gov/files/company_tickers.json`. `data.sec.gov/api/xbrl/companyfacts/CIK##########.json` has every reported number with its **filed** date (use the first-filed value, usable from the next trading day). `submissions` gives 8-K item 2.02 earnings-release dates. Checked working on 2026-10-06.
- **FRED / ALFRED** (free key): VIX, Treasury spreads, Baa spread; use ALFRED vintages for revised series.
- **Yahoo Finance** (unofficial): daily prices including ^VIX, ^VIX3M, HYG, IEF, ^TNX, ^IRX; earnings dates with surprise; dated upgrades and downgrades. Estimates are final consensus, not as-of.
- **FINRA** short interest: twice monthly, published about 8 business days later (lag to publication).
- Paid or limited: Alpha Vantage and FMP free tiers (restated statements), Sharadar (paid), GDELT (full history via BigQuery).

## Results in this repo

| Test | Result |
|---|---|
| `experiments/shared_brain.py`: one AI trained on crypto and stocks together vs separate AIs (daily, 5-day target) | Worse everywhere: crypto AUC 0.495/0.498 vs 0.520/0.522; stocks 0.460/0.508 vs 0.486/0.520 (2020-2022 / 2023-2026). Not adopted |
| `experiments/stock_regime.py`: halve the buy list's exposure when VIX > VIX3M, when HYG/IEF fell 2%+ in 21 days, or both | All lower Sharpe in 2016-2022 (0.66-0.78 vs 0.83) and 2023-2026 (1.85-2.03 vs 2.05), with no smaller worst fall. Not adopted |

## Next step (not started)

Learn slow fundamental tilts (net share issuance, buyback yield, gross profitability) on a wide universe: S&P 500 with historical membership, EDGAR first-filed values, monthly decisions; then test them as a tilt on the buy list with the usual rule (better in both 2016-2022 and 2023-2026). Several gigabytes of downloads and a few hours of processing.
