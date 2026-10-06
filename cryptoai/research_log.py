"""Every idea tested so far, with its result and verdict: the record behind the Strategy lab page.

Binance ranks bots by recent ROI. This ranks nothing by its best period: every entry was fixed before results were
seen, chosen on one period and checked on another (mostly 2022-2024 / 2025-2026), after fees, against a benchmark.
Details are in the experiment named in each row and in docs/research/. Add a row whenever an experiment finishes.
"""
import pandas as pd

ADOPTED, NOT_ADOPTED, PARTLY, FINDING = "Adopted", "Not adopted", "Partly", "Finding"

LOG = pd.DataFrame([
    # (date, area, idea, result on data it did not choose on, verdict, where)
    ("2026-10-05", "Model inputs", "15-minute and 1-hour chart patterns for the 4h models",
     "Higher walk-forward AUC", ADOPTED, "docs/research/reading-crypto-charts-for-day-trading.md"),
    ("2026-10-05", "Model inputs", "Order flow (taker buy/sell volume)", "Lower AUC", NOT_ADOPTED,
     "docs/research/reading-crypto-charts-for-day-trading.md"),
    ("2026-10-05", "Model inputs", "Futures funding rate and premium", "Lower AUC", NOT_ADOPTED,
     "docs/research/reading-crypto-charts-for-day-trading.md"),
    ("2026-10-05", "Model inputs", "Chart images (ChartScanAI-style)", "No gain", NOT_ADOPTED,
     "docs/research/chartscanai.md"),
    ("2026-10-05", "Model inputs", "Technical analysis bundle (Fibonacci, patterns, S/R, ...)", "No gain",
     NOT_ADOPTED, "docs/research/technical-analysis-concepts.md"),
    ("2026-10-05", "Model inputs", "Moon phase (placebo)", "Noise level: about 0.003 AUC", FINDING,
     "docs/research/technical-analysis-concepts.md"),
    ("2026-10-05", "Model training", "Sliding windows, recency weights, ensembles", "No gain", NOT_ADOPTED,
     "experiments/training_practices.py"),
    ("2026-10-05", "Model training", "LightGBM baseline vs the 50-day rule (one-time holdout)",
     "-29.3% vs +54.0% for the 50-day rule", NOT_ADOPTED, "experiments/baseline_lightgbm.py"),
    ("2026-10-05", "Trading rules", "Stop-loss in AI Smart", "Stops sold near short-term bottoms", NOT_ADOPTED,
     "docs/backTestResult"),
    ("2026-10-05", "Trading rules", "Meta-label filter on AI Smart buys", "No gain", NOT_ADOPTED,
     "docs/research/other-ai-trading-models.md"),
    ("2026-10-05", "Trading rules", "Shock dip-buy: 10% fall in 1 hour, sell 4 hours later",
     "Beat random entries on both check periods", ADOPTED, "experiments/shock_dip_buy.py"),
    ("2026-10-05", "Trading rules", "Trend rules x AI, learned combiner", "Lower Sharpe than the 50-day rule "
     "(trend x AI: about half the drawdown, half the return)", PARTLY, "experiments/trend_ai_strategies.py"),
    ("2026-10-06", "Altcoins", "AI on NEAR and ZEC (with and without training on them)",
     "Sharpe 0.75 vs 1.83 for the 50-day rule", NOT_ADOPTED, "experiments/altcoins_near_zec.py"),
    ("2026-10-06", "Risk", "Drop warning (P of a sharp fall before a rise) as AI Smart's exit",
     "Sharpe 0.42 vs 0.36; worst fall -31.5% vs -40.1%", ADOPTED, "experiments/exit_timing.py"),
    ("2026-10-06", "Trend rule", "50-day rule + volatility sizing / drop brake", "Sharpe 0.85 vs 0.92",
     NOT_ADOPTED, "experiments/trend_core.py"),
    ("2026-10-06", "Trend rule", "50-day rule on the top 8/12/16 coins by volume", "Sharpe 0.28 vs 0.93",
     NOT_ADOPTED, "experiments/trend_universe.py"),
    ("2026-10-06", "Trend rule", "Enter over 3/5/10 days instead of at once", "Sharpe 0.79 vs 0.93", NOT_ADOPTED,
     "experiments/split_entries.py"),
    ("2026-10-06", "Diagnostic", "Does the AI know anything the trend doesn't?",
     "Yes: IC +0.049 after removing trend (t 3.3)", FINDING, "experiments/ai_vs_trend.py"),
    ("2026-10-06", "Trading rules", "Size by calibrated AI confidence", "Sharpe 1.58 before fees, 0.27 after "
     "(vs 0.96): 187-340 trades a year", NOT_ADOPTED, "experiments/calibrated_sizing.py"),
    ("2026-10-06", "Trading rules", "Hold only while the 50-day rule AND the AI agree",
     "Sharpe 1.24 vs 0.96, worst fall -12.5% vs -28%; but lost on 2022-2024", PARTLY,
     "experiments/low_turnover_ai.py"),
    ("2026-10-06", "Risk", "Drop warning + volatility forecast (HAR) / options volatility (DVOL)",
     "AUC -0.002 / -0.053", NOT_ADOPTED, "experiments/drop_features.py"),
    ("2026-10-06", "Model inputs", "Whale positioning (open interest, top traders vs crowd, trade size)",
     "Direction -0.001 AUC; drop warning +0.010 / +0.012, not reliable enough (80%)", NOT_ADOPTED,
     "experiments/whale_features.py"),
    ("2026-10-06", "DCA", "Weekly buying: 2x below the 50-day average, 0.5x above", "$1.073 vs $1.066 per $1 "
     "(plain DCA); also ahead on 2022-2024", ADOPTED, "experiments/dca_ai.py"),
    ("2026-10-06", "DCA", "Weekly buying sized by the AI", "$1.059 vs $1.066 per $1 (plain DCA)", NOT_ADOPTED,
     "experiments/dca_ai.py"),
    ("2026-10-06", "Trading rules", "Shock dip-buy trigger scaled to each coin's volatility",
     "Total +17.5% vs +83.1% for the fixed 10% rule", NOT_ADOPTED, "experiments/vol_scaled_shock.py"),
    ("2026-10-06", "Learnt policy", "AI learns what/when/how much to buy and sell: return model + 216 sizing "
     "policies searched on 2022-2024", "Sharpe 0.69 vs 0.95 (50-day rule); also behind on 2022-2024 (0.91 vs "
     "0.98); deflated Sharpe 3%", NOT_ADOPTED, "experiments/policy_learning.py"),
    ("2026-10-06", "Trading rules", "Don't sell when the sale wouldn't cover the fees (unless a stop is needed)",
     "Tiny-profit guard: Sharpe 1.079 vs 1.085 (2022-2024), 0.450 vs 0.416 (2025-2026), about neutral. "
     "Never selling at a loss unless -5%/-10%: worst fall -44%/-48% vs -38% in 2022", NOT_ADOPTED,
     "experiments/fee_guard.py"),
    ("2026-10-06", "US stocks", "Trade 17 Binance-listed US stocks with 50/100/200-day trend rules instead of "
     "holding", "Best rule (chosen 2016-2022) beat holding on 0 of 17 stocks in 2023-2026", NOT_ADOPTED,
     "cryptoai/stocks.py"),
    ("2026-10-06", "US stocks", "Hold the 5 strongest stocks by 12-1 month momentum, monthly", "Sharpe 0.63 vs "
     "0.81 (2016-2022), 1.44 vs 1.72 (2023-2026) for holding all equally", NOT_ADOPTED, "cryptoai/stocks.py"),
    ("2026-10-06", "US stocks", "Stock AI (P(up, 5 days), pooled over 17 stocks, SPY as market context) as a risk "
     "filter or position sizer", "AUC 0.478 (2019-2022), 0.529 (2023-now); risk filter Sharpe 0.51 vs 0.68 for "
     "holding in 2019-2022", NOT_ADOPTED, "cryptoai/stockai.py"),
    ("2026-10-06", "US stocks", "Buy-for-hold list: equal weight across stocks above their 200-day average, "
     "reviewed monthly", "Sharpe 0.83 vs 0.61 (2016-2022), 2.05 vs 1.91 (2023-2026) for holding all equally; "
     "worst fall -46% vs -59%", ADOPTED, "experiments/stock_hold_picks.py"),
    ("2026-10-06", "US stocks", "Sell everything in a market crash (S&P 500 below its 200-day average, or 10% "
     "below its high)", "Cut the 2020/2022 worst fall (-46% vs -59%) but Sharpe 1.47 / 1.67 vs 1.91 in 2023-2026",
     PARTLY, "experiments/stock_hold_picks.py"),
    ("2026-10-06", "US stocks", "Rank stocks by 'potential': an AI predicting each stock's 3-month return vs SPY",
     "Ranking IC -0.05 (2019-2022), +0.11 but t 1.6 (2023-2026); tilting the buy list to its top half: Sharpe "
     "0.07 vs -0.12, then 2.00 vs 2.05", NOT_ADOPTED, "experiments/stock_potential.py"),
    ("2026-10-06", "Diagnostic", "How much prices moved after each kind of crypto AI reading (potential growth "
     "or decline)", "Next 1 day: readings of 55%+ were followed by +0.30% / +0.17% on average (2022-2024 / "
     "2025-2026), 48% or less by -0.07% / -0.12%; next 4 hours: about no difference in 2025-2026", FINDING,
     "cryptoai/outlook.py"),
    ("2026-10-06", "US stocks", "One top pick a month from the buy list (by 5-day AI, 3-month potential, strongest "
     "trend or steadiest)", "Steadiest chosen (2021-2022: +5% vs -16% for the list) but lagged the list in 2023-2026 "
     "(+172% vs +921%); no one-stock rule beat the list. Shown as 'if you only buy one'", PARTLY,
     "experiments/stock_top_pick.py"),
    ("2026-10-06", "Execution", "Buy with a limit order 1-3% below the price (wait 1-7 days, else market) instead "
     "of at market", "Every plan paid more on average: crypto +0.00% to +0.34%, stocks +0.16% to +0.97%", NOT_ADOPTED,
     "experiments/entry_price.py"),
    ("2026-10-06", "Diagnostic", "Which patterns the AI relies on (2025-2026, unseen)", "Mostly Bitcoin's "
     "short-term moves reversing; 15-minute patterns add ~nothing on new data", FINDING, "cryptoai/explain.py"),
], columns=["Date", "Area", "Idea", "Result (data not used for choosing)", "Verdict", "Details"])
