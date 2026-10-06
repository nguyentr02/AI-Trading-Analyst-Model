"""What patterns the AI has learnt, and whether they held on data it never saw.

    python -m cryptoai explain

For the "Next 1 day" model (4h candles): a copy is trained only on data before HOLDOUT_START and judged on the
rest, so nothing here is graded on data the model learnt from.
1. Reliance: for each group of related inputs (trend, momentum, overbought/oversold, ...), the drop in AUC when
   that group's values are shuffled across rows (grouped permutation importance). A big drop = the AI leans on it.
2. What each pattern shows: for the inputs the AI relies on most, the holdout rows are split into fifths by the
   input's value; per fifth, the AI's average P(up) and how often the price really was up 1 day later.
Saved to models/explain_4h.json for the dashboard; the live service refreshes it after each daily close.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config, model

NAME = "4h"
HOLDOUT_START = pd.Timestamp("2025-01-01", tz="UTC")
OUT = config.MODEL_DIR / f"explain_{NAME}.json"
REPEATS = 3

GROUPS = {
    "Longer trend": ["dist_ema20", "dist_ema50", "dist_ema200", "ret_50", "ret_100", "ema50_slope", "from_high_100"],
    "Recent momentum": ["ret_1", "ret_3", "ret_6", "ret_12", "ret_24", "macd", "macd_hist"],
    "Overbought / oversold": ["rsi_14", "rsi_7", "bb_pctb", "range_pos", "h_rsi_14"],
    "Volatility": ["bb_width", "atr_pct", "vol_10", "vol_50", "vol_ratio", "m_rv_4h", "m_rv_24h", "m_rv_ratio",
                   "m_semivar_ratio", "m_skew_24h"],
    "Volume": ["volume_z", "m_vol_last_hour_share", "m_taker_last_hour"],
    "15-minute and 1-hour patterns": ["m_ret_15m", "m_ret_30m", "m_ret_1h", "m_ret_2h", "m_rsi_14", "m_efficiency_4h",
                                      "m_efficiency_24h", "m_up_share_4h", "m_from_high_4h", "m_from_low_4h",
                                      "h_dist_ema20", "h_macd_hist", "candle_body"],
    "Bitcoin and the market": ["btc_ret_1", "btc_ret_6", "btc_ret_24", "btc_dist_ema50", "btc_dist_ema200",
                               "btc_vol_50", "btc_rsi_14", "rel_btc_24", "rel_btc_6", "xs_rank_24"],
    "Time of day and week": ["dow", "hour"],
}

LABELS = {
    "dist_ema20": "price vs its 20-candle average", "dist_ema50": "price vs its 50-candle (8-day) average",
    "dist_ema200": "price vs its 200-candle (33-day) average", "ret_50": "8-day return", "ret_100": "17-day return",
    "ema50_slope": "slope of the 8-day average", "from_high_100": "distance below the 17-day high",
    "ret_1": "last 4h return", "ret_3": "12h return", "ret_6": "1-day return", "ret_12": "2-day return",
    "ret_24": "4-day return", "macd": "MACD", "macd_hist": "MACD histogram", "rsi_14": "RSI (14 candles)",
    "rsi_7": "RSI (7 candles)", "bb_pctb": "position in the Bollinger bands", "range_pos": "position in the recent range",
    "h_rsi_14": "1-hour RSI", "bb_width": "Bollinger band width", "atr_pct": "average true range",
    "vol_10": "volatility, last 10 candles", "vol_50": "volatility, last 50 candles", "vol_ratio": "volatility rising vs falling",
    "m_rv_4h": "15-minute volatility, last 4h", "m_rv_24h": "15-minute volatility, last day",
    "m_rv_ratio": "volatility today vs recent", "m_semivar_ratio": "downside vs upside volatility",
    "m_skew_24h": "lopsided moves, last day", "volume_z": "volume vs normal",
    "m_vol_last_hour_share": "share of the candle's volume in its last hour", "m_taker_last_hour": "buyers vs sellers, last hour",
    "m_ret_15m": "last 15-minute move", "m_ret_30m": "last 30-minute move", "m_ret_1h": "last 1-hour move",
    "m_ret_2h": "last 2-hour move", "m_rsi_14": "15-minute RSI", "m_efficiency_4h": "how straight the last 4h moved",
    "m_efficiency_24h": "how straight the last day moved", "m_up_share_4h": "share of rising 15-minute bars, last 4h",
    "m_from_high_4h": "distance below the 4h high", "m_from_low_4h": "distance above the 4h low",
    "h_dist_ema20": "price vs its 20-hour average", "h_macd_hist": "1-hour MACD histogram", "candle_body": "candle body size",
    "btc_ret_1": "Bitcoin's last 4h move", "btc_ret_6": "Bitcoin's 1-day move", "btc_ret_24": "Bitcoin's 4-day move",
    "btc_dist_ema50": "Bitcoin vs its 8-day average", "btc_dist_ema200": "Bitcoin vs its 33-day average",
    "btc_vol_50": "Bitcoin's volatility", "btc_rsi_14": "Bitcoin's RSI", "rel_btc_24": "strength vs Bitcoin, 4 days",
    "rel_btc_6": "strength vs Bitcoin, 1 day", "xs_rank_24": "rank among the 4 coins, 4-day return",
    "dow": "day of the week", "hour": "hour of the day",
}


DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CATEGORIES = {"dow": lambda v: DAYS[v], "hour": lambda v: f"the {v:02d}:00-{(v + 4) % 24:02d}:00 UTC candle"}


def compute(top=6):
    ds = model.dataset(NAME, refresh=False)
    cols = model.feature_cols(ds)
    horizon = config.MODELS[NAME]["horizon"]
    step = pd.Timedelta(config.MODELS[NAME]["timeframe"])
    labeled = ds.dropna(subset=["y"])
    train = labeled[labeled.index <= HOLDOUT_START - (horizon + 1) * step]
    test = labeled[labeled.index >= HOLDOUT_START]
    m = model._new_model().fit(train[cols], train["y"])
    base_p = m.predict_proba(test[cols])[:, 1]
    base_auc = roc_auc_score(test["y"], base_p)

    rng = np.random.default_rng(0)
    reliance = []
    for group, feats in GROUPS.items():
        feats = [f for f in feats if f in cols]
        if not feats:
            continue
        drops = []
        for _ in range(REPEATS):
            shuffled = test[cols].copy()
            idx = rng.permutation(len(shuffled))
            shuffled[feats] = shuffled[feats].to_numpy()[idx]  # shuffle the group's rows together
            drops.append(base_auc - roc_auc_score(test["y"], m.predict_proba(shuffled)[:, 1]))
        reliance.append({"group": group, "auc_drop": float(np.mean(drops)), "inputs": len(feats)})
    reliance.sort(key=lambda r: -r["auc_drop"])

    # single inputs, to pick the ones to explain
    single = []
    for f in cols:
        shuffled = test[cols].copy()
        shuffled[f] = rng.permutation(shuffled[f].to_numpy())
        single.append((base_auc - roc_auc_score(test["y"], m.predict_proba(shuffled)[:, 1]), f))
    single.sort(reverse=True)

    patterns = []
    for drop, f in single[:top]:
        x = test[f]
        categorical = f in CATEGORIES
        bins = x.astype(int) if categorical else pd.qcut(x.rank(method="first"), 5, labels=False)
        rows = []
        for b in sorted(bins.unique()):
            mask = (bins == b).to_numpy()
            name = CATEGORIES[f](int(b)) if categorical else f"fifth {b + 1}"
            rows.append({"bucket": name, "from": float(x[mask].min()), "to": float(x[mask].max()),
                         "ai_p_up": float(base_p[mask].mean()), "actual_up": float(test["y"][mask].mean()),
                         "rows": int(mask.sum())})
        patterns.append({"feature": f, "label": LABELS.get(f, f), "auc_drop": float(drop), "categorical": categorical,
                         "group": next((g for g, fs in GROUPS.items() if f in fs), "Other"), "buckets": rows})
    out = {"model": NAME, "computed": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
           "train_until": str(train.index.max()), "holdout_from": str(HOLDOUT_START), "holdout_rows": len(test),
           "holdout_auc": float(base_auc), "base_rate": float(test["y"].mean()), "reliance": reliance,
           "patterns": patterns}
    with config.atomic(OUT) as tmp:
        tmp.write_text(json.dumps(out, indent=2))
    return out


def load():
    try:
        return json.loads(OUT.read_text())
    except (OSError, ValueError):
        return None


def sentence(p):
    """Plain-English summary of one pattern: lowest vs highest fifth, or the best and worst day/hour."""
    b = p["buckets"]
    if p.get("categorical"):
        best, worst = max(b, key=lambda r: r["ai_p_up"]), min(b, key=lambda r: r["ai_p_up"])
        return (f"The AI is most hopeful on {best['bucket']} ({best['ai_p_up']:.0%} expected, {best['actual_up']:.0%} "
                f"real) and least on {worst['bucket']} ({worst['ai_p_up']:.0%} expected, {worst['actual_up']:.0%} real).")
    lo, hi = b[0], b[-1]
    return (f"When {p['label']} is in its lowest fifth, the AI expects a rise {lo['ai_p_up']:.0%} of the time and the "
            f"price really rose {lo['actual_up']:.0%} of the time; in its highest fifth, {hi['ai_p_up']:.0%} expected, "
            f"{hi['actual_up']:.0%} real.")
