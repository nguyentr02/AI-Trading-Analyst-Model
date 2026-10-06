"""Dashboard. Run with:  .venv\\Scripts\\streamlit run app.py"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from cryptoai import (advisor, altcoins, auth, backtest, config, data, droprisk, explain, live, market, metrics, model,
                      notify, outlook, paper, portfolio, preview, research_log, signals, stockai, stockpaper, stocks,
                      whales)

ASSETS = config.ROOT / "assets"
st.set_page_config(page_title="Crypto AI", page_icon=str(ASSETS / "icon.svg"), layout="wide")
st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "icon.svg"), size="large")

# ---------------- sign-in for online access ----------------
# Keeps the refresh token in the browser (localStorage), so reloading the page doesn't sign you out.
_TOKEN_STORE = st.components.v2.component(
    "cryptoai_token_store",
    html="<div></div>",
    js="""
export default function (component) {
  const { data, setStateValue } = component
  const KEY = "cryptoai_refresh_token"
  if (!data) return
  if (data.action === "set" && data.token) localStorage.setItem(KEY, data.token)
  if (data.action === "clear") localStorage.removeItem(KEY)
  if (data.action === "read") setStateValue("stored", localStorage.getItem(KEY) || "none")
}
""",
)


def token_store(action, token=None):
    return _TOKEN_STORE(key="token_store", data={"action": action, "token": token}, height=0,
                        on_stored_change=lambda: None)


def online_request():
    """True when the page came through the Cloudflare tunnel (it adds a CF-Connecting-IP header)."""
    headers = st.context.headers
    return bool(headers.get("Cf-Connecting-Ip") or headers.get("cf-connecting-ip"))


def start_session(pair):
    st.session_state["access_token"], st.session_state["refresh_token"] = pair


def require_login():
    """Sign-in for online access: username + password, then a 15-minute access token renewed with a 7-day
    refresh token (see cryptoai/auth.py). Opening the dashboard on this PC or the home network skips it.

    The browser-token component is mounted exactly once per run (Streamlit refuses a repeated key), so each
    branch below makes its one token_store call and then returns or stops.
    """
    if not online_request():
        return
    if not auth.is_set():
        st.error("Online access is switched off: no login has been set on the PC running the AI.")
        st.stop()
    s = st.session_state

    # 0. Sign-out was requested on the Sign out page: revoke, forget, and clear the browser's copy.
    if s.pop("sign_out", False):
        auth.revoke(s.pop("refresh_token", None))
        s.pop("access_token", None)
        s["token_checked"] = True
        token_store("clear")
        st.title("Signed out")
        st.caption("Reload the page to sign in again.")
        st.stop()
    # 1. A valid access token in this session.
    if auth.verify_access(s.get("access_token")):
        token_store("set", s["refresh_token"])
        return
    # 2. Access token expired: renew it with the refresh token.
    if s.get("refresh_token"):
        pair = auth.refresh(s.pop("refresh_token"))
        s.pop("access_token", None)
        if pair:
            start_session(pair)
            token_store("set", pair[1])
            return
    # 3. A new visit: look for a refresh token saved in this browser (this run's one mount).
    if not s.get("token_checked"):
        stored = token_store("read").stored
        if stored is None:
            st.caption("Checking sign-in…")
            st.stop()
        s["token_checked"] = True
        pair = auth.refresh(stored) if stored != "none" else None
        if pair:
            start_session(pair)
            st.rerun()
    else:
        token_store("clear")  # no valid tokens: drop any stale copy kept in the browser
    # 4. Sign in.
    st.title("Crypto AI")
    with st.form("sign_in", width=420):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", type="primary"):
            if auth.check_login(username, password):
                start_session(auth.issue_tokens(username))
                st.rerun()
            st.error("Wrong username or password.")
    st.stop()


def page_sign_out():
    # require_login() has already mounted the token component this run; let it do the sign-out on a rerun.
    st.session_state["sign_out"] = True
    st.rerun()


require_login()

# Coinbase palette, matching .streamlit/config.toml. Plotly needs explicit colors per theme.
if st.context.theme.type == "dark":
    BLUE, GREY, UP, DOWN = "#578BFA", "#8A919E", "#27AD75", "#F0616D"
else:
    BLUE, GREY, UP, DOWN = "#0052FF", "#8A919E", "#098551", "#CF202F"
GRID = "rgba(138,145,158,0.18)"
# How far to trust each model, from walk-forward and permutation tests (see docs/research/).
TRUST = {
    "4h_next": "Most accurate, but it changes often: trading every change loses to fees, so use it to time "
               "entries and exits. ",
    "1d": "Weakest: no better than chance in 2023-2024, so treat it with extra caution. ",
}


def style(fig, height):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0),
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


# Explanations shown when hovering a table's column header. One meaning per name across the whole dashboard.
COLUMN_HELP = {
    # prices and markets
    "#": "Rank by market cap (CoinGecko).",
    "Coin": "The coin (or pair) this row is about.",
    "Pair": "The Binance trading pair.",
    "Stock": "The stock or ETF this row is about.",
    "Price": "Last traded price.",
    "Price (Binance)": "Last price of the stock's perpetual futures on Binance (trades 24/7).",
    "1h": "Price change over the last hour.",
    "24h": "Price change over the last 24 hours.",
    "7d": "Price change over the last 7 days.",
    "30d": "Price change over the last 30 days.",
    "90d": "Price change over the last 90 days.",
    "1 month": "Price change over the last month (about 21 trading days).",
    "1 year": "Price change over the last year.",
    "Last 7 days": "Hourly price over the last 7 days.",
    "Volume 24h": "Value traded in the last 24 hours.",
    "Market cap": "Price x coins in circulation.",
    "24h range": "Lowest and highest price in the last 24 hours.",
    "Fully diluted value": "Price x the maximum number of coins that will ever exist.",
    "Circulating supply": "Coins in circulation now.",
    "Supply issued": "Circulating supply as a share of the maximum supply.",
    "All-time high": "The highest price ever.",
    "From all-time high": "How far the price is below its all-time high.",
    "From 1y high": "How far the price is below its highest close of the last year.",
    "Typical daily move": "Standard deviation of daily returns over 30 days: how much the price usually moves in a day.",
    "Typical yearly swing": "Annualised volatility over the last year: how much the price usually moves in a year.",
    "Days with 10%+ moves (90d)": "Days in the last 90 with a rise or fall of 10% or more.",
    "Worst fall (10y)": "The deepest fall from a peak in the last 10 years.",
    # trend and signals
    "Trend": "Daily close vs its 200-day average: above = long-term uptrend.",
    "Trend rule": "The 50-day trend rule: hold while the daily close is above its 50-day average, cash below.",
    "Long-term trend": "Daily close vs its 200-day average.",
    "Short-term trend": "Daily close vs its 50-day average.",
    "vs 200-day": "How far the daily close is above (+) or below (-) its 200-day average.",
    "AI next 1 day": "The AI's call and chance of a rise over the next day (next-1-day model).",
    "AI: P(up, 5 days)": "The stock AI's chance the price is higher 5 trading days from now. Weak: see Stock signals.",
    "New signal": "The signal the model switched to.",
    "Prediction": "Which model: next 4 hours, next 1 day or next 3 days.",
    "P(up)": "The model's chance the price is higher at the end of its window.",
    # trades and accounts
    "Side": "BUY or SELL.",
    "Amount": "Number of coins or shares.",
    "Avg cost": "Average price paid per coin, including fees.",
    "Value": "What the holding is worth at today's price.",
    "P&L": "Profit or loss in dollars: value now minus what it cost.",
    "P&L %": "Profit or loss as a share of what it cost.",
    "Invested": "Share of the sleeve held in the coin rather than cash.",
    "Total (USDT)": "Value of the trade.",
    "Total (USD)": "Value of the trade.",
    "Fee": "Exchange fee paid on the trade (0.1%).",
    "Reason": "Why the account traded.",
    "Account": "The paper-trading account.",
    "Total P&L": "Balance now minus the starting balance.",
    "Realised": "Profit or loss locked in by sales, after fees.",
    "Unrealised": "Profit or loss on coins still held, at today's price vs what they cost.",
    "Fees paid": "All exchange fees paid so far (slippage is in the fill prices).",
    "Trades": "Number of trades so far.",
    "vs buy & hold": "This account's balance minus the buy & hold benchmark's.",
    "Invested now": "Share of the account held in coins rather than cash.",
    "Advice": "What the AI suggests for this holding.",
    "Why": "The reason for the advice.",
    "Share": "Share of your portfolio in this holding.",
    # research and testing
    "Strategy": "The trading approach being tested.",
    "Period": "The years tested. Rules are chosen on the earlier period and checked once on the later one.",
    "Return": "Total gain or loss over the period, after fees.",
    "Sharpe": "Return per unit of risk (annualised). Higher is better; 1 is good.",
    "Worst fall": "The deepest fall from a peak during the period (max drawdown).",
    "Hold, Sharpe 2023-26": "Sharpe of simply holding the stock in 2023-2026.",
    "Best rule, Sharpe 2023-26": "Sharpe of the best trend rule (chosen on 2016-2022) in 2023-2026.",
    "AUC": "How well the model ranks rises above falls on data it didn't train on. 0.5 is a coin flip.",
    "Accuracy": "Share of correct up/down calls on data the model didn't train on.",
    "Training rows": "Candles the model learnt from.",
    "Kept": "Whether the new model was kept (it must beat a coin flip on unseen data).",
    "Data up to": "The last candle the model trained on.",
    "Date": "When the test was run.",
    "Area": "What kind of idea was tested.",
    "Idea": "What was tested.",
    "Result (data not used for choosing)": "The result on data the idea was not chosen or tuned on.",
    "Verdict": "Adopted, not adopted, partly, or a finding.",
    "Details": "Where the code and full results are.",
    "NEAR $": "$1,000 in NEAR at the start of 2025, at the end of the test.",
    "ZEC $": "$1,000 in ZEC at the start of 2025, at the end of the test.",
    "Sharpe (both)": "Sharpe of NEAR and ZEC together.",
    "AI accuracy (AUC)": "The AI's AUC on NEAR and ZEC (0.5 is a coin flip).",
    "50-day rule on": "Which coins the 50-day trend rule traded.",
    "Sharpe 2022-2024": "Sharpe on 2022-2024 (where choices were made).",
    "Sharpe 2025-2026": "Sharpe on 2025-2026 (the one-time check).",
    "Return 2025-2026": "Total return on 2025-2026, after fees.",
    "Event": "What kind of whale event.",
    "Size": "Dollar size of the trade or liquidations.",
    "Detail": "More about the event.",
    "Coin pair": "A Binance pair, like BTC/USDT.",
    # backtest page
    "Year": "Calendar year of the test.",
    "AI strategy": "The AI strategy's return that year, after fees.",
    "AI return": "The AI strategy's total return, after fees.",
    "AI Sharpe": "The AI strategy's return per unit of risk. Higher is better.",
    "AI worst drop": "The AI strategy's deepest fall from a peak.",
    "Buy & hold": "Buying at the start and holding to the end, the same period.",
    "Hold return": "Buy & hold's total return over the same period.",
    "Hold Sharpe": "Buy & hold's return per unit of risk over the same period.",
    "Hold worst drop": "Buy & hold's deepest fall from a peak over the same period.",
    "Chance Sharpe > 0": "Probabilistic Sharpe ratio: the chance the strategy's true Sharpe is above zero.",
    "Chance beats hold": "The chance the strategy's true Sharpe is above buy & hold's.",
    "Fee per trade": "The exchange fee assumed per buy or sell (Binance's is 0.1%).",
    # buy for hold
    "Do now": "BUY: in the tested buy-for-hold list (above its 200-day average). AVOID for now: below it.",
    "Price now": "Live Binance price (stock perpetual futures).",
    "Invest": "Your budget split equally across the stocks marked BUY.",
    "Shares": "Shares that amount buys at the price now (fractional).",
    "Each weekly step": "Buy a quarter of it each week for 4 weeks, instead of all at once.",
    "Sell if below (200-day avg)": "A floor, not a target: its 200-day average price. Sell only if the stock is below this at a monthly check. It rises as the stock rises.",
    "Room above the sell-if-below line": "How far the price is above its floor now. Negative: below it (not a buy).",
}
COLUMN_HELP_PREFIX = {"Time (": "Date and time of the event, in the time zone chosen at the top.",
                      "When (": "When the signal changed, in the time zone chosen at the top.",
                      "Trained (": "When the model was retrained, in the time zone chosen at the top.",
                      "AI next": "The AI's view for this coin over that window: potential growth, potential decline or no clear direction."}


def _help_for(col):
    if col in COLUMN_HELP:
        return COLUMN_HELP[col]
    return next((h for p, h in COLUMN_HELP_PREFIX.items() if str(col).startswith(p)), None)


def table(data, column_config=None, **kwargs):
    """st.dataframe with an explanation on every column header (shown on hover), from COLUMN_HELP."""
    df = data.data if hasattr(data, "data") and hasattr(data, "to_html") else data  # a pandas Styler
    config = dict(column_config or {})
    for col in df.columns:
        text = _help_for(col)
        if not text:
            continue
        cfg = config.get(col)
        if cfg is None:
            config[col] = st.column_config.Column(help=text)
        elif isinstance(cfg, dict) and not cfg.get("help"):
            config[col] = {**cfg, "help": text}
    return st.dataframe(data, column_config=config, **kwargs)


@st.cache_data(ttl=300, max_entries=8, show_spinner="Loading AI signals…")
def get_signals(tf, model_version=None):
    """`model_version` (the model's training time) makes a retrained model show up immediately.

    Uses the data the live service keeps up to date, rather than downloading it again here. If the service
    isn't running, it downloads the latest candles itself.
    """
    return signals.current(tf, refresh=not service_running())


def service_running():
    s = live.load_status()
    if s is None:
        return False
    return (pd.Timestamp.now(tz="UTC") - pd.Timestamp(s["updated"])).total_seconds() < 180


def model_version(tf):
    m = model.load_metrics(tf)
    return m["trained_at"] if m else None


@st.cache_resource
def live_feed():
    """One Binance WebSocket connection shared by every browser tab."""
    return market.LiveFeed()


@st.cache_data(ttl=10, show_spinner=False)
def get_tickers():
    """REST fallback, used only until the WebSocket has delivered its first update."""
    return market.tickers()


@st.cache_data(ttl=60, show_spinner=False)
def get_coingecko():
    return market.coingecko()


def usd(v):
    """Compact dollar amount: $1.73T, $26.5B, $902.7M."""
    if v is None or pd.isna(v):
        return "–"
    for div, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= div:
            return f"${v / div:,.2f}{unit}"
    return f"${v:,.2f}"


def amount(v):
    if v is None or pd.isna(v):
        return "–"
    for div, unit in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= div:
            return f"{v / div:,.2f}{unit}"
    return f"{v:,.2f}"


# Keep the timeframe and symbol selection when moving between pages.
st.session_state.setdefault("model", "4h")
st.session_state.setdefault("chart_interval", "4h")
st.session_state.setdefault("chart_ind", ["MA", "Volume", "AI signal"])
st.session_state.setdefault("sym", config.SYMBOLS[0])
st.session_state.setdefault("alt_chart_interval", "4h")
st.session_state.setdefault("alt_chart_ind", ["MA", "Volume"])
st.session_state.setdefault("stock_chart_interval", "1D")
st.session_state.setdefault("stock_chart_ind", ["MA", "Volume"])
for k in ("model", "sym", "chart_interval", "chart_ind", "alt_chart_interval", "alt_chart_ind", "stock_chart_interval",
          "stock_chart_ind"):
    st.session_state[k] = st.session_state[k]

# Time zone for every time shown (data and logs stay in UTC; only the display changes).
TIMEZONES = {"UTC": "UTC", "Vietnam": "Asia/Ho_Chi_Minh"}
TZ_LABELS = {"UTC": "UTC", "Vietnam": "Vietnam time"}


def tz_label():
    return TZ_LABELS[st.session_state.get("tz") or "UTC"]


def local(t):
    """A UTC time (Timestamp, ISO string, Series or DatetimeIndex) in the chosen time zone."""
    zone = TIMEZONES[st.session_state.get("tz") or "UTC"]
    if isinstance(t, pd.Series):
        t = pd.to_datetime(t, utc=True, format="ISO8601") if t.dtype == object else t
        return (t.dt.tz_localize("UTC") if t.dt.tz is None else t).dt.tz_convert(zone)
    if isinstance(t, pd.DatetimeIndex):
        return (t.tz_localize("UTC") if t.tz is None else t).tz_convert(zone)
    t = pd.Timestamp(t)
    return (t.tz_localize("UTC") if t.tzinfo is None else t).tz_convert(zone)


def when(t, fmt="%b %d, %H:%M"):
    """A time formatted for display with its zone, e.g. 'Oct 06, 14:00 Vietnam time'."""
    return f"{local(t):{fmt}} {tz_label()}"


# ---------------- market (live) ----------------
COIN_NAMES = {"BTC/USDT": "Bitcoin", "ETH/USDT": "Ethereum", "BNB/USDT": "BNB", "SOL/USDT": "Solana"}
def coin_label(sym):
    """'Bitcoin  BTC', or just 'BNB' when the name and ticker are the same."""
    name, tick = COIN_NAMES.get(sym, coin(sym)), coin(sym)
    return tick if name == tick else f"{name}  {tick}"


COIN_COLORS = {"BTC/USDT": "#F7931A", "ETH/USDT": "#627EEA", "BNB/USDT": "#F3BA2F", "SOL/USDT": "#9945FF"}


@st.cache_data(ttl=300, show_spinner=False)
def week_closes(sym):
    """Hourly closes over the last 7 days, for sparklines."""
    return recent_candles(sym, "1h", limit=168)["close"].tolist()


def page_market():
    st.title("Market")
    st.caption("Live prices from Binance, market data from CoinGecko.")
    live_market()
    whale_panel()
    supply_table()


@st.fragment(run_every=30)
def whale_panel():
    """Whale activity from the live service's whale watch (cryptoai/whales.py): big trades, liquidations, leverage."""
    st.subheader("Whale activity")
    s = whales.load_status()
    if s is None or (pd.Timestamp.now(tz="UTC") - pd.Timestamp(s["updated"])).total_seconds() > 15 * 60:
        st.caption("Whale watch not running. It starts with the live learning service.")
        return
    cards = st.container(horizontal=True, gap="medium")
    for sym in config.SYMBOLS:
        t = s["trades_1h"].get(sym, {})
        buy, sell = t.get("BUY", 0), t.get("SELL", 0)
        oi = s["oi"].get(sym, {})
        net = buy - sell
        cards.metric(f"{coin(sym)} whales, last hour",
                     f"{'+' if net >= 0 else '−'}{usd(abs(net)) if net else '$0'} net",
                     f"open interest {oi['change_1h']:+.1%} in 1h" if oi else None, border=True, delta_color="off",
                     help=f"Spot market buys minus sells of ${whales.BIG_TRADE[sym] / 1e6:g}M or more in one trade "
                          f"(bought {usd(buy) if buy else '$0'}, sold {usd(sell) if sell else '$0'}), and the change "
                          "in futures open interest (leveraged positions) over the last hour.")
    liq = s["liquidations_1h"]
    st.caption(f"Liquidations in the last hour: longs {usd(liq['LONG']) if liq['LONG'] else '$0'}, shorts "
               f"{usd(liq['SHORT']) if liq['SHORT'] else '$0'} (Binance reports at most one per coin per second, so "
               f"these are undercounts). Alerts go out on bursts of whale trades, liquidation cascades over "
               f"{usd(whales.LIQ_ALERT)} in 5 minutes, and unusual 1-hour open interest jumps. Whale data mostly "
               "signals volatility ahead, not direction.")
    ev = whales.load_events(500)
    ev = ev[ev["kind"].isin(["trade", "cascade", "oi"]) | (ev["usd"] >= 1_000_000)].tail(12).iloc[::-1]
    if len(ev):
        kinds = {"trade": "Big trade", "liquidation": "Liquidation", "cascade": "Liquidation cascade",
                 "oi": "Open interest jump"}
        table(pd.DataFrame({f"Time ({tz_label()})": local(ev["time"]).dt.strftime("%b %d, %H:%M:%S"),
                                   "Event": ev["kind"].map(kinds), "Coin": ev["symbol"].str.replace("/USDT", ""),
                                   "Side": ev["side"], "Size": ev["usd"].map(lambda v: usd(v) if v else ""),
                                   "Detail": ev["note"].fillna("")}),
                     hide_index=True, width="stretch", alt="Recent whale events")


@st.fragment(run_every=config.LIVE_REFRESH)
def live_market():
    feed = live_feed()
    tk = feed.tickers()
    if tk is None:
        try:
            tk = get_tickers()
        except Exception as e:
            st.error(f"Could not reach Binance: {e}")
            return
    if feed.error:
        st.warning(f"Live feed reconnecting, prices may be a few seconds old: {feed.error}")
    try:
        cg = get_coingecko()
    except Exception as e:
        cg = pd.DataFrame(columns=["symbol"])
        st.warning(f"CoinGecko unavailable, market cap data hidden: {e}")
    df = tk.merge(cg, on="symbol", how="left")
    col = lambda name: df.get(name, pd.Series(index=df.index, dtype=float))  # CoinGecko columns may be missing
    dollars = lambda v: "–" if pd.isna(v) else f"${v:,.2f}"

    # Coin cards: price, 24h change and a 7-day sparkline, like the top of an exchange's markets page.
    cards = st.container(horizontal=True, gap="medium")
    for r in df.itertuples():
        try:
            spark = week_closes(r.symbol)
        except Exception:
            spark = None
        cards.metric(coin_label(r.symbol).replace("  ", " · "), f"${r.price:,.2f}", f"{r.change_24h:+.2%}",
                     border=True, chart_data=spark, chart_type="area",
                     help="Price now, change over 24 hours, and the last 7 days of hourly prices.")

    st.subheader("Markets")
    markets = pd.DataFrame({
        "#": col("rank"),
        "Coin": [coin_label(s) for s in df["symbol"]],
        "Price": df["price"].map(dollars),
        "1h": col("change_1h"),
        "24h": df["change_24h"],
        "7d": col("change_7d"),
        "Last 7 days": [week_closes(s) for s in df["symbol"]],
        "Volume 24h": col("total_volume_usd").map(usd),
        "Market cap": col("market_cap").map(usd),
    })
    colored_table(markets, ["1h", "24h", "7d"], "Price, change, 7-day trend, volume and market cap for each coin",
                  column_config={
                      "#": st.column_config.NumberColumn(width="small", format="%d"),
                      "Last 7 days": st.column_config.LineChartColumn(width="medium"),
                      "Volume 24h": st.column_config.TextColumn(help="All exchanges, from CoinGecko"),
                  })

    st.caption(f"Prices stream from Binance and redraw every {config.LIVE_REFRESH}s. Market cap, volume and "
               f"1h/7d change come from CoinGecko every 60s. Last update {when(tk['updated'].max(), '%H:%M:%S')}.")


@st.fragment(run_every=60)
def supply_table():
    """Slow-changing CoinGecko data, redrawn once a minute rather than with every price tick."""
    try:
        df = get_tickers().merge(get_coingecko(), on="symbol", how="left")
    except Exception as e:
        st.caption(f"Supply data unavailable: {e}")
        return
    col = lambda name: df.get(name, pd.Series(index=df.index, dtype=float))
    dollars = lambda v: "–" if pd.isna(v) else f"${v:,.2f}"
    st.subheader("Supply and valuation")
    supply = pd.DataFrame({
        "Coin": [coin_label(s) for s in df["symbol"]],
        "24h range": [f"{dollars(lo)} – {dollars(hi)}" for lo, hi in zip(df["low_24h"], df["high_24h"])],
        "Fully diluted value": col("fdv").map(usd),
        "Circulating supply": col("circulating_supply").map(amount),
        "Supply issued": col("circulating_supply") / col("max_supply"),
        "All-time high": col("ath").map(dollars),
        "From all-time high": col("from_ath"),
    })
    colored_table(supply, ["From all-time high"], "24h range, valuation, supply and all-time high for each coin",
                  column_config={"Supply issued": st.column_config.ProgressColumn(
                      format="percent", min_value=0, max_value=1,
                      help="Circulating supply as a share of the maximum supply. Blank when there is no maximum.")})


def colored_table(df, pct, alt, column_config=None):
    """Dataframe with percentage columns formatted and colored green (up) or red (down)."""
    table(
        df.style.format({k: "{:+.2%}" for k in pct}, na_rep="–")
        .map(lambda v: f"color: {UP}" if pd.notna(v) and v > 0 else f"color: {DOWN}" if pd.notna(v) and v < 0 else "",
             subset=pct),
        hide_index=True, width="stretch", alt=alt, column_config=column_config,
    )


# ---------------- signals ----------------
SIGNAL_STYLE = {  # badge color and icon for each signal
    "BULLISH": ("green", ":material/trending_up:"),
    "BEARISH": ("red", ":material/trending_down:"),
    "NEUTRAL": ("gray", ":material/trending_flat:"),
}
SIGNAL_ARROW = {"BULLISH": "▲ Potential growth", "BEARISH": "▼ Potential decline", "NEUTRAL": "● No clear direction"}
SIGNAL_TEXT = {"BULLISH": "Potential growth", "BEARISH": "Potential decline", "NEUTRAL": "No clear direction"}
WINDOW_TEXT = {"4h_next": "the next 4 hours", "4h": "the next day", "1d": "the next 3 days"}


def page_signals():
    st.title("Signals")
    service_badge()
    signal_board()


def service_badge():
    """Whether the always-on learning service is running, from the status file it updates every minute."""
    s = live.load_status()
    if s is None:
        st.badge("Live learning service not running", icon=":material/cloud_off:", color="gray",
                 help="Start it with live.bat, or set it up to start automatically with setup_autostart.ps1.")
        return
    age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(s["updated"])).total_seconds()
    if age > 180:
        st.badge(f"Live learning service offline since {when(s['updated'])}",
                 icon=":material/cloud_off:", color="red")
    elif s["state"] == "listening":
        last = max(s.get(f"last_learn_{n}", "") for n in config.MODELS)
        st.badge(f"Live learning: listening for candle closes · last learned {when(last, '%H:%M')}",
                 icon=":material/sensors:", color="green",
                 help="The service retrains and updates signals a few minutes after every 4h and 1d candle close.")
    else:
        st.badge(f"Live learning: {s['state']}", icon=":material/sync:", color="orange")


def signal_row(name, sig, live_now=None, look=None):
    """One prediction inside a coin card: potential growth or decline, how much (from what really happened after
    similar readings), and the live provisional reading."""
    spec = config.MODELS[name]
    color, icon = SIGNAL_STYLE[sig.signal]
    line = st.container(horizontal=True, vertical_alignment="center", gap="small")
    line.markdown(spec["label"], width="content")
    line.badge(SIGNAL_TEXT[sig.signal], icon=icon, color=color)
    st.progress(float(sig.prob_up), text=f"Chance of a rise {sig.prob_up:.0%}")
    b = outlook.describe(name, float(sig.prob_up), look)
    if b:
        st.caption(f"How much: after readings like this (2022-2026, data the AI never trained on) the price moved "
                   f"**{b['mean']:+.2%}** on average over {WINDOW_TEXT[name]}, typically {b['p25']:+.1%} to "
                   f"{b['p75']:+.1%}, and rose {b['up_share']:.0%} of the time.",
                   help="A buy and a later sell cost about 0.2% in fees, so moves smaller than that don't pay. "
                        "'Typically' is the middle half of outcomes; a quarter were worse and a quarter better.")
    if live_now:
        arrow = {"BULLISH": ":green[▲]", "BEARISH": ":red[▼]"}.get(live_now["signal"], ":gray[●]")
        st.caption(f"Live now {arrow} {live_now['prob_up']:.1%} · {live_now['candle_progress']:.0%} into the "
                   f"{spec['timeframe']} candle", help="Provisional: the models re-run on the live price every "
                   "minute, treating the candle still forming as if it closed now. It shows where the signal is "
                   "heading; the confirmed signal above updates when the candle closes.")


DROP_STYLE = {"Normal": ("gray", ":material/check_circle:"), "Elevated": ("orange", ":material/warning:"),
              "High": ("red", ":material/trending_down:")}


def drop_row(reading):
    """The drop warning inside a coin card: chance a sharp fall comes before a rise in the next 3 days."""
    color, icon = DROP_STYLE[reading["level"]]
    line = st.container(horizontal=True, vertical_alignment="center", gap="small")
    line.markdown("Drop risk, 3 days", width="content")
    line.badge(reading["level"], icon=icon, color=color)
    st.progress(min(float(reading["p_drop"]), 1.0), text=f"P(sharp drop first) {reading['p_drop']:.0%}")


@st.fragment(run_every=60)
def signal_board():
    """Every coin's three predictions at once. Re-checks every minute so new signals appear by themselves."""
    names = [n for n in config.MODELS if model.load(n) is not None]
    if not names:
        st.error("No model yet. Run `train.bat`, or press **Retrain models** on the Backtest page.")
        return
    sigs = {n: get_signals(n, model_version(n)).set_index("symbol") for n in names}
    live_preview = preview.load()
    if live_preview and preview.age_seconds(live_preview) > 5 * 60:
        live_preview = None  # the service has stopped updating it; don't show stale "live" numbers
    drops = (droprisk.load() or {}).get("coins", {})
    look = outlook.load()

    cards = st.columns(len(config.SYMBOLS), gap="medium")
    for card, sym in zip(cards, config.SYMBOLS):
        with card.container(border=True):
            close = sigs[names[0]].loc[sym, "price"]
            name, tick = COIN_NAMES[sym], coin(sym)
            st.markdown(f"#### {name}" + (f" :gray[{tick}]" if name != tick else ""))
            st.caption(f"Last close ${close:,.2f}")
            for n in names:
                now = (live_preview or {}).get("models", {}).get(n, {}).get(sym)
                signal_row(n, sigs[n].loc[sym], now, look)
            if sym in drops:
                drop_row(drops[sym])

    with st.container(border=True):
        st.markdown("**How to read these**")
        st.caption(f"Each prediction says whether the coin has **potential growth** (chance of a rise "
                   f"{config.ENTER_PROB:.0%} or more), **potential decline** ({config.EXIT_PROB:.0%} or less), or no "
                   "clear direction, over its window. **How much** comes from what really happened after similar "
                   "readings on data the AI never trained on: the average move and its typical range. The moves "
                   "are small (tenths of a percent), so a single signal is a tilt in the odds, not a forecast; "
                   "a buy and a sell together cost about 0.2% in fees.")
        if look:
            weak = [config.MODELS[n]["label"] for n, m in look["models"].items()
                    if m["check"]["2025-2026"]["high_mean"] - m["check"]["2025-2026"]["low_mean"] < 0.001]
            if weak:
                st.caption(f"**{', '.join(weak)}:** growth and decline readings have been followed by almost the "
                           "same moves in 2025-2026 (less than 0.1% apart), so treat them as timing hints only.")
        for n in names:
            st.caption(f"**{config.MODELS[n]['label']}:** " + TRUST.get(n, "The main signal for direction.").strip())
        if drops:
            st.caption(f"**Drop risk:** the chance that, within 3 days, the price first falls about 2× the coin's "
                       f"normal daily move before rising as much (about 1 in 5 candles on average). Elevated from "
                       f"{droprisk.CAUTION_AT:.0%}: in testing, not buying then helped. High from {droprisk.SELL_AT:.0%}: "
                       "selling then cut the worst drop from -40% to -32% on 2025-2026. Single warnings are often "
                       "wrong; it helps by avoiding the worst falls.")
        st.caption("**Live now** is provisional: the models re-run on the live price every minute, as if the "
                   "forming candle closed now. Alerts and advice use the confirmed signals.")
        st.caption("Estimates with a small edge, not advice. Size positions so being wrong is affordable.")

    st.subheader("Recent signal changes")
    if not config.SIGNAL_LOG.exists():
        st.caption("No signal changes logged yet. The live learning service logs them at every candle close.")
        return
    log = pd.read_csv(config.SIGNAL_LOG)
    log = log[log["changed"] & log["timeframe"].isin(config.MODELS) & log["symbol"].isin(config.SYMBOLS)]
    log = log.tail(30).iloc[::-1]
    table(
        pd.DataFrame({
            f"When ({tz_label()})": local(pd.to_datetime(log["checked_at"], format="ISO8601", utc=True))
            .dt.strftime("%b %d, %H:%M"),
            "Coin": log["symbol"].map(coin_label),
            "Prediction": log["timeframe"].map(lambda n: config.MODELS[n]["label"]),
            "New signal": log["signal"].map(SIGNAL_ARROW),
            "P(up)": log["prob_up"].round(2),
            "Price": log["price"],
        }).style.map(lambda v: f"color: {UP}" if v.startswith("▲") else f"color: {DOWN}" if v.startswith("▼") else "",
                     subset=["New signal"]),
        hide_index=True, width="stretch", alt="Recent changes in the AI's signals",
        column_config={
            "P(up)": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1, width="medium"),
            "Price": st.column_config.NumberColumn(format="dollar"),
        },
    )


# ---------------- chart ----------------
CHART_INTERVALS = {"15m": "15m", "1h": "1h", "4h": "4h", "1D": "1d"}  # button label -> Binance interval
INDICATORS = ["MA", "Bollinger", "Volume", "AI signal"]
MA_COLORS = {7: "#F4B000", 25: "#C855E8", 99: "#8A919E"}  # Binance's default MA(7/25/99)
INITIAL_CANDLES = 150  # shown at first; zoom out or pan to see the rest


@st.cache_data(ttl=10, show_spinner=False)
def recent_candles(sym, interval, limit=400):
    """Latest candles straight from Binance, including the one still forming (like an exchange chart)."""
    rows = data.exchange().fetch_ohlcv(sym, interval, limit=limit)
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df.set_index("time")


@st.cache_data(ttl=300, max_entries=6, show_spinner=False)
def ai_history(name, version):
    """The model's P(up) for recent closed candles of every coin. `version` refreshes it after a retrain."""
    tf = config.MODELS[name]["timeframe"]
    raw = {s: df.iloc[-(400 + 250):] for s, df in data.closed(tf, refresh=False).items()}
    return model.predict_history(raw, name)


def coin(sym):
    return sym.split("/")[0]


def price_text(x):
    """A price with enough decimals for its size: 85,592.10 / 2.1340 / 0.006327."""
    decimals = 2 if x >= 100 else 4 if x >= 1 else max(4, 3 - int(np.floor(np.log10(x)))) if x > 0 else 2
    return f"{x:,.{decimals}f}"


def page_chart():
    bar = st.container(horizontal=True, vertical_alignment="bottom", gap="medium")
    sym = bar.segmented_control("Coin", config.SYMBOLS, key="sym", required=True, format_func=coin)
    interval = bar.segmented_control("Interval", list(CHART_INTERVALS), key="chart_interval", required=True)
    shown = bar.pills("Indicators", INDICATORS, key="chart_ind", selection_mode="multi")

    main, side = st.columns([3.3, 1], gap="medium")
    with main:
        price_header(sym)
        price_chart(sym, interval, tuple(shown))
    with side:
        ai_outlook(sym)


@st.fragment(run_every=2)
def price_header(sym):
    """Exchange-style header: pair, big live price, 24h change and a stats strip."""
    tk = live_feed().tickers()
    if tk is None:
        tk = get_tickers()
    t = tk.set_index("symbol").loc[sym]
    try:
        cap = get_coingecko().set_index("symbol").loc[sym, "market_cap"]
    except Exception:
        cap = None

    head = st.container(horizontal=True, vertical_alignment="center", gap="large")
    head.metric(f"{coin(sym)} / USDT", f"${t.price:,.2f}", f"{t.change_24h:+.2%} 24h", width="content")
    # Secondary stats small, as on an exchange, so the price stays the focus.
    for label, value in (("24h high", f"${t.high_24h:,.2f}"), ("24h low", f"${t.low_24h:,.2f}"),
                         ("24h volume", usd(t.quote_volume_24h)), ("Market cap", usd(cap))):
        head.markdown(f":gray[{label}]  \n**{value}**", width="content")


@st.fragment(run_every=10)
def price_chart(sym, interval, shown):
    """Candles (with the forming one), indicators, volume and the AI's P(up), refreshed every 10 seconds."""
    tf = CHART_INTERVALS[interval]
    try:
        df = recent_candles(sym, tf)
    except Exception as e:
        st.warning(f"Could not load candles from Binance: {e}")
        return
    step = df.index[1] - df.index[0]
    x = local(df.index).tz_localize(None)  # shown in the chosen time zone
    ai_models = [n for n, s in config.MODELS.items() if s["timeframe"] == tf] if "AI signal" in shown else []

    rows = ["price"] + (["volume"] if "Volume" in shown else []) + (["ai"] if ai_models else [])
    heights = {"price": 0.64, "volume": 0.14, "ai": 0.22}
    fig = make_subplots(rows=len(rows), cols=1, shared_xaxes=True, vertical_spacing=0.035,
                        row_heights=[heights[r] / sum(heights[x] for x in rows) for r in rows])
    row = {r: i + 1 for i, r in enumerate(rows)}

    fig.add_trace(go.Candlestick(
        x=x, open=df.open, high=df.high, low=df.low, close=df.close, name=coin(sym),
        increasing=dict(line_color=UP, fillcolor=UP), decreasing=dict(line_color=DOWN, fillcolor=DOWN),
        showlegend=False), row=1, col=1)

    if "MA" in shown:
        for n, color in MA_COLORS.items():
            fig.add_trace(go.Scatter(x=x, y=df.close.rolling(n).mean(), name=f"MA({n})",
                                     line=dict(color=color, width=1.2), hovertemplate="%{y:,.2f}"), row=1, col=1)
    if "Bollinger" in shown:
        mid, sd = df.close.rolling(20).mean(), df.close.rolling(20).std()
        fig.add_trace(go.Scatter(x=x, y=mid + 2 * sd, name="BB upper", line=dict(color=BLUE, width=1),
                                 opacity=0.6, hovertemplate="%{y:,.2f}", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=mid - 2 * sd, name="BOLL(20, 2)", line=dict(color=BLUE, width=1),
                                 opacity=0.6, fill="tonexty", fillcolor="rgba(87,139,250,0.07)",
                                 hovertemplate="%{y:,.2f}"), row=1, col=1)

    # Current price line with a price tag on the axis, coloured by the forming candle's direction.
    last = df.iloc[-1]
    last_color = UP if last.close >= last.open else DOWN
    fig.add_hline(y=last.close, line=dict(color=last_color, width=1, dash="dot"), row=1, col=1)
    fig.add_annotation(x=1, xref="paper", y=last.close, yref="y", text=f" {price_text(last.close)} ", showarrow=False,
                       xanchor="left", font=dict(color="white", size=11), bgcolor=last_color)

    if "volume" in row:
        colors = [UP if c >= o else DOWN for o, c in zip(df.open, df.close)]
        fig.add_trace(go.Bar(x=x, y=df.volume, name="Volume", marker_color=colors, opacity=0.55,
                             showlegend=False, hovertemplate="%{y:,.0f}"), row=row["volume"], col=1)

    if ai_models:
        for name in ai_models:
            hist = ai_history(name, model_version(name))
            if hist is None or sym not in hist:
                continue
            p = hist[sym].reindex(df.index)
            main_model = name == ai_models[-1]
            fig.add_trace(go.Scatter(
                x=x, y=p, name=f"AI P(up) {config.MODELS[name]['label'].lower()}",
                line=dict(color=BLUE if main_model else GREY, width=2 if main_model else 1.2),
                hovertemplate="%{y:.0%}"), row=row["ai"], col=1)
        for y in (config.ENTER_PROB, config.EXIT_PROB):
            fig.add_hline(y=y, line=dict(color=GREY, width=1, dash="dot"), row=row["ai"], col=1)
        fig.update_yaxes(tickformat=".0%", row=row["ai"], col=1)

    # Open on the latest candles with room on the right, like an exchange chart, and keep the viewer's
    # zoom across the 10-second refreshes (uirevision) until they switch coin or interval.
    view = df.iloc[-INITIAL_CANDLES:]
    pad = (view.high.max() - view.low.min()) * 0.06
    fig.update_xaxes(range=[x[-len(view)], x[-1] + 6 * step])
    fig.update_yaxes(range=[view.low.min() - pad, view.high.max() + pad], row=1, col=1)
    fig.update_layout(
        height=620, margin=dict(l=0, r=70, t=10, b=0), uirevision=f"{sym}-{interval}",
        hovermode="x unified", dragmode="pan", xaxis_rangeslider_visible=False, bargap=0.15,
        legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom", font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=GRID, showspikes=True, spikemode="across", spikesnap="cursor",
                     spikedash="dot", spikethickness=1, spikecolor=GREY)
    fig.update_yaxes(gridcolor=GRID, side="right", showspikes=True, spikemode="across", spikesnap="cursor",
                     spikedash="dot", spikethickness=1, spikecolor=GREY)
    st.plotly_chart(fig, width="stretch", alt=f"{coin(sym)} {interval} candlestick chart",
                    config={"scrollZoom": True, "displaylogo": False,
                            "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"]})
    note = "Scroll to zoom, drag to pan, double-click to reset."
    if ai_models:
        note += (" The AI line is the model's P(up) at each closed candle. Past values come from the model "
                 "trained on that data, so they look better than reality; see Backtest for honest results.")
    elif "AI signal" in shown:
        note += " The AI line appears on the 4h and 1D intervals, the timeframes the models are built on."
    st.caption(note)


def ai_outlook(sym):
    """Side panel: what each model expects for this coin."""
    st.subheader("AI outlook", icon=":material/psychology:")
    for name, spec in config.MODELS.items():
        if model.load(name) is None:
            continue
        sig = get_signals(name, model_version(name)).set_index("symbol").loc[sym]
        color = {"BULLISH": "green", "BEARISH": "red"}.get(sig.signal, "gray")
        icon = {"BULLISH": ":material/trending_up:", "BEARISH": ":material/trending_down:"}.get(
            sig.signal, ":material/trending_flat:")
        with st.container(border=True):
            top = st.container(horizontal=True, vertical_alignment="center")
            top.markdown(f"**{spec['label']}**")
            top.badge(SIGNAL_TEXT[sig.signal], icon=icon, color=color)
            st.progress(float(sig.prob_up), text=f"P(up) {sig.prob_up:.0%}")
            st.caption(TRUST.get(name, "The main signal.").strip())
    st.caption(f"Potential growth at a {config.ENTER_PROB:.0%}+ chance of a rise, potential decline at "
               f"{config.EXIT_PROB:.0%} or less. "
               "Estimates with a small edge, not advice.")


# ---------------- backtest ----------------
def page_backtest():
    st.title("Backtest")
    bar = st.container(horizontal=True, vertical_alignment="bottom", gap="medium")
    name = bar.segmented_control("Prediction", list(config.MODELS), key="model", required=True,
                                 format_func=lambda n: config.MODELS[n]["label"])
    sym = bar.segmented_control("Coin", config.SYMBOLS, key="sym", required=True, format_func=coin)
    if model.load(name) is None:
        st.error("No model yet. Run `train.bat`, or press **Retrain models** below.")
        retrain_button()
        return
    model_status(name)

    st.session_state.setdefault("bt_enter", config.ENTER_PROB)
    st.session_state.setdefault("bt_exit", config.EXIT_PROB)
    with st.expander("Strategy settings", icon=":material/tune:"):
        c1, c2 = st.columns(2)
        enter = c1.slider("Buy when P(up) is above", 0.50, 0.70, key="bt_enter", step=0.01)
        exit_ = c2.slider("Sell when P(up) is below", 0.30, 0.55, key="bt_exit", step=0.01)
        st.caption("The strategy holds the coin or cash, never shorts, and pays a "
                   f"{config.FEE:.2%} fee each time it buys or sells.")
    enter, exit_ = st.session_state["bt_enter"], min(st.session_state["bt_exit"], st.session_state["bt_enter"])

    oos = model.load_oos(name)
    ctf = config.MODELS[name]["timeframe"]
    eq, s = backtest.run(oos[oos["symbol"] == sym], ctf, enter, exit_)
    ai, bh = s["strategy"], s["buy_hold"]

    kpis = st.container(horizontal=True, gap="medium")
    kpis.metric("AI strategy return", f"{ai['total_return']:+.0%}",
                f"{ai['total_return'] - bh['total_return']:+.0%} vs buy & hold", border=True)
    kpis.metric("Buy & hold return", f"{bh['total_return']:+.0%}", border=True)
    kpis.metric("Worst drop", f"{ai['max_drawdown']:.0%}", f"buy & hold {bh['max_drawdown']:.0%}",
                delta_color="off", delta_arrow="off", border=True,
                help="Largest fall from a previous high (max drawdown).")
    kpis.metric("Sharpe ratio", f"{ai['sharpe']:.2f} ± {ai['sharpe_se']:.2f}", f"buy & hold {bh['sharpe']:.2f}",
                delta_color="off", delta_arrow="off", border=True,
                help="Return per unit of risk, per year. The ± is its standard error: the true value is likely "
                     "within about two of these.")
    kpis.metric("Trades", s["num_trades"], f"{s['win_rate']:.0%} won", delta_color="off", delta_arrow="off",
                border=True)
    kpis.metric("Time invested", f"{s['time_in_market']:.0%}", f"fees {s['fees_paid_pct']:.0%} in total",
                delta_color="off", delta_arrow="off", border=True)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.05)
    for col_, label, color, width in (("buy_hold", "Buy & hold", GREY, 1.6), ("strategy", "AI strategy", BLUE, 2.2)):
        fig.add_trace(go.Scatter(x=local(eq.index).tz_localize(None), y=eq[col_], name=label, line=dict(color=color, width=width),
                                 hovertemplate="%{y:.2f}x"), row=1, col=1)
        drawdown = eq[col_] / eq[col_].cummax() - 1
        fig.add_trace(go.Scatter(x=local(eq.index).tz_localize(None), y=drawdown, name=f"{label} drop", showlegend=False,
                                 line=dict(color=color, width=1), fill="tozeroy", fillcolor=_rgba(color, 0.18),
                                 hovertemplate="%{y:.0%}"), row=2, col=1)
    ticks = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
    fig.update_yaxes(type="log", title_text="Growth of $1", tickvals=ticks,
                     ticktext=[f"{t:g}×" for t in ticks], row=1, col=1)
    fig.update_yaxes(tickformat=".0%", title_text="Drop from high", row=2, col=1)
    fig = style(fig, 520)
    fig.update_yaxes(side="right")
    st.plotly_chart(fig, width="stretch", alt=f"Growth of $1 and drawdowns for the AI strategy on {coin(sym)}")
    st.caption(f"Walk-forward test from {eq.index[0]:%b %Y}: every period is predicted by a model trained only on "
               f"earlier data, so this is how the AI would have done without seeing the future.")

    trust_section(ai, bh)
    q = backtest.PERIODS_PER_YEAR[ctf]
    rets = backtest.returns(oos[oos["symbol"] == sym], ctf, enter, exit_)
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Year by year**")
        years = [{"Year": str(y), "AI return": (1 + g.strategy).prod() - 1, "Hold return": (1 + g.buy_hold).prod() - 1,
                  "AI Sharpe": metrics.sharpe(g.strategy, q), "Hold Sharpe": metrics.sharpe(g.buy_hold, q)}
                 for y, g in rets.groupby(rets.index.year) if len(g) > q / 12]
        table(
            pd.DataFrame(years).style.format({"AI return": "{:+.0%}", "Hold return": "{:+.0%}",
                                              "AI Sharpe": "{:.2f}", "Hold Sharpe": "{:.2f}"})
            .map(lambda v: f"color: {UP}" if v > 0 else f"color: {DOWN}" if v < 0 else "",
                 subset=["AI return", "Hold return"]),
            hide_index=True, width="stretch", alt="AI strategy and buy & hold results for each year")
    with right:
        st.markdown("**How fees change the result**")
        fees = []
        for f in (0.0, 0.0005, 0.00075, 0.001, 0.002):
            r = backtest.returns(oos[oos["symbol"] == sym], ctf, enter, exit_, fee=f).strategy
            fees.append({"Fee per trade": f"{f * 100:g}%" + ("  (Binance standard)" if f == 0.001
                         else "  (paying fees in BNB)" if f == 0.00075 else ""),
                         "AI return": (1 + r).prod() - 1, "Sharpe": metrics.sharpe(r, q)})
        table(pd.DataFrame(fees).style.format({"AI return": "{:+.0%}", "Sharpe": "{:.2f}"}),
                     hide_index=True, width="stretch", alt="AI strategy results at different trading fees")

    st.subheader("All coins")
    rows, basket = [], []
    for sy, g in oos.groupby("symbol"):
        _, ss = backtest.run(g, ctf, enter, exit_)
        a = ss["strategy"]
        rows.append({"Coin": coin_label(sy),
                     "AI strategy": a["total_return"], "Buy & hold": ss["buy_hold"]["total_return"],
                     "AI worst drop": a["max_drawdown"], "Hold worst drop": ss["buy_hold"]["max_drawdown"],
                     "Sharpe": f"{a['sharpe']:.2f} ± {a['sharpe_se']:.2f}", "Chance Sharpe > 0": a["psr"],
                     "Chance beats hold": a["psr_vs_hold"], "Trades": ss["num_trades"]})
        basket.append(backtest.returns(g, ctf, enter, exit_))
    # Equal-weight basket: a quarter of the money follows the AI on each coin (or a share of the coins trading).
    b = pd.concat(basket, keys=range(len(basket))).groupby(level=1).mean()
    bm = metrics.summary(b.strategy, q, bench=b.buy_hold)
    rows.append({"Coin": "All four together", "AI strategy": (1 + b.strategy).prod() - 1,
                 "Buy & hold": (1 + b.buy_hold).prod() - 1,
                 "AI worst drop": metrics.max_drawdown((1 + b.strategy).cumprod()),
                 "Hold worst drop": metrics.max_drawdown((1 + b.buy_hold).cumprod()),
                 "Sharpe": f"{bm['sharpe']:.2f} ± {bm['sharpe_se']:.2f}", "Chance Sharpe > 0": bm["psr"],
                 "Chance beats hold": bm["psr_vs_hold"], "Trades": sum(r["Trades"] for r in rows)})
    pct = ["AI strategy", "Buy & hold", "AI worst drop", "Hold worst drop"]
    table(
        pd.DataFrame(rows).style.format({**{k: "{:+.0%}" for k in pct}, "Chance Sharpe > 0": "{:.0%}",
                                         "Chance beats hold": "{:.0%}"})
        .map(lambda v: f"color: {UP}" if v > 0 else f"color: {DOWN}" if v < 0 else "", subset=["AI strategy", "Buy & hold"]),
        hide_index=True, width="stretch", alt="Backtest results for every coin and for all four together")
    st.caption(f"Sharpe uses a 0% risk-free rate and {q:,} periods a year (crypto trades every day), with fees "
               f"of {config.FEE:.1%} per trade and no slippage. Details: docs/research/sharpe-ratio.md.")

    st.subheader("Learning history")
    history = model.load_log()
    h = history[history["model"] == name] if not history.empty else history
    if h.empty:
        st.caption("No retraining recorded yet.")
        return
    h = h.assign(trained_at=local(h["trained_at"]).dt.tz_localize(None), last_candle=local(h["last_candle"]).dt.tz_localize(None))
    fig = go.Figure(go.Scatter(x=h["trained_at"], y=h["oos_auc"], mode="lines+markers", name="AUC",
                               line=dict(color=BLUE, width=2), hovertemplate="%{y:.4f}"))
    fig.add_hline(y=config.MIN_AUC, line=dict(color=GREY, width=1, dash="dot"),
                  annotation_text="minimum to accept", annotation_position="bottom right")
    fig.update_yaxes(title_text="Walk-forward AUC")
    st.plotly_chart(style(fig, 240), width="stretch", alt="Model accuracy after each retrain")
    with st.expander("Every retrain", icon=":material/list:"):
        table(
            h.iloc[::-1][["trained_at", "last_candle", "rows", "oos_auc", "oos_accuracy", "accepted"]].rename(columns={
                "trained_at": f"Trained ({tz_label()})", "last_candle": "Data up to", "rows": "Training rows",
                "oos_auc": "AUC", "oos_accuracy": "Accuracy", "accepted": "Kept"}),
            hide_index=True, width="stretch", alt="Every retraining run",
            column_config={"Accuracy": st.column_config.NumberColumn(format="percent"),
                           f"Trained ({tz_label()})": st.column_config.DatetimeColumn(format="MMM D, HH:mm"),
                           "Data up to": st.column_config.DatetimeColumn(format="MMM D, HH:mm")})
    st.caption("The model retrains at every candle close on all data so far. A new model replaces the old one only "
               "if it still beats a coin flip on data it didn't train on.")


def trust_section(ai, bh):
    """How much to trust the backtest: risk-adjusted metrics and the chance the edge is real."""
    st.markdown("**How much to trust this**")
    row = st.container(horizontal=True, gap="medium")
    row.metric("Chance Sharpe is above 0", f"{ai['psr']:.0%}", border=True,
               help="Probabilistic Sharpe ratio: the chance the strategy's true Sharpe is positive, given how "
                    "long the test is and how jumpy the returns are.")
    row.metric("Chance it beats buy & hold", f"{ai['psr_vs_hold']:.0%}", border=True,
               help="The chance the strategy's true Sharpe is above buy & hold's Sharpe on the same period.")
    row.metric(f"After {config.TRIALS_TESTED} variants tried", f"{ai['deflated_sharpe']:.0%}", border=True,
               help="Deflated Sharpe ratio: the chance above 0, corrected for having picked the best of the "
                    "variants we tested. Below 50% means the result could easily be luck.")
    row.metric("Sortino ratio", f"{ai['sortino']:.2f}", f"buy & hold {bh['sortino']:.2f}", delta_color="off",
               delta_arrow="off", border=True, help="Like Sharpe, but only counts falls as risk, not rises.")
    row.metric("Calmar ratio", f"{ai['calmar']:.2f}", f"buy & hold {bh['calmar']:.2f}", delta_color="off",
               delta_arrow="off", border=True, help="Yearly growth divided by the worst drop.")


def model_status(name):
    """Card with the model's honest accuracy and a retrain button."""
    m = model.load_metrics(name)
    with st.container(border=True, horizontal=True, vertical_alignment="center", gap="large"):
        if m:
            base = max(m["baseline_up_rate"], 1 - m["baseline_up_rate"])
            st.metric("Walk-forward AUC", f"{m['oos_auc']:.3f}", help="0.5 is a coin flip.", width="content")
            st.metric("Accuracy", f"{m['oos_accuracy']:.1%}", f"{m['oos_accuracy'] - base:+.1%} vs always guessing",
                      delta_arrow="off", width="content")
            st.markdown(f":gray[Last trained]  \n**{when(m['trained_at'])}**",
                        width="content")
            st.markdown(f":gray[Training rows]  \n**{m['rows']:,}**", width="content")
        retrain_button()


def _rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def retrain_button():
    if st.button("Retrain models", icon=":material/refresh:", help="Downloads fresh data and retrains (a few minutes)"):
        with st.spinner("Training…"):
            for n in config.MODELS:
                model.train(n)
        st.cache_data.clear()
        st.rerun()


# ---------------- portfolio ----------------
ACTION_STYLE = {  # badge color and icon for each kind of advice
    "BUY": ("green", ":material/add_shopping_cart:"),
    "SELL": ("red", ":material/sell:"),
    "HOLD": ("blue", ":material/pause_circle:"),
    "WAIT": ("gray", ":material/schedule:"),
    "NOT TRACKED": ("gray", ":material/help:"),
}


@st.cache_data(ttl=60, max_entries=4, show_spinner="Working out the advice…")
def get_advice(versions, portfolio_mtime):
    """Advice for the saved portfolio. Recomputed when a model retrains or the portfolio is saved."""
    return advisor.current(refresh=False)


def advice_section():
    st.subheader("AI advice", icon=":material/tips_and_updates:")
    versions = tuple(model_version(n) for n in (advisor.DIRECTION, advisor.TIMING))
    if None in versions:
        st.caption("The advice appears once the models are trained.")
        return
    mtime = config.PORTFOLIO_FILE.stat().st_mtime if config.PORTFOLIO_FILE.exists() else 0
    advice = get_advice(versions, mtime)

    act = advice[advice["action"].isin(["BUY", "SELL"])]
    if len(act):
        cols = st.columns(min(len(act), 4), gap="medium")
        for col, r in zip(cols * 2, act.itertuples()):
            color, icon = ACTION_STYLE[r.action]
            tick = coin(r.symbol)
            with col.container(border=True):
                st.badge(f"{r.action.title()} {tick}", icon=icon, color=color)
                st.metric("Suggested amount", f"${r.amount_usdt:,.0f}", f"about {r.amount_coin:.4f} {tick}",
                          delta_color="off", delta_arrow="off")
                st.caption(r.reason)
                if isinstance(r.timing, str):
                    st.caption(f":material/schedule: {r.timing}")
    else:
        st.caption("No buying or selling suggested right now.")

    rest = advice[~advice["action"].isin(["BUY", "SELL"])]
    with st.container(border=True):
        for r in rest.itertuples():
            color, icon = ACTION_STYLE[r.action]
            line = st.container(horizontal=True, vertical_alignment="center", gap="small")
            line.badge(f"{r.action.title()} {coin(r.symbol)}", icon=icon, color=color)
            line.caption(r.reason, width="content")

    s = notify.settings()
    channels = [c for c, on in (("Windows notifications", s.get("windows")), ("Zalo", s.get("zalo"))) if on]
    st.caption(
        "Rules: sell when the next-1-day P(up) is at or below "
        f"{config.EXIT_PROB:.0%}, buy at or above {config.ENTER_PROB:.0%}; next 4 hours is only a timing hint. "
        f"Buys split your spare cash equally, at most {config.MAX_PER_COIN:.0%} of the portfolio per coin. "
        + (f"Alerts on changes: {', '.join(channels)}." if channels else "Alerts are off.")
        + " Estimates with a small edge, not financial advice.")


def page_portfolio():
    st.title("Portfolio")
    holdings = portfolio.load()
    advice_section()
    if holdings:
        portfolio_overview(holdings)
    else:
        with st.container(border=True):
            st.markdown("**No holdings yet**")
            st.caption("Add what you own and your spare cash below, to get advice for your own portfolio.")

    with st.expander("Edit holdings and cash", icon=":material/edit:", expanded=not holdings):
        cash = st.number_input("Spare cash (USDT)", min_value=0.0, value=float(portfolio.load_cash()), step=50.0,
                               help="Money available for buying. The AI uses it to suggest buy amounts.")
        edited = st.data_editor(
            pd.DataFrame(holdings, columns=["symbol", "amount", "avg_cost"]),
            num_rows="dynamic", width="stretch", key="holdings_editor",
            column_config={
                "symbol": st.column_config.TextColumn("Coin pair", help="A Binance pair, like BTC/USDT",
                                                      validate=r"^[A-Z0-9]+/[A-Z]+$", required=True),
                "amount": st.column_config.NumberColumn("Amount", format="%.6f", min_value=0, required=True),
                "avg_cost": st.column_config.NumberColumn("Average cost (USDT)", format="%.4f", min_value=0,
                                                          required=True),
            },
        )
        if st.button("Save", icon=":material/save:", type="primary"):
            portfolio.save(edited.dropna().to_dict("records"), cash=cash)
            st.toast("Saved", icon=":material/check:")
            st.rerun()
        st.caption("Saved only on this computer, in portfolio.json.")


def portfolio_overview(holdings):
    try:
        pv = portfolio.valued(holdings)
    except Exception as e:
        st.error(f"Could not price holdings: {e}")
        return
    total, cost = pv["value"].sum(), (pv["amount"] * pv["avg_cost"]).sum()
    pnl = total - cost

    left, right = st.columns([1.5, 1], gap="large", vertical_alignment="center")
    with left:
        st.metric("Total balance", f"${total:,.2f}", f"{pnl:+,.2f} USDT ({pnl / cost:+.1%})" if cost else None)
        stats = st.container(horizontal=True, gap="large")
        stats.metric("Invested", f"${cost:,.2f}", width="content")
        stats.metric("Holdings", len(pv), width="content")
        names = [n for n in ("4h", "4h_next") if model.load(n) is not None]
        sigs = {n: get_signals(n, model_version(n)).set_index("symbol")["signal"] for n in names}
        if "4h" in sigs:
            bearish = int((pv["symbol"].map(sigs["4h"]) == "BEARISH").sum())
            stats.metric("Potential decline (next 1 day)", bearish, width="content")
    with right:
        fig = go.Figure(go.Pie(labels=pv["symbol"].map(coin), values=pv["value"], hole=0.62, sort=False,
                               marker=dict(colors=[COIN_COLORS.get(s, GREY) for s in pv["symbol"]]),
                               textinfo="label+percent", hovertemplate="%{label}: $%{value:,.2f}<extra></extra>"))
        fig.update_layout(height=240, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
        st.plotly_chart(fig, width="stretch", alt="Share of the portfolio in each coin")

    st.subheader("Holdings")
    tbl = pd.DataFrame({
        "Coin": pv["symbol"].map(coin_label),
        "Amount": pv["amount"],
        "Price": pv["price"],
        "Value": pv["value"],
        "Allocation": pv["value"] / total if total else 0,
        "Average cost": pv["avg_cost"],
        "P&L": pv["pnl"],
        "P&L %": pv["pnl_pct"],
    })
    for n in names:
        tbl[f"AI {config.MODELS[n]['label'].lower()}"] = pv["symbol"].map(sigs[n]).map(SIGNAL_ARROW).fillna("not tracked")
    signal_cols = [c for c in tbl if c.startswith("AI ")]
    table(
        tbl.style.format({"P&L %": "{:+.1%}", "P&L": "{:+,.2f}"})
        .map(lambda v: f"color: {UP}" if v > 0 else f"color: {DOWN}" if v < 0 else "", subset=["P&L", "P&L %"])
        .map(lambda v: f"color: {UP}" if v.startswith("▲") else f"color: {DOWN}" if v.startswith("▼") else "",
             subset=signal_cols),
        hide_index=True, width="stretch", alt="Your holdings with value, profit or loss, and AI signals",
        column_config={
            "Amount": st.column_config.NumberColumn(format="%.6f"),
            "Price": st.column_config.NumberColumn(format="dollar"),
            "Value": st.column_config.NumberColumn(format="dollar"),
            "Average cost": st.column_config.NumberColumn(format="dollar"),
            "Allocation": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        },
    )


# ---------------- paper trading ----------------
PAPER_COLORS = {"ai": "#0052FF", "trend_dip": "#F4B000", "trend_ai": "#C855E8", "trend": "#8A919E", "hold": "#B5BAC3"}


BOOKS = {b.key: b for b in paper.BOOKS}


def page_paper(book=paper.MAIN):
    st.title("Paper trading" if book is paper.MAIN else "Altcoin paper trading")
    if book is not paper.MAIN:
        st.caption("Experimental: the AI was trained on BTC, ETH, BNB and SOL and is applied to these altcoins as is. "
                   "In testing on NEAR and ZEC, the 50-day trend rule did far better than the AI.")
    acct = paper.load(book)
    if acct is None:
        with st.container(border=True, width=560):
            st.markdown("**Start a 4-week paper trial**")
            st.caption(f"Coins: {', '.join(coin(s) for s in book.symbols)}. Five pretend accounts start together and "
                       "trade live: AI Smart, Trend + dip-buy, Trend × AI + dip-buy, and two benchmarks (trend rule, "
                       "buy & hold). Nothing real is bought or sold.")
            cash = st.number_input("Starting balance per account (USDT)", min_value=100.0, value=1000.0, step=100.0,
                                   key=f"paper_cash_{book.key}")
            if st.button("Start trial", type="primary", icon=":material/play_arrow:", key=f"paper_start_{book.key}"):
                with st.spinner("Starting…"):
                    if book is not paper.MAIN:
                        altcoins.compute()
                    paper.open_account(cash, book=book)
                    paper.step_ai(book=book)
                    paper.step_daily(book=book)
                st.rerun()
        return
    paper_dashboard(book.key)


def paper_stats(acct, prices, v, t, now):
    """Binance-style running stats per account: P&L split into realised and unrealised, fees, trades, vs buy & hold."""
    start, names = acct["start_cash"], acct["names"]
    runtime = now - pd.Timestamp(acct["opened"])
    rows = []
    for a, sleeves in acct["accounts"].items():
        unreal = 0.0
        for s, sl in sleeves.items():
            qty, cost = paper.holdings(sl)
            unreal += qty * prices[s] - cost
        total = v[a] - start
        mine = t[t["account"] == a]
        rows.append({"Account": names[a], "Total P&L": total, "Realised": total - unreal, "Unrealised": unreal,
                     "Fees paid": float(mine["fee"].sum()), "Trades": len(mine), "vs buy & hold": v[a] - v["hold"],
                     "Invested now": 1 - sum(sl["cash"] for sl in sleeves.values()) / v[a] if v[a] else 0})
    df = pd.DataFrame(rows).sort_values("Total P&L", ascending=False)
    signed = ["Total P&L", "Realised", "Unrealised", "vs buy & hold"]
    table(
        df.style.format({**{c: "{:+,.2f}" for c in signed}, "Fees paid": "${:,.2f}"})
        .map(lambda x: f"color: {UP}" if isinstance(x, float) and x > 0.005 else
             f"color: {DOWN}" if isinstance(x, float) and x < -0.005 else "", subset=signed),
        hide_index=True, width="stretch", alt="Profit and loss breakdown for every paper account",
        column_config={"Invested now": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
    st.caption(f"Running {runtime.days} d {runtime.seconds // 3600} h. Realised: profit or loss locked in by sales "
               "(after fees). Unrealised: the coins still held, at today's price vs what they cost. Fees are 0.1% "
               "of every trade; slippage is in the fill prices. vs buy & hold: this account minus the buy & hold "
               "benchmark.")


def page_paper_alts():
    page_paper(paper.ALTS)


def _live_prices():
    tk = live_feed().tickers()
    if tk is None:
        tk = get_tickers()
    return dict(zip(tk["symbol"], tk["price"]))


@st.cache_data(ttl=10, show_spinner=False)
def _alt_prices(symbols):
    return data.prices(list(symbols))


def _ai_column(book):
    """{symbol: 'arrow P(up)'} for the wallet table: confirmed signals (main coins) or the altcoin readings."""
    if book is paper.MAIN:
        sig = get_signals("4h", model_version("4h")).set_index("symbol")
        return {s: SIGNAL_ARROW[sig.loc[s, "signal"]] + f" {sig.loc[s, 'prob_up']:.0%}" for s in book.symbols}
    coins = (altcoins.load() or {}).get("coins", {})
    out = {}
    for s in book.symbols:
        p = coins.get(s, {}).get("p_up_1d")
        out[s] = "" if p is None else SIGNAL_ARROW[signals.label(p)] + f" {p:.0%}"
    return out


@st.fragment(run_every=10)
def paper_dashboard(book_key="main"):
    book = BOOKS[book_key]
    acct = paper.load(book)
    prices = _live_prices() if book is paper.MAIN else _alt_prices(book.symbols)
    v = paper.value(acct, prices)
    start, names = acct["start_cash"], acct["names"]
    opened, ends = pd.Timestamp(acct["opened"]), pd.Timestamp(acct["ends"])
    now = pd.Timestamp.now(tz="UTC")
    if paper.active(acct):
        left = ends - now
        st.caption(f"Trial: {local(opened):%b %d, %H:%M} → {when(ends)} · day {(now - opened).days + 1} of "
                   f"{(ends - opened).days} · {left.days} days {left.seconds // 3600} h left. Each account started with "
                   f"${start:,.0f} (${start / len(book.symbols):,.0f} per coin). Fills at the live Binance price with "
                   f"{acct['fee']:.1%} fee and {acct['slippage']:.2%} slippage. Nothing real is traded.")
    else:
        st.badge("Trial finished", icon=":material/flag:", color="blue")
        if acct.get("final_report"):
            st.caption(f"Final report: docs/backTestResult/{Path(acct['final_report']).name}")

    for c in acct.get("rule_changes", []):
        st.caption(f":material/rule: Rule change on {when(c['time'])}: {c['change']}")
    cards = st.container(horizontal=True, gap="medium")
    for a in sorted(v, key=lambda k: -v[k]):
        cards.metric(names[a], f"${v[a]:,.2f}", f"{v[a] - start:+,.2f} ({v[a] / start - 1:+.2%})", border=True)
    paper_stats(acct, prices, v, paper.trades(book), now)

    hist = paper.balance_history(book)
    if len(hist):
        live_row = pd.DataFrame([{"time": now, **v}])
        h = pd.concat([hist, live_row], ignore_index=True)
        fig = go.Figure()
        for a in reversed(list(names)):
            if a not in h:
                continue
            bench = a in ("trend", "hold")
            fig.add_trace(go.Scatter(x=local(h["time"]).dt.tz_localize(None), y=h[a], name=names[a], mode="lines",
                                     hovertemplate="$%{y:,.2f}",
                                     line=dict(color=PAPER_COLORS[a], width=1.5 if bench else 2.4,
                                               dash="dot" if bench else None)))
        fig.add_hline(y=start, line=dict(color=GREY, width=1, dash="dash"))
        fig.update_yaxes(tickprefix="$", side="right")
        st.plotly_chart(style(fig, 320), width="stretch", alt="Balance of every paper account over time")

    with st.expander("What each account does", icon=":material/info:"):
        st.markdown(
            "- **AI Smart**: invests by the AI's next-1-day confidence, sells half if the 3-day view is still up, takes "
            "half profit at +10%. " + (
                "Decides at every 4h close and on the live readings once an action holds "
                f"{paper.CONFIRM_MINUTES} minutes (then {paper.COOLDOWN_MINUTES} min per coin; not backtested).\n"
                if book.live else "Decides at every 4h close.\n") +
            "- **Trend + dip-buy**: holds a coin while its daily close is above its 50-day average (the rule with the "
            "highest return in testing).\n"
            "- **Trend × AI + dip-buy**: average of the 20/50/100/200-day trend rules, sized by the AI's next-3-days "
            "P(up) and rebalanced daily (about half the drawdown and half the return in testing).\n"
            f"- **Dip-buy** (first three): a {paper.SHOCK_DROP:.0%}+ fall within an hour buys up to half that coin's "
            "sleeve, sold 4 hours later (tested in experiments/shock_dip_buy.py).\n"
            "- **Benchmarks**: the 50-day trend rule alone, and buy & hold.")

    which = st.segmented_control("Account", list(names), default="ai", key=f"paper_account_{book.key}",
                                 format_func=names.get) or "ai"
    st.subheader("Wallet")
    left_col, right_col = st.columns([2.2, 1], gap="large")
    ai_col = _ai_column(book)
    rows = []
    for sym, sl in acct["accounts"][which].items():
        px = prices[sym]
        qty, cost = paper.holdings(sl)
        val = qty * px
        rows.append({"Coin": coin_label(sym), "Amount": qty, "Avg cost": cost / qty if qty else None, "Price": px,
                     "Value": val, "P&L": val - cost if qty else None, "P&L %": val / cost - 1 if qty else None,
                     "Invested": val / (val + sl["cash"]) if val + sl["cash"] else 0,
                     "AI next 1 day": ai_col[sym]})
    cash = sum(sl["cash"] for sl in acct["accounts"][which].values())
    rows.append({"Coin": "USDT (cash)", "Amount": cash, "Avg cost": None, "Price": 1.0, "Value": cash, "P&L": None,
                 "P&L %": None, "Invested": None, "AI next 1 day": ""})
    wallet = pd.DataFrame(rows)
    with left_col:
        table(
            wallet.style.format({"P&L %": "{:+.2%}", "P&L": "{:+,.2f}"}, na_rep="–")
            .map(lambda x: f"color: {UP}" if isinstance(x, (int, float)) and x > 0 else
                 f"color: {DOWN}" if isinstance(x, (int, float)) and x < 0 else "", subset=["P&L", "P&L %"])
            .map(lambda x: f"color: {UP}" if str(x).startswith("▲") else f"color: {DOWN}" if str(x).startswith("▼") else "",
                 subset=["AI next 1 day"]),
            hide_index=True, width="stretch", alt=f"Coins and cash held by the {names[which]} account",
            column_config={"Amount": st.column_config.NumberColumn(format="%.6f"),
                           "Avg cost": st.column_config.NumberColumn(format="dollar"),
                           "Price": st.column_config.NumberColumn(format="dollar"),
                           "Value": st.column_config.NumberColumn(format="dollar"),
                           "Invested": st.column_config.ProgressColumn("Sleeve invested", format="percent",
                                                                       min_value=0, max_value=1)})
    with right_col:
        alloc = wallet[wallet["Value"] > 0.01]
        ticks = alloc["Coin"].str.split("  ").str[-1]
        fig = go.Figure(go.Pie(labels=ticks, values=alloc["Value"], hole=0.62, sort=False,
                               marker=dict(colors=[{**COIN_COLORS, **altcoins.COLORS}.get(f"{c}/USDT", "#C9CDD4")
                                                   for c in ticks]),
                               textinfo="label+percent", hovertemplate="%{label}: $%{value:,.2f}<extra></extra>"))
        fig.update_layout(height=240, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
        st.plotly_chart(fig, width="stretch", alt=f"Allocation of the {names[which]} account")

    st.subheader("Trade history")
    t = paper.trades(book)
    t = t[t["account"] == which].iloc[::-1]
    if t.empty:
        st.caption("No trades yet in this account.")
        return
    table(
        pd.DataFrame({f"Time ({tz_label()})": local(t["time"]).dt.strftime("%b %d, %H:%M"), "Pair": t["symbol"],
                      "Side": t["side"],
                      "Price": t["price"], "Amount": t["quantity"], "Total (USDT)": t["total"], "Fee": t["fee"],
                      "Reason": t["reason"]})
        .style.map(lambda x: f"color: {UP}; font-weight: 600" if x == "BUY" else f"color: {DOWN}; font-weight: 600"
                   if x == "SELL" else "", subset=["Side"]),
        hide_index=True, width="stretch", alt=f"Trades of the {names[which]} account",
        column_config={"Price": st.column_config.NumberColumn(format="dollar"),
                       "Amount": st.column_config.NumberColumn(format="%.6f"),
                       "Total (USDT)": st.column_config.NumberColumn(format="dollar"),
                       "Fee": st.column_config.NumberColumn(format="dollar")})


# ---------------- altcoins ----------------
@st.cache_data(ttl=1800, show_spinner="Scanning Binance for the biggest movers (about 15 seconds)…")
def get_alt_scan():
    return altcoins.scan()


@st.cache_data(ttl=300, show_spinner="Running the AI on the altcoins…")
def get_alt_readings(version):
    return altcoins.readings()


def page_altcoins():
    st.title("Altcoin market")
    st.caption("Experimental: the live AI does not trade altcoins; a separate paper trial does (Altcoins → Paper "
               "trading). Testing found the AI reads them barely better than a coin flip, while the 50-day trend "
               "rule did far better on them.")
    try:
        r = get_alt_readings(model_version("4h"))
    except Exception as e:
        st.error(f"Could not compute the readings: {e}")
        r = pd.DataFrame()
    alt_chart_section(r)

    st.subheader("AI reading")
    per_row = 3
    for i in range(0, len(r), per_row):
        cols = st.columns(per_row, gap="medium")
        for col, row in zip(cols, r.iloc[i:i + per_row].itertuples()):
            with col.container(border=True):
                alt_card(row)
    st.caption("The AI here is the live model trained on BTC, ETH, BNB and SOL, applied to each altcoin as is. "
               "Updated at every 4h close.")
    st.subheader("Biggest movers on Binance")
    alt_scan_table()


@st.cache_data(ttl=5, show_spinner=False)
def alt_ticker(sym):
    return data.exchange().fetch_ticker(sym)


def _alt_choices():
    """The watched altcoins, then the biggest movers (if the scan has run)."""
    try:
        movers = list(get_alt_scan()["symbol"])
    except Exception:
        movers = []
    return list(dict.fromkeys([*altcoins.WATCH, *movers]))


def _pick_from_table():
    """Clicking a row in the movers table opens that coin's chart."""
    rows = st.session_state["alt_scan_sel"].selection.rows
    if rows:
        st.session_state["alt_chart_coin"] = st.session_state["_alt_scan_symbols"][rows[0]]


def alt_chart_section(readings):
    """A live candle chart for the selected altcoin, shown once a coin is picked."""
    sym = st.selectbox("Coin chart", _alt_choices(), index=None, key="alt_chart_coin", format_func=coin,
                       placeholder="Select a coin to see its chart")
    if sym is None:
        st.caption("Pick a coin above, or click a row in the Biggest movers table below.")
        return
    bar = st.container(horizontal=True, vertical_alignment="bottom", gap="medium")
    interval = bar.segmented_control("Interval", list(CHART_INTERVALS), key="alt_chart_interval", required=True)
    shown = bar.pills("Indicators", [i for i in INDICATORS if i != "AI signal"], key="alt_chart_ind",
                      selection_mode="multi")
    main, side = st.columns([3.3, 1], gap="medium")
    with main:
        alt_price_header(sym)
        price_chart(sym, interval, tuple(shown))
    with side:
        st.subheader("AI reading", icon=":material/psychology:")
        row = readings[readings["symbol"] == sym] if len(readings) else readings
        if len(row):
            with st.container(border=True):
                alt_card(next(row.itertuples()))
            st.caption("Experimental: the AI was trained on BTC, ETH, BNB and SOL.")
        else:
            try:
                d = recent_candles(sym, "1d", limit=60)["close"].iloc[:-1]
                above = d.iloc[-1] > d.tail(50).mean()
                with st.container(border=True):
                    line = st.container(horizontal=True, vertical_alignment="center", gap="small")
                    line.markdown("Trend rule (50-day)", width="content")
                    line.badge("Hold" if above else "Cash", color="green" if above else "red",
                               icon=":material/trending_up:" if above else ":material/trending_down:")
                    st.caption(f"Daily close {d.iloc[-1] / d.tail(50).mean() - 1:+.1%} vs its 50-day average.")
            except Exception as e:
                st.caption(f"Trend unavailable: {e}")
            st.caption(f"The AI reading covers {', '.join(coin(s) for s in altcoins.WATCH)}; for other coins only "
                       "the trend rule is shown.")


@st.fragment(run_every=5)
def alt_price_header(sym):
    try:
        t = alt_ticker(sym)
    except Exception as e:
        st.caption(f"Price unavailable: {e}")
        return
    head = st.container(horizontal=True, vertical_alignment="center", gap="large")
    head.metric(f"{coin(sym)} / USDT", f"${price_text(t['last'])}", f"{(t.get('percentage') or 0) / 100:+.2%} 24h",
                width="content")
    for label, value in (("24h high", f"${price_text(t['high'])}"), ("24h low", f"${price_text(t['low'])}"),
                         ("24h volume", usd(t.get("quoteVolume")))):
        head.markdown(f":gray[{label}]  \n**{value}**", width="content")


def alt_card(row):
    """One altcoin's trend status, AI P(up) and drop risk."""
    st.markdown(f"#### {coin(row.symbol)}")
    st.caption(f"Last 4h close ${row.price:,.4g} · candle of {when(row.candle)}")
    line = st.container(horizontal=True, vertical_alignment="center", gap="small")
    line.markdown("Trend rule (50-day)", width="content")
    if row.above_50d:
        line.badge("Hold", icon=":material/trending_up:", color="green")
    else:
        line.badge("Cash", icon=":material/trending_down:", color="red")
    st.caption(f"Daily close {row.from_50d:+.1%} vs its 50-day average.")
    if pd.notna(row.p_up_1d):
        st.progress(float(row.p_up_1d), text=f"AI: P(up, next 1 day) {row.p_up_1d:.0%}")
    if pd.notna(row.p_up_3d):
        st.progress(float(row.p_up_3d), text=f"AI: P(up, next 3 days) {row.p_up_3d:.0%}")
    if pd.notna(row.p_drop):
        st.progress(min(float(row.p_drop), 1.0),
                    text=f"Drop risk, 3 days {row.p_drop:.0%} ({droprisk.level(row.p_drop)})")


def alt_scan_table():
    try:
        s = get_alt_scan()
    except Exception as e:
        st.error(f"Could not scan Binance: {e}")
        return
    table(
        pd.DataFrame({"Coin": s["symbol"].str.replace("/USDT", ""), "Price": s["price"],
                      "24h": s["change_24h"], "30d": s["change_30d"], "90d": s["change_90d"],
                      "Typical daily move": s["daily_vol"], "Days with 10%+ moves (90d)": s["days_10pct"],
                      "Trend rule": s["above_50d"].map({True: "▲ Hold", False: "▼ Cash"}),
                      "Volume 24h": s["volume_24h"].map(usd)})
        .style.format({"24h": "{:+.1%}", "30d": "{:+.0%}", "90d": "{:+.0%}", "Typical daily move": "{:.1%}",
                       "Price": "${:,.4g}"})
        .map(lambda x: f"color: {UP}" if isinstance(x, float) and x > 0 else
             f"color: {DOWN}" if isinstance(x, float) and x < 0 else "", subset=["24h", "30d", "90d"])
        .map(lambda x: f"color: {UP}" if str(x).startswith("▲") else f"color: {DOWN}", subset=["Trend rule"]),
        hide_index=True, width="stretch", alt="Most volatile liquid altcoins on Binance",
        on_select=_pick_from_table, selection_mode="single-row", key="alt_scan_sel")
    st.session_state["_alt_scan_symbols"] = list(s["symbol"])
    st.caption(f"Altcoins trading at least {usd(altcoins.MIN_VOLUME)} a day on Binance, ranked by their typical "
               "daily move over 30 days (BTC moves about 2% a day). Coins that just doubled usually give much of it "
               "back. Refreshed every 30 minutes. Click a row to open its chart above.")


def page_alt_learnt():
    st.title("What the AI learnt about altcoins")
    st.markdown(
        "- **Reading alts it never trained on:** the AI trained on BTC, ETH, BNB and SOL reads NEAR and ZEC about "
        "as well as BTC: slightly better than a coin flip.\n"
        "- **Training on them too** was chosen on 2022-2024 but did worse on 2025-2026, and it made the AI slightly "
        "worse on the four main coins. So NEAR and ZEC were not added to the live AI.\n"
        "- **The trend rule wins on alts.** On ZEC's 23x run, the AI kept selling part of the way up; the 50-day "
        "rule held on.\n"
        "- **More coins did not help the trend rule.** The most-traded alts each month are usually the ones in a "
        "hype cycle (LUNA before its crash, DOGE, PEPE), and the rule got whipsawed on them.\n"
        "- **Being tested live:** a separate 4-week paper trial runs the same five accounts on "
        f"{', '.join(coin(s) for s in altcoins.WATCH)} (Altcoins → Paper trading).")
    st.subheader("NEAR and ZEC, 2025-01 to 2026-10, $1000 each")
    st.caption("Tested on data the AI never trained on.")
    table(altcoins.NEAR_ZEC.style.format({"NEAR $": "${:,.0f}", "ZEC $": "${:,.0f}", "Sharpe (both)": "{:.2f}",
                                                 "AI accuracy (AUC)": "{:.3f}"}, na_rep="–"),
                 hide_index=True, width="stretch", alt="NEAR and ZEC test results")
    st.subheader("The 50-day rule on more coins")
    st.caption("Coins picked each month by trading volume, delisted coins included.")
    table(altcoins.UNIVERSE.style.format({"Sharpe 2022-2024": "{:.2f}", "Sharpe 2025-2026": "{:.2f}",
                                                 "Return 2025-2026": "{:+.1%}"}),
                 hide_index=True, width="stretch", alt="Trend rule on wider coin lists")
    st.caption("Sharpe: return per unit of risk (higher is better). AUC: 0.5 is a coin flip. Details in "
               "experiments/altcoins_near_zec.py and experiments/trend_universe.py.")


def page_patterns():
    st.title("Patterns the AI learnt")
    ex = explain.load()
    if ex is None:
        st.info("Not computed yet. It refreshes after each daily close, or run `python -m cryptoai explain`.")
        return
    st.caption(f"For the next-1-day model. A copy was trained only on data before {ex['holdout_from'][:10]} and is "
               f"judged here on the {ex['holdout_rows']:,} candles after it, so every number below comes from data "
               f"the model never learnt from. Updated {when(ex['computed'])}.")
    cards = st.container(horizontal=True, gap="medium")
    cards.metric("Accuracy on unseen data (AUC)", f"{ex['holdout_auc']:.3f}", border=True,
                 help="0.5 is a coin flip. Small but real: the AI ranks rising candles above falling ones a bit "
                      "more often than chance.")
    cards.metric("Candles that rose over the next day", f"{ex['base_rate']:.0%}", border=True)

    st.subheader("What the AI relies on")
    rel = pd.DataFrame(ex["reliance"])
    rel["Accuracy lost when shuffled"] = rel["auc_drop"]
    st.bar_chart(rel, x="group", y="Accuracy lost when shuffled", horizontal=True, sort="-Accuracy lost when shuffled",
                 x_label="AUC lost when this kind of pattern is shuffled", y_label="", color=BLUE,
                 alt="How much the AI relies on each kind of pattern")
    st.caption("Each kind of pattern is scrambled in turn; the bigger the loss in accuracy, the more the AI leans on "
               "it. Zero or below means the AI gets nothing from it on new data.")

    st.subheader("What the patterns show")
    st.caption("For the single inputs the AI relies on most: the unseen candles split into fifths by that input "
               "(lowest to highest), or by day/hour. Blue: how often the AI expected a rise. Grey: how often the "
               "price really rose. When the bars agree and slope the same way, the pattern held.")
    for p in ex["patterns"]:
        with st.container(border=True):
            st.markdown(f"**{p['label'][:1].upper() + p['label'][1:]}** · :gray[{p['group']}]")
            st.caption(explain.sentence(p))
            b = pd.DataFrame(p["buckets"])
            chart = pd.DataFrame({"bucket": b["bucket"], "AI expected a rise": b["ai_p_up"],
                                  "Price really rose": b["actual_up"]})
            st.bar_chart(chart, x="bucket", y=["AI expected a rise", "Price really rose"], color=[BLUE, GREY],
                         stack=False, sort=False, height=220, x_label="", y_label="Share of candles",
                         alt=f"Expected vs real rises by {p['label']}")


# ---------------- US stocks ----------------
@st.cache_data(ttl=6 * 3600, show_spinner="Loading 10 years of stock history (first time about 10 seconds)…")
def get_stock_analysis():
    return stocks.analyse_all()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_momentum_test():
    return stocks.momentum_test()


@st.cache_data(ttl=10, show_spinner=False)
def get_stock_prices():
    return stocks.live_prices()


def _pick_stock():
    rows = st.session_state["stock_table"].selection.rows
    if rows:
        st.session_state["stock_pick"] = st.session_state["_stock_order"][rows[0]]


def page_stocks():
    st.title("US stocks")
    st.caption("Live prices and charts from Binance's stock perpetual futures (24/7, the most traded way to hold US "
               "stocks on Binance; tokenized spot pairs like NVDAB/USDT also exist). Binance's stock history is only "
               "months long, so the hold-vs-trade tests below use 10 years of daily closes from Yahoo Finance.")
    try:
        a = get_stock_analysis()
    except Exception as e:
        st.error(f"Could not load stock history: {e}")
        return
    a = a[a.get("error").isna()] if "error" in a else a
    try:
        live = get_stock_prices()
    except Exception:
        live = {}

    pick = st.selectbox("Stock chart", list(a["ticker"]), index=None, key="stock_pick",
                        format_func=lambda t: f"{t} · {stocks.STOCKS[t]}", placeholder="Select a stock to see its chart")
    if pick:
        stock_detail(a.set_index("ticker").loc[pick], live.get(pick))

    st.subheader("Hold or trade?")
    beats = int(a["rule_beats_hold"].sum())
    st.markdown(f"**Trading with trend rules beat simply holding on {beats} of {len(a)} stocks** in 2023-2026 (rule "
                "chosen per stock on 2016-2022: buy above its 50, 100 or 200-day average, sell below). On stocks, "
                "unlike crypto, holding has been better.")
    tbl = pd.DataFrame({
        "Stock": a["ticker"] + " · " + a["name"],
        "Price (Binance)": [live.get(t, (np.nan,))[0] for t in a["ticker"]],
        "24h": [live.get(t, (np.nan, np.nan))[1] for t in a["ticker"]],
        "1 year": a["1y"], "From 1y high": a["from_high"],
        "Trend": np.where(a["above_200d"], "▲ above 200-day", "▼ below 200-day"),
        "Typical yearly swing": a["vol"], "Worst fall (10y)": a["worst_10y"],
        "Hold, Sharpe 2023-26": a["hold_check_sharpe"], "Best rule, Sharpe 2023-26":
            [r[f"{r['rule']}_check_sharpe"] if r["rule"] else np.nan for _, r in a.iterrows()],
    })
    st.session_state["_stock_order"] = list(a["ticker"])
    pct = ["24h", "1 year", "From 1y high", "Worst fall (10y)"]
    table(
        tbl.style.format({**{c: "{:+.1%}" for c in pct}, "Typical yearly swing": "{:.0%}", "Price (Binance)": "${:,.2f}",
                            "Hold, Sharpe 2023-26": "{:.2f}", "Best rule, Sharpe 2023-26": "{:.2f}"}, na_rep="–")
        .map(lambda x: f"color: {UP}" if isinstance(x, float) and x > 0 else
             f"color: {DOWN}" if isinstance(x, float) and x < 0 else "", subset=["24h", "1 year"])
        .map(lambda x: f"color: {UP}" if str(x).startswith("▲") else f"color: {DOWN}", subset=["Trend"]),
        hide_index=True, width="stretch", alt="US stocks on Binance: price, trend, risk, hold vs trade",
        on_select=_pick_stock, selection_mode="single-row", key="stock_table")
    st.caption("Click a row to open its chart. Sharpe: return per unit of risk. These are today's well-known stocks "
               "(survivorship bias), so their past returns look better than a stock picked in advance would have; "
               "hold vs trade on the same stock is the fair comparison.")

    m = get_momentum_test()
    with st.container(border=True):
        st.markdown("**Picking the strongest stocks each month (momentum)**")
        st.caption(f"Holding the 5 stocks with the best past-year return (skipping the last month), rebalanced "
                   f"monthly, vs holding all equally. Sharpe {m['momentum_choose_sharpe']:.2f} vs "
                   f"{m['equal_choose_sharpe']:.2f} on 2016-2022 and {m['momentum_check_sharpe']:.2f} vs "
                   f"{m['equal_check_sharpe']:.2f} on 2023-2026: more return lately but more risk, not better per "
                   f"unit of risk. Strongest now: {', '.join(m['top_now'])}.")


def stock_detail(row, live):
    """Chart from Binance and the hold/trade view for one stock."""
    sym = stocks.perp(row.name)
    bar = st.container(horizontal=True, vertical_alignment="bottom", gap="medium")
    interval = bar.segmented_control("Interval", list(CHART_INTERVALS), key="stock_chart_interval", required=True)
    shown = bar.pills("Indicators", [i for i in INDICATORS if i != "AI signal"], key="stock_chart_ind",
                      selection_mode="multi")
    main, side = st.columns([3.3, 1], gap="medium")
    with main:
        head = st.container(horizontal=True, vertical_alignment="center", gap="large")
        if live:
            head.metric(f"{row.name} / USDT (Binance)", f"${live[0]:,.2f}", f"{live[1]:+.2%} 24h", width="content")
            head.markdown(f":gray[24h volume]  \n**{usd(live[2])}**", width="content")
        head.markdown(f":gray[Last US close]  \n**${row.close:,.2f}**", width="content")
        price_chart(sym, interval, tuple(shown))
    with side:
        st.subheader("Hold view", icon=":material/savings:")
        with st.container(border=True):
            line = st.container(horizontal=True, vertical_alignment="center", gap="small")
            line.markdown("Long-term trend", width="content")
            line.badge("Up" if row.above_200d else "Down", color="green" if row.above_200d else "red",
                       icon=":material/trending_up:" if row.above_200d else ":material/trending_down:")
            st.caption(f"{row.from_200d:+.1%} vs its 200-day average · 1 year {row['1y']:+.0%} · "
                       f"{row.from_high:+.0%} from its 1-year high")
            st.caption(f"Typical yearly swing {row.vol:.0%}; worst fall in 10 years {row.worst_10y:.0%}. Hold only "
                       "what you could keep through a fall like that.")
        st.subheader("Trade test", icon=":material/swap_horiz:")
        with st.container(border=True):
            st.caption("Sharpe (return per unit of risk), 2023-2026, rule chosen on 2016-2022:")
            st.markdown(f"Hold: **{row.hold_check_sharpe:.2f}** · {row.rule} rule: **{row[f'{row.rule}_check_sharpe']:.2f}**")
            verdict = "Trading beat holding" if row.rule_beats_hold else "Holding beat trading"
            st.badge(verdict, color="green" if row.rule_beats_hold else "gray")
            st.caption(f"The {row.rule} rule would be {'in' if row.rule_in_now else 'out'} right now.")


def page_stock_signals():
    st.title("Stock signals")
    ai = stockai.load()
    try:
        a = get_stock_analysis()
    except Exception as e:
        st.error(f"Could not load stock history: {e}")
        return
    stock_buy_list(a)
    market_monitor()
    st.subheader("Every stock")
    if ai is None:
        st.info("The stock AI hasn't run yet. It retrains every day after the US close, or run "
                "`python -m cryptoai stock-ai`.")
        sig = {}
    else:
        sig = ai["signals"]
        trust = ai["auc"]["2023-now"]
        with st.container(border=True):
            st.markdown("**How much to trust the stock AI**")
            st.caption(f"It predicts the chance each stock is higher 5 trading days from now. Accuracy on data it "
                       f"never trained on: AUC {ai['auc']['2019-2022']:.3f} in 2019-2022 (worse than a coin flip) and "
                       f"{trust:.3f} since 2023. Trading on it "
                       f"{'beat' if ai['trade_on_ai'] else 'did not beat'} simply holding. Treat it as a weak hint; "
                       f"the trend columns and holding matter more. Updated {when(ai['computed'])}.")
    rows = []
    for _, r in a.iterrows():
        p = sig.get(r["ticker"], {}).get("p_up_5d")
        rows.append({"Stock": f"{r['ticker']} · {r['name']}", "Long-term trend": "▲ above 200-day" if r["above_200d"]
                     else "▼ below 200-day", "vs 200-day": r["from_200d"],
                     "Short-term trend": "▲ above 50-day" if r["above_50d"] else "▼ below 50-day",
                     "AI: P(up, 5 days)": p, "1 month": r["1m"], "Typical yearly swing": r["vol"]})
    df = pd.DataFrame(rows)
    table(
        df.style.format({"vs 200-day": "{:+.1%}", "1 month": "{:+.1%}", "Typical yearly swing": "{:.0%}",
                         "AI: P(up, 5 days)": "{:.0%}"}, na_rep="–")
        .map(lambda x: f"color: {UP}" if str(x).startswith("▲") else f"color: {DOWN}" if str(x).startswith("▼") else "",
             subset=["Long-term trend", "Short-term trend"])
        .map(lambda x: f"color: {UP}" if isinstance(x, float) and x > 0 else f"color: {DOWN}"
             if isinstance(x, float) and x < 0 else "", subset=["vs 200-day", "1 month"]),
        hide_index=True, width="stretch", alt="Trend and AI signal for each US stock",
        column_config={"AI: P(up, 5 days)": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
    st.caption("For stocks, testing found holding beat every trading rule (Stock backtest). Use these to decide what "
               "to buy and when to add in steps, not to jump in and out.")


@st.fragment(run_every=60)
def stock_buy_list(a):
    """Live 'buy for hold' recommendation: the tested monthly uptrend rule on today's Binance prices."""
    st.subheader("Buy for hold now", icon=":material/shopping_cart:")
    try:
        live = get_stock_prices()
    except Exception:
        live = {}
    b = stocks.buy_list(a, live)
    n = int(b["in"].sum())
    review = stocks.next_review().tz_convert("UTC")
    pick = stocks.top_pick(b)
    if pick is not None:
        rec = stocks.TOP_PICK_RECORD
        with st.container(border=True):
            line = st.container(horizontal=True, vertical_alignment="center", gap="small")
            line.markdown(f"**AI's top pick: {pick['ticker']} · {pick['name']}** at ${pick['price']:,.2f}",
                          width="content")
            line.badge("If you only buy one", icon=":material/star:", color="blue")
            st.caption(f"The steadiest stock in the buy list: a typical yearly swing of {pick['vol']:.0%} and a worst "
                       f"fall of {pick['worst_10y']:.0%} in 10 years; {pick['from_200d']:+.0%} above its sell-if-below "
                       f"line (${pick['avg200']:,.2f}). Of four ways to pick one stock, this did best when tested on "
                       f"2021-2022 ({rec['pick_2122']:+.0%} vs {rec['list_2122']:+.0%} for the whole list, a bear "
                       f"market), but it lagged the whole list in 2023-2026 ({rec['pick_2326']:+.0%} vs "
                       f"{rec['list_2326']:+.0%}). No one-stock pick beat holding the whole list, so the list below is "
                       "still the better choice. The pick is reviewed monthly with the list.")
    _, saved_cash = stocks.load_portfolio()
    budget = st.number_input("How much do you want to invest? (USD)", min_value=0.0, step=100.0,
                             value=float(saved_cash) if saved_cash >= 100 else 1000.0, key="buy_budget")
    each = budget / n if n else 0.0
    st.markdown(f"**What to buy:** the {n} stocks marked **BUY** in the table below, the same amount of each "
                f"(${each:,.0f} per stock in total).")
    st.caption("Why the same amount of each: no way of ranking which stocks have more potential has passed testing "
               "(past-year momentum, risk-adjusted momentum, low volatility, the 5-day AI, and an AI predicting each "
               "stock's 3-month return all failed to beat equal shares; see Research → Strategy lab).")
    st.markdown(f"**When to buy:** in 4 equal buys, one a week. On each buy day, buy ${each / 4:,.0f} of every stock "
                "marked BUY that day.")
    st.markdown("**At what price:** the market price at the moment you buy (the *Price now* column). Don't set a "
                "lower target: in testing, waiting with an order 1-3% below the price cost **0.2% to 1% more on "
                "average** than buying at market, because when the dip doesn't come the price runs away. Spreading the "
                "buys over 4 weeks already evens out the price you pay.")
    today = pd.Timestamp.now(tz="UTC")
    plan = pd.DataFrame({
        "Buy": [f"Buy {i + 1} of 4" for i in range(4)],
        "Day": [("Today, " if i == 0 else "") + f"{local(today + pd.Timedelta(days=7 * i)):%a %b %d}" for i in range(4)],
        "Each BUY stock": [each / 4] * 4, "Total that day": [budget / 4] * 4,
    })
    table(plan.style.format({"Each BUY stock": "${:,.0f}", "Total that day": "${:,.0f}"}), hide_index=True,
          width="content", alt="Buy calendar: four weekly buys",
          column_config={"Buy": st.column_config.Column(help="Which of the 4 buys."),
                         "Day": st.column_config.Column(help="The day to make this buy (in the time zone chosen at "
                                                             "the top). Any time that day is fine."),
                         "Each BUY stock": st.column_config.Column(help="Dollars to buy of each stock marked BUY on "
                                                                        "that day."),
                         "Total that day": st.column_config.Column(help="All of that day's buys together.")})
    st.caption("On each buy day, look at this table again: if a stock has switched to AVOID, skip it and split its "
               "amount over the others.")
    with st.container(border=True):
        st.markdown(f"**Hold until:** there is no fixed end date. Check once a month at the US close (next: "
                    f"**{when(review)}**). Keep a stock while its price is above its **sell-if-below line** (its 200-day average, a floor "
                    "that rises with the stock; there is no take-profit target). If it is below the line at a monthly check, sell it, and buy whatever newly shows "
                    "BUY. Between checks, do nothing, even if it dips.")
    rows = []
    for _, r in b.iterrows():
        buy = bool(r["in"])
        rows.append({"Stock": f"{r['ticker']} · {r['name']}", "Do now": "BUY" if buy else "AVOID for now",
                     "Price now": r["price"], "Invest": each if buy else 0.0,
                     "Shares": each / r["price"] if buy else 0.0, "Each weekly step": each / 4 if buy else 0.0,
                     "Sell if below (200-day avg)": r["avg200"], "Room above the sell-if-below line": r["from_200d"],
                     "Worst fall (10y)": r["worst_10y"]})
    df = pd.DataFrame(rows)
    table(df.style.format({"Price now": "${:,.2f}", "Invest": "${:,.0f}", "Shares": "{:.3f}",
                           "Each weekly step": "${:,.0f}", "Sell if below (200-day avg)": "${:,.2f}",
                           "Room above the sell-if-below line": "{:+.1%}", "Worst fall (10y)": "{:.0%}"})
          .map(lambda x: f"color: {UP}; font-weight: 600" if x == "BUY" else
               f"color: {GREY}" if x == "AVOID for now" else "", subset=["Do now"])
          .map(lambda x: f"color: {UP}" if isinstance(x, float) and x > 0 else
               f"color: {DOWN}" if isinstance(x, float) and x < 0 else "", subset=["Room above the sell-if-below line"]),
          hide_index=True, width="stretch", alt="Which stocks to buy now, how much, and their sell lines")
    st.caption(f"Live Binance prices, refreshed every minute. Shares are fractional (Binance stock tokens and futures "
               f"allow that). Why this rule: holding an equal share of every stock above its 200-day average, checked "
               f"monthly, beat holding all {len(b)} stocks in both test periods (Sharpe 0.83 vs 0.61 on 2016-2022, "
               f"2.05 vs 1.91 on 2023-2026) with a smaller worst fall (-46% vs -59%). A stock far above its sell line "
               "has more room before a sell but has already run up; 'Worst fall' shows how much each has dropped "
               "before. Tested on past data, not a guarantee.")


@st.fragment(run_every=60)
def market_monitor():
    """US market crash monitor: S&P 500 vs its high and 200-day average, live."""
    try:
        s = stocks.market_status(get_stock_prices()["SPY"][0])
    except Exception as e:
        st.caption(f"Market monitor unavailable: {e}")
        return
    color = {"Normal": "green", "Correction": "orange", "Bear market": "red"}[s["level"]]
    with st.container(border=True):
        line = st.container(horizontal=True, vertical_alignment="center", gap="small")
        line.markdown("**US market crash monitor**", width="content")
        line.badge(s["level"], color=color, icon=":material/monitor_heart:")
        st.caption(f"S&P 500 (SPY ${s['price']:,.2f}): {s['from_high']:+.1%} from its 1-year high, "
                   f"{'above' if s['above_200d'] else 'below'} its 200-day average (${s['avg200']:,.2f}), "
                   f"{s['today']:+.1%} today. Alerts (Windows, Zalo) when it falls 10% (correction) or 20% (bear "
                   "market) from its high, crosses its 200-day average, or drops 4%+ in a day.")
        st.caption("What testing found: selling everything at those points cut the worst fall in the 2020 and 2022 "
                   "crashes (-46% vs -59%) but missed the 2023-2026 rebound (Sharpe 1.47 vs 1.91). So the warning "
                   "tells you; the tested rule only sells a stock at the monthly review once it is below its 200-day "
                   "average.")


def page_stock_backtest():
    st.title("Stock backtest")
    try:
        a = get_stock_analysis().set_index("ticker")
    except Exception as e:
        st.error(f"Could not load stock history: {e}")
        return
    bar = st.container(horizontal=True, vertical_alignment="bottom", gap="medium")
    t = bar.selectbox("Stock", list(a.index), key="stock_bt", format_func=lambda x: f"{x} · {stocks.STOCKS[x]}")
    rule = bar.segmented_control("Trend rule", [50, 100, 200], key="stock_bt_rule", default=200, required=True,
                                 format_func=lambda n: f"{n}-day")
    c = stocks.history(t)
    hold = (1 + c.pct_change().fillna(0)).cumprod()
    pos = (c > c.rolling(rule).mean()).astype(float).shift(1).fillna(0)
    trade = (1 + (pos * c.pct_change() - pos.diff().abs() * stocks.COST).fillna(0)).cumprod()
    chart = pd.DataFrame({"Hold": 1000 * hold, f"{rule}-day rule": 1000 * trade})
    chart.index = local(chart.index).tz_localize(None)
    st.line_chart(chart, color=[BLUE, GREY], height=340, y_label="Balance of $1,000 (USD)", x_label="",
                  alt=f"{t}: holding vs the {rule}-day trend rule")
    row = a.loc[t]
    rows = []
    for name, key in (("Hold", "hold"), (f"{rule}-day rule", f"sma{rule}")):
        for label, period in (("choose", "2016-2022"), ("check", "2023-2026")):
            rows.append({"Strategy": name, "Period": period, "Return": row[f"{key}_{label}_return"],
                         "Sharpe": row[f"{key}_{label}_sharpe"], "Worst fall": row[f"{key}_{label}_worst"]})
    table(pd.DataFrame(rows).style.format({"Return": "{:+.0%}", "Sharpe": "{:.2f}", "Worst fall": "{:+.0%}"},
                                                 na_rep="–"),
                 hide_index=True, width="stretch", alt=f"{t} hold vs trend rule by period")
    st.caption(f"Rule: hold {t} while its daily close is above its {rule}-day average, else cash; 0.1% cost per trade. "
               "Across all 17 stocks, the best rule (chosen on 2016-2022) beat holding on none in 2023-2026.")

    ai = stockai.load()
    if ai:
        st.subheader("Trading on the stock AI")
        names = {"hold": "Hold all", "ai_gate": "AI risk filter (out below 45%)", "ai_tilt": "AI-sized"}
        rows = [{"Strategy": names[k], "Period": p, "Return": v[p]["return"], "Sharpe": v[p]["sharpe"],
                 "Worst fall": v[p]["worst"]} for k, v in ai["strategies"].items() for p in v]
        table(pd.DataFrame(rows).style.format({"Return": "{:+.0%}", "Sharpe": "{:.2f}", "Worst fall": "{:+.0%}"}),
                     hide_index=True, width="stretch", alt="Stock AI strategies vs holding")
        st.caption(f"Equal-weight basket of all the stocks, decisions on Mondays. The AI variant chosen on 2019-2022 "
                   f"({names[ai['chosen']]}) {'beat' if ai['trade_on_ai'] else 'did not beat'} holding in both periods.")


def page_stock_portfolio():
    st.title("Stock portfolio")
    holdings, cash = stocks.load_portfolio()
    try:
        a = get_stock_analysis()
        live = {t: p[0] for t, p in get_stock_prices().items()}
    except Exception:
        a, live = pd.DataFrame(), {}
    if holdings:
        adv, total, plan = stocks.advise(holdings, cash, a, live)
        cards = st.container(horizontal=True, gap="medium")
        invested = (adv["shares"] * adv["avg_cost"]).sum()
        cards.metric("Total value", f"${total:,.2f}", border=True)
        cards.metric("Profit / loss", f"{adv['pnl'].sum():+,.2f}", f"{adv['pnl'].sum() / invested:+.2%}" if invested else None,
                     border=True)
        cards.metric("Spare cash", f"${cash:,.2f}", border=True)
        st.subheader("Advice")
        table(
            pd.DataFrame({"Stock": adv["ticker"], "Advice": adv["action"], "Value": adv["value"], "Share": adv["weight"],
                          "P&L %": adv["pnl_pct"], "Why": adv["reason"]})
            .style.format({"Value": "${:,.2f}", "P&L %": "{:+.1%}"}, na_rep="–")
            .map(lambda x: "color: #F4B000; font-weight: 600" if x == "TRIM" else "", subset=["Advice"]),
            hide_index=True, width="stretch", alt="Advice for each stock you hold",
            column_config={"Share": st.column_config.ProgressColumn("Share of portfolio", format="percent", min_value=0,
                                                                    max_value=1),
                           "Why": st.column_config.TextColumn(width="large")})
        if plan:
            st.info(plan, icon=":material/savings:")
        st.caption("Prices from Binance's stock futures. Hold-first advice: in testing, holding beat trend-rule trading "
                   "on every stock here. Not financial advice.")
    else:
        with st.container(border=True):
            st.markdown("**No stocks yet**")
            st.caption("Add what you own below to get hold-first advice for your stocks.")
    with st.expander("Edit stocks and cash", icon=":material/edit:", expanded=not holdings):
        new_cash = st.number_input("Spare cash for stocks (USD)", min_value=0.0, value=float(cash), step=50.0)
        edited = st.data_editor(
            pd.DataFrame(holdings, columns=["ticker", "shares", "avg_cost"]), num_rows="dynamic", width="stretch",
            key="stock_holdings_editor",
            column_config={"ticker": st.column_config.SelectboxColumn("Stock", options=list(stocks.STOCKS), required=True),
                           "shares": st.column_config.NumberColumn("Shares", format="%.4f", min_value=0, required=True),
                           "avg_cost": st.column_config.NumberColumn("Average cost (USD)", format="%.2f", min_value=0,
                                                                     required=True)})
        if st.button("Save", icon=":material/save:", type="primary", key="stock_save"):
            stocks.save_portfolio(edited.dropna().to_dict("records"), new_cash)
            st.toast("Saved", icon=":material/check:")
            st.rerun()
        st.caption("Saved only on this computer, in stock_portfolio.json.")


def page_stock_paper():
    st.title("Stock paper trading")
    acct = stockpaper.load()
    if acct is None:
        with st.container(border=True, width=560):
            st.markdown("**Start a 4-week stock paper trial**")
            st.caption("Six pretend accounts: the buy-for-hold list (monthly), hold all stocks, all in QQQ, buy in 4 "
                       "weekly steps, the 200-day trend rule, and the AI risk filter. Decisions at the US close, at "
                       "Binance stock prices. Nothing real is traded.")
            cash = st.number_input("Starting balance per account (USD)", min_value=100.0, value=1000.0, step=100.0,
                                   key="stock_paper_cash")
            if st.button("Start trial", type="primary", icon=":material/play_arrow:", key="stock_paper_start"):
                with st.spinner("Starting…"):
                    stockpaper.open_account(cash)
                st.rerun()
        return
    stock_paper_dashboard()


@st.fragment(run_every=30)
def stock_paper_dashboard():
    acct = stockpaper.load()
    try:
        prices = {t: p[0] for t, p in get_stock_prices().items()}
    except Exception:
        st.warning("Could not reach Binance for prices.")
        return
    v = stockpaper.value(acct, prices)
    start, names = acct["start_cash"], acct["names"]
    st.caption(f"Trial: {local(acct['opened']):%b %d, %H:%M} → {when(acct['ends'])}. Decisions at the US close "
               "(16:00 New York), filled at that moment's Binance stock-futures price. Nothing real is traded.")
    cards = st.container(horizontal=True, gap="medium")
    for k in sorted(v, key=lambda k: -v[k]):
        cards.metric(names[k], f"${v[k]:,.2f}", f"{v[k] - start:+,.2f} ({v[k] / start - 1:+.2%})", border=True)
    hist = stockpaper.balance_history()
    if len(hist) > 1:
        h = hist.set_index(local(hist["time"]).dt.tz_localize(None))[list(names)].rename(columns=names)
        st.line_chart(h, height=300, y_label="Balance (USD)", x_label="", alt="Balance of every stock paper account")
    with st.expander("What each account does", icon=":material/info:"):
        st.markdown(
            "- **Buy-for-hold list (monthly)**: an equal share of every stock above its 200-day average, rebalanced "
            "at each month's last US close (the tested rule on Stock signals).\n"
            "- **Hold all**: every stock (ETFs excluded) bought equally at the start, never sold.\n"
            "- **Nasdaq-100 ETF**: all in QQQ, never sold.\n"
            "- **Buy in 4 weekly steps**: the same stocks, bought a quarter at a time each week.\n"
            "- **200-day trend rule**: each stock held only while above its 200-day average (did not beat holding "
            "in testing).\n"
            "- **AI risk filter**: each stock held unless the stock AI's P(up, 5 days) is below 45% (did not beat "
            "holding in testing).")
    t = stockpaper.trades().iloc[::-1]
    if len(t):
        st.subheader("Trade history")
        table(pd.DataFrame({f"Time ({tz_label()})": local(t["time"]).dt.strftime("%b %d, %H:%M"),
                                   "Account": t["account"].map(names), "Stock": t["symbol"], "Side": t["side"],
                                   "Price": t["price"], "Total (USD)": t["total"], "Reason": t["reason"]})
                     .style.map(lambda x: f"color: {UP}; font-weight: 600" if x == "BUY" else
                                f"color: {DOWN}; font-weight: 600" if x == "SELL" else "", subset=["Side"]),
                     hide_index=True, width="stretch", alt="Stock paper trades",
                     column_config={"Price": st.column_config.NumberColumn(format="dollar"),
                                    "Total (USD)": st.column_config.NumberColumn(format="dollar")})


def page_strategy_lab():
    st.title("Strategy lab")
    st.caption("Every idea tested so far. Unlike a bot marketplace that ranks by the last few days' ROI, each idea "
               "here was fixed before its results were seen, chosen on one period and judged on another, after "
               "fees, against a benchmark (usually buy & hold and the 50-day trend rule).")
    log = research_log.LOG
    counts = log["Verdict"].value_counts()
    cards = st.container(horizontal=True, gap="medium")
    cards.metric("Strategy variants compared", config.TRIALS_TESTED, border=True,
                 help="Counted for the deflated Sharpe ratio, which corrects for picking the best of many tries.")
    cards.metric("Ideas tested", len(log), border=True)
    cards.metric("Adopted", int(counts.get(research_log.ADOPTED, 0)), border=True)
    cards.metric("Not adopted", int(counts.get(research_log.NOT_ADOPTED, 0)), border=True)
    pick = st.segmented_control("Show", ["All", *log["Verdict"].unique()], default="All", key="lab_filter") or "All"
    shown = log if pick == "All" else log[log["Verdict"] == pick]
    colors = {research_log.ADOPTED: UP, research_log.NOT_ADOPTED: DOWN, research_log.PARTLY: "#F4B000"}
    table(shown.iloc[::-1].style.map(lambda x: f"color: {colors[x]}; font-weight: 600" if x in colors else "",
                                            subset=["Verdict"]),
                 hide_index=True, width="stretch", alt="Every tested idea with its result and verdict",
                 column_config={"Idea": st.column_config.TextColumn(width="large"),
                                "Result (data not used for choosing)": st.column_config.TextColumn(width="large")})
    st.caption("Most ideas fail; that is expected. The plain 50-day trend rule has been the hardest benchmark to "
               "beat. The AI's value so far is mostly in risk: the drop warning, and holding less when it is unsure.")


pages = {
    "Top coins": [
        st.Page(page_market, title="Market", icon=":material/monitoring:", url_path="market", default=True),
        st.Page(page_signals, title="Signals", icon=":material/bolt:", url_path="signals"),
        st.Page(page_chart, title="Chart", icon=":material/candlestick_chart:", url_path="chart"),
        st.Page(page_backtest, title="Backtest", icon=":material/history:", url_path="backtest"),
        st.Page(page_portfolio, title="Portfolio", icon=":material/account_balance_wallet:", url_path="portfolio"),
        st.Page(page_paper, title="Paper trading", icon=":material/science:", url_path="paper"),
    ],
    "Altcoins": [
        st.Page(page_altcoins, title="Altcoin market", icon=":material/rocket_launch:", url_path="altcoins"),
        st.Page(page_paper_alts, title="Altcoin paper trading", icon=":material/science:", url_path="altcoins-paper"),
        st.Page(page_alt_learnt, title="What the AI learnt", icon=":material/school:", url_path="altcoins-learnt"),
    ],
    "US stocks": [
        st.Page(page_stocks, title="Stock market", icon=":material/show_chart:", url_path="stocks"),
        st.Page(page_stock_signals, title="Stock signals", icon=":material/bolt:", url_path="stock-signals"),
        st.Page(page_stock_backtest, title="Stock backtest", icon=":material/history:", url_path="stock-backtest"),
        st.Page(page_stock_portfolio, title="Stock portfolio", icon=":material/account_balance_wallet:",
                url_path="stock-portfolio"),
        st.Page(page_stock_paper, title="Stock paper trading", icon=":material/science:", url_path="stock-paper"),
    ],
    "Research": [
        st.Page(page_patterns, title="Patterns the AI learnt", icon=":material/psychology:", url_path="patterns"),
        st.Page(page_strategy_lab, title="Strategy lab", icon=":material/biotech:", url_path="strategy-lab"),
    ],
}
if online_request():
    pages["Account"] = [st.Page(page_sign_out, title="Sign out", icon=":material/logout:", url_path="sign-out")]
nav = st.navigation(pages, position="top")
st.container(horizontal=True, horizontal_alignment="right").segmented_control(
    "Time zone", list(TIMEZONES), key="tz", default="UTC", required=True, bind="query-params",
    format_func=lambda z: "UTC" if z == "UTC" else "Vietnam (UTC+7)", label_visibility="collapsed")
nav.run()
