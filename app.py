"""Dashboard. Run with:  .venv\\Scripts\\streamlit run app.py"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from cryptoai import advisor, auth, backtest, config, data, live, market, metrics, model, notify, portfolio, signals

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
for k in ("model", "sym", "chart_interval", "chart_ind"):
    st.session_state[k] = st.session_state[k]


# ---------------- market (live) ----------------
COIN_NAMES = {"BTC/USDT": "Bitcoin", "ETH/USDT": "Ethereum", "BNB/USDT": "BNB", "SOL/USDT": "Solana"}
def coin_label(sym):
    """'Bitcoin  BTC', or just 'BNB' when the name and ticker are the same."""
    name, tick = COIN_NAMES.get(sym, sym), coin(sym)
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
    supply_table()


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
               f"1h/7d change come from CoinGecko every 60s. Last update {tk['updated'].max():%H:%M:%S} UTC.")


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


def colored_table(table, pct, alt, column_config=None):
    """Dataframe with percentage columns formatted and colored green (up) or red (down)."""
    st.dataframe(
        table.style.format({k: "{:+.2%}" for k in pct}, na_rep="–")
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
SIGNAL_ARROW = {"BULLISH": "▲ Bullish", "BEARISH": "▼ Bearish", "NEUTRAL": "● Neutral"}


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
        st.badge(f"Live learning service offline since {s['updated'][:16].replace('T', ' ')} UTC",
                 icon=":material/cloud_off:", color="red")
    elif s["state"] == "listening":
        last = max(s.get(f"last_learn_{n}", "") for n in config.MODELS)
        st.badge(f"Live learning: listening for candle closes · last learned {last[11:16]} UTC",
                 icon=":material/sensors:", color="green",
                 help="The service retrains and updates signals a few minutes after every 4h and 1d candle close.")
    else:
        st.badge(f"Live learning: {s['state']}", icon=":material/sync:", color="orange")


def signal_row(spec, sig):
    """One prediction inside a coin card: label and badge on one line, probability bar below."""
    color, icon = SIGNAL_STYLE[sig.signal]
    line = st.container(horizontal=True, vertical_alignment="center", gap="small")
    line.markdown(spec["label"], width="content")
    line.badge(sig.signal.title(), icon=icon, color=color)
    st.progress(float(sig.prob_up), text=f"P(up) {sig.prob_up:.0%}")


@st.fragment(run_every=60)
def signal_board():
    """Every coin's three predictions at once. Re-checks every minute so new signals appear by themselves."""
    names = [n for n in config.MODELS if model.load(n) is not None]
    if not names:
        st.error("No model yet. Run `train.bat`, or press **Retrain models** on the Backtest page.")
        return
    sigs = {n: get_signals(n, model_version(n)).set_index("symbol") for n in names}

    cards = st.columns(len(config.SYMBOLS), gap="medium")
    for card, sym in zip(cards, config.SYMBOLS):
        with card.container(border=True):
            close = sigs[names[0]].loc[sym, "price"]
            name, tick = COIN_NAMES[sym], coin(sym)
            st.markdown(f"#### {name}" + (f" :gray[{tick}]" if name != tick else ""))
            st.caption(f"Last close ${close:,.2f}")
            for n in names:
                signal_row(config.MODELS[n], sigs[n].loc[sym])

    with st.container(border=True):
        st.markdown("**How to read these**")
        st.caption(f"P(up) is the chance the price is higher at the end of each window. Bullish at "
                   f"{config.ENTER_PROB:.0%} or more, bearish at {config.EXIT_PROB:.0%} or less.")
        for n in names:
            st.caption(f"**{config.MODELS[n]['label']}:** " + TRUST.get(n, "The main signal for direction.").strip())
        st.caption("Estimates with a small edge, not advice. Size positions so being wrong is affordable.")

    st.subheader("Recent signal changes")
    if not config.SIGNAL_LOG.exists():
        st.caption("No signal changes logged yet. The live learning service logs them at every candle close.")
        return
    log = pd.read_csv(config.SIGNAL_LOG)
    log = log[log["changed"] & log["timeframe"].isin(config.MODELS) & log["symbol"].isin(config.SYMBOLS)]
    log = log.tail(30).iloc[::-1]
    st.dataframe(
        pd.DataFrame({
            "When (UTC)": pd.to_datetime(log["checked_at"], format="ISO8601").dt.strftime("%b %d, %H:%M"),
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
    ai_models = [n for n, s in config.MODELS.items() if s["timeframe"] == tf] if "AI signal" in shown else []

    rows = ["price"] + (["volume"] if "Volume" in shown else []) + (["ai"] if ai_models else [])
    heights = {"price": 0.64, "volume": 0.14, "ai": 0.22}
    fig = make_subplots(rows=len(rows), cols=1, shared_xaxes=True, vertical_spacing=0.035,
                        row_heights=[heights[r] / sum(heights[x] for x in rows) for r in rows])
    row = {r: i + 1 for i, r in enumerate(rows)}

    fig.add_trace(go.Candlestick(
        x=df.index, open=df.open, high=df.high, low=df.low, close=df.close, name=coin(sym),
        increasing=dict(line_color=UP, fillcolor=UP), decreasing=dict(line_color=DOWN, fillcolor=DOWN),
        showlegend=False), row=1, col=1)

    if "MA" in shown:
        for n, color in MA_COLORS.items():
            fig.add_trace(go.Scatter(x=df.index, y=df.close.rolling(n).mean(), name=f"MA({n})",
                                     line=dict(color=color, width=1.2), hovertemplate="%{y:,.2f}"), row=1, col=1)
    if "Bollinger" in shown:
        mid, sd = df.close.rolling(20).mean(), df.close.rolling(20).std()
        fig.add_trace(go.Scatter(x=df.index, y=mid + 2 * sd, name="BB upper", line=dict(color=BLUE, width=1),
                                 opacity=0.6, hovertemplate="%{y:,.2f}", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=mid - 2 * sd, name="BOLL(20, 2)", line=dict(color=BLUE, width=1),
                                 opacity=0.6, fill="tonexty", fillcolor="rgba(87,139,250,0.07)",
                                 hovertemplate="%{y:,.2f}"), row=1, col=1)

    # Current price line with a price tag on the axis, coloured by the forming candle's direction.
    last = df.iloc[-1]
    last_color = UP if last.close >= last.open else DOWN
    fig.add_hline(y=last.close, line=dict(color=last_color, width=1, dash="dot"), row=1, col=1)
    fig.add_annotation(x=1, xref="paper", y=last.close, yref="y", text=f" {last.close:,.2f} ", showarrow=False,
                       xanchor="left", font=dict(color="white", size=11), bgcolor=last_color)

    if "volume" in row:
        colors = [UP if c >= o else DOWN for o, c in zip(df.open, df.close)]
        fig.add_trace(go.Bar(x=df.index, y=df.volume, name="Volume", marker_color=colors, opacity=0.55,
                             showlegend=False, hovertemplate="%{y:,.0f}"), row=row["volume"], col=1)

    if ai_models:
        for name in ai_models:
            hist = ai_history(name, model_version(name))
            if hist is None or sym not in hist:
                continue
            p = hist[sym].reindex(df.index)
            main_model = name == ai_models[-1]
            fig.add_trace(go.Scatter(
                x=p.index, y=p, name=f"AI P(up) {config.MODELS[name]['label'].lower()}",
                line=dict(color=BLUE if main_model else GREY, width=2 if main_model else 1.2),
                hovertemplate="%{y:.0%}"), row=row["ai"], col=1)
        for y in (config.ENTER_PROB, config.EXIT_PROB):
            fig.add_hline(y=y, line=dict(color=GREY, width=1, dash="dot"), row=row["ai"], col=1)
        fig.update_yaxes(tickformat=".0%", row=row["ai"], col=1)

    # Open on the latest candles with room on the right, like an exchange chart, and keep the viewer's
    # zoom across the 10-second refreshes (uirevision) until they switch coin or interval.
    view = df.iloc[-INITIAL_CANDLES:]
    pad = (view.high.max() - view.low.min()) * 0.06
    fig.update_xaxes(range=[view.index[0], df.index[-1] + 6 * step])
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
            top.badge(sig.signal.title(), icon=icon, color=color)
            st.progress(float(sig.prob_up), text=f"P(up) {sig.prob_up:.0%}")
            st.caption(TRUST.get(name, "The main signal.").strip())
    st.caption(f"Bullish at {config.ENTER_PROB:.0%}+, bearish at {config.EXIT_PROB:.0%} or less. "
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
        fig.add_trace(go.Scatter(x=eq.index, y=eq[col_], name=label, line=dict(color=color, width=width),
                                 hovertemplate="%{y:.2f}x"), row=1, col=1)
        drawdown = eq[col_] / eq[col_].cummax() - 1
        fig.add_trace(go.Scatter(x=eq.index, y=drawdown, name=f"{label} drop", showlegend=False,
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
        st.dataframe(
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
        st.dataframe(pd.DataFrame(fees).style.format({"AI return": "{:+.0%}", "Sharpe": "{:.2f}"}),
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
    st.dataframe(
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
    fig = go.Figure(go.Scatter(x=h["trained_at"], y=h["oos_auc"], mode="lines+markers", name="AUC",
                               line=dict(color=BLUE, width=2), hovertemplate="%{y:.4f}"))
    fig.add_hline(y=config.MIN_AUC, line=dict(color=GREY, width=1, dash="dot"),
                  annotation_text="minimum to accept", annotation_position="bottom right")
    fig.update_yaxes(title_text="Walk-forward AUC")
    st.plotly_chart(style(fig, 240), width="stretch", alt="Model accuracy after each retrain")
    with st.expander("Every retrain", icon=":material/list:"):
        st.dataframe(
            h.iloc[::-1][["trained_at", "last_candle", "rows", "oos_auc", "oos_accuracy", "accepted"]].rename(columns={
                "trained_at": "Trained (UTC)", "last_candle": "Data up to", "rows": "Training rows",
                "oos_auc": "AUC", "oos_accuracy": "Accuracy", "accepted": "Kept"}),
            hide_index=True, width="stretch", alt="Every retraining run",
            column_config={"Accuracy": st.column_config.NumberColumn(format="percent"),
                           "Trained (UTC)": st.column_config.DatetimeColumn(format="MMM D, HH:mm"),
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
            st.markdown(f":gray[Last trained]  \n**{pd.Timestamp(m['trained_at']):%b %d, %H:%M} UTC**",
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
            stats.metric("Flagged bearish (next 1 day)", bearish, width="content")
    with right:
        fig = go.Figure(go.Pie(labels=pv["symbol"].map(coin), values=pv["value"], hole=0.62, sort=False,
                               marker=dict(colors=[COIN_COLORS.get(s, GREY) for s in pv["symbol"]]),
                               textinfo="label+percent", hovertemplate="%{label}: $%{value:,.2f}<extra></extra>"))
        fig.update_layout(height=240, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
        st.plotly_chart(fig, width="stretch", alt="Share of the portfolio in each coin")

    st.subheader("Holdings")
    table = pd.DataFrame({
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
        table[f"AI {config.MODELS[n]['label'].lower()}"] = pv["symbol"].map(sigs[n]).map(SIGNAL_ARROW).fillna("not tracked")
    signal_cols = [c for c in table if c.startswith("AI ")]
    st.dataframe(
        table.style.format({"P&L %": "{:+.1%}", "P&L": "{:+,.2f}"})
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


pages = [
    st.Page(page_market, title="Market", icon=":material/monitoring:", url_path="market", default=True),
    st.Page(page_signals, title="Signals", icon=":material/bolt:", url_path="signals"),
    st.Page(page_chart, title="Chart", icon=":material/candlestick_chart:", url_path="chart"),
    st.Page(page_backtest, title="Backtest", icon=":material/history:", url_path="backtest"),
    st.Page(page_portfolio, title="Portfolio", icon=":material/account_balance_wallet:", url_path="portfolio"),
]
if online_request():
    pages.append(st.Page(page_sign_out, title="Sign out", icon=":material/logout:", url_path="sign-out"))
nav = st.navigation(pages, position="top")
nav.run()
