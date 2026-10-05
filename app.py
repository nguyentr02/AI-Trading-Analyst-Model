"""Dashboard. Run with:  .venv\\Scripts\\streamlit run app.py"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from cryptoai import backtest, config, data, live, market, model, portfolio, signals

ASSETS = config.ROOT / "assets"
st.set_page_config(page_title="Crypto AI", page_icon=str(ASSETS / "icon.svg"), layout="wide")
st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "icon.svg"), size="large")

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
SIGNAL_ICON = {"BULLISH": "▲ BULLISH", "BEARISH": "▼ BEARISH", "NEUTRAL": "● NEUTRAL"}


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


@st.cache_data(ttl=300, max_entries=8, show_spinner="Fetching latest candles…")
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


@st.cache_data(ttl=300)
def get_candles(sym, tf):
    return data.drop_open_candle(data.update(sym, tf), tf)


# Keep the timeframe and symbol selection when moving between pages.
st.session_state.setdefault("model", "4h")
st.session_state.setdefault("chart_interval", "4h")
st.session_state.setdefault("chart_ind", ["MA", "Volume", "AI signal"])
st.session_state.setdefault("sym", config.SYMBOLS[0])
for k in ("model", "sym", "chart_interval", "chart_ind"):
    st.session_state[k] = st.session_state[k]


def controls(symbol=True):
    """Timeframe and symbol pickers shown at the top of the model pages."""
    c1, c2, _ = st.columns([2.4, 1.6, 3])
    tf = c1.segmented_control("Prediction", list(config.MODELS), key="model", required=True,
                              format_func=lambda n: config.MODELS[n]["label"])
    sym = c2.selectbox("Symbol", config.SYMBOLS, key="sym") if symbol else None
    if model.load(tf) is None:
        st.error("No model yet. Run `python -m cryptoai train` or press **Retrain models** on the Backtest page.")
        st.stop()
    return tf, sym


# ---------------- market (live) ----------------
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

    cols = st.columns(len(df))
    for c, r in zip(cols, df.itertuples()):
        c.metric(r.symbol, f"${r.price:,.2f}", f"{r.change_24h:+.2%} 24h")
        c.caption(f"Market cap **{usd(getattr(r, 'market_cap', None))}**"
                  + (f" · rank #{int(r.rank)}" if pd.notna(getattr(r, "rank", None)) else ""))

    col = lambda name: df.get(name, pd.Series(index=df.index, dtype=float))  # CoinGecko columns may be missing
    dollars = lambda v: "–" if pd.isna(v) else f"${v:,.2f}"

    # Two narrower tables instead of one wide one, so neither needs a horizontal scrollbar.
    st.subheader("Price action")
    price = pd.DataFrame({
        "Coin": df["symbol"],
        "Price": df["price"].map(dollars),
        "1h": col("change_1h"),
        "24h": df["change_24h"],
        "7d": col("change_7d"),
        "24h high": df["high_24h"].map(dollars),
        "24h low": df["low_24h"].map(dollars),
        "Volume 24h (Binance)": df["quote_volume_24h"].map(usd),
    })
    colored_table(price, ["1h", "24h", "7d"], "Live price, change and 24h range for each coin")

    st.subheader("Market data")
    mkt = pd.DataFrame({
        "Coin": df["symbol"],
        "Market cap": col("market_cap").map(usd),
        "FDV": col("fdv").map(usd),
        "Volume 24h (all exchanges)": col("total_volume_usd").map(usd),
        "Circulating supply": col("circulating_supply").map(amount),
        "Max supply": col("max_supply").map(amount),
        "All-time high": col("ath").map(dollars),
        "From ATH": col("from_ath"),
    })
    colored_table(mkt, ["From ATH"], "Market cap, supply and all-time high for each coin")
    st.caption(f"Live. Price, 24h change, high/low and Binance volume streamed from Binance and redrawn every "
               f"{config.LIVE_REFRESH}s. Market cap, supply, 1h/7d change and all-exchange volume from CoinGecko, "
               f"refreshed every 60s. Last update {tk['updated'].max():%H:%M:%S} UTC.")


def colored_table(table, pct, alt):
    """Dataframe with percentage columns formatted and colored green (up) or red (down)."""
    st.dataframe(
        table.style.format({k: "{:+.2%}" for k in pct}, na_rep="–")
        .map(lambda v: f"color: {UP}" if pd.notna(v) and v > 0 else f"color: {DOWN}" if pd.notna(v) and v < 0 else "",
             subset=pct),
        hide_index=True, width="stretch", alt=alt,
    )


def page_market():
    st.title("Market")
    live_market()


# ---------------- signals ----------------
def page_signals():
    st.title("Signals")
    tf, _ = controls(symbol=False)
    signal_board(tf)


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
                 help="The service retrains and updates signals within seconds of every 4h and 1d candle close.")
    else:
        st.badge(f"Live learning: {s['state']}", icon=":material/sync:", color="orange")


@st.fragment(run_every=60)
def signal_board(tf):
    """Re-checks every minute so a signal from a just-closed candle appears without reloading the page."""
    service_badge()
    sig = get_signals(tf, model_version(tf))
    cols = st.columns(len(sig))
    for c, r in zip(cols, sig.itertuples()):
        c.metric(r.symbol, f"{r.price:,.2f}" if r.price >= 1 else f"{r.price:.5f}", SIGNAL_ICON[r.signal],
                 delta_color="normal" if r.signal == "BULLISH" else "inverse" if r.signal == "BEARISH" else "off")
        c.caption(f"P(up, {config.MODELS[tf]['label'].lower()}): **{r.prob_up:.0%}**")
    st.caption(
        f"BULLISH when P(up) ≥ {config.ENTER_PROB:.0%}, BEARISH when ≤ {config.EXIT_PROB:.0%}. "
        f"Based on the last closed {config.MODELS[tf]['timeframe']} candle. "
        + TRUST.get(tf, "")
        + f"Signals are statistical estimates with a small edge, not "
        f"advice. Size positions so being wrong is affordable."
    )
    if config.SIGNAL_LOG.exists():
        st.subheader("Recent signal changes")
        log = pd.read_csv(config.SIGNAL_LOG)
        changes = log[log["changed"]].tail(20).iloc[::-1]
        st.dataframe(changes[["checked_at", "symbol", "timeframe", "price", "prob_up", "signal"]],
                     hide_index=True, width="stretch")
    else:
        st.info("Run `python -m cryptoai signals` on a schedule to log signal changes (alerts).")


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
    tf, sym = controls()
    model_status(tf)
    oos = model.load_oos(tf)
    g = oos[oos["symbol"] == sym]
    c1, c2 = st.columns(2)
    enter = c1.slider("Enter when P(up) >", 0.50, 0.70, config.ENTER_PROB, 0.01)
    exit_ = c2.slider("Exit when P(up) <", 0.30, 0.55, config.EXIT_PROB, 0.01)
    ctf = config.MODELS[tf]["timeframe"]
    eq, s = backtest.run(g, ctf, enter, min(exit_, enter))
    st_, bh = s["strategy"], s["buy_hold"]

    m = st.columns(5)
    m[0].metric("Strategy return", f"{st_['total_return']:.0%}", f"{st_['total_return'] - bh['total_return']:+.0%} vs hold")
    m[1].metric("Max drawdown", f"{st_['max_drawdown']:.0%}", f"hold: {bh['max_drawdown']:.0%}", delta_color="off")
    m[2].metric("Sharpe", f"{st_['sharpe']:.2f}", f"hold: {bh['sharpe']:.2f}", delta_color="off")
    m[3].metric("Trades", s["num_trades"], f"win rate {s['win_rate']:.0%}", delta_color="off")
    m[4].metric("Time in market", f"{s['time_in_market']:.0%}", f"fees {s['fees_paid_pct']:.0%}", delta_color="off")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq.strategy, name="AI strategy", line=dict(color=BLUE, width=2),
                             hovertemplate="%{y:.2f}x"))
    fig.add_trace(go.Scatter(x=eq.index, y=eq.buy_hold, name="Buy & hold", line=dict(color=GREY, width=2),
                             hovertemplate="%{y:.2f}x"))
    fig.update_yaxes(type="log", title="Growth of $1 (log)")
    st.plotly_chart(style(fig, 450), width="stretch")
    st.caption(f"Walk-forward out-of-sample from {eq.index[0]:%Y-%m-%d}: each period is predicted by a model "
               f"trained only on earlier data. Fee {config.FEE:.2%} per trade side. Long or flat only.")

    history = model.load_log()
    if not history.empty:
        with st.expander("Learning history"):
            h = history[history["model"] == tf]
            fig = go.Figure(go.Scatter(x=h["trained_at"], y=h["oos_auc"], mode="lines+markers",
                                       line=dict(color=BLUE, width=2), name="Walk-forward AUC",
                                       hovertemplate="%{y:.4f}"))
            fig.add_hline(y=config.MIN_AUC, line=dict(color=GREY, width=1, dash="dot"),
                          annotation_text="minimum to accept", annotation_position="right")
            st.plotly_chart(style(fig, 260), width="stretch")
            st.dataframe(h.iloc[::-1][["trained_at", "last_candle", "rows", "oos_auc", "oos_accuracy", "accepted"]],
                         hide_index=True, width="stretch")
            st.caption("The model retrains every day on all data up to the last closed candle. "
                       "A new model replaces the old one only if it still beats a coin flip out of sample.")

    with st.expander("All symbols"):
        rows = []
        for sy, gg in oos.groupby("symbol"):
            _, ss = backtest.run(gg, ctf, enter, min(exit_, enter))
            rows.append({"symbol": sy, "strategy": ss["strategy"]["total_return"],
                         "buy_hold": ss["buy_hold"]["total_return"],
                         "strategy_dd": ss["strategy"]["max_drawdown"], "buy_hold_dd": ss["buy_hold"]["max_drawdown"],
                         "sharpe": ss["strategy"]["sharpe"], "trades": ss["num_trades"]})
        st.dataframe(pd.DataFrame(rows).style.format(
            {"strategy": "{:.0%}", "buy_hold": "{:.0%}", "strategy_dd": "{:.0%}", "buy_hold_dd": "{:.0%}",
             "sharpe": "{:.2f}"}), hide_index=True, width="stretch")


def model_status(tf):
    info, btn = st.columns([5, 1], vertical_alignment="center")
    metrics = model.load_metrics(tf)
    if metrics:
        info.caption(
            f"{config.MODELS[tf]['label']} model trained {metrics['trained_at'][:16].replace('T', ' ')} UTC. "
            f"Out-of-sample accuracy **{metrics['oos_accuracy']:.1%}** "
            f"(coin-flip baseline {max(metrics['baseline_up_rate'], 1 - metrics['baseline_up_rate']):.1%}), "
            f"AUC **{metrics['oos_auc']:.3f}**."
        )
    if btn.button("Retrain models", icon=":material/refresh:", help="Downloads fresh data and retrains (≈1 min)",
                  width="stretch"):
        with st.spinner("Training…"):
            for t in config.MODELS:
                model.train(t)
        st.cache_data.clear()
        st.rerun()


# ---------------- portfolio ----------------
def page_portfolio():
    st.title("Portfolio")
    tf = st.session_state["model"]
    st.caption("Stored locally in portfolio.json. Symbols must be Binance pairs like BTC/USDT.")
    edited = st.data_editor(
        pd.DataFrame(portfolio.load(), columns=["symbol", "amount", "avg_cost"]),
        num_rows="dynamic", width="stretch",
        column_config={"amount": st.column_config.NumberColumn(format="%.6f"),
                       "avg_cost": st.column_config.NumberColumn("avg cost (USDT)", format="%.4f")},
    )
    if st.button("Save holdings", icon=":material/save:", type="primary"):
        portfolio.save(edited.dropna().to_dict("records"))
        st.success("Saved.")
        st.rerun()

    holdings = portfolio.load()
    if holdings:
        try:
            pv = portfolio.valued(holdings)
        except Exception as e:
            st.error(f"Could not price holdings: {e}")
        else:
            a, b, c = st.columns(3)
            total, cost = pv["value"].sum(), (pv["amount"] * pv["avg_cost"]).sum()
            a.metric("Total value", f"${total:,.2f}")
            b.metric("Unrealised P&L", f"${total - cost:,.2f}", f"{(total - cost) / cost:+.1%}" if cost else None)
            sig = get_signals(tf, model_version(tf))
            sig_map = dict(zip(sig["symbol"], sig["signal"]))
            pv["AI signal"] = pv["symbol"].map(sig_map).fillna("not tracked")
            c.metric("Positions flagged BEARISH", int((pv["AI signal"] == "BEARISH").sum()))
            st.dataframe(pv.style.format({"price": "{:,.4f}", "value": "${:,.2f}", "pnl": "${:,.2f}",
                                          "pnl_pct": "{:+.1%}", "avg_cost": "{:,.4f}"}),
                         hide_index=True, width="stretch")


nav = st.navigation(
    [
        st.Page(page_market, title="Market", icon=":material/monitoring:", url_path="market", default=True),
        st.Page(page_signals, title="Signals", icon=":material/bolt:", url_path="signals"),
        st.Page(page_chart, title="Chart", icon=":material/candlestick_chart:", url_path="chart"),
        st.Page(page_backtest, title="Backtest", icon=":material/history:", url_path="backtest"),
        st.Page(page_portfolio, title="Portfolio", icon=":material/account_balance_wallet:", url_path="portfolio"),
    ],
    position="top",
)
nav.run()
