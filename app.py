"""Dashboard. Run with:  .venv\\Scripts\\streamlit run app.py"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from cryptoai import backtest, config, data, market, model, portfolio, signals

ASSETS = config.ROOT / "assets"
st.set_page_config(page_title="Crypto AI", page_icon=str(ASSETS / "icon.svg"), layout="wide")
st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "icon.svg"), size="large")

# Coinbase palette, matching .streamlit/config.toml. Plotly needs explicit colors per theme.
if st.context.theme.type == "dark":
    BLUE, GREY, UP, DOWN = "#578BFA", "#8A919E", "#27AD75", "#F0616D"
else:
    BLUE, GREY, UP, DOWN = "#0052FF", "#8A919E", "#098551", "#CF202F"
GRID = "rgba(138,145,158,0.18)"
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


@st.cache_data(ttl=300, show_spinner="Fetching latest candles…")
def get_signals(tf):
    return signals.current(tf)


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
st.session_state.setdefault("tf", "1d")
st.session_state.setdefault("sym", config.SYMBOLS[0])
for k in ("tf", "sym"):
    st.session_state[k] = st.session_state[k]


def controls(symbol=True):
    """Timeframe and symbol pickers shown at the top of the model pages."""
    c1, c2, _ = st.columns([1.2, 1.6, 4])
    tf = c1.radio("Timeframe", config.TIMEFRAMES, key="tf", horizontal=True)
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

    table = pd.DataFrame({
        "Coin": df["symbol"],
        "Price": df["price"].map(lambda v: f"${v:,.2f}"),
        "1h": df.get("change_1h"),
        "24h": df["change_24h"],
        "7d": df.get("change_7d"),
        "24h high": df["high_24h"].map(lambda v: f"${v:,.2f}"),
        "24h low": df["low_24h"].map(lambda v: f"${v:,.2f}"),
        "Volume 24h (Binance)": df["quote_volume_24h"].map(usd),
        "Volume 24h (all exchanges)": df.get("total_volume_usd", pd.Series(dtype=float)).map(usd),
        "Market cap": df.get("market_cap", pd.Series(dtype=float)).map(usd),
        "FDV": df.get("fdv", pd.Series(dtype=float)).map(usd),
        "Circulating supply": df.get("circulating_supply", pd.Series(dtype=float)).map(amount),
        "Max supply": df.get("max_supply", pd.Series(dtype=float)).map(amount),
        "All-time high": df.get("ath", pd.Series(dtype=float)).map(lambda v: "–" if pd.isna(v) else f"${v:,.2f}"),
        "From ATH": df.get("from_ath"),
    })
    pct = ["1h", "24h", "7d", "From ATH"]
    st.dataframe(
        table.style.format({k: "{:+.2%}" for k in pct}, na_rep="–")
        .map(lambda v: f"color: {UP}" if pd.notna(v) and v > 0 else f"color: {DOWN}" if pd.notna(v) and v < 0 else "",
             subset=pct),
        hide_index=True, width="stretch",
    )
    st.caption(f"Live. Price, 24h change, high/low and Binance volume streamed from Binance and redrawn every "
               f"{config.LIVE_REFRESH}s. Market cap, supply, 1h/7d change and all-exchange volume from CoinGecko, "
               f"refreshed every 60s. Last update {tk['updated'].max():%H:%M:%S} UTC.")


def page_market():
    st.title("Market")
    live_market()


# ---------------- signals ----------------
def page_signals():
    st.title("Signals")
    tf, _ = controls(symbol=False)
    sig = get_signals(tf)
    cols = st.columns(len(sig))
    for c, r in zip(cols, sig.itertuples()):
        c.metric(r.symbol, f"{r.price:,.2f}" if r.price >= 1 else f"{r.price:.5f}", SIGNAL_ICON[r.signal],
                 delta_color="normal" if r.signal == "BULLISH" else "inverse" if r.signal == "BEARISH" else "off")
        c.caption(f"P(up in {config.HORIZON[tf]} candles): **{r.prob_up:.0%}**")
    st.caption(
        f"BULLISH when P(up) ≥ {config.ENTER_PROB:.0%}, BEARISH when ≤ {config.EXIT_PROB:.0%}. "
        f"Based on the last closed {tf} candle. Signals are statistical estimates with a small edge, not "
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
def page_chart():
    st.title("Chart")
    tf, sym = controls()
    df = get_candles(sym, tf)
    lookback = st.slider("Candles shown", 60, 1000, 240, step=20)
    view = df.iloc[-lookback:]
    raw = {s: get_candles(s, tf).iloc[-(lookback + 250):] for s in config.SYMBOLS}
    prob = model.predict_history(raw, tf)[sym].reindex(view.index)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04,
                        subplot_titles=(f"{sym} · {tf}", "Model P(up)"))
    fig.add_trace(go.Candlestick(x=view.index, open=view.open, high=view.high, low=view.low, close=view.close,
                                 increasing_line_color=UP, decreasing_line_color=DOWN, name="Price",
                                 showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=prob.index, y=prob, line=dict(color=BLUE, width=2), name="P(up)",
                             showlegend=False, hovertemplate="%{y:.0%}"), row=2, col=1)
    for y, lbl in ((config.ENTER_PROB, "enter"), (config.EXIT_PROB, "exit")):
        fig.add_hline(y=y, line=dict(color=GREY, width=1, dash="dot"), row=2, col=1,
                      annotation_text=lbl, annotation_position="right")
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    st.plotly_chart(style(fig, 650), width="stretch")
    st.caption("Past probabilities here come from the final model, which was trained on this data, so they look "
               "better than reality. Use the Backtest page for honest performance.")


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
    eq, s = backtest.run(g, tf, enter, min(exit_, enter))
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
            h = history[history["timeframe"] == tf]
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
            _, ss = backtest.run(gg, tf, enter, min(exit_, enter))
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
            f"{tf} model trained {metrics['trained_at'][:16].replace('T', ' ')} UTC. "
            f"Out-of-sample accuracy **{metrics['oos_accuracy']:.1%}** "
            f"(coin-flip baseline {max(metrics['baseline_up_rate'], 1 - metrics['baseline_up_rate']):.1%}), "
            f"AUC **{metrics['oos_auc']:.3f}**."
        )
    if btn.button("Retrain models", icon=":material/refresh:", help="Downloads fresh data and retrains (≈1 min)",
                  width="stretch"):
        with st.spinner("Training…"):
            for t in config.TIMEFRAMES:
                model.train(t)
        st.cache_data.clear()
        st.rerun()


# ---------------- portfolio ----------------
def page_portfolio():
    st.title("Portfolio")
    tf = st.session_state["tf"]
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
            sig_map = dict(zip(get_signals(tf)["symbol"], get_signals(tf)["signal"]))
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
