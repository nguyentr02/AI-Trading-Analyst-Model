"""Command line: python -m cryptoai --help lists every command (train, signals, advice, live, simulate, ...)."""
import argparse

import pandas as pd

from . import backtest, config, market, model, signals


def main():
    ap = argparse.ArgumentParser(prog="cryptoai")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train", help="download data, train and evaluate the models")
    sub.add_parser("signals", help="print current signals and log any changes")
    sub.add_parser("backtest", help="print backtest results from the last training run")
    sub.add_parser("market", help="print live price, volume and market cap")
    sub.add_parser("daily", help="daily learning: fetch new candles, retrain, keep the better model, log signals")
    sub.add_parser("live", help="always-on service: learn and signal at every candle close, catch up after downtime")
    sub.add_parser("advice", help="buy / sell / hold advice for your saved portfolio")
    sub.add_parser("notify-test", help="send a test alert to Windows and Zalo")
    zs = sub.add_parser("zalo-setup", help="connect your Zalo bot (message the bot from Zalo first)")
    zs.add_argument("token", help="bot token from bot.zaloplatforms.com")
    sim = sub.add_parser("simulate", help="paper-trade a coin over a past period from a starting balance")
    sim.add_argument("symbol", help="e.g. BNB/USDT")
    sim.add_argument("start", help="first day, e.g. 2026-01-01")
    sim.add_argument("end", help="day after the last day, e.g. 2026-06-01")
    sim.add_argument("--cash", type=float, default=1000.0, help="starting balance in USDT (default 1000)")
    sim.add_argument("--stop", choices=["fixed", "trailing"], help="add a stop-loss: fixed (from the buy price) "
                     "or trailing (from the highest close since buying)")
    sim.add_argument("--stop-pct", type=float, default=0.10, help="stop-loss distance, e.g. 0.10 = 10%% (default)")
    args = ap.parse_args()

    if args.cmd == "train":
        for name, spec in config.MODELS.items():
            print(f"Training {name} ({spec['label']}) ...")
            for k, v in model.train(name).items():
                print(f"  {k}: {v}")

    elif args.cmd == "advice":
        from . import advisor

        advice = advisor.current(refresh=True)
        for r in advice.itertuples():
            amount = f"  ~${r.amount_usdt:,.0f}" if r.action in ("BUY", "SELL") else ""
            print(f"{r.action:<12}{r.symbol:<10}{amount}")
            print(f"            {r.reason}")
            if isinstance(r.timing, str):
                print(f"            Timing: {r.timing}")

    elif args.cmd == "notify-test":
        from . import notify

        for channel, result in notify.send("Crypto AI: test alert", "If you can read this, alerts work.").items():
            print(f"{channel}: {result}")

    elif args.cmd == "zalo-setup":
        from . import notify

        chat_id, name = notify.zalo_find_chat_id(args.token)
        if not chat_id:
            print("No messages found. Open your bot in Zalo, send it any message (e.g. 'hi'), then run this again.")
            return
        s = notify.settings()
        s["zalo"] = {"token": args.token, "chat_id": chat_id}
        notify.save_settings(s)
        notify.zalo_send(args.token, chat_id, "Crypto AI is connected. You'll get buy/sell alerts here.")
        print(f"Connected to {name or chat_id}. A confirmation message was sent to your Zalo.")

    elif args.cmd == "simulate":
        from . import simulate

        stop = {"type": args.stop, "pct": args.stop_pct} if args.stop else None
        summary, trades, equity = simulate.simulate(args.symbol, args.start, args.end, cash=args.cash, stop=stop)
        out = config.ROOT / "reports"
        out.mkdir(exist_ok=True)
        tag = f"{args.symbol.replace('/', '_')}_{args.start}_{args.end}"
        if stop:
            tag += f"_{args.stop}{args.stop_pct:.0%}".replace("%", "pct")
        stop_text = f", {args.stop} stop-loss at {args.stop_pct:.0%}" if stop else ""
        print(f"\n{args.symbol} {args.start} to {args.end}, starting with ${args.cash:,.2f}, "
              f"fee {config.FEE:.2%} per trade{stop_text}\n")
        for name, r in summary.iterrows():
            stops = f"   {int(r['stops hit'])} stopped out" if stop and name != "Buy & hold" else ""
            print(f"  {name:<28} ${r['final balance']:>9,.2f}  {r['return']:+7.1%}   worst drop {r['worst drop']:+6.1%}"
                  f"   {int(r['trades']):>3} trades   win rate {r['win rate']:>4.0%}   fees ${r['fees paid']:,.2f}{stops}")
            print(f"  {'':<28} Sharpe {r['sharpe']:+.2f} +/- {r['sharpe_se']:.2f}   Sortino {r['sortino']:+.2f}"
                  f"   chance Sharpe > 0: {r['chance sharpe > 0']:.0%}")
        days = (pd.Timestamp(args.end) - pd.Timestamp(args.start)).days
        if days < 730:
            print(f"\n  Note: {days} days is short. The +/- on each Sharpe shows how uncertain it is; a difference "
                  "smaller than about twice that could be luck.")
        for name, t in trades.items():
            t.to_csv(out / f"{tag}_{name.split(' (')[0].replace(' ', '_').lower()}_trades.csv", index=False)
        equity.to_csv(out / f"{tag}_balance.csv")
        simulate.balance_chart(equity, args.symbol, args.cash).write_html(out / f"{tag}_balance.html")
        print(f"\nTrade lists, balance history and a balance chart (.html) saved in {out}")

    elif args.cmd == "live":
        from .live import LiveService

        LiveService().run()

    elif args.cmd == "daily":
        from datetime import datetime, timezone

        print(f"=== Daily learning run {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC ===")
        for tf in config.MODELS:
            prev = model.load_metrics(tf)
            m = model.train(tf, min_auc=config.MIN_AUC)
            change = f" (was {prev['oos_auc']})" if prev else ""
            verdict = "model updated" if m["accepted"] else f"REJECTED, below {config.MIN_AUC}: kept previous model"
            print(f"{tf}: data to {m['last_candle'][:16]}, {m['rows']} rows, "
                  f"walk-forward AUC {m['oos_auc']}{change}, accuracy {m['oos_accuracy']:.1%} -> {verdict}")
        allsig, changed = signals.check_and_log()
        print(allsig[["symbol", "timeframe", "price", "prob_up", "signal"]].to_string(index=False))
        for r in changed.itertuples():
            print(f"  CHANGED {r.symbol} {r.timeframe}: {r.signal} (P(up)={r.prob_up})")

    elif args.cmd == "signals":
        allsig, changed = signals.check_and_log()
        print(allsig[["symbol", "timeframe", "price", "prob_up", "signal"]].to_string(index=False))
        if changed.empty:
            print("\nNo signal changes since last check.")
        else:
            print("\n*** SIGNAL CHANGES ***")
            for r in changed.itertuples():
                print(f"  {r.symbol} {r.timeframe}: {r.signal} (P(up)={r.prob_up})")

    elif args.cmd == "backtest":
        for tf in config.MODELS:
            oos = model.load_oos(tf)
            if oos is None:
                print(f"No {tf} results - run train first.")
                continue
            print(f"\n=== {tf} (out-of-sample, fee {config.FEE:.2%}/side) ===")
            print(f"{'symbol':<10}{'strat':>9}{'b&h':>9}{'strat DD':>10}{'b&h DD':>9}{'sharpe':>8}{'trades':>8}")
            for sym, g in oos.groupby("symbol"):
                _, s = backtest.run(g, config.MODELS[tf]["timeframe"])
                st, bh = s["strategy"], s["buy_hold"]
                print(
                    f"{sym:<10}{st['total_return']:>9.0%}{bh['total_return']:>9.0%}"
                    f"{st['max_drawdown']:>10.0%}{bh['max_drawdown']:>9.0%}"
                    f"{st['sharpe']:>8.2f}{s['num_trades']:>8}"
                )

    elif args.cmd == "market":
        df = market.snapshot()
        for r in df.itertuples():
            print(f"\n{r.symbol}  ${r.price:,.2f}  ({r.change_24h:+.2%} 24h)")
            print(f"  24h high/low   ${r.high_24h:,.2f} / ${r.low_24h:,.2f}")
            print(f"  24h volume     ${r.quote_volume_24h:,.0f} on Binance")
            if "market_cap" in df:
                print(f"  24h volume     ${r.total_volume_usd:,.0f} all exchanges")
                print(f"  market cap     ${r.market_cap:,.0f}  (rank #{r.rank})")
                print(f"  circulating    {r.circulating_supply:,.0f}")
                print(f"  all-time high  ${r.ath:,.2f}  ({r.from_ath:+.1%} from ATH)")


if __name__ == "__main__":
    main()
