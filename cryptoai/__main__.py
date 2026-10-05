"""Command line: python -m cryptoai [train|signals|backtest|market|daily|live]"""
import argparse

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
    args = ap.parse_args()

    if args.cmd == "train":
        for name, spec in config.MODELS.items():
            print(f"Training {name} ({spec['label']}) ...")
            for k, v in model.train(name).items():
                print(f"  {k}: {v}")

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
