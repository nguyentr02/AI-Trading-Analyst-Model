"""Command line: python -m cryptoai [train|signals|backtest]"""
import argparse

from . import backtest, config, model, signals


def main():
    ap = argparse.ArgumentParser(prog="cryptoai")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train", help="download data, train and evaluate the models")
    sub.add_parser("signals", help="print current signals and log any changes")
    sub.add_parser("backtest", help="print backtest results from the last training run")
    args = ap.parse_args()

    if args.cmd == "train":
        for tf in config.TIMEFRAMES:
            print(f"Training {tf} ...")
            for k, v in model.train(tf).items():
                print(f"  {k}: {v}")

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
        for tf in config.TIMEFRAMES:
            oos = model.load_oos(tf)
            if oos is None:
                print(f"No {tf} results - run train first.")
                continue
            print(f"\n=== {tf} (out-of-sample, fee {config.FEE:.2%}/side) ===")
            print(f"{'symbol':<10}{'strat':>9}{'b&h':>9}{'strat DD':>10}{'b&h DD':>9}{'sharpe':>8}{'trades':>8}")
            for sym, g in oos.groupby("symbol"):
                _, s = backtest.run(g, tf)
                st, bh = s["strategy"], s["buy_hold"]
                print(
                    f"{sym:<10}{st['total_return']:>9.0%}{bh['total_return']:>9.0%}"
                    f"{st['max_drawdown']:>10.0%}{bh['max_drawdown']:>9.0%}"
                    f"{st['sharpe']:>8.2f}{s['num_trades']:>8}"
                )


if __name__ == "__main__":
    main()
