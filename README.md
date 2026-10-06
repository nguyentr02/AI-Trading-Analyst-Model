# Crypto AI: signals and tracker

An AI that watches **BTC, ETH, BNB and SOL** on Binance, learns from their price history, and estimates
the chance each coin goes up:

| Prediction | Built on | Updates |
|---|---|---|
| **Next 4 hours** | 4h candles plus the 15m and 1h charts | every 4 hours |
| **Next 1 day** | 4h candles plus the 15m and 1h charts | every 4 hours |
| **Next 3 days** | daily candles | once a day |

It streams live prices, retrains itself a few minutes after every candle close, and shows everything in a
dashboard alongside your holdings. It **does not place trades**. You make every decision.

Everything runs on your own computer. It uses Binance's free public data (no account or API key) and does
not need Claude or any other AI service to run or learn.

---

## Part 1: Set it up (once per PC)

### 1. Install the tools

1. **Python 3.14** from [python.org](https://www.python.org/downloads/). In the installer, tick
   **"Add python.exe to PATH"**.
2. **Git** from [git-scm.com](https://git-scm.com/download/win). The default options are fine.

### 2. Download the code and install

Open **PowerShell** (Start menu, type "PowerShell"), then run these one at a time:

```
cd $HOME\Documents
git clone https://github.com/nguyentr02/AI-Trading-Analyst-Model.git
cd AI-Trading-Analyst-Model
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

### 3. Train the AI for the first time

Double-click **`train.bat`** in the folder. It downloads the price history since 2019 and trains both
models. The first run takes a few minutes; it prints the results and waits for a key press when done.

### 4. Make it run automatically

Choose one.

**A. On an always-on (24/7) PC** — recommended for the PC that does the learning:

1. Right-click the Start button > **Terminal (Admin)** or **Windows PowerShell (Admin)**.
2. Run (change the path if you cloned somewhere else):
   ```
   cd $HOME\Documents\AI-Trading-Analyst-Model
   powershell -ExecutionPolicy Bypass -File setup_autostart.ps1 -AtStartup
   ```
3. Turn off sleep: **Settings > System > Power > Screen and sleep > "When plugged in, put my device to
   sleep after" = Never**. The screen turning off is fine.

The AI now starts when Windows boots, even if nobody logs in, so it survives Windows Update restarts.

**B. On an everyday PC** (runs while you are logged in), in a normal PowerShell:

```
powershell -ExecutionPolicy Bypass -File setup_autostart.ps1
```

Either way, two things now run in the background with no window:

| Background task | What it does | Log file |
|---|---|---|
| **Crypto AI live service** | Learns from every closed candle and updates the signals | `logs\live.log` |
| **Crypto AI dashboard** | The dashboard website at http://localhost:8501 | `logs\dashboard.log` |

If either one crashes, it restarts itself after 30 seconds.

**C. No auto-start:** double-click `dashboard.bat` to open the dashboard and `live.bat` to run the learning
service. They stop when you close them or shut down.

> Use only one of A, B or C on a PC. If auto-start is set up, don't also double-click `dashboard.bat` or
> `live.bat`, or you get two copies running.

---

## Part 2: Use it

### Open the dashboard

Go to **http://localhost:8501** in your browser.

From your phone or another computer on the **same Wi-Fi**, use `http://<PC-IP>:8501`. To find the PC's IP,
run `ipconfig` on it and look for "IPv4 Address" (for example `192.168.1.157`). The first time, Windows
Firewall may ask whether to allow Python; allow it on private networks.

### Open it from anywhere (online access)

The 24/7 PC can put the dashboard on the internet through a free Cloudflare tunnel, protected by a login.
No Cloudflare account, domain or router changes are needed.

1. Set the login once, on the 24/7 PC, in PowerShell in the project folder:
   ```
   .venv\Scripts\python -m cryptoai set-login
   ```
   Enter a username and a password of at least 10 characters. Only a salted hash of the password is
   saved (in `dashboard_auth.json`, kept off GitHub). Running it again changes the login and signs
   everyone out.
2. Add online access to the auto-start, in an **Admin** PowerShell:
   ```
   powershell -ExecutionPolicy Bypass -File setup_autostart.ps1 -AtStartup -Online
   ```
   This adds a third background task, **Crypto AI online access**. The first run downloads Cloudflare's
   official `cloudflared` into the `tools` folder.
3. The dashboard's address looks like `https://random-words.trycloudflare.com`. It changes whenever the
   tunnel restarts, so every new address is sent to you as a Windows notification and on Zalo (set up Zalo
   as in *Advice for your portfolio*). The current address is also in `logs\public_url.txt`.

Opened through that address, the dashboard asks for the username and password. After signing in you get an
**access token** (valid 15 minutes, renewed automatically) and a **refresh token** (valid 7 days, saved in
your browser so reloading doesn't sign you out; it is swapped for a new one each time it is used). **Sign
out** in the top bar cancels the refresh token. Opening the dashboard on the PC itself or on your home Wi-Fi
needs no login.

The tunnel refuses to start until a login is set, so the dashboard is never online unprotected. To take it
offline again, run the setup command from step 2 without `-Online`.

### The pages

| Page | What you see |
|---|---|
| **Market** | Live prices updating every second, 24h change, high/low, volume, market cap, supply, all-time high, and whale activity (big trades, liquidations, open interest) |
| **Signals** | The AI's current call and drop risk for each coin, and a log of recent signal changes. The badge at the top shows whether the live learning service is running |
| **Chart** | Candlestick chart with the model's P(up) underneath |
| **Backtest** | How the strategy would have done on data the model had not seen, model accuracy, **Learning history**, and a **Retrain models** button |
| **Portfolio** | AI advice (buy / sell / hold, with amounts) for your holdings and spare cash, plus value, profit/loss and allocation |
| **Paper trading** | The 4-week trial: five pretend accounts trading live, balances, profit split into realised and unrealised, fees, result vs buy & hold, wallets and trade history |
| **Altcoins → Altcoin market** | Experimental: trend status, AI P(up) and drop risk for NEAR, ZEC, XRP, DOGE, AVAX and LINK, and the most volatile liquid altcoins on Binance |
| **Altcoins → Altcoin paper trading** | A separate 4-week trial of the same five accounts on those six altcoins |
| **Altcoins → What the AI learnt** | What testing found about altcoins |
| **Research → Strategy lab** | Every idea tested so far, with its result on data it wasn't chosen on, and whether it was adopted |

The navbar has three menus: **Top coins** (the pages for BTC, ETH, BNB and SOL), **Altcoins** and **Research**.

Use the **Prediction** switch (Next 4 hours, Next 1 day, Next 3 days) at the top of Signals, Chart and Backtest.

### Reading a signal

- **P(up)** is the model's estimated chance the price is higher at the end of the prediction window
  (4 hours, 1 day or 3 days from the last closed candle).
- **BULLISH** when P(up) is 55% or more, **BEARISH** at 48% or less, **NEUTRAL** in between.
- New "next 4 hours" and "next 1 day" signals are ready about 3 minutes after each 4h candle closes:
  00:00, 04:00, 08:00, 12:00, 16:00 and 20:00 UTC (07:00, 11:00, 15:00, 19:00, 23:00, 03:00 in Vietnam, UTC+7).
  The "next 3 days" signal updates once a day, shortly after 00:00 UTC.
- The Signals page refreshes itself every minute, so you don't need to reload.
- **Live now** (under each signal) is a provisional reading, updated every minute: the models re-run on the
  live Binance price as if the candle still forming closed now. It shows where the signal is heading before
  the candle closes. The confirmed signal, the alerts and the portfolio advice still update at candle closes.

**How much to trust it:** the models are right 52–53% of the time. That is a small edge, which is normal for
honest price prediction.

| Prediction | Accuracy (AUC*) | Track record | How to use it |
|---|---|---|---|
| **Next 4 hours** | 0.546 | Beat a coin flip in every year since 2021 | Timing only. Its edge is real before fees, but trading every change pays it all out in fees |
| **Next 1 day** | 0.533 | Beat a coin flip in every year; the version before the 15m/1h patterns passed a permutation test | The main signal |
| **Next 3 days** | 0.527 | No better than chance in 2023–2024; failed a permutation test | Extra caution |

\*AUC: how well the model ranks up moves above down moves on data it never trained on. 0.5 is a coin flip.

A sensible way to combine them: take the **next 1 day** signal as the direction, and use **next 4 hours** to
choose a better moment to enter or exit. Treat every signal as one input among several, and never risk money
you can't afford to lose.

**Drop risk, 3 days** (bottom of each coin card) is a separate warning: the chance that, within the next 3 days,
the price first falls about twice the coin's normal daily move before rising as much. On average about 1 in 5
candles is followed by such a drop, so 20–30% is normal. It is **Elevated** from 35% and **High** from 40%.
Its accuracy (AUC 0.57) is better than the direction predictions'. In testing, selling when it was High and
not buying while it was Elevated cut the worst fall of the Smart strategy from −40% to −32% on 2025–2026. Single
warnings are often wrong (about half the time the price was higher 3 days later); it helps by avoiding the
worst falls, not by timing every top. It retrains at every 4h close. See `experiments/exit_timing.py`.

### How it learns

The live service stays connected to Binance. Each time a candle closes it:

1. downloads the final data,
2. retrains the models for that timeframe on everything since 2019 up to that candle (at 4h closes, the drop
   warning too),
3. tests the new model on months it didn't train on, and keeps it only if it still beats a coin flip
   (otherwise it keeps the previous model),
4. logs the new signals.

If the PC was off or the internet was down, it downloads everything it missed when it comes back and learns
from it before continuing. Nothing is skipped.

### Advice for your portfolio

On the **Portfolio** page, open **Edit holdings and cash**, enter what you own (coin pair, amount, average
cost) and your **spare cash** in USDT, and press **Save**. The **AI advice** section then tells you, per coin:

| Advice | When |
|---|---|
| **Sell** (all of it) | You hold the coin and its next-1-day P(up) is 48% or less |
| **Hold** | You hold the coin and there is no sell signal |
| **Buy** (with a suggested amount) | You don't hold the coin and its next-1-day P(up) is 55% or more |
| **Wait** | No signal, or not enough spare cash |

Suggested buy amounts split your spare cash (plus what suggested sells would free up) equally across the
coins to buy, keeping any one coin to at most 30% of your portfolio. The next-4-hours signal adds a timing
hint, such as "a dip is likely in the next 4 hours; buying a few hours later may get a better price".
These are the same rules tested in the backtest; they are estimates with a small edge, not financial advice.

**Alerts:** after every 4h candle close, the live service re-checks the advice and alerts you when a coin
changes to Buy or Sell. Check the advice any time with `.venv\Scripts\python -m cryptoai advice`.

- **Windows notifications** are on by default. They only appear when someone is logged in to the PC.
- **Zalo**, to get alerts on your phone:
  1. Go to [bot.zaloplatforms.com](https://bot.zaloplatforms.com), sign in with Zalo, create a bot and copy
     its **bot token**.
  2. In the Zalo app, open your new bot and send it any message, for example "hi".
  3. On the PC running the AI, in PowerShell in the project folder, run:
     ```
     .venv\Scripts\python -m cryptoai zalo-setup <your bot token>
     ```
     You should get a "Crypto AI is connected" message in Zalo.
  4. Send a test alert to every channel any time with `.venv\Scripts\python -m cryptoai notify-test`.

  Alert settings, including the bot token, are saved in `notify.json`, which is kept off GitHub. To turn
  Windows notifications off, set `"windows": false` in that file.

### Whale alerts

The live service also watches Binance for big players (free public data) and alerts you on the same channels:

| Alert | When |
|---|---|
| **Whales buying / selling** | 3 or more market trades of $2M+ (BTC), $1M+ (ETH) or $300k+ (BNB, SOL) on the same side within a minute, or one trade 5 times that size |
| **Liquidation cascade** | More than $10M of leveraged longs (or shorts) force-closed within 5 minutes on our 4 coins |
| **Open interest jump** | Futures open interest (leveraged positions) changes 3%+ in an hour, far more than usual |

The **Market** page shows the last hour (whale buys minus sells, open interest change, liquidations) and recent
events. These alerts are information only; nothing trades on them. Research and our own test found whale data
mostly signals that a big move is coming, not which way (see `docs/research/whale-tracking.md`). To turn them
off, set `"whales": false` in `notify.json`. Every event is saved to `logs\whale_events.csv`.

### Paper trading (4-week trial)

The **Paper trading** page runs five pretend accounts that started together and trade live for 4 weeks.
Each starts with the same balance, split into one sleeve per coin (for $1,000: $250 each).

| Account | Strategy |
|---|---|
| **AI Smart** | Sized by the AI's next-1-day confidence, sells half if the 3-day view is still up, takes half profit at +10%. Decides at every 4h close and on the live readings once an action holds 10 minutes (then 1 hour per coin; these guards are not backtested) |
| **Trend + dip-buy** | Holds a coin while its daily close is above its 50-day average (highest return in testing) |
| **Trend x AI + dip-buy** | Average of the 20/50/100/200-day trend rules, sized by the AI's next-3-days P(up), rebalanced daily (about half the drawdown and half the return in testing) |
| Trend rule (benchmark) | The 50-day rule alone |
| Buy & hold (benchmark) | Bought at the start, never sold |

**Dip-buy** (the first three accounts): if a coin closes 10% or more below where it was an hour earlier
(checked at every 15-minute close), buy with up to half of that coin's sleeve and sell 4 hours later. Tested
in `experiments/shock_dip_buy.py`.

- **Fills** use the live Binance price with a 0.1% fee and 0.05% slippage. Nothing real is bought or sold.
- **Updates:** a notification with the standings every week (Windows and Zalo). After 4 weeks trading
  stops, a report is written to `docs/backTestResult`, and you are notified.
- **Everything is saved** in the `paper` folder (kept off GitHub): `account.json` (balances and positions),
  `trades.csv` (every trade with its reason) and `balance.csv` (every account's balance each hour). An
  earlier account is archived to `paper_archive_<time>` when a new trial starts.
- **Runs only while the learning service runs.** After downtime it catches up at the next close, but it
  cannot trade on moves it missed.

---

## Part 3: Look after it

### Check it is healthy

- **Signals page badge:**
  - green "Live learning: listening": all good.
  - orange: catching up or reconnecting, usually fine for a minute.
  - red "offline since ...": the service stopped. See Troubleshooting.
- **Backtest > Learning history:** one row per retrain. Many **REJECTED** rows in a row mean the market has
  changed in a way the model can't handle; worth a closer look.
- **`logs\live.log`:** what the service did and every signal change (lines marked `<-- CHANGED`).

### Update to a newer version

When the code on GitHub has been improved, in PowerShell in the project folder:

```
git pull
.venv\Scripts\pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File setup_autostart.ps1 -AtStartup
```

On an everyday PC, leave out `-AtStartup` and use a normal PowerShell. The last line restarts both background
tasks so they use the new code. On a 24/7 PC, it needs an Admin PowerShell, as in setup.

### Stop it

| To | Do |
|---|---|
| Stop it (and its auto-start) | `powershell -ExecutionPolicy Bypass -File setup_autostart.ps1 -Remove` (Admin PowerShell if you used `-AtStartup`) |
| Start it again | Run the setup command from step 4 again. It catches up on anything it missed while stopped |
| Restart it | Run the setup command from step 4 again. It stops the running copy first |

Don't stop it with **End** in Task Scheduler: that only stops the hidden launcher, and the AI keeps running.

### Moving to another PC

Repeat Part 1 on the new PC. Price data and models are rebuilt by `train.bat`. Your holdings are in
`portfolio.json`, which is not on GitHub for privacy; copy that file across if you want them.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Browser says **"Connection error"** or the page won't load | The dashboard isn't running. Run the setup command from step 4 again, or double-click `dashboard.bat`. Check `logs\dashboard.log` for errors |
| Signals badge is **red / offline** | Look at the end of `logs\live.log`. If the service stopped, run the setup command again. It catches up on anything it missed |
| `logs\live.log` says **"no connection to Binance"** | The internet is down or Binance is unreachable. It retries by itself and catches up once the connection is back |
| Binance is **blocked** where you are | The data comes from `api.binance.com` and `stream.binance.com`. If they are blocked in your country, the AI can't get data |
| **"No model yet"** on the dashboard | Run `train.bat` once |
| **"python is not recognized"** | Python isn't on PATH. Reinstall Python with "Add python.exe to PATH" ticked |
| Dashboard opens on **port 8502** instead of 8501 | Another copy is already running on 8501. Use that one, or stop the extra copy |
| PC **sleeps** and the AI stops | Set sleep to Never (step 4A). When the PC wakes up, the AI catches up |
| Online link says **"Online access is switched off"** | No login is set on the 24/7 PC: run `.venv\Scripts\python -m cryptoai set-login` |
| Online link **stopped working** | The tunnel restarted and has a new address: check Zalo, the Windows notification, or `logs\public_url.txt`. `logs\tunnel.log` shows what happened |

---

## Reference

### Files you can double-click

| File | What it does |
|---|---|
| `train.bat` | Downloads data, retrains both models, prints the backtest |
| `dashboard.bat` | Opens the dashboard in your browser (manual use) |
| `live.bat` | Runs the learning service (manual use; auto-start runs it for you) |
| `daily.bat` | One learning run, logged to `logs\daily.log` |
| `check_signals.bat` | Logs the current signals to `signals_output.txt` |

### Command line

```
.venv\Scripts\python -m cryptoai train      # download data, retrain, print accuracy
.venv\Scripts\python -m cryptoai backtest   # print backtest results per coin
.venv\Scripts\python -m cryptoai signals    # print current signals and log changes
.venv\Scripts\python -m cryptoai market     # print live price, volume, market cap
.venv\Scripts\python -m cryptoai daily      # one learning run
.venv\Scripts\python -m cryptoai live       # always-on learning service
.venv\Scripts\python -m cryptoai advice     # buy / sell / hold advice for your portfolio
.venv\Scripts\python -m cryptoai set-login  # set the login for online access
.venv\Scripts\python -m cryptoai tunnel     # put the dashboard online now (Ctrl+C to stop)
.venv\Scripts\python -m cryptoai simulate BNB/USDT 2026-01-01 2026-06-01 --cash 1000   # paper-trade a past period
```

### Folders

| Folder | Contents | On GitHub? |
|---|---|---|
| `cryptoai/` | The AI's code | Yes |
| `data/` | Downloaded price, funding and premium history | No, rebuilt by training |
| `models/` | Trained models, accuracy, learning history | No, rebuilt by training |
| `logs/` | `live.log`, `dashboard.log`, `daily.log`, service status | No |
| `docs/research/` | Research notes behind the model's design | Yes |

### How it works (for developers)

- `cryptoai/data.py`: public Binance data, cached in `data/`. Spot candles with taker-buy volume and trade
  count, plus futures funding rates and premium index (downloaded but currently unused by the model, see
  `UNUSED_FEATURES` in `config.py`).
- `cryptoai/features.py`: about 40 inputs. Each coin's own chart (returns, RSI, MACD, EMA distance,
  Bollinger, ATR, volatility, volume, long-term trend, distance from recent high, day and hour) plus market
  context: what BTC is doing, and each coin's strength against BTC and the other tracked coins.
- `cryptoai/model.py`: gradient-boosted trees, one model per timeframe, trained on all coins pooled.
  Evaluated **walk-forward**: each test period is predicted by a model trained only on earlier data.
- `cryptoai/live.py`: the always-on service (Binance kline WebSocket, catch-up, retrain at each candle close).
- `cryptoai/market.py`: live prices (Binance WebSocket) and market data (CoinGecko) for the Market page.
- `cryptoai/backtest.py`: long-or-flat with 0.1% fee per side, compared with buy and hold.
- `cryptoai/config.py`: coins, timeframes, horizons, thresholds. Edit here.
