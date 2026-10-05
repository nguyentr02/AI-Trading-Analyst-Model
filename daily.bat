@echo off
rem Daily learning: fetch new candles, retrain, keep the better model, log signals.
rem Manual fallback: the live service (live.bat) now does this automatically at every candle close.
cd /d "%~dp0"
if not exist logs mkdir logs
.venv\Scripts\python -m cryptoai daily >> logs\daily.log 2>&1
