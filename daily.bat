@echo off
rem Daily learning: fetch new candles, retrain, keep the better model, log signals.
rem Scheduled by schedule_daily.ps1 to run every morning after the 00:00 UTC daily candle closes.
cd /d "%~dp0"
if not exist logs mkdir logs
.venv\Scripts\python -m cryptoai daily >> logs\daily.log 2>&1
