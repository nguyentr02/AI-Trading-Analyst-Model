@echo off
cd /d "%~dp0"
.venv\Scripts\python -m cryptoai train
.venv\Scripts\python -m cryptoai backtest
pause
