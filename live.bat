@echo off
rem Always-on service: catches up on missed data, then learns and signals at every candle close.
rem Restarts itself 30 seconds after any crash. Output goes to logs\live.log.
cd /d "%~dp0"
if not exist logs mkdir logs
:loop
.venv\Scripts\python -m cryptoai live >> logs\live.log 2>&1
timeout /t 30 /nobreak > nul
goto loop
