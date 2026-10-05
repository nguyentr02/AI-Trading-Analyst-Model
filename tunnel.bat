@echo off
rem Puts the dashboard online through a Cloudflare tunnel and sends you the link (Windows + Zalo).
rem Needs a login first: .venv\Scripts\python -m cryptoai set-login. Output goes to logs\tunnel.log.
cd /d "%~dp0"
if not exist logs mkdir logs
:loop
.venv\Scripts\python -m cryptoai tunnel >> logs\tunnel.log 2>&1
timeout /t 30 /nobreak > nul
goto loop
