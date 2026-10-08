@echo off
rem For a PC that only views: shows the alerts the 24/7 PC sends (logs\alerts.csv, synced here by Syncthing)
rem as Windows notifications. Restarts itself 30 seconds after any crash. Output goes to alerts_viewer.log
rem (kept out of logs\, which Syncthing keeps identical to the 24/7 PC).
cd /d "%~dp0"
:loop
.venv\Scripts\python -m cryptoai alert-mirror >> alerts_viewer.log 2>&1
timeout /t 30 /nobreak > nul
goto loop
