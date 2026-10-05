@echo off
rem Dashboard for always-on use: no browser window, restarts itself 30 seconds after any crash.
rem Open it at http://localhost:8501 (or http://<this-PC's-IP>:8501 from other devices).
cd /d "%~dp0"
if not exist logs mkdir logs
:loop
.venv\Scripts\streamlit run app.py --server.headless true >> logs\dashboard.log 2>&1
timeout /t 30 /nobreak > nul
goto loop
