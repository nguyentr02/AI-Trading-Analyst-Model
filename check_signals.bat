@echo off
cd /d "%~dp0"
.venv\Scripts\python -m cryptoai signals >> signals_output.txt 2>&1
