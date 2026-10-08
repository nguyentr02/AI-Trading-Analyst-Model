@echo off
rem Pulls the latest code from GitHub and restarts the services if it changed (see update.ps1).
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File update.ps1
