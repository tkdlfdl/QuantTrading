@echo off
REM Portable nightly track-record wrapper for the cloud VM.
REM Python must be installed for ALL USERS and on the system PATH.
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
echo ===== %DATE% %TIME% ===== >> "%~dp0state\cron.log"
python -m live.run_daily >> "%~dp0state\cron.log" 2>&1
