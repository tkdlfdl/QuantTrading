@echo off
REM Portable intraday wrapper for the cloud VM.
REM Assumes Python is installed for ALL USERS and added to the system PATH,
REM so the SYSTEM-account scheduled task can resolve `python`.
REM %~dp0 = this live\ folder (with trailing backslash).
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
echo ===== %DATE% %TIME% ===== >> "%~dp0state\intraday.log"
python -m live.run_intraday --once --live >> "%~dp0state\intraday.log" 2>&1
