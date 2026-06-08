@echo off
REM Intraday execution tick - invoked hourly by Windows Task Scheduler.
REM No --force: the script's own ET market-hours gate decides whether to act.
REM --live: submits real PAPER orders to Alpaca (paper account, virtual money).
REM To return to dry-run, remove the --live flag below.
cd /d C:\Users\sailk\desktop\Trading
set PYTHONIOENCODING=utf-8
echo ===== %DATE% %TIME% ===== >> "C:\Users\sailk\desktop\Trading\live\state\intraday.log"
"C:\Users\sailk\AppData\Local\Programs\Python\Python312\python.exe" -m live.run_intraday --once --live >> "C:\Users\sailk\desktop\Trading\live\state\intraday.log" 2>&1
