@echo off
REM Daily paper-trading cycle - invoked by Windows Task Scheduler each weekday evening.
cd /d C:\Users\sailk\desktop\Trading
set PYTHONIOENCODING=utf-8
echo ===== %DATE% %TIME% ===== >> "C:\Users\sailk\desktop\Trading\live\state\cron.log"
"C:\Users\sailk\AppData\Local\Programs\Python\Python312\python.exe" -m live.run_daily >> "C:\Users\sailk\desktop\Trading\live\state\cron.log" 2>&1
