@echo off
REM Improvement Cycle - scheduled tasks "ImprovementCycleAM" (daily 06:00) and
REM "ImprovementCyclePM" (daily 19:00, after the 18:07 settle).
REM Runs one loop iteration per SELF_IMPROVEMENT_PLAN.md via headless Claude Code.
cd /d C:\Users\sailk\Desktop\Trading
set PYTHONIOENCODING=utf-8
REM allow long-running background agents (data fetches) up to 45 min
set CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=2700000
echo ============================================================= >> live\state\improvement_cycle.log
echo CYCLE START %date% %time% >> live\state\improvement_cycle.log
type live\improvement_cycle_prompt.md | "C:\Users\sailk\.local\bin\claude.exe" -p --dangerously-skip-permissions >> live\state\improvement_cycle.log 2>&1
echo CYCLE END %date% %time% >> live\state\improvement_cycle.log
