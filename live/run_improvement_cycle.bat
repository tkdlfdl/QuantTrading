@echo off
REM Weekly Improvement Cycle — scheduled task "ImprovementCycleWeekly" (Sat 10:00)
REM Runs one loop iteration per SELF_IMPROVEMENT_PLAN.md via headless Claude Code.
cd /d C:\Users\sailk\Desktop\Trading
set PYTHONIOENCODING=utf-8
echo ============================================================= >> live\state\improvement_cycle.log
echo CYCLE START %date% %time% >> live\state\improvement_cycle.log
type live\improvement_cycle_prompt.md | claude -p --dangerously-skip-permissions >> live\state\improvement_cycle.log 2>&1
echo CYCLE END %date% %time% >> live\state\improvement_cycle.log
