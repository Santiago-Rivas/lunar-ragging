@echo off
REM Metrics Analysis Batch Script for Windows

echo ===== Analyzing Metrics Results =====

REM Create plots directory if it doesn't exist
if not exist "plots" mkdir plots

echo.
echo Step 1: Generating plots...
python plot_metrics.py

echo.
echo Step 2: Generating analysis report...
python metrics_report.py

echo.
echo Step 3: Opening report in default browser...
start "" metrics_report.md

echo.
echo Step 4: Starting plot viewer...
python view_plots.py

echo.
echo Analysis complete! 
echo The report is available at metrics_report.md
echo The plots are available in the 'plots' directory
echo. 