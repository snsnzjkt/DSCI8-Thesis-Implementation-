@echo off
title Network IDS GUI Launcher
echo.
echo ================================================================
echo                Network IDS GUI - SCS-ID vs Baseline CNN
echo ================================================================
echo.
echo Starting GUI application...
echo.

cd /d "%~dp0"
python launch_gui.py

echo.
echo GUI application closed.
pause