@echo off
cd /d "%~dp0"
py -3 -m src.desktop_demo
if errorlevel 1 pause
