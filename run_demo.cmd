@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run setup_cpu.cmd once before starting the demo.
  pause
  exit /b 1
)
.venv\Scripts\python.exe demo.py --model YOLOKW8n
pause
