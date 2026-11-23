@echo off
cd /d "%~dp0"
"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" build.py
if errorlevel 1 (
  echo Build failed - see the message above.
  pause
  exit /b 1
)
start "" "%~dp0out\index.html"
