@echo off
rem Starts the capture. Creates .venv on the first start and reinstalls the packages when requirements.txt changed.
cd /d "%~dp0"
call :setup || (pause & exit /b 1)
.venv\Scripts\python -m poe2layout %*
pause
exit /b

:setup
if not exist .venv\Scripts\python.exe (
  py -3.14 -m venv .venv || py -3 -m venv .venv || exit /b 1
)
fc /b requirements.txt .venv\installed-requirements.txt >nul 2>&1
if errorlevel 1 (
  .venv\Scripts\python -m pip install -r requirements.txt || exit /b 1
  copy /y requirements.txt .venv\installed-requirements.txt >nul
)
exit /b 0
