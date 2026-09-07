@echo off
REM Actual backend startup, factored out of start.bat so its own command
REM line (with venv activation and uvicorn's flags) never has to survive
REM being embedded inside wt's already-fragile multi-tab syntax - a bare
REM filename there is far more robust than a long quoted/&&-chained string.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
uvicorn API.main:app --reload --host 0.0.0.0 --port 8000
