@echo off
REM Actual backend startup, factored out of start.bat so its own command
REM line (with venv activation and uvicorn's flags) never has to survive
REM being embedded inside wt's already-fragile multi-tab syntax - a bare
REM filename there is far more robust than a long quoted/&&-chained string.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
REM Invoked via run.py (uvicorn.run(), not the uvicorn CLI): click's CLI
REM parser expands "*" wildcards in raw argv on Windows by default, before
REM it even knows an argument belongs to --reload-exclude, turning
REM "API/data/*" into one argument per file already in API/data. See
REM run.py for details.
python run.py
