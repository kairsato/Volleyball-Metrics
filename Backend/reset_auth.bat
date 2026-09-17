@echo off
REM Recovery tool for a lost Share password - deletes auth_config.json (and
REM auth_sessions.json) so login goes back to its safe off-by-default state.
REM The server re-reads these files fresh on every request, so this takes
REM effect immediately - no restart needed, even while the server is running.
REM No confirmation prompt on purpose - this is meant for a "locked out,
REM just fix it" moment, and deleting these two files is low-risk/easily
REM redone (just set a new password again afterward).

set ROOT=%~dp0
set DATA_DIR=%ROOT%Backend\API\data
set AUTH_CONFIG=%DATA_DIR%\auth_config.json
set AUTH_SESSIONS=%DATA_DIR%\auth_sessions.json

echo Resetting Share's login...
echo Looking in: %DATA_DIR%
echo.

del /q "%AUTH_CONFIG%" 2>nul
del /q "%AUTH_SESSIONS%" 2>nul

if exist "%AUTH_CONFIG%" (
    echo FAILED to delete auth_config.json.
    echo It may be read-only, or this window may need to run as Administrator.
) else (
    echo Done. Login is off now - open the app with no password, then set a
    echo new one from Settings -^> Share if you want login back on.
)

echo.
pause
