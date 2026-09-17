@echo off
REM See Backend/_run_backend.bat's comment - same reasoning, just for the
REM Caddy reverse proxy. Resolves caddy.exe's real path itself (see
REM start.bat's original comment on why) so start.bat doesn't need to pass
REM anything in - this tab's wt commandline is just this bare filename.
cd /d "%~dp0"

REM The Caddyfile's site address is {$DEV_HOSTNAME} - pulled from whatever
REM hostname was saved in the app's Share settings dialog (App.tsx's
REM "Custom domain" field), which Backend/API/share.py mirrors as plain text
REM to Backend/API/data/hostname.txt for exactly this. Empty/missing file
REM (no hostname set yet) leaves DEV_HOSTNAME unset, same as before.
set "DEV_HOSTNAME="
if exist "Backend\API\data\hostname.txt" (
    set /p DEV_HOSTNAME=<"Backend\API\data\hostname.txt"
)

where caddy >nul 2>&1
if errorlevel 1 (
    set "CADDY_EXE=%LOCALAPPDATA%\Microsoft\WinGet\Packages\CaddyServer.Caddy_Microsoft.Winget.Source_8wekyb3d8bbwe\caddy.exe"
) else (
    set "CADDY_EXE=caddy"
)

REM With no hostname configured, Caddyfile's site address
REM ({$DEV_HOSTNAME}, www.{$DEV_HOSTNAME}) would collapse to the literal,
REM invalid hostname "www." and Caddy would refuse to start. Fall back to
REM Caddyfile.local's self-signed :443 setup in that case.
if "%DEV_HOSTNAME%"=="" (
    set "CADDY_CONFIG=Caddyfile.local"
) else (
    set "CADDY_CONFIG=Caddyfile"
)
"%CADDY_EXE%" run --config "%CADDY_CONFIG%"
