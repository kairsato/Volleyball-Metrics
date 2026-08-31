@echo off
REM Launches the backend (FastAPI/uvicorn) and frontend (Vite) dev servers,
REM each in its own terminal window. Both bind to 0.0.0.0 (--host 0.0.0.0
REM / vite.config.ts's server.host) so another device on the same LAN can
REM reach them via this machine's own IP, not just from this machine
REM itself - run `ipconfig` to find that IP, then open
REM http://<that IP>:5173 on the other device.

set ROOT=%~dp0

start "Backend" cmd /k "cd /d "%ROOT%Backend" && .venv\Scripts\activate.bat && uvicorn API.main:app --reload --host 0.0.0.0 --port 8000"
start "Frontend" cmd /k "cd /d "%ROOT%Frontend" && npm run dev"
