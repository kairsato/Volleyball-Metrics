@echo off
REM Launches the backend (FastAPI/uvicorn) and frontend (Vite) dev servers,
REM each in its own terminal window.

set ROOT=%~dp0

start "Backend" cmd /k "cd /d "%ROOT%Backend" && .venv\Scripts\activate.bat && uvicorn API.main:app --reload --port 8000"
start "Frontend" cmd /k "cd /d "%ROOT%Frontend" && npm run dev"
