@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not defined COMFY_PYTHON if exist "%~dp0..\..\python_env\python.exe" set "COMFY_PYTHON=%~dp0..\..\python_env\python.exe"
if not defined COMFY_PYTHON if exist "%~dp0..\..\.venv\Scripts\python.exe" set "COMFY_PYTHON=%~dp0..\..\.venv\Scripts\python.exe"
if not defined COMFY_PYTHON if exist "%~dp0..\..\venv\Scripts\python.exe" set "COMFY_PYTHON=%~dp0..\..\venv\Scripts\python.exe"
if not defined COMFY_PYTHON if exist "%~dp0..\..\..\python_embeded\python.exe" set "COMFY_PYTHON=%~dp0..\..\..\python_embeded\python.exe"
if not defined COMFY_PYTHON (
    echo ERROR: Set COMFY_PYTHON to the Python executable that runs ComfyUI.
    exit /b 1
)
"%COMFY_PYTHON%" "%~dp0install.py" %*
exit /b %errorlevel%
