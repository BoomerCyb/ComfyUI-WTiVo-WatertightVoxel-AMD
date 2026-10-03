@echo off
if "%WTIVO_PYTHON%"=="" (
  echo Set WTIVO_PYTHON to your ComfyUI ROCm python_env\python.exe first.
  exit /b 1
)
"%WTIVO_PYTHON%" "%~dp0wtivo.py" %*
