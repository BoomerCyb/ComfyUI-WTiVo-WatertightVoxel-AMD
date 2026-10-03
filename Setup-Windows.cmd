@echo off
call "%~dp0install_requirements.bat" %*
exit /b %errorlevel%
