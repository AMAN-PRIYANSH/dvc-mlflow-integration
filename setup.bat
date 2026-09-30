@echo off
REM ---------------------------------------------------------------
REM  One-time set-up (Windows). Double-click this file once.
REM  It makes a private Python environment (.venv) and installs
REM  DVC, MLflow, scikit-learn and Streamlit into it.
REM ---------------------------------------------------------------
cd /d "%~dp0"
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Set-up finished. Next: double-click start_gui.bat
pause
