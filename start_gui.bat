@echo off
REM ---------------------------------------------------------------
REM  Opens the GUI in your browser (http://localhost:8501).
REM  The GUI starts the MLflow server by itself (http://127.0.0.1:5000).
REM  Keep this window open while you use the GUI. Close it to stop.
REM ---------------------------------------------------------------
cd /d "%~dp0"
call .venv\Scripts\activate.bat
start "" cmd /c "timeout /t 8 >nul & start http://localhost:8501"
streamlit run app.py
