@echo off
REM Activate the virtual environment
call venv\Scripts\activate


REM Navigate back to the main directory and start the frontend
start cmd /k "streamlit run main.py"

REM Pause to keep the batch window open
pause
