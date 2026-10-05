@echo off
REM One-click setup + launch for Windows (run from the project folder)
cd /d "%~dp0"
if not exist venv ( python -m venv venv )
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
python tests\test_all.py
streamlit run app.py
