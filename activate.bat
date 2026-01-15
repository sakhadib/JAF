@echo off
REM Activate virtual environment and run the story generation script

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Installing dependencies...
pip install -q -r requirements.txt

echo.
echo Virtual environment ready!
echo.
echo Usage examples:
echo   python run.py --model deepseek/deepseek-r1
echo   python run.py --model openai/gpt-5.2 --threads 10
echo   python run.py --help
echo.
