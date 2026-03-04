@echo off
REM ============================================================
REM NOVA — Dorm Room AI Assistant — Setup Script (Windows)
REM ============================================================

echo === NOVA Setup ===

REM 1. Check Python version
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Install Python 3.11 from https://www.python.org/downloads/
    exit /b 1
)

python -c "import sys; v=sys.version_info; exit(0 if v.major==3 and 11<=v.minor<=12 else 1)" 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python 3.11 or 3.12 required (3.13 has torch/onnxruntime issues).
    python --version
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do echo Using: %%i

REM 2. Create virtual environment
if not exist ".venv3" (
    echo Creating virtual environment...
    python -m venv .venv3
)

call .venv3\Scripts\activate.bat
echo Virtual environment activated.

REM 3. Upgrade pip
pip install --upgrade pip setuptools wheel -q

REM 4. Detect GPU and install torch
echo Detecting GPU...
where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo NVIDIA GPU detected!
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>nul
    echo Installing PyTorch with CUDA support...
    pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 -q
) else (
    echo No NVIDIA GPU detected. Installing CPU-only PyTorch.
    pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu -q
)

REM 5. Install requirements
echo Installing server dependencies...
pip install -r server\requirements.txt -q

echo Installing client dependencies...
pip install -r client\requirements.txt -q

REM 6. Create data directories
echo Creating data directories...
if not exist "data\models\jointbert" mkdir data\models\jointbert
if not exist "data\models\piper" mkdir data\models\piper
if not exist "data\models\smart_turn" mkdir data\models\smart_turn
if not exist "data\speechbrain_model" mkdir data\speechbrain_model
if not exist "data\speaker_enrollment" mkdir data\speaker_enrollment

REM 7. Copy .env if needed
if not exist ".env" (
    echo Creating .env from template...
    copy .env.example .env
    echo IMPORTANT: Edit .env and add your API keys!
) else (
    echo .env already exists, skipping.
)

REM 8. Download models
if exist "scripts\download_models.py" (
    echo Downloading ML models...
    python scripts\download_models.py
) else (
    echo Model download script not found. Models will download on first run.
)

echo.
echo === Setup Complete! ===
echo.
echo Next steps:
echo   1. Edit .env with your API keys (at minimum: GROQ_API_KEY, ELEVEN_API_KEY)
echo   2. (Optional) Set up Gmail:  python server\setup_gmail.py
echo   3. Start server: python run_server.py
echo   4. Start client: python run_client.py
echo   5. Open UI: http://localhost:8000/ui/
