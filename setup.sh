#!/usr/bin/env bash
# ============================================================
# NOVA — Dorm Room AI Assistant — Setup Script (Linux/macOS)
# ============================================================
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== NOVA Setup ===${NC}"

# 1. Check Python version
PYTHON=""
for cmd in python3.11 python3.12 python3; do
    if command -v "$cmd" &>/dev/null; then
        version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ] && [ "$minor" -le 12 ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo -e "${RED}Error: Python 3.11 or 3.12 required (3.13 has torch/onnxruntime issues).${NC}"
    echo "Install Python 3.11: https://www.python.org/downloads/"
    exit 1
fi

echo -e "Using Python: $($PYTHON --version)"

# 2. Create virtual environment
if [ ! -d ".venv3" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    $PYTHON -m venv .venv3
fi

source .venv3/bin/activate
echo -e "Virtual environment activated: $(which python)"

# 3. Upgrade pip
pip install --upgrade pip setuptools wheel -q

# 4. Detect GPU and install appropriate torch
echo -e "${YELLOW}Detecting GPU...${NC}"
if command -v nvidia-smi &>/dev/null; then
    echo -e "${GREEN}NVIDIA GPU detected!${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
    echo "Installing PyTorch with CUDA support..."
    pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 -q
else
    echo -e "${YELLOW}No NVIDIA GPU detected. Installing CPU-only PyTorch.${NC}"
    pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu -q
fi

# 5. Install requirements
echo -e "${YELLOW}Installing server dependencies...${NC}"
pip install -r server/requirements.txt -q

echo -e "${YELLOW}Installing client dependencies...${NC}"
pip install -r client/requirements.txt -q

# 6. Create data directories
echo -e "${YELLOW}Creating data directories...${NC}"
mkdir -p data/models/jointbert
mkdir -p data/models/piper
mkdir -p data/models/smart_turn
mkdir -p data/speechbrain_model
mkdir -p data/speaker_enrollment

# 7. Copy .env if needed
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Creating .env from template...${NC}"
    cp .env.example .env
    echo -e "${RED}IMPORTANT: Edit .env and add your API keys!${NC}"
else
    echo ".env already exists, skipping."
fi

# 8. Download models
if [ -f "scripts/download_models.py" ]; then
    echo -e "${YELLOW}Downloading ML models...${NC}"
    python scripts/download_models.py
else
    echo -e "${YELLOW}Model download script not found. Models will download on first run.${NC}"
fi

echo ""
echo -e "${GREEN}=== Setup Complete! ===${NC}"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API keys (at minimum: GROQ_API_KEY, ELEVEN_API_KEY)"
echo "  2. (Optional) Set up Gmail:  python server/setup_gmail.py"
echo "  3. Start server: python run_server.py"
echo "  4. Start client: python run_client.py"
echo "  5. Open UI: http://localhost:8000/ui/"
