#!/bin/bash
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

echo -e "${CYAN}${BOLD}"
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║   👁  WitnessAI — One-Command Setup                     ║"
echo "  ║   Gemini Realtime 5fps | Deepgram STT | ElevenLabs TTS ║"
echo "  ╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

die() { echo -e "${RED}❌ $1${NC}"; exit 1; }
ok()  { echo -e "${GREEN}✅ $1${NC}"; }
warn(){ echo -e "${YELLOW}⚠  $1${NC}"; }

# ─────────────────────────────────────────────────────────────────
# Step 1: Find or install Python 3.12+
# ─────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}[1/5] Python 3.12+ check...${NC}"

PYTHON_BIN=""
for cmd in python3.13 python3.12; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON_BIN="$cmd"
        ok "$($cmd --version)"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    PY_VER=$(python3 --version 2>&1 | cut -d' ' -f2)
    MINOR=$(echo "$PY_VER" | cut -d'.' -f2)

    if [ "$MINOR" -lt 12 ]; then
        echo -e "${YELLOW}Python $PY_VER found — installing 3.12...${NC}"

        if [[ "$OSTYPE" == "darwin"* ]]; then
            command -v brew &>/dev/null || /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            brew install python@3.12
            PYTHON_BIN="python3.12"
        elif command -v apt-get &>/dev/null; then
            sudo apt-get install -y software-properties-common 2>/dev/null || true
            sudo add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null || true
            sudo apt-get update 2>/dev/null || true
            sudo apt-get install -y python3.12 python3.12-venv python3.12-dev 2>/dev/null \
                || die "apt install python3.12 failed. Run manually:\n  sudo apt install python3.12 python3.12-venv"
            PYTHON_BIN="python3.12"
        elif command -v dnf &>/dev/null; then
            sudo dnf install -y python3.12 || die "dnf install python3.12 failed"
            PYTHON_BIN="python3.12"
        else
            die "Cannot auto-install Python 3.12.\nInstall manually: https://www.python.org/downloads/\nThen re-run: bash setup.sh"
        fi

        command -v "$PYTHON_BIN" &>/dev/null || die "Python 3.12 install failed. Install manually and re-run."
        ok "$($PYTHON_BIN --version)"
    else
        PYTHON_BIN="python3"
        ok "Python $PY_VER"
    fi
fi

# ─────────────────────────────────────────────────────────────────
# Step 2: Create virtual environment
# ─────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}[2/5] Creating virtual environment...${NC}"

if [ -d ".venv" ]; then
    warn ".venv already exists — reusing it"
else
    "$PYTHON_BIN" -m venv .venv \
        || die "Failed to create venv. Try: $PYTHON_BIN -m ensurepip"
fi

# Activate
source .venv/bin/activate || die "Could not activate .venv"
VENV_PYTHON=".venv/bin/python"
VENV_PIP=".venv/bin/pip"
ok "venv active: $($VENV_PYTHON --version)"

# ─────────────────────────────────────────────────────────────────
# Step 3: Install packages (pip only — no curl/uv needed)
# ─────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}[3/5] Installing packages via pip...${NC}"
echo -e "${YELLOW}Packages: Vision Agents SDK, YOLOv8, FastAPI, Deepgram, ElevenLabs, OpenCV${NC}"
echo -e "${YELLOW}First run: 3-6 minutes. Grab a coffee. ☕${NC}\n"

"$VENV_PIP" install --upgrade pip --quiet

"$VENV_PIP" install \
    "vision-agents[getstream,gemini,ultralytics,deepgram,elevenlabs]" \
    "fastapi" \
    "uvicorn[standard]" \
    "python-multipart" \
    "opencv-python-headless" \
    "numpy" \
    "Pillow" \
    "python-dotenv" \
    "scipy" \
    || die "pip install failed. Check your internet connection."

ok "All packages installed"

# ─────────────────────────────────────────────────────────────────
# Step 4: Configure .env
# ─────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}[4/5] API key configuration...${NC}"
if grep -q "your_stream_api_key_here" .env 2>/dev/null; then
    echo -e "${YELLOW}Fill in .env with your 4 API keys (all free tiers):${NC}\n"
    printf "  %-22s %s\n" "STREAM_API_KEY"     "→ https://getstream.io/try-for-free"
    printf "  %-22s %s\n" "STREAM_API_SECRET"  "→ same dashboard"
    printf "  %-22s %s\n" "GOOGLE_API_KEY"     "→ https://aistudio.google.com"
    printf "  %-22s %s\n" "DEEPGRAM_API_KEY"   "→ https://deepgram.com"
    printf "  %-22s %s\n" "ELEVENLABS_API_KEY" "→ https://elevenlabs.io"
    echo -e "\n  Edit with: ${CYAN}nano .env${NC}"
else
    ok ".env already configured"
fi

# ─────────────────────────────────────────────────────────────────
# Step 5: Pre-download YOLOv8n weights
# ─────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}[5/5] Pre-downloading YOLOv8n weights (~6MB)...${NC}"
"$VENV_PYTHON" -c "
from ultralytics import YOLO
YOLO('yolov8n.pt')
print('YOLOv8n ready')
" && ok "YOLOv8n ready" || warn "Weights will auto-download on first run (non-critical)"

# ─────────────────────────────────────────────────────────────────
# Done — print exact commands to run
# ─────────────────────────────────────────────────────────────────
echo -e "\n${GREEN}${BOLD}"
echo "  ════════════════════════════════════════════════════════"
echo "  ✅ WitnessAI setup complete!"
echo "  ════════════════════════════════════════════════════════"
echo -e "${NC}"
echo -e "  ${BOLD}Run WitnessAI (copy-paste these exactly):${NC}"
echo ""
echo -e "  ${CYAN}source .venv/bin/activate${NC}"
echo -e "  ${CYAN}python main.py join${NC}"
echo ""
echo -e "  Dashboard → ${CYAN}http://localhost:8000${NC}"
echo -e "  API docs  → ${CYAN}http://localhost:8000/docs${NC}"
echo ""
