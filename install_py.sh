#!/bin/bash
# WitnessAI — Python 3.12 installer via pyenv (no apt needed)
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

die() { echo -e "${RED}❌ $1${NC}"; exit 1; }
ok()  { echo -e "${GREEN}✅ $1${NC}"; }
warn(){ echo -e "${YELLOW}⚠  $1${NC}"; }

echo -e "${CYAN}${BOLD}"
echo "  WitnessAI — Python 3.12 Installer"
echo -e "${NC}"

# ── Already have 3.12? ────────────────────────────────────────────
for cmd in python3.13 python3.12; do
    if command -v "$cmd" &>/dev/null; then
        ok "Already have $($cmd --version) — nothing to do!"
        echo -e "Run: ${CYAN}bash setup.sh${NC}"
        exit 0
    fi
done

# ── Install pyenv build dependencies first ────────────────────────
echo -e "${BOLD}[1/3] Installing build dependencies...${NC}"
sudo apt-get install -y \
    build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev libncursesw5-dev xz-utils \
    tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    curl git 2>/dev/null \
    || die "apt install of build deps failed. Are you online?"
ok "Build deps installed"

# ── Install pyenv ─────────────────────────────────────────────────
echo -e "\n${BOLD}[2/3] Installing pyenv...${NC}"
if [ -d "$HOME/.pyenv" ]; then
    warn "pyenv already installed — updating"
    cd "$HOME/.pyenv" && git pull --quiet && cd -
else
    curl https://pyenv.run | bash \
        || die "Could not download pyenv. Check internet connection."
fi

# Wire pyenv into current shell
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
ok "pyenv $(pyenv --version)"

# Add to ~/.bashrc so it persists
if ! grep -q 'PYENV_ROOT' ~/.bashrc; then
    echo '' >> ~/.bashrc
    echo '# pyenv (added by WitnessAI setup)' >> ~/.bashrc
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    ok "pyenv added to ~/.bashrc"
fi

# ── Build Python 3.12 ─────────────────────────────────────────────
echo -e "\n${BOLD}[3/3] Building Python 3.12.3 from source (~5-10 min)...${NC}"
warn "This compiles Python — it takes a while. Don't close the terminal."
echo ""

pyenv install 3.12.3 --skip-existing \
    || die "Python 3.12.3 build failed. Check output above."

pyenv global 3.12.3
ok "Python $(python --version) installed and set as default"

echo -e "\n${GREEN}${BOLD}"
echo "  ✅ Python 3.12 ready!"
echo -e "${NC}"
echo -e "Now run:"
echo -e "  ${CYAN}source ~/.bashrc${NC}   # reload shell"
echo -e "  ${CYAN}bash setup.sh${NC}      # install WitnessAI"
echo ""
