#!/usr/bin/env bash
# NutriLens — startup script
# Usage: bash start.sh
#        bash start.sh --debug
#        bash start.sh --port 8080

set -e
cd "$(dirname "$0")"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

echo -e "${GREEN}"
echo "  _   _       _        _ _                   "
echo " | \ | |     | |      (_) |                  "
echo " |  \| |_   _| |_ _ __ _| |     ___ _ __  ___ "
echo " | . \` | | | | __| '__| | |    / _ \ '_ \/ __|"
echo " | |\  | |_| | |_| |  | | |___|  __/ | | \__ \\"
echo " |_| \_|\__,_|\__|_|  |_|______\___|_| |_|___/"
echo -e "${NC}"
echo "  Food Intelligence · Flask App"
echo ""

# ── Check Python ──────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo -e "${RED}✗ python3 not found. Install Python 3.9+${NC}"
  exit 1
fi
echo -e "${GREEN}✓${NC} Python $(python3 --version | cut -d' ' -f2)"

# ── Check dependencies ────────────────────────────────────────────────────────
if ! python3 -c "import flask" 2>/dev/null; then
  echo -e "${YELLOW}⚠ Dependencies not installed. Running: pip install -r requirements.txt${NC}"
  pip install -r requirements.txt
fi
echo -e "${GREEN}✓${NC} Dependencies OK"

# ── Check model ───────────────────────────────────────────────────────────────
if [ -f "model_trained_101class.hdf5" ]; then
  echo -e "${GREEN}✓${NC} Model found"
else
  echo -e "${YELLOW}⚠ model_trained_101class.hdf5 not found — running in DEMO MODE${NC}"
  echo "  Train the model in Google Colab using NutriLens_Training.ipynb"
fi

# ── Load .env if present ──────────────────────────────────────────────────────
if [ -f ".env" ]; then
  export $(grep -v '^#' .env | xargs)
  echo -e "${GREEN}✓${NC} Loaded .env"
fi

# ── Create uploads dir ────────────────────────────────────────────────────────
mkdir -p static/uploads

# ── Parse args ────────────────────────────────────────────────────────────────
DEBUG_FLAG=""
PORT=5000
while [[ $# -gt 0 ]]; do
  case $1 in
    --debug) DEBUG_FLAG="--debug"; shift ;;
    --port)  PORT="$2"; shift 2 ;;
    *) shift ;;
  esac
done

echo ""
echo -e "  ${GREEN}→ Starting on http://127.0.0.1:${PORT}${NC}"
echo ""

python3 app.py $DEBUG_FLAG 127.0.0.1 $PORT
