#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "[INFO] Starting API check..."

# Activate virtualenv if exists
[ -d ".venv_api" ] && source .venv_api/bin/activate

# Get production models path from Config
MODELS_DIR=$(python -c "from config import Config_backend; print(Config_backend().models_dir)")

# Check if all model files exist
MISSING=0
for i in {1..7}; do
    if [ ! -f "$MODELS_DIR/Ensemble_h${i}.pkl" ]; then
        echo "[INFO] Missing model: Ensemble_h${i}.pkl"
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo "[INFO] Production models missing. Training now..."
    python pipeline.py train
    echo "[INFO] Training finished. Please restart the API."
    exit 0
fi

echo "[INFO] All production models found."

# Function to free a port if occupied
free_port() {
    PORT=$1
    if lsof -i tcp:$PORT &>/dev/null; then
        PID=$(lsof -ti tcp:$PORT)
        echo "[INFO] Port $PORT is in use. Killing process $PID..."
        kill -9 $PID
        sleep 1
        echo "[INFO] Previous process on port $PORT killed."
    fi
}

# Free FastAPI (8000) and Streamlit (8501) ports
free_port 8000
free_port 8501

echo "[INFO] Starting FastAPI server..."
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

echo "[INFO] FastAPI running with PID $FASTAPI_PID"


wait $FASTAPI_PID
