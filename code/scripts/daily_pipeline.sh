#!/bin/bash

set -e  # Exit on error
# Get absolute project directory 
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

# Setup logging
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

TODAY=$(date +%Y-%m-%d)

# Cross-platform date for yesterday
if [[ "$OSTYPE" == "darwin"* ]]; then
    YESTERDAY=$(date -v-1d +%Y-%m-%d)
else
    YESTERDAY=$(date -d "yesterday" +%Y-%m-%d)
fi

TODAY_MARKER=$(date +%Y%m%d)
LOG_FILE="$LOG_DIR/pipeline_$TODAY_MARKER.log"

# Logging function
log() { 
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

VENV_DIR="$PROJECT_DIR/.venv_api"

if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    PYTHON="$VENV_DIR/bin/python"

    python -m ipykernel install --user --name=venv_api --display-name="Python (venv_api)"

    # Add code directory to PYTHONPATH
    export PYTHONPATH="$PROJECT_DIR/code:$PROJECT_DIR:$PYTHONPATH"
    log "Virtual environment activated"

else
    PYTHON="python"
    export PYTHONPATH="$PROJECT_DIR/code:$PROJECT_DIR:$PYTHONPATH"
    log "  Virtual environment not found, using system Python"
fi


log "****************************************************************************************"
log " DAILY PIPELINE - $TODAY"
log "****************************************************************************************"

START=$(date +%s)

# Cache & flags
CACHE_DIR="$PROJECT_DIR/.cache"
mkdir -p "$CACHE_DIR"

RETRAIN_FLAG="$CACHE_DIR/retrain_$TODAY_MARKER.done"
EDA_FLAG="$CACHE_DIR/eda_done.flag"
PREDICT_FLAG="$CACHE_DIR/predict_$TODAY_MARKER.done"

# Check if today is Monday
IS_MONDAY=0 
if [ "$(date +%u)" -eq 1 ]; then
    IS_MONDAY=1
fi


# Get last date from CSV safely 
get_last_date() {
    local csv_file="$1"
    "$PYTHON" -c "import pandas as pd; import sys; df=pd.read_csv(sys.argv[1]); print(df.iloc[-1,0])" "$csv_file"
}

# ============================================================================
# STEP 1: DATA COLLECTION
# ============================================================================

log ""
log "STEP 1: Data Collection"

# Get dataset path (handles spaces)
DATASET=$("$PYTHON" -c "from config import get_collector_config; print(get_collector_config().dataset_merged_path)")

if [ ! -f "$DATASET" ]; then
    log " Dataset not found, running initial collection..."
    (cd "$PROJECT_DIR/code" && "$PYTHON" pipeline.py collect) 2>&1 | tee -a "$LOG_FILE"
else
    # Use helper function to safely get last date
    LAST_DATE=$(get_last_date "$DATASET")
    
    if [ "$LAST_DATE" == "$YESTERDAY" ]; then
        log " Dataset up-to-date (last: $YESTERDAY)"
    else
        log " Collecting data (last: $LAST_DATE, target: $YESTERDAY)..."
        (cd "$PROJECT_DIR/code" && "$PYTHON" pipeline.py collect) 2>&1 | tee -a "$LOG_FILE"
        rm -f "$PREDICT_FLAG"
        log " Collection completed"
    fi
fi


log ""
log "STEP 2: Feature Engineering"

# Get paths (handles spaces)
DATASET_FINAL=$("$PYTHON" -c "from config import get_preprocessor_config; print(get_preprocessor_config().dataset_production_final_path)")
PREPROCESS_PKL=$("$PYTHON" -c "from config import get_preprocessor_config; print(get_preprocessor_config().preprocessing_path)")

if [ ! -f "$DATASET_FINAL" ] || [ ! -f "$PREPROCESS_PKL" ]; then
    log " Processed dataset or preprocessor not found, running preprocessing..."
    (cd "$PROJECT_DIR/code" && "$PYTHON" pipeline.py preprocess) 2>&1 | tee -a "$LOG_FILE"
    rm -f "$PREDICT_FLAG"
else
    # Use helper function for both dates
    LAST_DATE_FINAL=$(get_last_date "$DATASET_FINAL")
    LAST_DATE_ORIG=$(get_last_date "$DATASET")
    
    if [ "$LAST_DATE_FINAL" == "$YESTERDAY" ] && [ "$LAST_DATE_ORIG" == "$YESTERDAY" ]; then
        log " Features up-to-date (last: $YESTERDAY)"
    else
        log " Processing features..."
        (cd "$PROJECT_DIR/code" && "$PYTHON" pipeline.py preprocess) 2>&1 | tee -a "$LOG_FILE"
        rm -f "$PREDICT_FLAG"
    fi
fi


if [ $IS_MONDAY -eq 1 ]; then
    log ""
    log "STEP 3: Model Training (Monday)"
    
    if [ -f "$RETRAIN_FLAG" ]; then
        log " Models already trained today"
    else
        log " Training models..."
        (cd "$PROJECT_DIR/code" && "$PYTHON" pipeline.py train) 2>&1 | tee -a "$LOG_FILE"
        touch "$RETRAIN_FLAG"
        rm -f "$PREDICT_FLAG"
        log " Models trained successfully"
    fi
else
    log ""
    log "STEP 3: Model Training (skipped - not Monday)"
fi

if [ ! -f "$EDA_FLAG" ]; then
    log ""
    log "STEP 4: Exploratory Data Analysis"
    
    NOTEBOOK="$PROJECT_DIR/code/eda.ipynb"
    OUTPUT_DIR="$PROJECT_DIR/code/Exploratory_result"
    mkdir -p "$OUTPUT_DIR"
    
    if [ -f "$NOTEBOOK" ]; then
        if command -v papermill &> /dev/null; then
            # Check if Jupyter kernel exists
            if jupyter kernelspec list 2>/dev/null | grep -q "venv_api"; then
                log " Running EDA notebook with kernel 'venv_api'..."
                papermill "$NOTEBOOK" "$OUTPUT_DIR/eda_$TODAY_MARKER.ipynb" \
                    -k venv_api \
                    --log-output 2>&1 | tee -a "$LOG_FILE"
                touch "$EDA_FLAG"
                log " EDA completed successfully"
            else
                log "  Jupyter kernel 'venv_api' not found"
                log "    Install with: python -m ipykernel install --user --name=venv_api"
                log " Skipping EDA for now"
            fi
        else
            log " Papermill not installed, skipping EDA"
        fi
    else
        log " Notebook not found at: $NOTEBOOK"
    fi
else
    log ""
    log "STEP 4: EDA ( already completed)"
    log " To re-run: rm -f .cache/eda_done.flag"
fi


log ""
log "STEP 5: Forecast Generation"

# Get paths (handles spaces)
PRED_FILE=$("$PYTHON" -c "from config import get_predict_config; print(get_predict_config().latest_prediction_path)")
MODELS_DIR=$("$PYTHON" -c "from config import get_backend_config; print(get_backend_config().models_dir)")
PREPROCESS_FILE=$("$PYTHON" -c "from config import get_preprocessor_config; print(get_preprocessor_config().preprocessing_path)")

if [ -f "$PREDICT_FLAG" ]; then
    log " Predictions already generated today"
else
    # Check if production models exist
    MISSING=0
    for h in {1..7}; do
        MODEL_PATH="$MODELS_DIR/Ensemble_h${h}.pkl"
        if [ ! -f "$MODEL_PATH" ]; then
            log " Missing: Ensemble_h${h}.pkl"
            MISSING=1
        fi
    done
    
    # Check preprocessor
    if [ ! -f "$PREPROCESS_FILE" ]; then
        log " Missing preprocessing.pkl, running preprocessing..."
        (cd "$PROJECT_DIR/code" && "$PYTHON" pipeline.py preprocess) 2>&1 | tee -a "$LOG_FILE"
        rm -f "$PREDICT_FLAG"
    fi
    
    # Train if missing models
    if [ $MISSING -eq 1 ]; then
        log " Production models missing, training now..."
        (cd "$PROJECT_DIR/code" && "$PYTHON" pipeline.py train) 2>&1 | tee -a "$LOG_FILE"
    fi
    
    # Generate predictions
    log "Generating 7-day forecast..."
    (cd "$PROJECT_DIR/code" && "$PYTHON" pipeline.py predict) 2>&1 | tee -a "$LOG_FILE"
    touch "$PREDICT_FLAG"
    
    # Display summary if prediction file exists
    if [ -f "$PRED_FILE" ]; then
        if command -v jq &> /dev/null; then
            NEXT_DATE=$(jq -r '.predictions.day_1.date' "$PRED_FILE" 2>/dev/null || echo "N/A")
            PRED_VALUE=$(jq -r '.predictions.day_1.predicted' "$PRED_FILE" 2>/dev/null || echo "N/A")
            NUM_DAYS=$(jq '.predictions | length' "$PRED_FILE" 2>/dev/null || echo "0")
            
            if [ "$NEXT_DATE" != "N/A" ] && [ "$PRED_VALUE" != "N/A" ]; then
                log " Forecast: $NUM_DAYS days | $NEXT_DATE -> ${PRED_VALUE} GWh"
            else
                log " Predictions generated (jq parse issue)"
            fi
        else
            log " Predictions generated (install 'jq' for detailed summary)"
        fi
    else
        log " Prediction file not found: $PRED_FILE"
    fi
fi

END=$(date +%s)
DURATION=$((END - START))
MIN=$((DURATION / 60))
SEC=$((DURATION % 60))

log ""
log "****************************************************************************************"
log "  PIPELINE COMPLETED IN ${MIN}m ${SEC}s"
log "****************************************************************************************"


# Remove old cache files (>7 days)
find "$CACHE_DIR" -name "*.done" -mtime +7 -delete 2>/dev/null || true
find "$CACHE_DIR" -name "*.flag" -mtime +7 -delete 2>/dev/null || true

# Remove old EDA notebooks (>3 days)
if [ -d "$PROJECT_DIR/code/Exploratory_result" ]; then
    find "$PROJECT_DIR/code/Exploratory_result" -name "*.ipynb" -mtime +3 -delete 2>/dev/null || true
fi

# Remove old logs (>45 days)
find "$LOG_DIR" -name "pipeline_*.log" -mtime +45 -delete 2>/dev/null || true

log " Cleanup completed"


log ""
log "****************************************************************************************"

# Cross-platform "tomorrow 06:00"
if [[ "$OSTYPE" == "darwin"* ]]; then
    NEXT_RUN=$(date -v+1d '+%Y-%m-%d 06:00')
else
    NEXT_RUN=$(date -d 'tomorrow 06:00' '+%Y-%m-%d %H:%M')
fi

log " All done! Next run: $NEXT_RUN"
log "****************************************************************************************"

exit 0