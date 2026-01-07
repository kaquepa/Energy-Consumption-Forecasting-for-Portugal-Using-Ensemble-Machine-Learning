from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import subprocess
import time
import psutil
import traceback
from pathlib import Path

from backend.weather_utils import  WeatherUtils 
from backend.energy_utils import EnergyUtils 
from backend.predictor_loader import MakingPrediction 
from config import Config_backend

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("main")

_PREDICTOR_CACHE = None
Config = Config_backend()
energy_utils = EnergyUtils()
make_prediction = MakingPrediction() 
weather_utils = WeatherUtils()
 

def get_predictor():
    """Return cached predictor instance."""
    global _PREDICTOR_CACHE
    if _PREDICTOR_CACHE is None:
        _PREDICTOR_CACHE = make_prediction.load_predictor()
    return _PREDICTOR_CACHE

app = FastAPI(
    title="Energy Consumption Forecasting for Portugal",
    description="API for energy consumption forecasting",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    """Health check and API information."""
    predictor = get_predictor()
    return {
        "api": "API for energy consumption forecasting",
        "version": "1.0.0",
        "status": "online",
        "model_loaded": predictor is not None,
        "endpoints": {
            "weather": [
                "/weather/current",
                "/weather/forecast/{days}"
            ],
            "energy": [
                "/energy/predict-next",
                "/energy/forecast/{days}"
            ],
            "info": [
                "/",
                "/health",
                "/model/info"
            ]
        }
    }

@app.get("/health")
def health_check():
    """Check service health status."""
    predictor = get_predictor()
    
    return {
        "status": "healthy" if predictor else "degraded",
        "model_loaded": predictor is not None,
        "model_metrics": predictor.model_metrics if predictor else None
    }

@app.get("/weather/current")
def weather_current():
    """Get current weather conditions."""
    try:
        data = weather_utils.get_current_weather()
        return {
            "status": "success",
            "data": data
        }
    except Exception as e:
        logger.error(f"Error fetching current weather: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/weather/forecast/{days}")
def weather_forecast(days: int):
    """Get weather forecast for N days."""
    if not 1 <= days <= 16:
        raise HTTPException(
            status_code=400,
            detail="Days must be between 1 and 16"
        )
    
    try:
        data = weather_utils.get_weather_forecast(days)
        return {
            "status": "success",
            "forecast_days": days,
            "data": data
        }
    except Exception as e:
        logger.error(f"Error fetching weather forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/energy/predict-next")
def energy_predict_next():
    """Get next-day energy consumption prediction."""
    result = energy_utils.get_next_day_prediction()
    
    if result is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded or prediction error"
        )
    
    return {
        "status": "success",
        "prediction": result
    }

@app.get("/energy/forecast/{days}")
def energy_forecast(days: int):
    """Get multi-day energy consumption forecast."""
    if not 1 <= days <= 7:
        raise HTTPException(
            status_code=400,
            detail="Days must be between 1 and 7"
        )
    
    result = energy_utils.get_energy_forecast(days)
    
    if result is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded or prediction error"
        )
    
    return {
        "status": "success",
        "forecast_horizon": days,
        "predictions": result
    }

@app.get("/model/info")
def model_info():
    """Get model information and performance metrics."""
    predictor = get_predictor()
    if not predictor:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    num_features = 0
    if hasattr(predictor, 'preprocessor') and hasattr(predictor.preprocessor, 'feature_names'):
        num_features = len(predictor.preprocessor.feature_names)
    
    dataset_size = 0
    period = "N/A"
    if hasattr(predictor, 'df_last'):
        dataset_size = 1
        period = f"{predictor.df_last['date'].iloc[0].date()} (last known)"
    
    metrics = {}
    if hasattr(predictor, 'metrics'):
        metrics = predictor.metrics
    
    return {
        "status": "success",
        "model": {
            "type": "Direct Multi-Horizon",
            "models": "RandomForest, LightGBM, XGBoost (competition)",
            "horizons": 7,
            "features": num_features
        },
        "performance": metrics,
        "dataset": {
            "last_date": period,
            "features": num_features
        }
    }

def is_streamlit_running(port=8501):
    """Check if Streamlit is already running on the given port."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline')
            if cmdline and isinstance(cmdline, list):
                cmd_str = ' '.join(cmdline)
                if 'streamlit' in cmd_str.lower() and str(port) in cmd_str:
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return False

def start_streamlit():
    """Start Streamlit dashboard in background if not already running."""
    if is_streamlit_running():
        logger.info("Streamlit already running on port 8501")
        return
    dashboard_path = Config.dashboard_dir / "app.py"
    
    if not dashboard_path.exists():
        logger.warning(f"Dashboard not found at {dashboard_path}")
        return
    logger.info("\nStarting Streamlit dashboard...")
    try:
        process = subprocess.Popen(
            [
                "streamlit",
                "run",
                str(dashboard_path),
                "--server.port=8501",
                "--server.headless=true",
                "--browser.gatherUsageStats=false"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(dashboard_path.parent)
        )
        
        time.sleep(2)
        
        if process.poll() is None:
            logger.info("Streamlit started on http://localhost:8501")
        else:
            stdout, stderr = process.communicate()
            logger.error(f"Streamlit failed to start: {stderr.decode()}")
            
    except FileNotFoundError:
        logger.error("Streamlit not found. Install with: pip install streamlit")
    except Exception as e:
        logger.error(f"Error starting Streamlit: {e}")

@app.on_event("startup")
async def startup_event():
    """Validate models and initialize predictor on API startup."""
    # Check ensemble models (h1 to h7)
    missing_models = []
    
    for h in range(1, 8):
        model_path = Config.models_dir / f"Ensemble_h{h}.pkl"
        if not model_path.exists():
            missing_models.append(f"Ensemble_h{h}.pkl")
            logger.error(f" Missing: Ensemble_h{h}.pkl")

    if missing_models:
        raise RuntimeError("Models not found - cannot start API")
    
    # Check preprocessor
    
    if not Config.preprocessing_path.exists():
        logger.error(f" Preprocessor not found: {Config.preprocessing_path}")
        raise RuntimeError("Preprocessor not found")
  
    # Load predictor
    try:
        predictor = get_predictor()
            
    except Exception as e:
        logger.error(f"\n Failed to load predictor: {e}")
        traceback.print_exc()
        raise
    logger.info(" Local:   http://localhost:8000")

    # Start Streamlit
    start_streamlit()

 
  

