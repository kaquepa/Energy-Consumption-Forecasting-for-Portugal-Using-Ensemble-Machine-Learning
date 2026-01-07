import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from predictor.main_predictor import EnergyPredictor
from backend.weather_utils import  WeatherUtils 
from config import Config_predict
Config = Config_predict()

logger = logging.getLogger("predictor_loader")

# Cached predictor
_predictor = None
_weather_cache = None
weather_forecast = WeatherUtils()


class MakingPrediction():
    def __init__(self):
        pass
    def _load_weather_forecast(self,days: int = 7):
        """Loads weather forecast only once, with safe fallback."""
        global _weather_cache
        if _weather_cache is not None:
            return _weather_cache
        
        try:
            weather_list = weather_forecast.get_weather_forecast(days)
            if not weather_list:
                raise ValueError("Empty weather list")
            
            # Convert list to dict indexed by day_1..day_7
            _weather_cache = {
                f"day_{i+1}": weather_list[i] for i in range(min(days, len(weather_list)))
            }
        except Exception as e:
            logger.warning(f"Error while fetching weather forecast: {e}")
            _weather_cache = {}
        
        return _weather_cache

    def load_predictor(self):
        """Loads full predictor + injects weather data for forecasting."""
        global _predictor
        
        if _predictor is not None:
            return _predictor
        
        try:
            dataset_path = Config.dataset_production_final_path  
            models_dir = Config.models_dir   
            features_path = Config.feature_names_path 
            preprocessor_path = Config.preprocessing_path 
            
            # Validate required files
            missing = []
            if not dataset_path.exists():
                missing.append(f"Dataset: {dataset_path}")
            if not models_dir.exists():
                missing.append(f"Models dir: {models_dir}")
            if not features_path.exists():
                missing.append(f"Features: {features_path}")
            if not preprocessor_path.exists():
                missing.append(f"Preprocessor: {preprocessor_path}")
            
            if missing:
                for m in missing:
                    logger.error(f" {m}")
                raise FileNotFoundError("Required files are missing.")
            
            # Initialize predictor
            predictor = EnergyPredictor()
            # Load weather forecast (7 days) 
            weather = self._load_weather_forecast(days=7)
            predictor.weather_forecast = weather
            
            # Log model metrics
            if hasattr(predictor, 'metrics'):
                metrics = predictor.metrics
                if metrics and isinstance(metrics, dict) and 1 in metrics:
                    m = metrics[1]
            
            _predictor = predictor
            return _predictor
            
        except Exception as e:
            logger.error(f"Failed to load predictor: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def get_predictor(self):
        """Returns cached predictor instance (with weather data)."""
        return self.load_predictor()