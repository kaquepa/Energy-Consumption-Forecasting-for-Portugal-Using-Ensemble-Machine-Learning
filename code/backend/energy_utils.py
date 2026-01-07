import logging
from typing import Optional, Dict, Any
from datetime import datetime 
import traceback
from backend.cache import cache
from backend.predictor_loader import MakingPrediction

make_prediction = MakingPrediction() 

# Safe config import with fallback
try:
    from config import Config_backend
    Config = Config_backend()
except:
    class Dashboard_Config:
        FORECAST_CACHE_TTL = 300
    Config = Dashboard_Config()

logger = logging.getLogger("energy_utils")

class EnergyUtils():
    def __init__(self):
        pass
    def get_energy_forecast(self,days: int) -> Optional[Dict[str, Any]]:
        """Get energy forecast for N days ahead."""
        predictor = make_prediction.load_predictor()
        if predictor is None:
            logger.error("Predictor not loaded")
            return None
        
        cache_key = f"forecast_energy_{days}"
        cached = cache.get(cache_key)
        if cached:
            logger.info(f"Returning cached {days} day forecast")
            return cached
        
        try:
            logger.info(f"Generating {days} day forecast...")
            
            # Use native predictor method (returns 7 days at once)
            result = predictor.predict_next_7_days()
            
            if not result or 'predictions' not in result:
                logger.error("Predictor returned invalid result")
                return None
            # Cache and return COMPLETE result
            cache.set(cache_key, result, Config.FORECAST_CACHE_TTL)
            return result
            
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_next_day_prediction(self):
        """Get next day (D+1) energy prediction."""
        predictor = make_prediction.load_predictor()
        if predictor is None:
            return None
        
        cache_key = "predict_next"
        cached = cache.get(cache_key)
        if cached:
            logger.info("Returning cached next day prediction")
            return cached
        
        try:
            logger.info("Generating next day prediction...")
            
            # Get full forecast and extract day_1
            full_forecast = predictor.predict_next_7_days()
            
            if not full_forecast or 'predictions' not in full_forecast:
                logger.error("Predictor returned invalid result")
                return None
            
            # Extract day_1
            day_1_prediction = full_forecast['predictions'].get('day_1')
            
            if day_1_prediction is None:
                logger.error("No prediction found for day_1")
                return None
            
            predicted_value = day_1_prediction['predicted']  
            mae = day_1_prediction.get('metrics', {}).get('MAE', 5.0)  
            result = {
                'predicted': predicted_value,
                'date': day_1_prediction['date'],
                'weather': day_1_prediction.get('weather'),
                'metrics': day_1_prediction.get('metrics', {}),
                'confidence_interval': {
                    'lower': round(predicted_value - mae, 2),
                    'upper': round(predicted_value + mae, 2)
                }
            }
            cache.set(cache_key, result, Config.FORECAST_CACHE_TTL)
            logger.info(f" Prediction: {result['predicted']} GWh for {result['date']}")
            return result
        
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            traceback.print_exc()
            return None