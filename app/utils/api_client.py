
import requests
import streamlit as st
from typing import Optional, Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIClient:
    """API client for fetching data from backend."""
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.timeout = 20
    
    def _get(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Make GET request to API."""
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
    
    def get_health(self) -> Optional[Dict[str, Any]]:
        """Get API health status."""
        return self._get("/health")
    
    def get_next_day_prediction(self) -> Optional[Dict[str, Any]]:
        """Get next day prediction."""
        result = self._get("/energy/predict-next")
        if result and result.get('status') == 'success':
            return result.get('prediction')
        return None
    
    def get_energy_forecast(self, days: int = 7) -> Optional[Dict[str, Any]]:
        """Get multi-day energy forecast."""
        result = self._get(f"/energy/forecast/{days}")
        if result and result.get('status') == 'success':
            return result.get('predictions')
        return None
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get model information and metrics."""
        result = self._get("/model/info")
        if result and result.get('status') == 'success':
            return result
        return None
    
    def get_current_weather(self) -> Optional[Dict[str, Any]]:
        """Get current weather."""
        result = self._get("/weather/current")
        if result and result.get('status') == 'success':
            return result.get('data')
        return None
    
    def get_weather_forecast(self, days: int = 7) -> Optional[List[Dict[str, Any]]]:
        """Get weather forecast."""
        result = self._get(f"/weather/forecast/{days}")
        if result and result.get('status') == 'success':
            return result.get('data')
        return None


@st.cache_data(ttl=60)
def get_cached_forecast(api_url: str, days: int) -> Optional[Dict[str, Any]]:
    """Get cached forecast data."""
    client = APIClient(api_url)
    return client.get_energy_forecast(days)


@st.cache_data(ttl=300)
def get_cached_weather(api_url: str, days: int) -> Optional[List[Dict[str, Any]]]:
    """Get cached weather data."""
    client = APIClient(api_url)
    return client.get_weather_forecast(days)
 