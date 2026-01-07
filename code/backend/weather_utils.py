import requests
import sys
from pathlib import Path
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config import Config_backend
Config = Config_backend()

from backend.cache import cache

# Global session with retry
session = requests.Session()
retry = Retry(total=3, backoff_factor=0.3)
session.mount("https://", HTTPAdapter(max_retries=retry))

class WeatherUtils():
    def __init__(self):
        pass
    def find_index_hour(self,hora_atual, lista_horas):
        """Finds the index of the closest hour."""
        dt = datetime.fromisoformat(hora_atual)
        hora_arredondada = dt.replace(minute=0, second=0).isoformat(timespec="minutes")
        if hora_arredondada in lista_horas:
            return lista_horas.index(hora_arredondada)
        
        datas = [datetime.fromisoformat(h) for h in lista_horas]
        diffs = [abs((dt - h).total_seconds()) for h in datas]
        return diffs.index(min(diffs))
    def get_current_weather(self):
        cache_key = "weather_current"
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={Config.LATITUDE}&longitude={Config.LONGITUDE}"
            f"&current=temperature_2m,wind_speed_10m"
            f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,"
            f"precipitation,shortwave_radiation"
        )
        
        response = session.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        # Processing
        current = data["current"]
        hourly = data["hourly"]
        horas = hourly["time"]
        idx = self.find_index_hour(current["time"], horas)
        
        result = {
            "temperatura": current["temperature_2m"],
            "vento": current["wind_speed_10m"],
            "time": current["time"],
            "radiacao_solar": hourly["shortwave_radiation"][idx],
            "humidade": hourly["relative_humidity_2m"][idx],
            "chuva": hourly["precipitation"][idx]
        }
        
        cache.set(cache_key, result, Config.WEATHER_CACHE_TTL)
        return result
    def get_weather_forecast(self,days: int):
        cache_key = f"weather_forecast_{days}"
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={Config.LATITUDE}&longitude={Config.LONGITUDE}"
            f"&daily=temperature_2m_mean,relative_humidity_2m_mean,"
            f"precipitation_sum,wind_speed_10m_max,shortwave_radiation_sum"
            f"&forecast_days={days}"
        )
        response = session.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        daily = data["daily"]
        result = []
        for i in range(len(daily["time"])):
            result.append({
                "date": daily["time"][i],
                "temperatura": daily["temperature_2m_mean"][i],
                "radiacao_solar": daily["shortwave_radiation_sum"][i],
                "humidade": daily["relative_humidity_2m_mean"][i],
                "chuva": daily["precipitation_sum"][i],
                "vento": daily["wind_speed_10m_max"][i],
            })
        cache.set(cache_key, result, Config.WEATHER_CACHE_TTL)
        return result
