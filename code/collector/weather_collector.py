import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from config import Config_collector

class WeatherCollector:
    def __init__(self): 
        self.Config = Config_collector()
        self.filepath = self.Config.WEATHER_PATH
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        
        self.lat = self.Config.LATITUDE 
        self.lon = self.Config.LONGITUDE 
        
        self.variables = [
            "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
            "relative_humidity_2m_mean", "precipitation_sum",
            "wind_speed_10m_mean", "shortwave_radiation_sum"
        ]
    
    def fetch_range(self, start, end):  
        """Collect historical data from  Open-Meteo"""
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={self.lat}&longitude={self.lon}"
            f"&start_date={start}&end_date={end}" 
            f"&daily={','.join(self.variables)}"
            f"&timezone=Europe%2FLisbon"
        )
        
        r = requests.get(url, timeout=30)
        data = r.json()
        
        if "daily" not in data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data["daily"])
        df.rename(columns={"time": "date"}, inplace=True)
        return df
    
    def update(self):
        """Apdate local weather data file"""
        today = datetime.now().strftime("%Y-%m-%d")
        if not os.path.exists(self.filepath):
            df = self.fetch_range(self.Config.START_DATE, today) 
            if not df.empty:
                df.to_csv(self.filepath, index=False)
            else:
                print("Error in initial fetch from API")
            return df
        
        df_old = pd.read_csv(self.filepath)
        last_date = df_old["date"].max()
        
        if last_date >= today:
            return df_old
       
     
        next_day = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        df_new = self.fetch_range(next_day, today) 
        if df_new.empty:
            return df_old
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all = df_all.drop_duplicates(subset=['date'], keep='last') 
        df_all = df_all.sort_values('date').reset_index(drop=True) 
        df_all.to_csv(self.filepath, index=False)
        return df_all