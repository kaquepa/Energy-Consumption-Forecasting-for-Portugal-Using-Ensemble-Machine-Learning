import pandas as pd
import numpy as np
from datetime import date, timedelta
import holidays
from preprocessor.logger import get_logger

logger = get_logger("FeatureEngineering")
class FeatureEngineering:
    """Feature Engineering Pipeline"""
    def __init__(self, input_path, output_path, is_production=False):
        self.input_path = input_path
        self.output_path = output_path
        self.is_production = is_production

        self.df = pd.read_csv(input_path)
        self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")
        self.df = self.df.sort_values("date").reset_index(drop=True)

        if not is_production:
            self.target_cols = [f"target_next_day_{h}" for h in range(1, 8)]
            self._create_targets()
        else:
            self.target_cols = []

    def _create_targets(self):
        logger.info("Creating target columns")
        for h in range(1, 8):
            self.df[f"target_next_day_{h}"] = self.df["corrigido_temperatura"].shift(-h)
    
    def fix_data_errors(self):
        exclude = ["date"] + self.target_cols
        for col in self.df.select_dtypes("number"):
            if col in exclude:
                continue
            self.df[col] = self.df[col].clip(lower=0)
    
    def consumption_features(self):
        c = self.df["corrigido_temperatura"]
        for lag in [1, 2, 3, 7, 14, 30]:
            self.df[f"consumption_lag_{lag}"] = c.shift(lag)
        for w in [3, 7, 14, 30]:
            self.df[f"rolling_mean_{w}"] = c.shift(1).rolling(w).mean()
        self.df["rolling_std_7"] = c.shift(1).rolling(7).std()
        self.df["rolling_std_30"] = c.shift(1).rolling(30).std()
        self.df["trend_7days"] = c.diff(7)
        self.df["trend_30days"] = c.diff(30)
    
    def weather_features(self):
        self.df["temp"] = self.df["temperature_2m_mean"]
        self.df["temp_lag_1"] = self.df["temp"].shift(1)
        self.df["temp_squared"] = self.df["temp"] ** 2
        self.df["humidity"] = self.df["relative_humidity_2m_mean"]
        self.df["rain_amount"] = self.df["precipitation_sum"]
        self.df["rain_flag"] = (self.df["rain_amount"] > 1).astype(int)
        self.df["wind_speed"] = self.df["wind_speed_10m_mean"]
        self.df["solar_radiation"] = self.df["shortwave_radiation_sum"]
    
    def interaction_features(self):
        self.df["temp_x_rain"] = self.df["temp"] * self.df["rain_flag"]
        self.df["temp_x_humidity"] = self.df["temp"] * self.df["humidity"]
        self.df["temp_x_wind"] = self.df["temp"] * self.df["wind_speed"]
    
    def time_features(self):
        self.df["weekday"] = self.df.date.dt.dayofweek
        self.df["month"] = self.df.date.dt.month
        self.df["is_weekend"] = (self.df.weekday >= 5).astype(int)
        self.df["weekday_sin"] = np.sin(2 * np.pi * self.df.weekday / 7)
        self.df["weekday_cos"] = np.cos(2 * np.pi * self.df.weekday / 7)
        self.df["month_sin"] = np.sin(2 * np.pi * self.df.month / 12)
        self.df["month_cos"] = np.cos(2 * np.pi * self.df.month / 12)
        self._holiday_features()

    def _holiday_features(self):
        """Mark Portuguese national holidays"""
        try:
            # Get years from dataset
            min_year = self.df["date"].min().year
            max_year = self.df["date"].max().year
            # Create Portuguese holiday calendar
            pt_holidays = holidays.Portugal(years=range(min_year, max_year + 1))
            
            # Mark if date is a holiday
            self.df["is_holiday"] = self.df["date"].apply(
                lambda x: 1 if x in pt_holidays else 0
            ).astype(int)
            
            # day between holiday and weekend
            self.df["is_bridge_day"] = 0
            
            for idx in self.df.index:
                date_val = self.df.loc[idx, "date"]
                weekday = self.df.loc[idx, "weekday"]
                
                # Thursday before Friday holiday
                if weekday == 3:  # Thursday
                    next_day = date_val + pd.Timedelta(days=1)
                    if next_day in pt_holidays:
                        self.df.loc[idx, "is_bridge_day"] = 1
                
                # Tuesday after Monday holiday
                if weekday == 1:  # Tuesday
                    prev_day = date_val - pd.Timedelta(days=1)
                    if prev_day in pt_holidays:
                        self.df.loc[idx, "is_bridge_day"] = 1
        except ImportError:
            self.df["is_holiday"] = 0
            self.df["is_bridge_day"] = 0
        except Exception as e:
            logger.error(f"Error marking holidays: {e}")
            self.df["is_holiday"] = 0
            self.df["is_bridge_day"] = 0
    
    def cleanup(self):
        if not self.is_production:
            # remove rows without targets
            self.df = self.df[self.df[self.target_cols].notna().all(axis=1)]
        else:
            logger.info(f"Production mode: keeping all {len(self.df)} rows")
        # fill remaining NaNs
        num_cols = self.df.select_dtypes("number").columns
        self.df[num_cols] = self.df[num_cols].fillna(self.df[num_cols].mean())

    def run(self):
        self.fix_data_errors()
        self.consumption_features()
        self.weather_features()
        self.interaction_features()
        self.time_features()
        self.cleanup()

        self.df.to_csv(self.output_path, index=False)
        logger.info(f"Feature engineering completed - {len(self.df)} rows")

        return self.df