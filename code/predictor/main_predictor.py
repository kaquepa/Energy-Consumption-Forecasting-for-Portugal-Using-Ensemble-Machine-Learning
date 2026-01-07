import json
import joblib
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict
import traceback

from predictor.logger import get_logger
from config import Config_predict

from backend.weather_utils import WeatherUtils
weather_forecast = WeatherUtils()
logger = get_logger("EnergyPredictor")

class EnergyPredictor:
    """correct energy predictor. """
    def __init__(self):
        self.Config = Config_predict()
        self._validate()
        self._load_data()
        self._load_preprocessor()
        self._load_models()
        self._load_metrics()

    def _validate(self):
        if not self.Config.dataset_production_final_path.exists():
            raise FileNotFoundError(self.Config.dataset_production_final_path)

        if not self.Config.preprocessing_path.exists():
            raise FileNotFoundError(self.Config.preprocessing_path)

        if not self.Config.models_dir.exists():
            raise FileNotFoundError(self.Config.models_dir)

    def _load_data(self):
        df = pd.read_csv(
            self.Config.dataset_production_final_path,
            parse_dates=["date"]
        )
        df = df.sort_values("date").reset_index(drop=True)
        if df.empty:
            raise ValueError("Production dataset is empty")
        # only last known day
        self.df_last = df.tail(1).copy()

    def _load_preprocessor(self):
        self.preprocessor = joblib.load(self.Config.preprocessing_path)
        if not hasattr(self.preprocessor, "feature_names"):
            raise ValueError("Invalid preprocessor")

    def _load_models(self):
        self.models = {}
        prod = self.Config.models_dir / "production"
        for h in range(1, 8):
            path = prod / f"Ensemble_h{h}.pkl"
            if path.exists():
                self.models[h] = joblib.load(path)
        if not self.models:
            raise RuntimeError("No models loaded")

    def _load_metrics(self):
        self.metrics = {}
        reg = self.Config.models_dir / "registry.json"

        if not reg.exists():
            return

        with open(reg) as f:
            data = json.load(f)

        for h, info in data.get("horizons", {}).items():
            self.metrics[int(h[1:])] = info.get("metrics", {})

    def _predict_next_7_days(self) -> Dict:
        X_base = self.preprocessor.transform(self.df_last)
        today = datetime.now().date()
        results = {}
        for h, model in self.models.items():
            y = float(model.predict(X_base)[0])

            date_h = (today + timedelta(days=h - 1)).strftime("%Y-%m-%d")
            m = self.metrics.get(h, {})

            results[f"day_{h}"] = {
                "date": date_h,
                "predicted": round(y, 2),
                "weather": self._weather(date_h),
                "metrics": m,
            }
        output = {
            "last_known": {
                "date": str(self.df_last["date"].iloc[0].date()),
                "consumption": float(self.df_last["corrigido_temperatura"].iloc[0]),
            },
            "predictions": results,
            "generated_at": datetime.now().isoformat(),
        }
        self._save(output)
        return output

    def _weather(self, date_str):
        try:
            weather = weather_forecast.get_weather_forecast(7)
            for d in weather:
                if d["date"] == date_str:
                    return d
        except Exception:
            pass
        return None

    def _save(self, data):
        out = self.Config.project_root /  "code" / "predictor" / "results" / "latest_prediction.json" #"predictor/results/latest_prediction.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def predict_next_7_days(self) -> Dict:
        X_base = self.preprocessor.transform(self.df_last)
        last_known_date = self.df_last["date"].iloc[0].date()
        results = {}
        for h, model in self.models.items():
            y = float(model.predict(X_base)[0])
            date_h = (last_known_date + timedelta(days=h)).strftime("%Y-%m-%d")
            m = self.metrics.get(h, {})
            results[f"day_{h}"] = {
                "date": date_h,
                "predicted": round(y, 2),
                "weather": self._weather(date_h),
                "metrics": m,
            }

        output = {
            "last_known": {
                "date": str(last_known_date),
                "consumption": float(self.df_last["corrigido_temperatura"].iloc[0]),
            },
            "predictions": results,
            "generated_at": datetime.now().isoformat(),
        }

        self._save(output)
        return output