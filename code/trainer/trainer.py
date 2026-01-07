import os
import json
import numpy as np
import pandas as pd
from typing import Dict

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
 
 
from trainer.logger import get_logger
from config import Config_trainer, Config_preprocessor
from trainer.utils_trainer import ShapSelector, ModelSaver
from preprocessor.pre_processor import Preprocessor

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
logger = get_logger("EnergyModelTrainer")

class EnergyModelTrainer:
    """Trains multiple models per horizon and selects the best one....Model Competition: RandomForest vs LightGBM vs XGBoost"""
    def __init__(self):
        self.Config = Config_trainer()
        
      
        self.csv_path = self.Config.dataset_final_path

        os.makedirs("results", exist_ok=True)
        self.results: Dict[str, Dict] = {}
        self.saver = ModelSaver()

    @staticmethod
    def metrics(y_true, y_pred) -> Dict[str, float]:
        return {
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "MAPE": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
            "R2": float(r2_score(y_true, y_pred)),
        }

    
    def load_data(self):
        """ Data loading"""
        df = pd.read_csv(self.csv_path, parse_dates=["date"])
        df = df.sort_values("date")

        # Rolling window
        cutoff_date = df["date"].max() - pd.Timedelta(days=self.Config.WINDOW_DAYS)
        df = df[df["date"] >= cutoff_date]

        target_cols = [f"target_next_day_{h}" for h in range(1, 8)]
        df = df.dropna(subset=target_cols)
        df = df.drop(columns=["date"])
        X = df.drop(columns=target_cols)
        y = df[target_cols]
        split = int(len(df) * (self.Config.TRAIN_SIZE / 100))

        self.X_train = X.iloc[:split].reset_index(drop=True)
        self.X_test  = X.iloc[split:].reset_index(drop=True)
        self.y_train = y.iloc[:split].values
        self.y_test  = y.iloc[split:].values
        self.feature_names = self.X_train.columns.tolist()

    def get_candidate_models(self):
        """Define the 3 candidate models for competition"""
        return {
            'RandomForest': RandomForestRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'LightGBM': LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=8,
                num_leaves=31,
                min_child_samples=20,
                random_state=42,
                verbosity=-1
            ),
            'XGBoost': XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=8,
                min_child_weight=3,
                random_state=42,
                verbosity=0
            )
        }

    def train_models(self) -> Dict[str, Dict[str, float]]:
        self.load_data()

        # SHAP feature selection (based on h1)
        logger.info("\nPerforming feature selection...")
        selector_model = LGBMRegressor(
            n_estimators=400,
            num_leaves=60,
            random_state=42,
            verbosity=-1,
        )
        selector_model.fit(self.X_train, self.y_train[:, 0])

        selector = ShapSelector(self.feature_names, top_k=30)
        selected_features = selector.select(selector_model, self.X_train)

        self.X_train = self.X_train[selected_features]
        self.X_test = self.X_test[selected_features]
        self.feature_names = selected_features

        with open(self.Config.selected_features_path, "w") as f:
            json.dump(self.feature_names, f, indent=2)

        logger.info(f"Selected {len(self.feature_names)} features")

        # Fit scaler
        scaler = StandardScaler()
        scaler.fit(self.X_train)

        # Apply scaling to train and test
        X_train_scaled = pd.DataFrame(
            scaler.transform(self.X_train),
            columns=self.feature_names,
            index=self.X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(self.X_test),
            columns=self.feature_names,
            index=self.X_test.index
        )

        # Save preprocessing pipeline
        preprocessor = Preprocessor(
            feature_names=self.feature_names,
            scaler=scaler,
        )
        preprocessor.save(Config_preprocessor().preprocessing_path)
        # Model competition - Train and select best per horizon
        for h in range(1, 8):
            y_train_h = self.y_train[:, h - 1]
            y_test_h = self.y_test[:, h - 1]

            candidates = self.get_candidate_models()
            results_h = {}

            # Train all candidates
            for model_name, model in candidates.items():
                model.fit(X_train_scaled, y_train_h)
                preds = model.predict(X_test_scaled)
                m = self.metrics(y_test_h, preds)

                results_h[model_name] = {
                    'model': model,
                    'metrics': m,
                    'rmse': m['RMSE']
                }

            # Select best model (lowest RMSE)
            best_name = min(results_h, key=lambda x: results_h[x]['rmse'])
            best_result = results_h[best_name]
            best_model = best_result['model']
            best_metrics = best_result['metrics']

            # Save only the best model
            self.saver.save_model(best_model, f"Ensemble_h{h}", best_metrics)
            self.results[f"Ensemble_h{h}"] = {
                **best_metrics,
                'model_type': best_name
            }

        return self.results

  