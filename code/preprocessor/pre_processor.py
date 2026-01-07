import joblib
import os
import pandas as pd
from datetime import datetime
from pathlib import Path

from preprocessor.logger import get_logger
logger = get_logger("Preprocessor")

class Preprocessor:
    def __init__(self, feature_names: list, scaler):
        if not isinstance(feature_names, list) or len(feature_names) == 0:
            raise ValueError("feature_names must be a non empty list")
        
        self.feature_names = feature_names         
        self.scaler = scaler                      
        self.meta = {
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": "final submission",
            "num_features": len(feature_names),
        }
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features for prediction (excludes targets)."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        
        feature_cols = [f for f in self.feature_names if not f.startswith('target_next_day_')]
        
        # Check missing features 
        missing = [f for f in feature_cols if f not in X.columns]
        if missing:
            raise ValueError(
                f"Missing features at prediction time: {missing}\n"
                f"Expected: {len(feature_cols)} features\n"
                f"Got: {len(X.columns)} columns"
            )
        

        X_out = X[feature_cols].copy()
        
        # Apply scaler (without fit)
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_out),
            columns=feature_cols,  
            index=X_out.index
        )
        
        return X_scaled
    
    def save(self, path: Path):
        """Persistence"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Preprocessor saved to {path}")
    
    @staticmethod
    def load(path: Path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Preprocessor not found: {path}")
        
        obj = joblib.load(path)
        if not isinstance(obj, Preprocessor):
            raise TypeError("Loaded object is not a Preprocessor")
        
        return obj
    
    def get_info(self):
        feature_cols = [f for f in self.feature_names 
                        if not f.startswith('target_next_day_')]
        
        return {
            "num_features": len(feature_cols),
            "num_targets": len(self.feature_names) - len(feature_cols),
            "features": feature_cols,
            "has_scaler": self.scaler is not None,
            "saved_at": self.meta["saved_at"],
            "version": self.meta["version"],
        }
    
    def __repr__(self):
        feature_cols = [f for f in self.feature_names 
                        if not f.startswith('target_next_day_')]
        return f"Preprocessor(num_features={len(feature_cols)})"