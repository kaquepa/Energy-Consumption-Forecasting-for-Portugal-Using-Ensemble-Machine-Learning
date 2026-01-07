import json
import joblib
import numpy as np
import pandas as pd
import shap
from typing import Dict, List
from shutil import copyfile
from sklearn.linear_model import RidgeCV
from config import Config_trainer

class ShapSelector:
    """Selects the most important features using SHAP values."""
    def __init__(self, feature_names: List[str], top_k: int = 30):
        self.feature_names = feature_names
        self.top_k = top_k

    def select(self, model, X_train: pd.DataFrame) -> List[str]:
        """Compute SHAP values and return the top-k most important features."""
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        mean_abs = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": mean_abs
        }).sort_values("importance", ascending=False)
        return importance_df.head(self.top_k)["feature"].tolist()

class ModelSaver:
    """Handles model persistence and promotion to production."""
    def __init__(self):
        self.Config = Config_trainer()
        self.registry = ModelRegistry()

        self.production_dir = self.Config.model_dir_path / "production"
        self.candidates_dir = self.Config.model_dir_path / "candidates"

        self.production_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model, name: str, metrics: Dict[str, float]) -> bool:
        """Save a trained model and promote it to production if it improves RMSE."""
        if not name.startswith("Ensemble_"):
            return False

        if not isinstance(metrics, dict) or "RMSE" not in metrics:
            raise ValueError(
                f"Invalid metrics for {name}. Expected {{'RMSE': ...}}, got {metrics}"
            )

        horizon = name.split("_")[-1]  # h1, h2, ...
        rmse = float(metrics["RMSE"])

        candidate_path = self.candidates_dir / f"{name}.pkl"
        production_path = self.production_dir / f"{name}.pkl"

        # Always save candidate
        joblib.dump(model, candidate_path)

        # auto-promote
        if not production_path.exists():
            copyfile(candidate_path, production_path)

            self.registry.register_new(
                horizon=horizon,
                rmse=rmse,
                model_path=str(production_path.relative_to(self.Config.project_root)),
                metrics=metrics
            )

            print(f"[PROMOTED - FIRST] {name} | RMSE={rmse:.4f}")
            return True

        # compare RMSE
        promoted = self.registry.update_if_better(
            horizon=horizon,
            rmse=rmse,
            model_path=str(production_path.relative_to(self.Config.project_root)),
            metrics=metrics
        )

        if promoted:
            copyfile(candidate_path, production_path)
            print(f"[PROMOTED] {name} | RMSE={rmse:.4f}")
        else:
            print(f"[REJECTED] {name} | RMSE={rmse:.4f}")

        return promoted

class ModelRegistry:
    """Stores and manages production models per forecasting horizon."""
    def __init__(self):
        self.Config = Config_trainer()
        self.path = self.Config.model_dir_path / "registry.json"

        if not self.path.exists():
            self.save({"horizons": {}})

    def load(self) -> Dict:
        """Load registry content from disk."""
        with open(self.path, "r") as f:
            return json.load(f)

    def save(self, data: Dict) -> None:
        """Persist registry content to disk."""
        with open(self.path, "w") as f:
            json.dump(data, f, indent=4)

    def get_current(self, horizon: str) -> Dict:
        """Get the currently deployed model information for a horizon."""
        data = self.load()
        return data["horizons"].get(horizon)

    def update_if_better(
        self,
        horizon: str,
        rmse: float,
        model_path: str,
        metrics: Dict[str, float]
    ) -> bool:
        """Update production model only if it already exists and RMSE improves."""

        data = self.load()
        current = data.get("horizons", {}).get(horizon)

        # Do not auto-promote if horizon does not exist
        if current is None:
            return False

        if rmse < current["rmse"]:
            data["horizons"][horizon] = {
                "rmse": float(rmse),
                "metrics": metrics,
                "path": model_path
            }
            self.save(data)
            return True

        return False
    def register_new(
        self,
        horizon: str,
        rmse: float,
        model_path: str,
        metrics: Dict[str, float]
    ) -> None:
        """Register a new horizon in the registry."""
        data = self.load()

        if "horizons" not in data:
            data["horizons"] = {}

        data["horizons"][horizon] = {
            "rmse": float(rmse),
            "metrics": metrics,
            "path": model_path
        }

        self.save(data)




