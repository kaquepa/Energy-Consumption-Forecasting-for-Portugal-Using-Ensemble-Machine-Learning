from pathlib import Path

class BaseConfig:
    """Base configuration class with utility methods."""
    
    def __init__(self):
        config_location = Path(__file__).parent.resolve()
        if config_location.name == 'code':
            self.project_root = config_location.parent
        else:
            # Fallback: assume we're at project root
            self.project_root = config_location
    def resolve(self, path: Path) -> Path:
        """Resolve path to absolute path."""
        return path.resolve()
    
    def ensure_dir(self, path: Path):
        """Create directory if it doesn't exist."""
        path.mkdir(parents=True, exist_ok=True)
    
    def check_files(self, paths, warn_only=True):
        """Check if files exist, warn or raise error."""
        missing = [p for p in paths if not p.exists()]
        if missing:
            msg = "\n".join(f" {p}" for p in missing)
            if warn_only:
                print(f"Missing files:\n{msg}")
            else:
                raise FileNotFoundError(f"Missing required files:\n{msg}")


class Config_collector(BaseConfig):
    """Configuration for data collection module."""
    
    def __init__(self):
        super().__init__()
        self.DATA_DIR_RAW = self.project_root / "data" / "raw"
        self.DATA_DIR_PROCESSED = self.project_root / "data" / "processed"
        
        # Ensure directories exist
        self.ensure_dir(self.DATA_DIR_RAW)
        self.ensure_dir(self.DATA_DIR_PROCESSED)
        
        # Raw data files
        self.REN_DATASET_PATH = self.DATA_DIR_RAW / "ren_dataset_raw.csv"
        self.REN_PRODUCTION_PATH = self.DATA_DIR_RAW / "ren_production_raw.csv"
        self.REN_CONSUMPTION_PATH = self.DATA_DIR_RAW / "ren_consumption_raw.csv"
        self.REN_LATEST_PATH = self.DATA_DIR_RAW / "ren_latest_row_raw.csv"
        self.WEATHER_PATH = self.DATA_DIR_RAW / "weather_raw.csv"
        
        # Processed data files
        self.dataset_merged_path = self.DATA_DIR_PROCESSED / "dataset_base.csv"
        self.dataset_production_path = self.DATA_DIR_PROCESSED / "dataset_production.csv"
        self.latest_with_weather_path = self.DATA_DIR_PROCESSED / "latest_row_with_weather.csv"
        
        # API configuration
        self.LATITUDE = 39.5   # Portugal center
        self.LONGITUDE = -8.0
        self.START_DATE = "2010-01-01"
        

class Config_preprocessor(BaseConfig):
    """Configuration for preprocessing/feature engineering module."""
    
    def __init__(self):
        super().__init__()
        
        # Input datasets (from collector)
        self.dataset_trainer_path = self.project_root / "data" / "processed" / "dataset_base.csv"
        self.dataset_production_path = self.project_root / "data" / "processed" / "dataset_production.csv"
        
        # Output directory (preprocessor results)
        self.data_dir = self.project_root / "data" / "preprocessor"
        self.ensure_dir(self.data_dir)
        
        # Output files
        self.dataset_trainer_final_path = self.data_dir / "dataset_trainer_final.csv"
        self.dataset_production_final_path = self.data_dir / "dataset_production_final.csv"
        self.preprocessing_path = self.data_dir / "preprocessing.pkl"
        
        # Feature engineering parameters
        self.LAG_DAYS = [1, 2, 3, 7, 14, 30]
        self.ROLLING_WINDOWS = [3, 7, 14, 30]


class Config_trainer(BaseConfig):
    """Configuration for model training module."""
    
    def __init__(self):
        super().__init__()
        
        # Input dataset (from preprocessor)
        self.dataset_final_path = self.project_root / "data" / "preprocessor" / "dataset_trainer_final.csv"
        self.preprocessing_path = self.project_root / "data" / "preprocessor" / "preprocessing.pkl"
        
        # Model directory 
        self.model_dir_path = self.project_root / "code" / "trainer" / "models"
        self.production_dir = self.model_dir_path / "production"
        self.candidates_dir = self.model_dir_path / "candidates"
        
        # Ensure directories exist
        self.ensure_dir(self.production_dir)
        self.ensure_dir(self.candidates_dir)
        
        # Model registry files
        self.selected_features_path = self.model_dir_path / "selected_features.json"
        self.registry_path = self.model_dir_path / "registry.json"
        
        # Training parameters
        self.TRAIN_SIZE = 80  # 80% train, 20% validation
        self.WINDOW_DAYS = 365 * 15  # 15 years
        self.N_HORIZONS = 7  # Forecast 1-7 days ahead
        
        # Model candidates
        self.MODELS = ['RandomForest', 'LightGBM', 'XGBoost']


class Config_predict(BaseConfig):
    """Configuration for prediction module."""
    
    def __init__(self):
        super().__init__()
        
        # Input dataset (from preprocessor)
        self.dataset_production_final_path = self.project_root / "data" / "preprocessor" / "dataset_production_final.csv"
        
        # Models directory 
        self.models_dir = self.project_root / "code" / "trainer" / "models"
        self.feature_names_path = self.project_root / "code" / "trainer" / "models" / "selected_features.json"
        self.preprocessing_path = self.project_root / "data" / "preprocessor" / "preprocessing.pkl"
        
        # Results directory 
        self.results_dir = self.project_root / "code" / "predictor" / "results"
        self.ensure_dir(self.results_dir)
        self.latest_prediction_path = self.results_dir / "latest_prediction.json"
        
        # Prediction parameters
        self.history_size = 90  # Days of history to keep
        self.N_HORIZONS = 7  # Predict 1-7 days ahead


class Config_backend(BaseConfig):
    """Configuration for FastAPI backend."""
    
    def __init__(self):
        super().__init__()
        
        # Model paths
        self.models_dir = self.project_root / "code" / "trainer" / "models" / "production"
        self.feature_names_path = self.project_root / "code" / "trainer" / "models" / "selected_features.json"
        self.preprocessing_path = self.project_root / "data" / "preprocessor" / "preprocessing.pkl"
        self.dashboard_dir = self.project_root / "app"
        
        
        # Dataset paths
        self.dataset_production_path = self.project_root / "data" / "preprocessor" / "dataset_production_final.csv"
        
        # API configuration
        self.API_HOST = "0.0.0.0"
        self.API_PORT = 8000
        self.CACHE_TTL = 300  # 5 minutes
        self.LATITUDE = 39.5   # Portugal center
        self.LONGITUDE = -8.0
        self.WEATHER_CACHE_TTL = 300  # 5 minutes
        self.FORECAST_CACHE_TTL = 600  # 10 minutes
        
        


class Config_dashboard(BaseConfig):
    """Configuration for Streamlit dashboard."""
    
    def __init__(self):
        super().__init__()
        
        # API configuration
        self.API_BASE_URL = "http://localhost:8000"
        
        # Dashboard settings
        self.PAGE_TITLE = "Energy Consumption Forecasting for Portugal"
        self.PAGE_ICON = "⚡"
        self.LAYOUT = "wide"
        
        # Display settings
        self.HISTORICAL_DAYS = 90
        self.FORECAST_DAYS = 7
        self.AUTO_REFRESH_SECONDS = 300  # 5 minutes
        # Colors (matching brand)
        self.COLORS = {
            'primary': '#1a5490',
            'success': '#28a745',
            'warning': '#ffc107',
            'danger': '#dc3545',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40',
            'gray': '#6c757d',
        }
         # Chart colors
        self.CHART_COLORS = {
            'historical_line': 'rgba(26, 84, 144, 1.0)',
            'forecast_line': 'rgba(26, 84, 144, 0.8)',
            'confidence_band': 'rgba(26, 84, 144, 0.15)',
            'weekend_marker': 'rgba(220, 53, 69, 0.6)',
            'today_line': 'rgba(40, 167, 69, 0.8)',
        }
        # Refresh interval
        self.AUTO_REFRESH_SECONDS = 60

        # Confidence thresholds
        self.CONFIDENCE_THRESHOLDS = {
            'high': 2.0,     # std_models < 2.0
            'medium': 3.0,   # std_models < 3.0
            'low': float('inf')  # std_models >= 3.0
        }
         # Cache TTL
        self.WEATHER_CACHE_TTL = 300  # 5 minutes
        self.FORECAST_CACHE_TTL = 600  # 10 minutes

        self.LATITUDE = 39.3999
        self.LONGITUDE = -8.2245




def get_collector_config():
    """Get collector configuration."""
    return Config_collector()

def get_preprocessor_config():
    """Get preprocessor configuration."""
    return Config_preprocessor()

def get_trainer_config():
    """Get trainer configuration."""
    return Config_trainer()

def get_predict_config():
    """Get predictor configuration."""
    return Config_predict()

def get_backend_config():
    """Get backend configuration."""
    return Config_backend()

def get_dashboard_config():
    """Get dashboard configuration."""
    return Config_dashboard()

