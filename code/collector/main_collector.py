import pandas as pd
from config import Config_collector
from collector.logger import get_logger
from collector.energy_collector import RENRawCollector
from collector.energy_transformer import RENTransformer
from collector.weather_collector import WeatherCollector

logger = get_logger("MainCollector")

class MainCollector:
    """Orchestrates full data collection pipeline."""
    def __init__(self):
        self.Config = Config_collector()
        print("init collect ...")

    def collect_energy(self):
        """Collect and transform REN energy data."""
        RENRawCollector().update()
        return RENTransformer().process()

    def collect_weather(self):
        """Collect weather data."""
        return WeatherCollector().update()

    def merge_datasets(self):
        """Merge energy and weather datasets."""
        ren_train = pd.read_csv(self.Config.REN_DATASET_PATH)
        ren_prod = pd.read_csv(self.Config.REN_PRODUCTION_PATH)
        weather = pd.read_csv(self.Config.WEATHER_PATH)

        ren_train["date"] = pd.to_datetime(ren_train["date"])
        ren_prod["date"] = pd.to_datetime(ren_prod["date"])
        weather["date"] = pd.to_datetime(weather["date"])

        train = ren_train.merge(weather, on="date", how="inner")
        prod = ren_prod.merge(weather, on="date", how="inner")
        train.to_csv(self.Config.dataset_merged_path, index=False)
        prod.to_csv(self.Config.DATA_DIR_PROCESSED / "dataset_production.csv", index=False)
        return train, prod

    def collect(self):
        """Run full pipeline."""
        logger.info("Starting full data collection")
        self.collect_energy()
        self.collect_weather()
        self.merge_datasets()
     