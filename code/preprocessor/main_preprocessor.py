from preprocessor.feature_engineering import FeatureEngineering
from config import Config_preprocessor
from preprocessor.logger import get_logger

logger = get_logger("MainPreprocessor")
class MainPreprocessor:
    def __init__(self):
        self.Config = Config_preprocessor()

    def main(self):
        # Dataset for train
        fe_train = FeatureEngineering(
            input_path=self.Config.dataset_trainer_path,
            output_path=self.Config.dataset_trainer_final_path,
            is_production=False
        )
        fe_train.run()

        # Dataset for predict
        fe_prod = FeatureEngineering(
            input_path=self.Config.dataset_production_path,
            output_path=self.Config.dataset_production_final_path,
            is_production=True
        )
        df_prod = fe_prod.run()
        logger.info("Preprocessing finished successfully")
        return df_prod