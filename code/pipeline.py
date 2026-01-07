import sys
from preprocessor.main_preprocessor import MainPreprocessor
from trainer.trainer import EnergyModelTrainer
from predictor.main_predictor import EnergyPredictor
from trainer.logger import get_logger
import traceback
from collector.main_collector import MainCollector

logger = get_logger('Pipeline')

class CollectorPipeline:
    """Data collection - daily"""
    def __init__(self):
        self.collector = MainCollector()
    
    def run(self):
        try:
            self.collector.collect()
            logger.info(" Data collection completed successfully!")
        except Exception as e:
            logger.error(f" Collection error: {e}")
            traceback.print_exc()
            raise


class PreprocessorPipeline:
    """Feature engineering - daily"""
    def __init__(self):
        self.preprocessor = MainPreprocessor()
    
    def run(self):
        try:
            self.preprocessor.main()
            logger.info(" Feature engineering completed successfully!")
        except Exception as e:
            logger.error(f" Preprocessing error: {e}")
            traceback.print_exc()
            raise


class TrainerPipeline:
    """Model training - weekly (Mondays)"""
    def __init__(self):
        self.trainer = EnergyModelTrainer()
    
    def run(self):
        try:
            results = self.trainer.train_models()
            return results
            
        except Exception as e:
            logger.error(f" Training error: {e}")
            traceback.print_exc()
            raise

class PredictorPipeline:
    """Generate forecasts - daily"""
    def __init__(self):
        self.predictor = EnergyPredictor()
    
    def run(self):
        try:
            result = self.predictor.predict_next_7_days()
            logger.info(f" Forecast for the next 7 days: {result}")
            return result
        except Exception as e:
            logger.error(f" Prediction error: {e}")
            traceback.print_exc()
            raise


def run_collection():
    """Step 1: Data collection only"""
    pipeline = CollectorPipeline()
    pipeline.run()


def run_preprocessing():
    """Step 2: Feature engineering """
    pipeline = PreprocessorPipeline()
    pipeline.run()


def run_training():
    """Step 3: Model training only"""
    pipeline = TrainerPipeline()
    pipeline.run()


def run_prediction():
    """Step 4: Forecast generation"""
    pipeline = PredictorPipeline()
    pipeline.run()


def run_full_pipeline():
    """Run complete pipeline: collect --> preprocess --> train --> predict""" 
    try:
        # Step 1: Collection
        run_collection()
        # Step 2: Preprocessing
        run_preprocessing()
        # Step 3: Training
        run_training()
        logger.info("")
        # Step 4: Prediction
        run_prediction()
    except Exception as e:
        logger.error("")
        raise


if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "collect":
            run_collection()
        
        elif mode == "preprocess":
            run_preprocessing()
        
        elif mode == "train":
            run_training()
        
        elif mode == "predict":
            run_prediction()
        
        elif mode == "full":
            run_full_pipeline()
        
        else:
            logger.error(f" Unknown command: {mode}")
            sys.exit(1)
    else:
        logger.error("Usage: python pipeline.py <command>")
        logger.error("")
        sys.exit(0)
