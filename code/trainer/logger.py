import logging
import os
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def get_logger(name="EnergyModelTrainer"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        timestamp = datetime.now().strftime("%Y-%m-%d")
        file_handler = logging.FileHandler(f"{LOG_DIR}/{name}_{timestamp}.log")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            "%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
