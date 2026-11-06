import os
import random
import logging
from pathlib import Path
import yaml
import numpy as np
import tensorflow as tf

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Safely loads YAML configuration across operating systems."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path.resolve()}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def setup_logger(log_path: str = "outputs/logs/pipeline.log") -> logging.Logger:
    """Configures file and console logging."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("PlantDiseaseApp")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(module)s: %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
    return logger

def set_seeds(seed: int = 42):
    """Sets random seeds for reproducibility across Python, NumPy, and TensorFlow."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)