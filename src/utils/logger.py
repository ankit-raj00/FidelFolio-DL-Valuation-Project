import logging
import os
from datetime import datetime

def setup_logger(name="FidelFolio_Logger", log_dir="logs", model_name="pipeline"):
    """
    Sets up a modular logger that writes to a timestamped file and stdout.
    
    Args:
        name (str): Name of the logger.
        log_dir (str): Directory to save log files.
        model_name (str): Name of the model (or prefix) for the log file.
        
    Returns:
        logging.Logger: Configured logger.
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{model_name}_{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    # Create Logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clean up existing handlers to avoid duplicates if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # Create Handlers
    # 1. File Handler
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)
    
    # 2. Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add Handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
