import logging

class Logger:
    # Configure Logging
    def setup_logging():
        # Create logger
        logger = logging.getLogger('weather_pipeline')
        logger.setLevel(logging.INFO)
        
        # Create formatters
        detailed_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        
        # File handler (existing)
        file_handler = logging.FileHandler('etl_pipeline.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler (new)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger