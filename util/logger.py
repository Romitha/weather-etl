"""Module provides a logging utility for configuring and managing log outputs.

for the weather pipeline application.
"""
import logging


class Logger:
    """A utility class for setting up and managing logging for the application."""

    # Configure Logging
    @staticmethod
    def setup_logging():
        """Configure and initializes logging for the weather pipeline.

        Sets up a logger with both a file handler and a console handler,
        each with appropriate formatting. The logger outputs messages to
        a log file and the console for better monitoring.

        :return: A configured logger instance.
        :rtype: logging.Logger
        """
        # Create logger
        logger = logging.getLogger("weather_pipeline")
        logger.setLevel(logging.INFO)

        # Create formatters
        detailed_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_formatter = logging.Formatter("%(levelname)s: %(message)s")

        # File handler (existing)
        file_handler = logging.FileHandler("etl_pipeline.log")
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
