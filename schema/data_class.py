"""Weather-related data structures."""
import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np

logger = logging.getLogger("weather_pipeline")
# Define Data Structures


@dataclass
class Location:
    """Represents a geographical location with associated metadata."""

    name: str
    region: str
    country: str
    latitude: float
    longitude: float
    timezone: str
    last_updated: str


@dataclass
class CurrentWeather:
    """Represents the current weather conditions for a location."""

    temperature_c: float
    condition_text: str
    last_updated: str

    def __post_init__(self):
        """Post-initialization method to validate and convert the temperature field."""
        self.temperature_c = self.safe_convert(self.temperature_c)

    def safe_convert(self, value):
        """Safely converts a value to a float, logging any invalid conversions.

        :param value: The value to convert.
        :type value: Any
        :return: The converted value, or NaN if conversion fails.
        :rtype: float
        """
        try:
            return float(value)
        except (ValueError, TypeError):
            logger = logging.getLogger("weather_pipeline")
            logger.warning(f"Invalid value encountered during CurrentWeather initialization: {value}")
            return np.nan


@dataclass
class Forecast:
    """Represents a single day's weather forecast."""

    date: str
    max_temp_c: float
    min_temp_c: float
    avg_temp_c: float
    total_precip_mm: float
    max_wind_kph: float
    condition_text: str

    def __post_init__(self):
        """Post-initialization method to validate and convert numerical fields."""
        # Validate that temperature values are valid floats
        self.max_temp_c = self.safe_convert(self.max_temp_c)
        self.min_temp_c = self.safe_convert(self.min_temp_c)
        self.avg_temp_c = self.safe_convert(self.avg_temp_c)
        self.total_precip_mm = self.safe_convert(self.total_precip_mm)
        self.max_wind_kph = self.safe_convert(self.max_wind_kph)

    def safe_convert(self, value):
        """Safely converts a value to a float, logging any invalid conversions.

        :param value: The value to convert.
        :type value: Any
        :return: The converted value, or NaN if conversion fails.
        :rtype: float
        """
        try:
            return float(value)
        except (ValueError, TypeError):
            # Log the invalid data
            logger = logging.getLogger("weather_pipeline")
            logger.warning(f"Invalid value encountered during Forecast initialization: {value}")
            return np.nan


@dataclass
class WeatherData:
    """Aggregates weather data, including location, current weather, and forecasts."""

    location: Location
    current_weather: CurrentWeather
    forecasts: List[Forecast] = field(default_factory=list)
