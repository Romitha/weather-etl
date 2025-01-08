from dataclasses import dataclass, field
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger('weather_pipeline')
# Define Data Structures
@dataclass
class Location:
    name: str
    region: str
    country: str
    latitude: float
    longitude: float
    timezone: str
    last_updated: str

@dataclass
class CurrentWeather:
    temperature_c: float
    condition_text: str
    last_updated: str

    def __post_init__(self):
        # Validate temperature_c field
        self.temperature_c = self.safe_convert(self.temperature_c)

    def safe_convert(self, value):
        """Safely convert temperature to float."""
        try:
            return float(value)
        except (ValueError, TypeError):
            logger = logging.getLogger('weather_pipeline')
            logger.warning(f"Invalid value encountered during CurrentWeather initialization: {value}")
            return np.nan

@dataclass
class Forecast:
    date: str
    max_temp_c: float
    min_temp_c: float
    avg_temp_c: float
    total_precip_mm: float
    max_wind_kph: float
    condition_text: str

    def __post_init__(self):
        # Validate that temperature values are valid floats
        self.max_temp_c = self.safe_convert(self.max_temp_c)
        self.min_temp_c = self.safe_convert(self.min_temp_c)
        self.avg_temp_c = self.safe_convert(self.avg_temp_c)
        self.total_precip_mm = self.safe_convert(self.total_precip_mm)
        self.max_wind_kph = self.safe_convert(self.max_wind_kph)

    def safe_convert(self, value):
        """Safely convert values to float and log invalid ones."""
        try:
            return float(value)
        except (ValueError, TypeError):
            # Log the invalid data
            logger = logging.getLogger('weather_pipeline')
            logger.warning(f"Invalid value encountered during Forecast initialization: {value}")
            return np.nan

@dataclass
class WeatherData:
    location: Location
    current_weather: CurrentWeather
    forecasts: List[Forecast] = field(default_factory=list)