"""Weather-related data structures."""
import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np

logger = logging.getLogger("weather_pipeline")
# Define Data Structures


@dataclass
class Location:
    """Represents a geographical location with associated metadata.

    :param name: The name of the location.
    :type name: str
    :param region: The region or state the location belongs to.
    :type region: str
    :param country: The country the location is in.
    :type country: str
    :param latitude: The latitude of the location.
    :type latitude: float
    :param longitude: The longitude of the location.
    :type longitude: float
    :param timezone: The timezone of the location.
    :type timezone: str
    :param last_updated: The timestamp of the last update for the location data.
    :type last_updated: str
    """

    name: str
    region: str
    country: str
    latitude: float
    longitude: float
    timezone: str
    last_updated: str


@dataclass
class CurrentWeather:
    """Represents the current weather conditions for a location.

    :param temperature_c: The current temperature in Celsius.
    :type temperature_c: float
    :param condition_text: A textual description of the current weather condition.
    :type condition_text: str
    :param last_updated: The timestamp of the last update for the weather data.
    :type last_updated: str
    """

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
    """Represents a single day's weather forecast.

    :param date: The date of the forecast.
    :type date: str
    :param max_temp_c: The maximum temperature in Celsius for the day.
    :type max_temp_c: float
    :param min_temp_c: The minimum temperature in Celsius for the day.
    :type min_temp_c: float
    :param avg_temp_c: The average temperature in Celsius for the day.
    :type avg_temp_c: float
    :param total_precip_mm: The total precipitation in millimeters for the day.
    :type total_precip_mm: float
    :param max_wind_kph: The maximum wind speed in kilometers per hour.
    :type max_wind_kph: float
    :param condition_text: A textual description of the day's weather condition.
    :type condition_text: str
    """

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
    """Aggregates weather data, including location, current weather, and forecasts.

    :param location: The location information.
    :type location: Location
    :param current_weather: The current weather conditions.
    :type current_weather: CurrentWeather
    :param forecasts: A list of weather forecasts for upcoming days.
    :type forecasts: List[Forecast]
    """

    location: Location
    current_weather: CurrentWeather
    forecasts: List[Forecast] = field(default_factory=list)
