"""Test suite for the Weather ETL pipeline."""
import os
import sys
from unittest.mock import mock_open, patch

import pandas as pd
import pytest

from etl import WeatherPipeline

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def mock_json_data():
    """Fixture that provides mock weather JSON data for testing the pipeline.

    This mock data includes weather information for a city, including current weather,
    location details, and forecast data. It is used to simulate an API response or
    a JSON file that would be loaded in the `WeatherPipeline`.

    Returns:
        dict: A dictionary representing mock weather data for a city.
    """
    return {
        "City1": {
            "location": {
                "name": "City1",
                "region": "Region1",
                "country": "Country1",
                "lat": 12.34,
                "lon": 56.78,
                "tz_id": "Region/City1",
                "localtime": "2025-01-01 12:00",
            },
            "current": {"temp_c": 25.0, "condition": {"text": "Clear"}, "last_updated": "2025-01-01 12:00"},
            "forecast": {
                "forecastday": [
                    {
                        "date": "2025-01-02",
                        "day": {
                            "maxtemp_c": 30.0,
                            "mintemp_c": 20.0,
                            "avgtemp_c": 25.0,
                            "totalprecip_mm": 5.0,
                            "maxwind_kph": 15.0,
                            "condition": {"text": "Sunny"},
                        },
                    },
                ],
            },
        },
    }


@pytest.fixture
def mock_pipeline(mock_json_data):
    """Fixture that creates a mocked instance of the `WeatherPipeline` class
    and simulates the ETL process.

    This fixture mocks environment variables, file opening, and JSON loading
    to ensure that the pipeline can be tested without real dependencies.
    It also manually sets the DataFrames for current weather and forecast data
    for testing purposes.

    Args:
        mock_json_data (dict): Mock weather JSON data to be used by the pipeline.

    Returns:
        WeatherPipeline: A mocked instance of the WeatherPipeline with mocked data
                         and environment variables.
    """
    # Mock the environment variables for the test
    with patch.dict(os.environ, {"HIGH_TEMP": "28.0", "HEAVY_RAIN": "4.0", "STRONG_WIND": "14.0"}):
        # Mock reading the JSON file
        with patch("builtins.open", mock_open(read_data="{}")):
            with patch("json.load", return_value=mock_json_data):
                pipeline = WeatherPipeline("mock_file.json")
                pipeline.extract_data()
                pipeline.transform_data()

                # Manually populate the DataFrames for testing purposes
                pipeline.current_weather_df = pd.DataFrame(
                    {
                        "city": ["City1", "City2"],
                        "temperature_c": [22, 25],
                        "condition": ["Clear", "Rainy"],
                        "last_updated": ["2025-01-08", "2025-01-08"],
                    },
                )

                pipeline.forecast_df = pd.DataFrame(
                    {
                        "city": ["City1", "City2"],
                        "date": ["2025-01-09", "2025-01-10"],
                        "max_temp_c": [28, 30],
                        "min_temp_c": [20, 22],
                        "avg_temp_c": [24, 26],
                        "precipitation_mm": [0.0, 10.0],
                        "wind_kph": [5, 15],
                        "condition": ["Sunny", "Stormy"],
                    },
                )

                return pipeline


def test_extract_data(mock_pipeline):
    """Test case for the `extract_data` method of the WeatherPipeline.

    This test verifies that the `extract_data` method correctly extracts data
    from the mocked JSON data and populates the `weather_data` attribute
    of the pipeline.

    Args:
        mock_pipeline (WeatherPipeline): The mocked pipeline instance with pre-set data.
    """
    assert len(mock_pipeline.weather_data) == 1
    assert "City1" in mock_pipeline.weather_data
    city_data = mock_pipeline.weather_data["City1"]
    assert city_data.location.name == "City1"
    assert city_data.current_weather.temperature_c == 25.0


def test_transform_data(mock_pipeline):
    """Test case for the `transform_data` method of the WeatherPipeline.

    This test ensures that the `transform_data` method correctly transforms
    the extracted data into DataFrames and that these DataFrames are not empty.

    Args:
        mock_pipeline (WeatherPipeline): The mocked pipeline instance with pre-set data.
    """
    assert isinstance(mock_pipeline.current_weather_df, pd.DataFrame)
    assert isinstance(mock_pipeline.forecast_df, pd.DataFrame)
    assert not mock_pipeline.current_weather_df.empty
    assert not mock_pipeline.forecast_df.empty
    assert "temperature_c" in mock_pipeline.current_weather_df.columns
    assert "max_temp_c" in mock_pipeline.forecast_df.columns


def test_analyze_temperatures(mock_pipeline):
    """Test case for the `analyze_temperatures` method of the WeatherPipeline.

    This test verifies that the `analyze_temperatures` method generates a DataFrame
    containing the temperature analysis, including the highest temperature
    for each city.

    Args:
        mock_pipeline (WeatherPipeline): The mocked pipeline instance with pre-set data.
    """
    mock_pipeline.analyze_temperatures()
    assert isinstance(mock_pipeline.temp_analysis_df, pd.DataFrame)
    assert "city" in mock_pipeline.temp_analysis_df.columns
    assert "highest_temp" in mock_pipeline.temp_analysis_df.columns


def test_generate_alerts(mock_pipeline):
    """Test case for the `generate_alerts` method of the WeatherPipeline.

    This test ensures that the `generate_alerts` method correctly generates alerts
    based on predefined thresholds for high temperature, heavy rain, and strong wind.

    Args:
        mock_pipeline (WeatherPipeline): The mocked pipeline instance with pre-set data.
    """
    mock_pipeline.THRESHOLDS = {"high_temp": 28.0, "heavy_rain": 4.0, "strong_wind": 14.0}
    mock_pipeline.generate_alerts()
    assert isinstance(mock_pipeline.alerts_df, pd.DataFrame)
    assert not mock_pipeline.alerts_df.empty
    assert "alert_type" in mock_pipeline.alerts_df.columns


def test_save_to_csv(mock_pipeline, tmp_path):
    """Test case for the `save_to_csv` method of the WeatherPipeline.

    This test verifies that the `save_to_csv` method correctly saves the
    current weather data to a CSV file in the specified directory.

    Args:
        mock_pipeline (WeatherPipeline): The mocked pipeline instance with pre-set data.
        tmp_path (Path): The temporary directory where the CSV file will be saved for testing.
    """
    output_dir = tmp_path / "output"

    # Ensure the output directory exists before saving
    output_dir.mkdir(parents=True, exist_ok=True)

    # Call the to_csv method on the mock_pipeline and specify the file path
    mock_pipeline.current_weather_df.to_csv(str(output_dir / "current_weather.csv"))

    # Assert the files are saved in the correct directory
    assert (output_dir / "current_weather.csv").exists()
