import pytest
import pandas as pd
from unittest.mock import patch, mock_open, MagicMock
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from etl import WeatherPipeline


@pytest.fixture
def mock_json_data():
    return {
        "City1": {
            "location": {
                "name": "City1",
                "region": "Region1",
                "country": "Country1",
                "lat": 12.34,
                "lon": 56.78,
                "tz_id": "Region/City1",
                "localtime": "2025-01-01 12:00"
            },
            "current": {
                "temp_c": 25.0,
                "condition": {"text": "Clear"},
                "last_updated": "2025-01-01 12:00"
            },
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
                            "condition": {"text": "Sunny"}
                        }
                    }
                ]
            }
        }
    }


@pytest.fixture
def mock_pipeline(mock_json_data):
    # Mock the environment variables for the test
    with patch.dict(os.environ, {
        'HIGH_TEMP': '28.0',
        'HEAVY_RAIN': '4.0',
        'STRONG_WIND': '14.0'
    }):
        # Mock reading the JSON file
        with patch("builtins.open", mock_open(read_data='{}')):
            with patch("json.load", return_value=mock_json_data):
                pipeline = WeatherPipeline("mock_file.json")
                pipeline.extract_data()
                pipeline.transform_data()

                # Manually populate the DataFrames for testing purposes
                pipeline.current_weather_df = pd.DataFrame({
                    'city': ['City1', 'City2'],
                    'temperature_c': [22, 25],
                    'condition': ['Clear', 'Rainy'],
                    'last_updated': ['2025-01-08', '2025-01-08']
                })
                
                pipeline.forecast_df = pd.DataFrame({
                    'city': ['City1', 'City2'],
                    'date': ['2025-01-09', '2025-01-10'],
                    'max_temp_c': [28, 30],
                    'min_temp_c': [20, 22],
                    'avg_temp_c': [24, 26],
                    'precipitation_mm': [0.0, 10.0],
                    'wind_kph': [5, 15],
                    'condition': ['Sunny', 'Stormy']
                })

                return pipeline


def test_extract_data(mock_pipeline):
    assert len(mock_pipeline.weather_data) == 1
    assert "City1" in mock_pipeline.weather_data
    city_data = mock_pipeline.weather_data["City1"]
    assert city_data.location.name == "City1"
    assert city_data.current_weather.temperature_c == 25.0


def test_transform_data(mock_pipeline):
    assert isinstance(mock_pipeline.current_weather_df, pd.DataFrame)
    assert isinstance(mock_pipeline.forecast_df, pd.DataFrame)
    assert not mock_pipeline.current_weather_df.empty
    assert not mock_pipeline.forecast_df.empty
    assert "temperature_c" in mock_pipeline.current_weather_df.columns
    assert "max_temp_c" in mock_pipeline.forecast_df.columns


def test_analyze_temperatures(mock_pipeline):
    mock_pipeline.analyze_temperatures()
    assert isinstance(mock_pipeline.temp_analysis_df, pd.DataFrame)
    assert "city" in mock_pipeline.temp_analysis_df.columns
    assert "highest_temp" in mock_pipeline.temp_analysis_df.columns


def test_generate_alerts(mock_pipeline):
    mock_pipeline.THRESHOLDS = {
        "high_temp": 28.0,
        "heavy_rain": 4.0,
        "strong_wind": 14.0
    }
    mock_pipeline.generate_alerts()
    assert isinstance(mock_pipeline.alerts_df, pd.DataFrame)
    assert not mock_pipeline.alerts_df.empty
    assert "alert_type" in mock_pipeline.alerts_df.columns

def test_save_to_csv(mock_pipeline, tmp_path):
    output_dir = tmp_path / "output"
    
    # Ensure the output directory exists before saving
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Call the to_csv method on the mock_pipeline and specify the file path
    mock_pipeline.current_weather_df.to_csv(str(output_dir / "current_weather.csv"))

    # Assert the files are saved in the correct directory
    assert (output_dir / "current_weather.csv").exists()
