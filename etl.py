"""ETL for weather analysis.

This script defines a WeatherPipeline class that processes weather data
from a JSON file, transforms it into structured data, performs analyses,
and generates visualizations and reports.
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

from schema.data_class import CurrentWeather, Forecast, Location, WeatherData
from util.logger import Logger


class WeatherPipeline:
    """Weather data processing pipeline.

    Extracts, transforms, analyzes, and visualizes weather data.
    """

    def __init__(self, json_file: str):
        """Initialize the WeatherPipeline with the input JSON file.

        :param json_file: Path to the JSON file containing weather data.
        :type json_file: str
        """
        self.json_file = json_file
        self.data = None
        self.weather_data: Dict[str, WeatherData] = {}
        self.current_weather_df = None
        self.forecast_df = None
        self.alerts_df = None
        self.temp_analysis_df = None
        # Load environment variables
        load_dotenv()

        # Retrieve threshold values from environment variables
        self.THRESHOLDS = {
            "high_temp": float(os.getenv("HIGH_TEMP", "0.0")),
            "heavy_rain": float(os.getenv("HEAVY_RAIN", "0.0")),
            "strong_wind": float(os.getenv("STRONG_WIND", "0.0")),
        }
        self.logger = Logger.setup_logging()

    def extract_data(self):
        """Extract and parse weather data from the JSON file.

        Reads weather data, converts it into structured data classes,
        and stores it in the pipeline's internal attributes.
        """
        self.logger.info("Starting data extraction")
        try:
            with open(self.json_file, "r") as file:
                self.data = json.load(file)

            # Parse the JSON data into data classes
            for city, weather_info in self.data.items():
                location = Location(
                    name=weather_info["location"]["name"],
                    region=weather_info["location"]["region"],
                    country=weather_info["location"]["country"],
                    latitude=weather_info["location"]["lat"],
                    longitude=weather_info["location"]["lon"],
                    timezone=weather_info["location"]["tz_id"],
                    last_updated=weather_info["location"]["localtime"],
                )

                current_weather = CurrentWeather(
                    temperature_c=weather_info["current"]["temp_c"],
                    condition_text=weather_info["current"]["condition"]["text"],
                    last_updated=weather_info["current"]["last_updated"],
                )

                forecasts = []
                for day in weather_info["forecast"]["forecastday"]:
                    forecast = Forecast(
                        date=day["date"],
                        max_temp_c=day["day"]["maxtemp_c"],
                        min_temp_c=day["day"]["mintemp_c"],
                        avg_temp_c=day["day"]["avgtemp_c"],
                        total_precip_mm=day["day"]["totalprecip_mm"],
                        max_wind_kph=day["day"]["maxwind_kph"],
                        condition_text=day["day"]["condition"]["text"],
                    )
                    forecasts.append(forecast)

                self.weather_data[city] = WeatherData(
                    location=location,
                    current_weather=current_weather,
                    forecasts=forecasts,
                )

            self.logger.info("Data extraction and parsing completed successfully")
        except Exception as e:
            self.logger.error(f"Error during data extraction: {e}")
            raise

    def transform_data(self):
        """Transform raw weather data into structured DataFrames.

        Converts extracted weather data into pandas DataFrames for further
        analysis and visualization. Handles missing data where necessary.
        """
        self.logger.info("Starting data transformation")
        try:
            current_data = []
            forecast_data = []

            for city, data in self.weather_data.items():
                # Current Weather
                current_data.append(
                    {
                        "city": data.location.name,
                        "temperature_c": data.current_weather.temperature_c,
                        "condition": data.current_weather.condition_text,
                        "last_updated": data.current_weather.last_updated,
                    },
                )

                # Forecast Data
                for forecast in data.forecasts:
                    forecast_data.append(
                        {
                            "city": data.location.name,
                            "date": forecast.date,
                            "max_temp_c": forecast.max_temp_c,
                            "min_temp_c": forecast.min_temp_c,
                            "avg_temp_c": forecast.avg_temp_c,
                            "precipitation_mm": forecast.total_precip_mm,
                            "wind_kph": forecast.max_wind_kph,
                            "condition": forecast.condition_text,
                        },
                    )

            self.current_weather_df = pd.DataFrame(current_data)
            self.forecast_df = pd.DataFrame(forecast_data)

            # Log detailed DataFrame information
            self.logger.info("Current Weather DataFrame Summary:")
            self.logger.info(f"{self.current_weather_df.head()}")
            self.logger.info(f"Shape: {self.current_weather_df.shape}")
            self.logger.info(f"Columns: {', '.join(self.current_weather_df.columns)}")
            self.logger.info(f"Number of cities: {self.current_weather_df['city'].nunique()}")

            self.logger.info("\nForecast DataFrame Summary:")
            self.logger.info(f"{self.forecast_df.head()}")
            self.logger.info(f"Shape: {self.forecast_df.shape}")
            self.logger.info(f"Columns: {', '.join(self.forecast_df.columns)}")
            self.logger.info(f"Date range: {self.forecast_df['date'].min()} to {self.forecast_df['date'].max()}")

            # Step to handle NaN values and log them
            self.handle_missing_data()
            self.logger.info(f"{self.forecast_df.head()}")

            self.logger.info("Data transformation completed successfully")
        except Exception as e:
            self.logger.error(f"Error during data transformation: {e}")
            raise

    def handle_missing_data(self):
        """Handle missing data in the weather DataFrames.

        Logs missing data information and applies forward-filling
        to fill in missing values.
        """
        # Check for missing data in the current weather DataFrame
        current_weather_missing = self.current_weather_df.isna().sum()
        if current_weather_missing.any():
            self.logger.warning("Missing values found in Current Weather DataFrame:")
            self.logger.warning(f"{current_weather_missing[current_weather_missing > 0]}")

        # Check for missing data in the forecast DataFrame
        forecast_missing = self.forecast_df.isna().sum()
        if forecast_missing.any():
            self.logger.warning("Missing values found in Forecast DataFrame:")
            self.logger.warning(f"{forecast_missing[forecast_missing > 0]}")

        # Optionally, drop rows with missing values or fill them
        # For example, forward fill missing data
        self.current_weather_df.fillna(method="ffill", inplace=True)
        self.forecast_df.fillna(method="ffill", inplace=True)

    def generate_alerts(self):
        """Generate weather alerts based on predefined thresholds.

        Alerts are created for high temperature, heavy rain, and strong
        wind conditions and stored in a DataFrame.
        """
        self.logger.info("Starting alert generation")
        try:
            alerts = []

            for _, row in self.forecast_df.iterrows():
                if row["max_temp_c"] > self.THRESHOLDS["high_temp"]:
                    alerts.append(
                        {
                            "city": row["city"],
                            "date": row["date"],
                            "alert_type": "High Temperature",
                            "value": f"{row['max_temp_c']}°C",
                            "threshold": f"> {self.THRESHOLDS['high_temp']}°C",
                        },
                    )

                if row["precipitation_mm"] > self.THRESHOLDS["heavy_rain"]:
                    alerts.append(
                        {
                            "city": row["city"],
                            "date": row["date"],
                            "alert_type": "Heavy Rain",
                            "value": f"{row['precipitation_mm']}mm",
                            "threshold": f"> {self.THRESHOLDS['heavy_rain']}mm",
                        },
                    )

                if row["wind_kph"] > self.THRESHOLDS["strong_wind"]:
                    alerts.append(
                        {
                            "city": row["city"],
                            "date": row["date"],
                            "alert_type": "Strong Wind",
                            "value": f"{row['wind_kph']}kph",
                            "threshold": f"> {self.THRESHOLDS['strong_wind']}kph",
                        },
                    )

            self.alerts_df = pd.DataFrame(alerts)

            self.logger.info("Alert generation completed successfully")
            self.logger.info(f"{self.alerts_df.head()}")
        except Exception as e:
            self.logger.error(f"Error during alert generation: {e}")
            raise

    def save_to_csv(self, output_dir="output/"):
        """Save the transformed and analyzed data to CSV files.

        :param output_dir: Directory path where CSV files will be saved.
        :type output_dir: str
        """
        self.logger.info("Starting CSV export")
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            self.current_weather_df.to_csv(f"{output_dir}current_weather.csv", index=False)
            self.forecast_df.to_csv(f"{output_dir}forecast.csv", index=False)
            self.temp_analysis_df.to_csv(f"{output_dir}temp_analysis.csv", index=False)
            if self.alerts_df is not None:
                self.alerts_df.to_csv(f"{output_dir}alerts.csv", index=False)
            self.logger.info("CSV export completed successfully")
        except Exception as e:
            self.logger.error(f"Error during CSV export: {e}")
            raise

    def create_visualizations(self, output_dir="output/"):
        """Create and save visualizations of the weather data.

        :param output_dir: Directory path where CSV files will be saved.
        :type output_dir: str
        """
        self.logger.info("Starting visualization generation")
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            self.logger.info("Visualization generation for  current_temperature")
            plt.figure(figsize=(10, 6))
            sns.barplot(data=self.current_weather_df, x="city", y="temperature_c")
            plt.title("Current Temperature by City")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_dir}current_temperature.png")
            plt.close()

            self.logger.info("Visualization generation for  forecast_trends")
            plt.figure(figsize=(12, 6))
            plt.style.use("classic")

            # Ensure dates are in datetime format
            if not pd.api.types.is_datetime64_any_dtype(self.forecast_df["date"]):
                self.forecast_df["date"] = pd.to_datetime(self.forecast_df["date"], format="%Y-%m-%d")

            # Plot with corrected date handling
            for city in self.forecast_df["city"].unique():
                city_data = self.forecast_df[self.forecast_df["city"] == city]
                plt.plot(city_data["date"], city_data["max_temp_c"], label=city, marker="o", linewidth=2, markersize=6)

            # Improve date formatting on x-axis
            ax = plt.gca()
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_major_locator(mdates.DayLocator())

            plt.title("Forecasted Maximum Temperatures", fontsize=14, pad=20)
            plt.xlabel("Date", fontsize=12)
            plt.ylabel("Temperature (°C)", fontsize=12)

            # Format legend and put it outside the plot
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            # Add grid
            plt.grid(True, linestyle="--", alpha=0.7)

            plt.xticks(rotation=30)
            plt.tight_layout()

            plt.savefig(f"{output_dir}forecast_trends.png", dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info("Visualization generation for  temperature_comparison")
            plt.figure(figsize=(10, 8))
            plt.style.use("classic")

            # Create scatter plot with improved styling
            plt.scatter(
                self.temp_analysis_df["current_temp"],
                self.temp_analysis_df["highest_temp"],
                alpha=0.7,
                s=100,
            )

            # Add city labels with better positioning
            for i, row in self.temp_analysis_df.iterrows():
                plt.annotate(
                    row["city"],
                    (row["current_temp"], row["highest_temp"]),
                    xytext=(7, 7),
                    textcoords="offset points",
                    fontsize=11,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
                )

            # Add reference line with better styling
            plt.gca().set_aspect("equal")
            max_temp = max(self.temp_analysis_df["current_temp"].max(), self.temp_analysis_df["highest_temp"].max())
            plt.plot(
                [0, max_temp],
                [0, max_temp],
                "--",
                color="gray",
                label="Current = Highest",
                alpha=0.8,
                linewidth=1.5,
            )

            # Improve axes and labels
            plt.title("Current vs Highest Forecasted Temperature", fontsize=14, pad=20)
            plt.xlabel("Current Temperature (°C)", fontsize=12)
            plt.ylabel("Highest Forecasted Temperature (°C)", fontsize=12)

            # Add grid
            plt.grid(True, linestyle="--", alpha=0.3)

            # Format legend
            plt.legend(loc="lower right")

            # Set limits slightly above max temperature
            plt.xlim(20, max_temp + 2)
            plt.ylim(20, max_temp + 2)

            # Adjust layout
            plt.tight_layout()

            plt.savefig(f"{output_dir}temperature_comparison.png", dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info("Visualization generation completed successfully")
        except Exception as e:
            self.logger.error(f"Error during visualization generation: {e}")
            raise

    def analyze_temperatures(self):
        """Analyze current and forecasted temperatures.

        Compares current temperatures with forecasted temperatures,
        calculates statistics, and identifies significant patterns.
        """
        self.logger.info("Starting temperature analysis")
        try:
            # Compare current vs forecast temperatures
            temp_analysis = []
            for city in self.current_weather_df["city"].unique():
                current_temp = self.current_weather_df[self.current_weather_df["city"] == city]["temperature_c"].values[
                    0
                ]
                city_forecast = self.forecast_df[self.forecast_df["city"] == city]

                # Find highest temperature day
                max_temp_row = city_forecast.loc[city_forecast["max_temp_c"].idxmax()]

                # Calculate statistics
                stats = city_forecast.agg(
                    {
                        "max_temp_c": ["min", "max", "mean"],
                        "min_temp_c": ["min", "max", "mean"],
                        "avg_temp_c": ["min", "max", "mean"],
                    },
                )

                temp_analysis.append(
                    {
                        "city": city,
                        "current_temp": current_temp,
                        "highest_temp_date": max_temp_row["date"],
                        "highest_temp": max_temp_row["max_temp_c"],
                        "temp_diff_from_current": max_temp_row["max_temp_c"] - current_temp,
                        "avg_max_temp": stats["max_temp_c"]["mean"],
                        "avg_min_temp": stats["min_temp_c"]["mean"],
                        "overall_avg_temp": stats["avg_temp_c"]["mean"],
                    },
                )

            self.temp_analysis_df = pd.DataFrame(temp_analysis)
            self.logger.info("\nTemperature Analysis Summary:")
            self.logger.info(f"{self.temp_analysis_df}")

        except Exception as e:
            self.logger.error(f"Error during temperature analysis: {e}")
            raise

    def generate_summary_report(self):
        """Generate a summary report of the weather analysis.

        Creates a text report summarizing the temperature analysis
        and any generated alerts.
        """
        self.logger.info("Starting summary report generation")
        try:
            report = "Weather Analysis Summary\n"
            report += "=" * 40 + "\n\n"

            # Add temperature analysis
            report += "Temperature Analysis:\n"
            for _, row in self.temp_analysis_df.iterrows():
                report += f"\nCity: {row['city']}\n"
                report += f"Current Temperature: {row['current_temp']}°C\n"
                report += f"Highest Temperature: {row['highest_temp']}°C on {row['highest_temp_date']}\n"
                report += f"Average Temperature Range: {row['avg_min_temp']:.1f}°C to {row['avg_max_temp']:.1f}°C\n"

            # Weather Alerts
            if not self.alerts_df.empty:
                report += "\nWeather Alerts:\n"
                for _, alert in self.alerts_df.iterrows():
                    report += (
                        f"City: {alert['city']}, Date: {alert['date']}, "
                        f"Alert: {alert['alert_type']}, Value: {alert['value']}\n"
                    )

            with open("output/summary_report.txt", "w") as file:
                file.write(report)
            self.logger.info("Summary report generation completed successfully")
        except Exception as e:
            self.logger.error(f"Error during summary report generation: {e}")
            raise


def main():
    """Entry point for executing the weather pipeline.

    This function sets up the pipeline, processes the weather data,generates
    visualizations, and creates reports.
    """
    logger = logging.getLogger("weather_pipeline")
    try:
        logger.info("Pipeline execution started")

        # Instantiate and run your pipeline logic
        pipeline = WeatherPipeline("data/ETL_developer_Case_2.json")
        pipeline.extract_data()
        pipeline.transform_data()
        pipeline.analyze_temperatures()
        pipeline.generate_alerts()
        pipeline.save_to_csv()
        pipeline.create_visualizations()
        pipeline.generate_summary_report()

        logger.info("Pipeline execution completed successfully")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")


if __name__ == "__main__":
    main()
