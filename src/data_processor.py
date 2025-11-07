import time
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import os
import numpy as np
#there is a week delay for weather data so ealiest date you can pull from i a week before current day
#if pulling data, pull 10 min before as there is a min delay before a new location can be requested, max 10 per hr

class WeatherDataProcessor:
    """A module to fetch, clean, and preprocess weather data from Open-Meteo API."""

    def __init__(self, cache_dir='.cache', retries=5, backoff_factor=0.2):
        """Initialize the API client with caching and retry settings."""
        cache_session = requests_cache.CachedSession(cache_dir, expire_after=-1)
        retry_session = retry(cache_session, retries=retries, backoff_factor=backoff_factor)
        self.client = openmeteo_requests.Client(session=retry_session)
        self.url = "https://archive-api.open-meteo.com/v1/archive"

    def fetch_city_data(self, latitude: float, longitude: float, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch daily precipitation data for a given location."""
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ["precipitation_sum"],
            "timezone": "America/New_York",
            "precipitation_unit": "inch"
        }
        
        try:
            responses = self.client.weather_api(self.url, params=params)
            response = responses[0]
            
            daily = response.Daily()
            precipitation_sum = daily.Variables(0).ValuesAsNumpy()
            
            data = {
                "date": pd.date_range(
                    start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                    end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=daily.Interval()),
                    inclusive="left"
                ),
                "precipitation_sum": precipitation_sum
            }
            return pd.DataFrame(data)
        
        except openmeteo_requests.Client.OpenMeteoRequestsError as e:
            print(f"Error fetching data: {e}")
            return None

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the precipitation data."""
        df = df.drop_duplicates(subset="date")
        df["precipitation_sum"] = df["precipitation_sum"].interpolate(method="linear").fillna(0)
        df["precipitation_sum"] = df["precipitation_sum"].clip(lower=0, upper=50)
        return df

    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize precipitation values to 0-1 range."""
        max_precip = df["precipitation_sum"].max()
        if max_precip > 0:
            df["precipitation_normalized"] = df["precipitation_sum"] / max_precip
        else:
            df["precipitation_normalized"] = df["precipitation_sum"]
        return df

    def load_data(self, city_name: str, data_dir: str = "../data") -> pd.DataFrame:
        """Load preprocessed data from a CSV file."""
        file_path = f"{data_dir}/{city_name.lower().replace(' ', '_')}_daily.csv"
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            print(f"Data file for {city_name} not found.")
            return None

    def process_and_save(self, cities: list, output_dir: str = "../data", delay: int = 60):  
        """Fetch, clean, normalize, and save data for all cities."""
        os.makedirs(output_dir, exist_ok=True)
        
        for i, city in enumerate(cities):
            print(f"Fetching data for {city['name']} ({i+1}/{len(cities)})...")
            df = self.fetch_city_data(city["latitude"], city["longitude"], "2000-11-22", "2025-04-02")
            
            if df is not None:
                df = self.clean_data(df)
                df = self.normalize_data(df)
                
                file_path = f"{output_dir}/{city['name'].lower().replace(' ', '_')}_daily.csv"
                df.to_csv(file_path, index=False)
                print(f"Saved data for {city['name']} to {file_path}")
            
            # Only sleep between cities, not after the last one
            if i < len(cities) - 1:
                print(f"Waiting {delay} seconds before next request (API rate limit)...")
                time.sleep(delay)

# List of 10 cities
cities = [
    {"name": "Tallahassee", "latitude": 30.4382, "longitude": -84.2806},
    {"name": "New York", "latitude": 40.7128, "longitude": -74.0060},
    {"name": "Los Angeles", "latitude": 34.0522, "longitude": -118.2437},
    {"name": "Chicago", "latitude": 41.8781, "longitude": -87.6298},
    {"name": "Houston", "latitude": 29.7604, "longitude": -95.3698},
    {"name": "Phoenix", "latitude": 33.4484, "longitude": -112.0740},
    {"name": "San Francisco", "latitude": 37.7749, "longitude": -122.4194},
    {"name": "Boston", "latitude": 42.3601, "longitude": -71.0589},
    {"name": "Seattle", "latitude": 47.6062, "longitude": -122.3321},
    {"name": "Miami", "latitude": 25.7617, "longitude": -80.1918},
]

if __name__ == "__main__":
    processor = WeatherDataProcessor()
    processor.process_and_save(cities)
    print("All city data fetched and saved.")
