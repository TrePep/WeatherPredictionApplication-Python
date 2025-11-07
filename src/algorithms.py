# algorithms.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from prophet import Prophet

# -------------------------------
# Clustering Algorithm Component
# -------------------------------
class ClusteringAlgorithm:
    def __init__(self, data_dir='data'): #dir with CSVs
        self.data_dir = data_dir
        self.cityFiles = self._load_city_files()

    def _load_city_files(self):
        city_files = {}
        for file in os.listdir(self.data_dir):
            if file.endswith('.csv'):
                city_name = file.replace('_daily.csv', '').replace('_', ' ').title()
                city_files[city_name] = os.path.join(self.data_dir, file)
        return city_files

    def load_city_data(self, city):
        if city not in self.cityFiles:
            raise FileNotFoundError(f"CSV file for {city} not found.")
        df = pd.read_csv(self.cityFiles[city], parse_dates=['date'])
        if df.empty:
            raise ValueError(f"CSV file for {city} is empty.")
        return df

    def compute_yearly_averages(self, selectedCities):
        yearlyData = {}
        for city in selectedCities:
            df = self.load_city_data(city)
            df = df[(df['date'].dt.year >= 2001) & (df['date'].dt.year <= 2024)]
            df['Year'] = df['date'].dt.year
            yearlyMean = df.groupby('Year')['precipitation_sum'].mean()
            yearlyData[city] = yearlyMean
        return pd.DataFrame(yearlyData)

    def run_kmeans(self, data, k):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(data.T)
        return kmeans

# ----------------------------------
# Forecasting Algorithm Component
# ----------------------------------
class ForecastingAlgorithm:
    @staticmethod
    def predict_with_prophet(df: pd.DataFrame, target_column: str, city: str, forecast_days: int = 30) -> pd.DataFrame:
        """
        Predict target_column for a specific city using Prophet.

        Parameters:
            df (pd.DataFrame): DataFrame containing the data.
            target_column (str): The column to predict (e.g., 'precipitation_sum').
            city (str): The city for which to forecast.
            forecast_days (int): The number of days to forecast.

        Returns:
            pd.DataFrame: Forecasted values with dates.
        """
        city_df = df[df['city'] == city].copy()
        city_df['date'] = pd.to_datetime(city_df['date']).dt.tz_localize(None)
        
        df_prophet = city_df[['date', target_column]].rename(columns={'date': 'ds', target_column: 'y'})
        
        model = Prophet(daily_seasonality=True)
        model.fit(df_prophet)
        
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        
        result = forecast[['ds', 'yhat']].tail(forecast_days).rename(
            columns={'ds': 'date', 'yhat': f'predicted_{target_column}'}
        )
        
        return result

    @staticmethod
    def predict_all_cities(df: pd.DataFrame, target_column: str, forecast_days: int = 30) -> pd.DataFrame:
        """
        Apply Prophet forecast to all unique cities present in the DataFrame.

        Returns:
            pd.DataFrame: Concatenated forecast results for all cities.
        """
        results = []
        for city in df['city'].unique():
            forecast = ForecastingAlgorithm.predict_with_prophet(df, target_column, city, forecast_days)
            forecast['city'] = city
            results.append(forecast)
        
        return pd.concat(results, ignore_index=True)

# -------------------------------
# Anomaly Detection Component
# -------------------------------
class AnomalyDetector:
    """
    Detect anomalies in time series data using a rolling window Z-score approach.
    """
    def __init__(self, window_size: int = 30, threshold: float = 3.0):
        self.window_size = window_size
        self.threshold = threshold

    def detect(self, time_series: pd.Series | np.ndarray) -> np.ndarray:
        """
        Detects anomalies using a rolling window.

        Returns:
            np.ndarray: Boolean array marking anomalies.
        """
        # Convert to pandas Series if necessary
        if not isinstance(time_series, pd.Series):
            time_series = pd.Series(time_series)

        anomalies = np.full(len(time_series), False)
        for i in range(self.window_size, len(time_series)):
            window = time_series.iloc[i - self.window_size:i]
            mean = window.mean()
            std = window.std()
            if std == 0 or pd.isna(std):
                std = 1e-10
            z_score = abs(time_series.iloc[i] - mean) / std
            anomalies[i] = z_score > self.threshold

        return anomalies
