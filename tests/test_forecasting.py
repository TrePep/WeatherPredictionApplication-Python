#run with 
# py -m pytest tests/test_forecasting.py
#or 
#python -m pytest tests/test_forecasting.py 
#from this folder 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from algorithms import ForecastingAlgorithm, AnomalyDetector
import pandas as pd
import numpy as np

def generate_mock_df():
    dates = pd.date_range(start="2024-01-01", periods=60)
    data = []
    for city in ["New York", "Los Angeles"]:
        for date in dates:
            data.append({
                "city": city,
                "date": date,
                "precipitation_sum": np.random.rand()
            })
    return pd.DataFrame(data)

def test_predict_with_prophet():
    df = generate_mock_df()
    result = ForecastingAlgorithm.predict_with_prophet(df, "precipitation_sum", "New York", forecast_days=10)
    assert isinstance(result, pd.DataFrame)
    assert "predicted_precipitation_sum" in result.columns
    assert len(result) == 10
    assert pd.api.types.is_datetime64_any_dtype(result["date"])

def test_predict_all_cities():
    df = generate_mock_df()
    result = ForecastingAlgorithm.predict_all_cities(df, "precipitation_sum", forecast_days=7)
    assert isinstance(result, pd.DataFrame)
    assert "city" in result.columns
    assert "predicted_precipitation_sum" in result.columns
    assert result["city"].nunique() == 2
    assert len(result) == 14  # 2 cities × 7 days

def test_anomaly_detector_detects_known_outlier():
    values = np.concatenate([np.ones(30), [10]])  # 30 normal, 1 spike
    detector = AnomalyDetector(window_size=10, threshold=2.0)
    result = detector.detect(values)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(values)
    assert result[-1] == True
