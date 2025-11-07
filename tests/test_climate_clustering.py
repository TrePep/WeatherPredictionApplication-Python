import sys
import os
import pytest
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from climate_clustering import analize  

cities = {
    'Tallahassee': 'data/tallahassee_daily.csv',
    'New York': 'data/new york_daily.csv',
    'Los Angeles': 'data/los angeles_daily.csv',
    'Houston': 'data/houston_daily.csv',
    'Chicago': 'data/chicago_daily.csv',
    'Miami': 'data/miami_daily.csv',
    'Boston': 'data/boston_daily.csv',
    'Phoenix': 'data/phoenix_daily.csv',
    'San Francisco': 'data/san francisco_daily.csv',
    'Seattle': 'data/seattle_daily.csv'
}

#Valid test where K = # of cities
def test_analize_with_valid_cities_and_k():
    selected_cities = ['Tallahassee', 'Miami', 'Chicago'] 
    clusters = 3  
    figure = analize(selected_cities, clusters)
    assert isinstance(figure, plt.Figure), "The result should be a matplotlib figure."

#Invalid K values
def test_invalid_k_value():
    selected_cities = ['Tallahassee', 'Miami', 'Chicago']
    clusters = -1  
    with pytest.raises(ValueError):
        analize(selected_cities, clusters)

#Error when no cities are selected
def test_no_cities_selected():
    selected_cities = []  
    clusters = 3  
    with pytest.raises(ValueError):
        analize(selected_cities, clusters)

#Non existant CSV file
def test_missing_csv_file():
    selected_cities = ['Tallahassee', 'NonExistentCity']  
    clusters = 1  
    with pytest.raises(FileNotFoundError):
        analize(selected_cities, clusters)

#Empty CSV file
def test_empty_csv_file():
    selected_cities = ['Tallahassee']  
    clusters = 3  
    frame = pd.DataFrame()  
    with pytest.MonkeyPatch.context() as m:
        m.setattr(pd, 'read_csv', lambda *args, **kwargs: frame)
        with pytest.raises(ValueError):
            analize(selected_cities, clusters)

#Test if it works for all cities
def test_large_number_of_cities():
    selected_cities = list(cities.keys())  
    clusters = 3  
    figure = analize(selected_cities, clusters)
    assert isinstance(figure, plt.Figure), "The result should be a matplotlib figure."

#Test if it can cluster a few cities and more clustering than allowed
def test_few_cities_for_clustering():
    selected_cities = ['Tallahassee', 'Miami']  
    clusters = 3  
    with pytest.raises(ValueError, match="Number of clusters must be less than or equal to the number of selected cities."):
        analize(selected_cities, clusters)


#Test if city name is not a sring
def test_non_string_city_names():
    selected_cities = [123, 456, 789]
    clusters = 3
    with pytest.raises(TypeError):
        analize(selected_cities, clusters)

#Test if invalid city name
def test_invalid_city_names():
    selected_cities = ['InvalidCity1', 'InvalidCity2']
    clusters = 3
    with pytest.raises(ValueError):
        analize(selected_cities, clusters)
