# src/visualizer.py
import os
import pandas as pd
import matplotlib.pyplot as plt

import algorithms
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from tkinter import Listbox, MULTIPLE
import numpy as np

class AnomalyVisualizer:
    """
    Visualizes precipitation data and detected anomalies using Tkinter menus.
    """
    def __init__(self, master, data_dir='../data'):
        self.master = master
        self.data_dir = data_dir
        self.city_files = self._find_csv_files()

    def _find_csv_files(self):
        files = {}
        for file in os.listdir(self.data_dir):
            if file.endswith('.csv'):
                city_name = file.split('.')[0].replace('_', ' ').title()
                files[city_name] = os.path.join(self.data_dir, file)
        return files

    def select_cities_dialog(self):
        top = tk.Toplevel(self.master)
        top.title("Select Cities for Anomaly Analysis")

        available_cities = list(self.city_files.keys())
        selected_cities_var = tk.StringVar(value=available_cities)

        listbox = Listbox(top, listvariable=selected_cities_var, selectmode=MULTIPLE, width=30, height=10)
        listbox.pack(padx=10, pady=10)

        selected_files = []

        def get_selected():
            selected_indices = listbox.curselection()
            selected_cities = [available_cities[i] for i in selected_indices]
            selected_files.extend([self.city_files[city] for city in selected_cities])
            top.destroy()

        select_button = ttk.Button(top, text="Select", command=get_selected)
        select_button.pack(pady=5)

        cancel_button = ttk.Button(top, text="Cancel", command=top.destroy)
        cancel_button.pack(pady=5)

        self.master.wait_window(top)
        return selected_files

    def get_window_size_dialog(self):
        window_size = simpledialog.askinteger(
            "Input", "Enter window size for the anomaly detector:",
            parent=self.master, initialvalue=30, minvalue=1
        )
        return window_size

    def plot_anomalies(self, file_paths, window_size=30, threshold=3.0):
        if not file_paths:
            messagebox.showinfo("Info", "No cities selected for anomaly analysis.")
            return

        detector = algorithms.AnomalyDetector(window_size=window_size, threshold=threshold)

        num_plots = len(file_paths)
        cols = 2
        rows = (num_plots + 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows), sharex=True,
                                 gridspec_kw={'hspace': 0.5, 'wspace': 0.3})

        axes = np.ravel(axes)

        for i, file_path in enumerate(file_paths):
            df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
            precipitation = df['precipitation_sum']
            anomalies = detector.detect(precipitation)

            ax = axes[i]
            ax.plot(df.index, precipitation, label='Precipitation', linewidth=0.5)
            anomaly_dates = df.index[anomalies]
            anomaly_values = precipitation[anomalies]
            ax.scatter(anomaly_dates, anomaly_values, color='red', label='Anomalies', s=20)

            location_name = os.path.basename(file_path).split('.')[0].replace('_', ' ').title()
            ax.set_title(location_name)
            ax.set_ylabel('Precipitation (inch)')
            ax.tick_params(axis='x', rotation=45)

        for j in range(i + 1, len(axes)):
            if isinstance(axes[j], plt.Axes):
                fig.delaxes(axes[j])

        fig.text(0.5, 0.02, 'Date', ha='center', va='center')
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.savefig('anomalies_subplots.png')
        plt.close()

    def plot_scatter_overlay(self, file_paths):
        if not file_paths:
            messagebox.showinfo("Info", "No cities selected for scatter plot.")
            return

        fig, ax = plt.subplots(figsize=(15, 8))
        for file_path in file_paths:
            df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
            location_name = os.path.basename(file_path).split('.')[0].replace('_', ' ').title()
            ax.scatter(df.index, df['precipitation_sum'], label=location_name, s=10, alpha=0.7)
        ax.set_xlabel('Date')
        ax.set_ylabel('Precipitation (inch)')
        ax.set_title('Overlay Scatter Plot of Precipitation for Selected Cities')
        ax.legend(loc='upper right', fontsize='small', ncol=2)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('scatter_overlay.png')
        plt.close()

class ClusteringVisualizer:
    def plot(self, df, kmeans, k):
        labels = kmeans.labels_ #get labels via kmeans
        cluster_assignments = {i: [] for i in range(k)} #dict to store cities
        for i, city in enumerate(df.columns):
            cluster_assignments[labels[i]].append(city)

        print("\nCluster Assignments:") #print cluster assignments
        for cluster, cities in cluster_assignments.items():
            print(f"Cluster {cluster+1}: {', '.join(cities)}")

        fig, axes = plt.subplots(1, 2, figsize=(18, 7)) #2 subplots

        for city in df.columns:
            axes[0].plot(df.index, df[city], label=city, marker='o', markersize=5)
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Average Precipitation (inches)')
        axes[0].set_title('Average Precipitation for US Cities (2001–2024)')
        axes[0].legend()
        axes[0].tick_params(axis='x', rotation=45)

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#ff7f0e', '#8c564b', '#2ca02c']
        for i, city in enumerate(df.columns):
            color = colors[labels[i] % len(colors)]
            axes[1].scatter(df.index, df[city], label=city, color=color, s=40, alpha=0.6)
            axes[1].text(df.index[-1], df[city].iloc[-1], city, fontsize=8, color=color) #text label for each city

        cluster_centers = kmeans.cluster_centers_ #plot cluster centers (X)
        for i in range(k):
            axes[1].scatter(df.index, cluster_centers[i], label=f'Cluster {i+1} Center', s=200, marker='X')

        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Average Precipitation (inches)')
        axes[1].set_title('K-Means Clustering of US City Precipitation')
        axes[1].legend()
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()
        return fig

class PredictionVisualizer:
    @staticmethod
    def plot_precipitation_forecast(forecast_df, city=None):
        if city:
            forecast_df = forecast_df[forecast_df['city'] == city]

        dates = forecast_df['date']
        values = forecast_df['predicted_precipitation_sum']

        plt.figure(figsize=(12, 6))
        plt.plot(dates, values, marker='o', label='Forecasted Precipitation')
        ymax = max(values.max(), 1.4)
        plt.axhspan(0.01, 0.1, facecolor='lightblue', alpha=0.3, label='Light rain')
        plt.axhspan(0.1, 0.5, facecolor='deepskyblue', alpha=0.3, label='Moderate rain')
        plt.axhspan(0.5, 1.0, facecolor='dodgerblue', alpha=0.3, label='Heavy rain')
        plt.axhspan(1.0, ymax, facecolor='navy', alpha=0.3, label='Very heavy rain')
        plt.title(f"📅 30-Day Precipitation Forecast{' for ' + city if city else ''}")
        plt.xlabel("Date")
        plt.ylabel("Precipitation (inches)")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend(loc='upper left')
        plt.ylim(0, ymax * 1.1)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_precipitation_forecast_all_cities(forecast_df):
        plt.figure(figsize=(14, 7))
        for city in forecast_df['city'].unique():
            city_df = forecast_df[forecast_df['city'] == city]
            plt.plot(city_df['date'], city_df['predicted_precipitation_sum'], marker='o', label=city)
        max_val = forecast_df['predicted_precipitation_sum'].max()
        ymax = max(max_val, 1.5)
        plt.axhspan(0.01, 0.1, facecolor='lightblue', alpha=0.2, label='Light rain')
        plt.axhspan(0.1, 0.5, facecolor='deepskyblue', alpha=0.2, label='Moderate rain')
        plt.axhspan(0.5, 1.0, facecolor='dodgerblue', alpha=0.2, label='Heavy rain')
        plt.axhspan(1.0, ymax, facecolor='navy', alpha=0.2, label='Very heavy rain')
        plt.title("📊 30-Day Precipitation Forecast for Cities")
        plt.xlabel("Date")
        plt.ylabel("Precipitation (inches)")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.ylim(0, ymax * 1.1)
        plt.legend(loc='upper left', fontsize='small', ncol=2)
        plt.tight_layout()
        plt.savefig('prediction_plot.png')
        plt.show()
