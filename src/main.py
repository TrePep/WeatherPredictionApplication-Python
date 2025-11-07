# main.py
import os
import sys
import tkinter as tk
from tkinter import messagebox, simpledialog
from tkinter import ttk, messagebox, filedialog
import subprocess
from visualizer import AnomalyVisualizer, ClusteringVisualizer, PredictionVisualizer
from algorithms import ForecastingAlgorithm
from algorithms import ClusteringAlgorithm 
import pandas as pd

def display_help():
    help_text = (
        "Available commands:\n"
        "- Process Data: Optionally create/update the data in the data folder\n"
        "- Clustering: Run KMeans clustering analysis\n"
        "- Predict Trends: Display precipitation trend predictions\n"
        "- Time Series: Analyze and visualize precipitation time series and anomalies"
    )
    messagebox.showinfo("Help", help_text)

def run_data_processing():
    # Check if data files already exist
    data_dir = 'data'
    if os.path.exists(data_dir):
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('_daily.csv')]
        if len(csv_files) >= 10:  # We expect 10 city files
            response = messagebox.askyesno("Data Exists", 
                f"Found {len(csv_files)} data files already exist.\n"
                "Do you want to refresh the data? (This will take several minutes due to API rate limits)")
            if not response:
                messagebox.showinfo("Process Data", "Using existing data files.")
                return
    
    # Show warning about time required
    response = messagebox.askyesno("Process Data", 
        "Data processing will fetch weather data for 10 cities from 2000-2025.\n"
        "This will take approximately 10-15 minutes due to API rate limits.\n"
        "Do you want to continue?")
    
    if not response:
        return
        
    try:
        # Import and run data processor directly to avoid subprocess issues
        from data_processor import WeatherDataProcessor, cities
        
        messagebox.showinfo("Processing", "Data processing started. This will take several minutes...")
        
        # Use the correct data directory path
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        processor = WeatherDataProcessor()
        # Use shorter delay for testing (10 seconds instead of 60)
        processor.process_and_save(cities, output_dir, delay=10)
        
        messagebox.showinfo("Process Data", "Data processing complete. Data files are updated.")
    except Exception as e:
        messagebox.showerror("Error", f"Error processing data:\n{e}")

def run_climate_clustering():
    algorithm = ClusteringAlgorithm() #initialize clustering algorithm
    cities = list(algorithm.cityFiles.keys()) #get list of cities

    root = tk.Tk() #makes window
    root.title("City Precipitation Clustering")

    frame = ttk.Frame(root, padding=20) #Jframe
    frame.grid(row=0, column=0)
    #Label for the listbox
    ttk.Label(frame, text="Select Cities:").grid(column=0, row=0, sticky='w')
    #listbox for multiple cities
    city_listbox = tk.Listbox(frame, selectmode=tk.MULTIPLE, height=10, exportselection=False)
    for city in cities:
        city_listbox.insert(tk.END, city)
    city_listbox.grid(column=0, row=1, rowspan=10, sticky='w')

    ttk.Label(frame, text="Number of Clusters (k):").grid(column=1, row=0, sticky='w') #label for number of clusters
    k_entry = ttk.Entry(frame, width=5)
    k_entry.grid(column=1, row=1, sticky='w')

    def get_selected_cities(): #Get list of selected cities
        selected_indices = city_listbox.curselection()
        return [cities[i] for i in selected_indices]

    def run():  # run clustering algorithm
        selected = get_selected_cities()
        try:
            k = int(k_entry.get())
            if k < 1 or len(selected) < k:
                if len(selected) == 0:
                    raise ValueError("No cities selected.")  # Raise before messagebox
                elif k < 1 or len(selected) < k:
                    raise ValueError("Invalid number of clusters.")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid integer for the number of clusters.")
            raise ValueError("Invalid number of clusters.")  # Raise after showing error
        
        if not selected:
            messagebox.showerror("No Cities Selected", "Please select at least one city.")
            raise ValueError("No cities selected.")  # Raise before messagebox

        frame = algorithm.compute_yearly_averages(selected)
        kmeans = algorithm.run_kmeans(frame, k)  # run kmeans clustering
        visualizer = ClusteringVisualizer()
        return visualizer.plot(frame, kmeans, k)  # returns the figure

    def save(): #Save the figure (PNG)
        fig = run()
        if fig: #save the figure
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if file_path:
                fig.savefig(file_path)
                messagebox.showinfo("Success", f"Charts saved to {file_path}")

    ttk.Button(frame, text="Run Clustering", command=run).grid(column=1, row=2, sticky='w', pady=10)
    ttk.Button(frame, text="Download Charts", command=save).grid(column=1, row=3, sticky='w', pady=10)

    root.mainloop()

def predict_trends():
    data_directory = 'data'
    predictor = ForecastingAlgorithm()
    all_city_data = []

    for filename in os.listdir(data_directory):
        if filename.endswith('_daily.csv'):
            city_name = filename.replace('_daily.csv', '')
            filepath = os.path.join(data_directory, filename)
            df = pd.read_csv(filepath)
            df['city'] = city_name  
            all_city_data.append(df)

    if all_city_data:
        combined_df = pd.concat(all_city_data, ignore_index=True)
        try:
            forecast_results = predictor.predict_all_cities(combined_df, target_column='precipitation_sum', forecast_days=30)
            PredictionVisualizer.plot_precipitation_forecast_all_cities(forecast_results)
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")
    else:
        messagebox.showerror("Error", "No data files found. Please run 'Process Data' first.")

def run_time_series_analysis():
    data_directory = 'data'
    anomaly_visualizer = AnomalyVisualizer(root, data_directory)
    selected_files = anomaly_visualizer.select_cities_dialog()
    if selected_files:
        window_size = anomaly_visualizer.get_window_size_dialog()
        if window_size is not None:
            try:
                anomaly_visualizer.plot_anomalies(selected_files, window_size=window_size)
                anomaly_visualizer.plot_scatter_overlay(selected_files)
            except Exception as e:
                messagebox.showerror("Error", f"Time series analysis failed: {e}")
    else:
        messagebox.showinfo("Info", "No cities selected for time series analysis.")

def exit_app():
    root.destroy()

# Create the main window
root = tk.Tk()
root.title("Climate Analyzer")
root.geometry("300x400")

# Title label
tk.Label(root, text="Climate Analyzer", font=("Helvetica", 14)).pack(pady=10)

# Add UI Buttons
tk.Button(root, text="Process Data", width=20, command=run_data_processing).pack(pady=5)
tk.Button(root, text="Clustering", width=20, command=run_climate_clustering).pack(pady=5)
tk.Button(root, text="Predict Trends", width=20, command=predict_trends).pack(pady=5)
tk.Button(root, text="Time Series", width=20, command=run_time_series_analysis).pack(pady=5)
tk.Button(root, text="Help", width=20, command=display_help).pack(pady=5)
tk.Button(root, text="Exit", width=20, command=exit_app).pack(pady=10)

root.mainloop()
