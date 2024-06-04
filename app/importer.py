import pandas as pd
import os

def import_data():
    # Set variable for the folder path
    folder_main_path = r"C:\Users\ruudvanbrink\OneDrive - Jumbo Supermarkten B.V\Avans\Jaar 4\Artificial Intelligence\Rolling Forecast Bigbags"

    # Load datasets
    dataset = pd.read_csv('./data/dataset.csv', delimiter=",")
    weather_data = pd.read_csv('./data/weather_history.csv', delimiter=";")
    sales_data = pd.read_csv('./data/sales_dataset.csv', delimiter=",")

    # Convert columns to correct format
    dataset['Date'] = pd.to_datetime(dataset['Date'], format="%Y-%m-%d")
    weather_data['date'] = pd.to_datetime(weather_data['date'], format="%d-%m-%Y")
    sales_data['Date'] = pd.to_datetime(sales_data['Date'], format="%d-%m-%Y")

    weather_data['temp_max'] = weather_data['temp_max'].str.replace(',', '.', regex=False).str.extract(r'(-?\d+\.?\d*)').astype(float)

    # Set 'date' column as index
    dataset['date_copy'] = dataset['Date']  # Create a copy of the 'date' column
    dataset.set_index('date_copy', inplace=True)
        
    # Set frequency to daily
    dataset = dataset.asfreq('D')

    # Set 'date' column as index
    weather_data['date_copy'] = weather_data['date']  # Create a copy of the 'date' column
    weather_data.set_index('date_copy', inplace=True)
        
    # Set frequency to daily
    weather_data = weather_data.asfreq('D')

    return dataset, weather_data, sales_data
