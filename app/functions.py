import pandas as pd
import numpy as np
import os

def categorize_temperature(temp):
    return int(temp / 10)

def categorize_sales(sales):
    # Define bucket boundaries
    buckets = [0, 100000, 150000, 200000, 250000, 300000]
    bucket_labels = range(len(buckets) - 1)  # 5 buckets: 0, 1, 2, 3, 4
    
    for i in range(len(buckets) - 1):
        if buckets[i] <= sales < buckets[i + 1]:
            return i
    return len(buckets) - 2

def extract_date_features(data):
    data = data.copy()
    
    data['quarter'] = data['Date'].dt.quarter
    data['month'] = data['Date'].dt.month
    data['week'] = data['Date'].dt.isocalendar().week.astype(int)
    data['weekday'] = data['Date'].dt.weekday
    return data

def calculate_moving_average(dataset, column, date, days):
    filtered_data = dataset[(dataset['Date'] < date)]
    
    if len(filtered_data) < days:
            return 0
        
    moving_average = filtered_data[column].tail(days).mean()
    return moving_average

def calculate_weekday_moving_average(dataset, column, date, weekday, size):
    filtered_data = dataset[(dataset['Date'] < date) & (dataset['Date'].dt.weekday == weekday)]
    
    if len(filtered_data) < size:
        return 0
    
    weekday_moving_average = filtered_data[column].tail(size).mean()
    return weekday_moving_average 

def check_date_in_dataframe(date_str, df, col):
    date = pd.to_datetime(date_str)
    return date in pd.to_datetime(df[col]).values

def create_lagged_features(data, target_data, lag_start, lag_end, column):
    data = data.copy()
    data.set_index('Date', inplace=True, drop=False)
    target_data = target_data.copy()
    
    if column == 'temp_max':
        target_date_col = 'date'
    else:
        target_date_col = 'Date'
    
    target_data.set_index(target_date_col, inplace=True, drop=False)
    
    # Ensure 'Date' column is in datetime format
    data['Date'] = pd.to_datetime(data['Date'])
    target_data[target_date_col] = pd.to_datetime(target_data[target_date_col])
    
    for i in data.index:
        date_str = i.strftime('%d-%m-%Y')
        
        for lag_number in range(lag_start, lag_end + 1):
            lag_date = i - pd.DateOffset(days=lag_number)
            lag_date_str = lag_date.strftime('%d-%m-%Y')

            # Check which dataframe contains the date
            if lag_date in data.index:
                lagged_value = data.loc[lag_date, column]
            elif lag_date in target_data.index:
                lagged_value = target_data.loc[lag_date, column]
            else:
                lagged_value = np.nan

            # Categorize value
            if column == 'temp_max':
                lagged_value_cat = categorize_temperature(lagged_value)
            elif column == 'Sales':
                lagged_value_cat = categorize_sales(lagged_value)
            
            # Add column with categorized data if necessary
            lagged_col_name = f'{column}_lag_{lag_number}'
            if lagged_col_name not in data.columns:
                data[lagged_col_name] = np.nan

            # Add value to correct row
            data.at[i, lagged_col_name] = lagged_value_cat
    return data

def add_forecast_to_history(new_forecast): 
    # Path to the file
    folder_main_path = r"C:\Users\ruudvanbrink\OneDrive - Jumbo Supermarkten B.V\Avans\Jaar 4\Artificial Intelligence\Rolling Forecast Bigbags"
    forecast_history_path = os.path.join(folder_main_path, r"Data\forecast_history.csv")
    
    # Read the file, specifying 'Date' as the index column
    forecast_history = pd.read_csv(forecast_history_path, delimiter=';', index_col=0)
    
    # Ensure forecast_history DataFrame has predefined columns
    if forecast_history.empty:
        columns = ['Forecast_X-1', 'Forecast_X-2', 'Forecast_X-3', 'Forecast_X-4', 'Forecast_X-5', 'Forecast_X-6', 'Forecast_X-7', 'Actual', 'AbsoluteError_X-1', 'AbsoluteError_X-2', 'AbsoluteError_X-3', 'AbsoluteError_X-4', 'AbsoluteError_X-5', 'AbsoluteError_X-6', 'AbsoluteError_X-7']
        forecast_history = pd.DataFrame(columns=columns)
    
    # Update the forecast_history DataFrame with new forecast data
    for i, (index, row) in enumerate(new_forecast.iterrows()):
        if isinstance(index, int):
            date = row['Date']  # No need to convert if index is an integer
        else:
            date = index.strftime('%d-%m-%Y')  # Convert date to dd-mm-yyyy format
        forecast_value = row['Forecast']
        forecast_col = f'Forecast_X-{i+1}'

        # Check if the date exists in the forecast_history DataFrame
        if date not in forecast_history.index:
            # If date does not exist, add a new row
            forecast_history.loc[date] = [None] * len(forecast_history.columns)

        # Update the forecast value in the corresponding column
        forecast_history.loc[date, forecast_col] = forecast_value

    # Save the updated forecast_history DataFrame back to forecast_history.csv
    forecast_history.to_csv(forecast_history_path)

    print("Forecast history updated successfully.")

