import pandas as pd
from app.functions import *

def preprocess_data(data, weather_data, sales_data):
    data = data.copy()
    
    # If column temp_max not exists, find corresponding temperature from weather_data and add as column
    if 'temp_max' not in data.columns:
        # Merge temperature data from weather_data based on date
        data = pd.merge(data, weather_data[['date', 'temp_max']], left_on='Date', right_on='date', how='left')
        
        # Merge sales data from sales_data based on date
        data = pd.merge(data, sales_data[['Date', 'Sales']], left_on='Date', right_on='Date', how='left')
    
    # Add lagged, categorized temperature features
    data = create_lagged_features(data, weather_data, 1, 7, 'temp_max')
    
    # Add lagged, categorized sales features
    data = create_lagged_features(data, sales_data, 8, 14, 'Sales')
    
    # Extract date features
    data = extract_date_features(data)
    
    print(data)
          
    # Calculate moving averages Sales
    for days in [14, 28, 56, 128]:
        moving_averages = []
        for date in data['Date']:
            moving_average = calculate_moving_average(sales_data, 'Sales', date, days)
            moving_averages.append(moving_average)
       
        # Assign the list of moving averages to a new column in the DataFrame
        data[f'Sales_MA_{days}'] = moving_averages
                
    # Drop 'date' column
    data.drop(columns=['date', 'date_copy', 'temp_max', 'Sales'], inplace=True, errors='ignore')
    
    return data
