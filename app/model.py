import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from preprocessor import preprocess_data

def train_xgboost_model(train_data):
    # Preprocess trainingdata
    train_data = preprocess_data(train_data)
    
    # Set 'date'column to index and frequency to daily
    train_data.set_index('Date', inplace=True)
    train_data = train_data.asfreq('D')
    
    # Split features and target variable
    X_train = train_data.drop(columns=['Bb'])
    y_train = train_data['Bb']
    
    # Scale numerical features
    scaler = MinMaxScaler()
    numerical_columns = X_train.select_dtypes(include=np.number).columns.tolist()
    X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])

    # Define XGBoost model
    model = XGBRegressor()

    # Train model
    model.fit(X_train, y_train)
    return model, scaler

def forecast_xgboost_model(forecast_data, weather_data, sales_data, model, scaler):
    # Drop target column
    if 'Bb' in forecast_data.columns:
        forecast_data = forecast_data.drop(columns=['Bb'])
    
    # Preprocess testset
    forecast_data = preprocess_data(forecast_data, weather_data, sales_data)
    
    # Set 'date'column to index and frequency to daily
    #forecast_data['Date'] = pd.to_datetime(forecast_data['Date'], format="%Y-%m-%d")
    forecast_data.set_index('Date', inplace=True)
    forecast_data = forecast_data.asfreq('D')
    
    # Scale numerical features
    numerical_columns = forecast_data.select_dtypes(include=np.number).columns.tolist()
    forecast_data[numerical_columns] = scaler.transform(forecast_data[numerical_columns])
    
    forecast = model.predict(forecast_data)
    
    return forecast

def plot_forecast(train_data, forecast):

    fig, ax = plt.subplots(figsize=(8,4))
    # Plot the full forecast
    ax.plot(train_data['Date'][-56:], train_data['Bb'][-56:], label='Training Data')
    ax.plot(forecast['Date'][-7:], forecast['Bb'][-7:], label='Forecast Data')
    #axs[0].plot(validation_data['Date'], validation_data['Bb'], label='Validation Data')
    ax.legend()
    
    # Plot only the last 28 days
    #last_21_days_train = train_data.iloc[-21:]
    #axs[1].plot(last_21_days_train['Date'], last_21_days_train['Bb'], label='Last 3 + 1 weeks - Training Data')
    #axs[1].plot(forecast['Date'], forecast[-21:], label='Last 3 + 1 weeks - Forecast Data')
    #axs[1].plot(last_21_days_validation['Date'], last_21_days_validation['Bb'], label='Last 3 + 1 weeks - Validation Data')
    #axs[1].legend()

    fig.tight_layout() 
    
    # Calculate MAE and MAPE
    #mae = mean_absolute_error(forecast['Bb'], forecast[-21:])
    #mape = mean_absolute_percentage_error(last_21_days_validation['Bb'], forecast[-21:]) * 100
    
    # Print MAE and MAPE
    #print(f'MAE: {mae:.2f}')
    #print(f'MAPE: {mape:.2f}%')

    return fig