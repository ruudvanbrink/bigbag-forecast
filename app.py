import streamlit as st
import pickle
import locale
import math

from app.importer import *
from app.model import *
from app.functions import add_forecast_to_history

locale.setlocale(locale.LC_ALL, 'de_DE')

# Laad het model
#@st.cache_data()
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()
    
# Laad de scaler met pickle
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.set_page_config(layout="wide")
with st.sidebar: 
    st.header('Menu')
    st.page_link("app.py", label='Forecast', icon=':material/bar_chart_4_bars:')


# Title
st.title("Jumbo Bigbag Forecast DC Breda")

# Laad data
dataset, weather_data, sales_data = import_data()
dataset['Date'] = pd.to_datetime(dataset['Date'])

# New data
forecast_data = [
    ['2024-04-08', 20.3],
    ['2024-04-09', 14.2],
    ['2024-04-10', 15.7], 
    ['2024-04-11', 14.0],
    ['2024-04-12', 21.4],
    ['2024-04-13', 23.7], 
    ['2024-04-14', 15.9],
    #['2024-04-15', 11.5]
]

forecast_df = pd.DataFrame(forecast_data, columns=['Date', 'temp_max'])
forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])

#Generate forecast
forecast = forecast_xgboost_model(forecast_df, weather_data, sales_data, model, scaler)

forecast = pd.DataFrame({
    'Date': forecast_df['Date'],
    'Bb': forecast    
})
forecast = forecast.rename(columns={'Bb':'Forecast'}) 

# add_forecast_to_history(forecast)

# Create chart
chart_data = pd.merge(dataset, forecast, how='outer', on='Date', sort=True)
tab1, tab2, tab3 = st.tabs(["1 jaar", "6 maanden", "3 maanden"])

with tab1:
   n = 365
   chart_data = chart_data.iloc[-n:]
   st.line_chart(chart_data, x='Date', color=['#4287f5','#FFA500'], height = 400)

with tab2:
   n = 182
   chart_data = chart_data.iloc[-n:]
   st.line_chart(chart_data, x='Date', color=['#4287f5', '#FFA500'], height = 400)

with tab3:
   n = 91
   chart_data = chart_data.iloc[-n:]
   st.line_chart(chart_data, x='Date', color=['#4287f5','#FFA500'], height = 400)

# Data section
st.divider()
st.header('Data')
col1, col2, col3, col4 = st.columns(4)

with col1:
    productivity = st.slider(label='Productiviteit per lijn (bigbags/uur)', min_value=1, max_value=50, value=16, step=1,)

with col2:
    hours = st.slider(label='Productieve tijd per dienst', min_value=4.0, max_value=10.0, value=6.75, step=0.25)
    

# Perform calculation based on slider value
forecast['Operators'] = round(forecast['Forecast'] / (productivity * hours))
forecast['Operators'] = np.clip(forecast['Operators'], a_min=1, a_max=15)
forecast['Ploegen'] = (forecast['Operators'] / 5).apply(math.ceil)
forecast['Capaciteit'] = round(forecast['Operators'] * productivity * hours)
forecast['Resterend'] = forecast['Forecast'].apply(math.floor) - forecast['Capaciteit']
forecast['Resterend'] = np.clip(forecast['Resterend'], a_min=0, a_max = (productivity * hours))

# Show table
forecast['Date'] = forecast['Date'].dt.strftime('%A %d %B %Y')
forecast['Forecast'] = forecast['Forecast'].astype(int).apply(lambda x: f'{x:,}'.replace(',', '.'))
st.dataframe(forecast)

