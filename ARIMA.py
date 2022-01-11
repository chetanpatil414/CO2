import streamlit as st

def main():
	""" Deploying streamlit App with Docker"""

	st.title("Streamlit App")
	st.header("Deploying Streamlit with Docker on GCP")



import pandas as pd
from datetime import date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
from statsmodels.tsa.arima_model import ARIMA

START = "2015-01-01"
#TODAY = date.today().strftime("%Y-%m-%d")
End = "2025-01-01"

st.title('CO2 EMISSION FORECASTING')
n_years = st.sidebar.slider('Years of Prediction:', 1, 10)
period = n_years * 365

data = pd.read_excel("C:/Users/PATIL'S/Desktop/data Science project/CO2 dataset.xlsx")
st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Year'], y=data['CO2'], name="CO2 Emission"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Year','CO2']]
df_train = df_train.rename(columns={"Year": "ds", "CO2": "y"})



m = Prophet()
data['Dates'] = pd.to_datetime(data['Year'], format='%Y')
air_qlt = data.drop(['Year'],axis=1)
air_qlt.rename(columns={'CO2': 'y', 'Dates': 'ds'}, inplace=True)
m.fit(air_qlt)
future_prices = m.make_future_dataframe(periods=period)
# m.fit(df_train)
# future = m.make_future_dataframe(periods=period)
forecast = m.predict(future_prices)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
