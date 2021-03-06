import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib.pylab
import matplotlib.pyplot as plt
import scipy.stats as stats
import datetime
import itertools
import warnings
from math import sqrt
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
from pandas.tseries.offsets import DateOffset
import warnings
warnings.filterwarnings("ignore") # specify to ignore warning messages
import streamlit as st
from plotly import graph_objs as go

st.title('CO2 EMISSION FORECASTING')
def user_input_features():
    Years = st.sidebar.number_input('Years of Prediction:', min_value=1, max_value=40, value=1, step=1)
    return Years 

df = user_input_features()+1
#st.subheader('User Input parameters')
#st.write(df)

import pandas as pd 
from datetime import datetime


def dateparse(dates):
    return datetime.strptime(dates, '%Y')


data = pd.read_excel("CO2 dataset.xlsx",
                           parse_dates=['Year'],
                           index_col='Year',
                           engine='openpyxl')



#Model Building
future_dates=[data.index[-1]+ DateOffset(years=x)for x in range(0,df)]
future_data=pd.DataFrame(index=future_dates[1:],columns=data.columns)

final_arima = ARIMA(data['CO2'],order = (3,1,4))
final_arima = final_arima.fit()

final_arima.fittedvalues.tail()

future_data['CO2'] = final_arima.predict(start = 215, end = 250, dynamic= True) 


# Plot raw data
st.header("Past data Analysis")
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data.index,y=data['CO2'], name="CO2 Emission"))
	st.plotly_chart(fig)


plot_raw_data()

future_data.tail(df)

st.sidebar.subheader(f'Forecasting for {df-1} year')
st.sidebar.table(future_data.tail(df))


# Plot raw data
st.header("Future Data Analysis")
def plot_result_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=future_data.index,y=future_data['CO2'], name="CO2 Emission"))
	st.plotly_chart(fig)


plot_result_data()
