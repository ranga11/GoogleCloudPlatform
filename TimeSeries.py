#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests, pandas as pd, numpy as np
from pandas import DataFrame
from io import StringIO
import time, json
import pandas_datareader.data as web
import datetime as dt
from datetime import date, timedelta
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import *
from matplotlib import style
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
import warnings
rcParams['figure.figsize'] = 16,8

def main():
    start = dt.datetime(2016,1,1)
    end = dt.datetime.today()
    style.use('ggplot')
    df = web.DataReader('AMZN', 'yahoo', start, end)
    df.to_csv('amzn.csv')
    df = pd.read_csv('amzn.csv', parse_dates = True, index_col = 0)
    group = df.groupby('Date')
    Daily_ClosePrice = group['Close'].mean()
    Daily_ClosePrice.head()
    df['Date'] = df.index
    df['Date'] = pd.to_datetime(df['Date'])
    indexed_df = df.set_index('Date')
    ts = indexed_df['Adj Close']
    # visualize the time series to see how AMZN stock Close price trends over time
    plt.figure(figsize=(16,8))
    plt.plot(ts)
    plt.title('AMZN 3 years trend')
    plt.xlabel('Year',fontsize=20)
    plt.ylabel('Adjusted Close Price',fontsize=20)
    ts_week = ts.resample('W').mean()
    ts_week
    plt.figure(figsize=(16,8))
    plt.plot(ts_week)

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=52,center=False).mean() 
    rolstd = timeseries.rolling(window=52,center=False).std()

    #Plot rolling statistics:
    plt.figure(figsize=(16,8))
    orig = plt.plot(timeseries, color='green',label='AMZN Data')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.xlabel('Time(Days)',fontsize=15)
    plt.ylabel('$(Dollar)',fontsize=15)
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
ts_week_log = np.log(ts_week)
test_stationarity(ts_week_log)
decomposition = seasonal_decompose(ts_week)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(16,8))
plt.subplot(411)
plt.plot(ts_week_log[-80:], label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend[-80:], label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal[-80:],label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual[-80:], label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
ts_week_log_diff = ts_week_log - ts_week_log.shift()
plt.figure(figsize=(16,8))
plt.plot(ts_week_log_diff,color='green')
plt.legend(loc='best')
plt.xlabel('Time(Days)',fontsize=15)
plt.ylabel('$(Dollar)',fontsize=15)
plt.title('Seasonality')
ts_week_log_diff.dropna(inplace=True)
test_stationarity(ts_week_log_diff)

if __name__=="__main__":
    main()


# In[ ]:




