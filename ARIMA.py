#!/usr/bin/env python
# coding: utf-8

# ## ARIMA MODEL FOR AMZN STOCK PREDICTION
# 
# ### Time Series Analysis
# 
# Any data that collectively represents how a system/process/behaviour changes over time is known as Time Series data. Time series forecasting uses information regarding historical values and associated patterns to predict future activity. In this project, we are going to predict tomorrow's closing price of Amazon Stock (AMZN) by analysing the stock data for over a time period(3 years) until today. Predicting the performance of stock market is one of the most difficult tasks as the share prices are highly volatile for various reasons. Thus, forecasting prices with high accuracy rates is is difficult. ARIMA is a very popular statistical method for time series forecasting. ARIMA models take into account the past values to predict the future values. 
# 
# ### ARIMA
# 
# ARIMA (Auto-Regressive Integrated Moving Average) is a technique for modelling time series data for forecasting or predicting the future data points in the series by taking into consideration the following parameters:
# 
# 1. Pattern or trend of growth/decline
# 2. Rate of change of growth/decline
# 3. Noise between consecutive data points
# 
# 

# ### Importing the necessary Libraries

# In[25]:


import requests
#import pandas as pd
import numpy as np
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
#get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
import warnings
rcParams['figure.figsize'] = 16,8


# ### Reading Data
# 
# Data is pulled from Yahoo Finance with start date as 1/1/2016 and end date as current date so as to take into consideration the most recent trend for better predictions.

# In[3]:


start = dt.datetime(2016,1,1)
end = dt.datetime.today()


# In[4]:


style.use('ggplot')


# In[5]:


df = web.DataReader('AMZN', 'yahoo', start, end)


# In[6]:


df.to_csv('amzn.csv')
df = pd.read_csv('amzn.csv', parse_dates = True, index_col = 0)


# ### Feature Description
# 
# The data pulled from Yahoo Finance is stored in a .csv file. The AMZN stock data has the following features:
# 
# 1. Date - in format: yyyy-mm-dd
# 2. High - Highest price reached in the day
# 3. Low	 - Lowest price reached in the day
# 4. Open - price of the stock at market open 
# 5. Close - price of the stock at market close
# 6. Volume - Number of shares traded
# 7. Adj Close - stock's closing price on any given day of trading that has been amended to include any distributions and corporate actions that occurred at any time before the next day's open
# 

# In[7]:


df.head(5)


# In[8]:


group = df.groupby('Date')
Daily_ClosePrice = group['Close'].mean()

Daily_ClosePrice.head()


# In[9]:


df.tail(5)


# In[10]:


# indexing the dataframe by Date
df['Date'] = df.index


# In[11]:


# convert the date column into a time series with daily frequency

df['Date'] = pd.to_datetime(df['Date'])


# In[12]:


indexed_df = df.set_index('Date')


# In[13]:


ts = indexed_df['Adj Close']
ts.head()


# In[14]:


# visualize the time series to see how AMZN stock Close price trends over time
plt.figure(figsize=(16,8))
plt.plot(ts)
plt.title('AMZN 3 years trend')
plt.xlabel('Year',fontsize=20)
plt.ylabel('Adjusted Close Price',fontsize=20)


# In[15]:


# resampling by week

ts_week = ts.resample('W').mean()
ts_week


# In[16]:


plt.figure(figsize=(16,8))
plt.plot(ts_week)


# ### Check for Stationarity of the Time series
# 
# Stationary Time Series data does not have any upward or downward trend or seasonal effects. Mean or variance are consistent over time
# 
# Non-Stationary Time Series data show trends, seasonal effects, and other structures depend on time. Forecasting performance is dependent on the time of observation. Mean and variance change over time and a drift in the model is captured.
# 
# We are using a statistical method called Dickey-Fuller test to check if our time series is stationary or not.

# In[17]:


# check for stationarity

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


# In[18]:


test_stationarity(ts_week)


# #### Because the test statistic is more than the 5% critical value and the p-value is larger than 0.05, the moving average is not constant over time and the null hypothesis of the Dickey-Fuller test cannot be rejected. This shows that the weekly time series is not stationary. Before you can apply ARIMA models for forecasting, you need to transform this time series into a stationary time series.

# In[19]:


# apply a non linear log transform

ts_week_log = np.log(ts_week)


# In[20]:


test_stationarity(ts_week_log)


# #### The Dickey-Fuller test results confirm that the series is still non-stationary. Again the test statistic is larger than the 5% critical value and the p-value larger than 0.05

# In[21]:


# remove trend and seasonality with decomposition

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


# #### Time series are stationary if they do not have trend or seasonal effects. We are going to use the difference transform to remove the time series' dependence on time.

# In[22]:


# remove trend and seasonality with differencing
ts_week_log_diff = ts_week_log - ts_week_log.shift()
plt.figure(figsize=(16,8))
plt.plot(ts_week_log_diff,color='green')
plt.legend(loc='best')
plt.xlabel('Time(Days)',fontsize=15)
plt.ylabel('$(Dollar)',fontsize=15)
plt.title('Seasonality')


# In[23]:


ts_week_log_diff.dropna(inplace=True)
test_stationarity(ts_week_log_diff)


# #### The above graph shows how the rolling mean and rolling standard deviation are comparitively consistent over time after the time series transformation. We can proceed to use this transformed data for training our ARIMA model and forecasting the upcoming week's stock price.
# 
# ### ARIMA Model

# In[24]:


size = int(len(ts_week_log)*(0.7))
train, test = ts_week_log[0:size], ts_week_log[size:len(ts_week_log)]
history = [x for x in train]
predictions = list()

print('Printing Predicted vs Expected Values...')
print('\n')
for t in range(len(test)):
    model = ARIMA(history, order=(2,1,1)) #The order(p,d,q) of the model
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()[0]
    yhat = output[0]
    predictions.append(float(yhat))
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (np.exp(yhat), np.exp(obs)))

    
error = mean_squared_error(test, predictions)
r2 = r2_score(test, predictions)
print(r2)

print('\n')
print('Printing Mean Squared Error of Predictions...')
print('Test MSE: %.6f' % error)


# In[26]:


predictions_series = pd.Series(predictions, index = test.index)


# In[27]:


plt.figure(figsize=(16,8))
fig, ax = plt.subplots()
ax.set(title='Prediction (ARIMA)', xlabel='Date(weekly)', ylabel='$(Dollar)')
ax.plot(ts_week[-70:-5], label='observed', color='r')
ax.plot(np.exp(predictions_series), color='g', label='rolling one-step out-of-sample forecast')
legend = ax.legend(loc='upper left')
legend.get_frame().set_facecolor('w')


# In[28]:


r2


# ### Tuning hyper parameters for our ARIMA model
# 
# We are implementing grid search to find the best combination of p, d, q values for our model

# In[29]:


import warnings
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

order=(2,1,1)
# evaluate an ARIMA model for a given order (p,d,q) and return RMSE
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    X = X.astype('float32')
    train_size = int(len(X) * 0.50)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        # model_fit = model.fit(disp=0)
        model_fit = model.fit(trend='nc', disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    mse = mean_squared_error(test, predictions)
    rmse = sqrt(mse)
    return rmse

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, (order))
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s RMSE=%.3f' % (order,mse))
                except:
                    continue
    #print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))

# evaluate parameters
p_values = range(0, 5)
d_values = range(0, 3)
q_values = range(0, 5)
warnings.filterwarnings("ignore")
evaluate_models(ts_week_log, p_values, d_values, q_values)


# #### The combination of p,d,q values for which the RMSE is lowest is chosen as the best combination. In our case, there are multiple combinations with least RMSE of 0.034 and we will be comparing the accuracy of the corresponding model, to find the best one.
# 
# #### 1. Order = (0,1,2)

# In[34]:


size = int(len(ts_week_log)*(0.7))
train, test = ts_week_log[0:size], ts_week_log[size:len(ts_week_log)]
history = [x for x in train]
predictions = list()

print('Printing Predicted vs Expected Values...')
print('\n')
for t in range(len(test)):
    model = ARIMA(history, order=(3,1,0)) #The order(p,d,q) of the model
    model_fit = model.fit(disp=5)
    output = model_fit.forecast()[0]
    yhat = output[0]
    predictions.append(float(yhat))
    obs = test[t]
    history.append(obs)
    #print('predicted=%f, expected=%f' % (np.exp(yhat), np.exp(obs)))

    
error = mean_squared_error(test, predictions)
r2 = r2_score(test, predictions)
print(r2)

print('\n')
print('Printing Mean Squared Error of Predictions...')
print('Test MSE: %.6f' % error)


# #### Accuracy for Order = (0,1,2) is 78.92%
# 
# #### 2. Order = (0,1,3)

# In[33]:


size = int(len(ts_week_log)*(0.7))
train, test = ts_week_log[0:size], ts_week_log[size:len(ts_week_log)]
history = [x for x in train]
predictions = list()

print('Printing Predicted vs Expected Values...')
print('\n')
for t in range(len(test)):
    model = ARIMA(history, order=(0,1,3)) #The order(p,d,q) of the model
    model_fit = model.fit(disp=5)
    output = model_fit.forecast()[0]
    yhat = output[0]
    predictions.append(float(yhat))
    obs = test[t]
    history.append(obs)
    #print('predicted=%f, expected=%f' % (np.exp(yhat), np.exp(obs)))

    
error = mean_squared_error(test, predictions)
r2 = r2_score(test, predictions)
print(r2)

print('\n')
print('Printing Mean Squared Error of Predictions...')
print('Test MSE: %.6f' % error)


# #### Accuracy for Order = (0,1,3) = 79.38%
# 
# #### 3. Order = (0,1,2)

# In[30]:


size = int(len(ts_week_log)*(0.7))
train, test = ts_week_log[0:size], ts_week_log[size:len(ts_week_log)]
history = [x for x in train]
predictions = list()

print('Printing Predicted vs Expected Values...')
print('\n')
for t in range(len(test)):
    model = ARIMA(history, order=(0,1,2)) #The order(p,d,q) of the model
    model_fit = model.fit(disp=5)
    output = model_fit.forecast()[0]
    yhat = output[0]
    predictions.append(float(yhat))
    obs = test[t]
    history.append(obs)
    #print('predicted=%f, expected=%f' % (np.exp(yhat), np.exp(obs)))

    
error = mean_squared_error(test, predictions)
r2 = r2_score(test, predictions)
print(r2)

print('\n')
print('Printing Mean Squared Error of Predictions...')
print('Test MSE: %.6f' % error)


# In[31]:


predictions_series = pd.Series(predictions, index = test.index)


# In[32]:


plt.figure(figsize=(16,8))
fig, ax = plt.subplots()
ax.set(title='Prediction (ARIMA)', xlabel='Date(weekly)', ylabel='$(Dollar)')
ax.plot(ts_week[-70:-5], label='observed', color='r')
ax.plot(np.exp(predictions_series), color='g', label='rolling one-step out-of-sample forecast')
legend = ax.legend(loc='upper left')
legend.get_frame().set_facecolor('w')


# #### Accuracy for Order = (0,1,2) = 79.49%

# ### Observations
# 
# The above graph represents the comparison of our observed stock close prices vs forecasted close price for the upcoming week with an accuracy of 79.49%
# 

# In[1]:


from prettytable import PrettyTable


# In[13]:


result_hyperparam = PrettyTable()

result_hyperparam.field_names = ["Model(ARIMA(p,d,q))", "Accuracy"]

result_hyperparam.add_row(['ARIMA(2,2,1)',79.23])
result_hyperparam.add_row(['ARIMA(0,1,3)',79.32])
result_hyperparam.add_row(['ARIMA(3,1,0)',78.92])
result_hyperparam.add_row(['ARIMA(0,1,2)',79.49])


print(result_hyperparam)


# In[11]:


result_predictedvsexpected = PrettyTable()

result_predictedvsexpected.field_names = ["date", "predicted","expected"]

result_predictedvsexpected.add_row(['2019-04-21',1843.78,1857.53])
result_predictedvsexpected.add_row(['2019-04-14',1847.59,1844.03])
result_predictedvsexpected.add_row(['2019-04-07',1766.52,1821.01])
result_predictedvsexpected.add_row(['2019-03-31',1799.97,1775.57])
result_predictedvsexpected.add_row(['2019-03-24',1699.47,1777.06])

print(result_predictedvsexpected)


# 
# ### Conclusion
# 
# Predicting stock market prices is really a difficult task, as the market keeps changing continuously with time following a trend based on various factors. 
# All Stock market data are a time series problem and we have analysed and transformed our Amazon stock price data treating its seasonality and trend as these affect the prediction.
# We have modelled a ML algorithm called ARIMA(Auto-Regressive Integrated Moving Average) to forecast the stock market prices and tuned the hyper parameters to get a model that predicts with better accuracy.
# 

# In[ ]:




