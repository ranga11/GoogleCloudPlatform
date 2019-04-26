# Databricks notebook source
#import gcloud
import sys
sys.path.insert(0, "/home/vranga_11/.local/lib/python3.6/site-packages")
import numpy as np
import pandas as pd
import datetime as dt
from datetime import date
from datetime import datetime
import matplotlib.pyplot as plt
import fix_yahoo_finance as yf 
import pyspark as spark
from io import StringIO
import time, json
from datetime import date, timedelta
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.metrics import *
from matplotlib import style
import matplotlib.pylab as plt
#%matplotlib inline
from matplotlib.pylab import rcParams
import warnings
from prettytable import PrettyTable
rcParams['figure.figsize'] = 16,8
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_list, struct
from pyspark.sql.types import FloatType, StructField, StructType, StringType, TimestampType
from sklearn.metrics import mean_squared_error

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# start = datetime.now() # gets today's date
# start = start.replace(year=start.year-3) # gets the date before 3 years
# end = datetime.now() # gets today's date

# # COMMAND ----------

# def get_csv():
#     amzn_data = yf.download('AMZN',start,end) # getting amazon stock from yahoo finance
#     amzn_data.to_csv('AMZN.csv', index=True, header=True)
#     amzn_data.Close.plot() # Plotting the close value of stocks
#     plt.show()
#     return;

# # COMMAND ----------

# get_csv()

# df = spark \
#         .read \
#         .format("csv") \
#         .option("header", "true") \
#         .option("inferSchema", "true") \
#         .load(storage_bucket + "/" + amzn)
sc = SparkContext()
spark = SparkSession(sc)
bucket = spark._jsc.hadoopConfiguration().get("fs.gs.system.bucket")
project = spark._jsc.hadoopConfiguration().get("fs.gs.project.id")

conf = {
    # Input Parameters
    "mapred.bq.project.id": flash-freehold-237222,
    "mapred.bq.gcs.bucket": freehold-1,
    "mapred.bq.temp.gcs.path": input_directory,
    "mapred.bq.input.project.id": project,
    "mapred.bq.input.dataset.id": "flash-freehold-237222",
    "mapred.bq.input.table.id": "flash-freehold-237222:Arima_test.Amazondata",
}


put_directory = "gs://gs://{}/tmp/Amazondata-{}", ".format(bucket)


df = spark.sparkContext.newAPIHadoopRDD(
    "com.google.cloud.hadoop.io.bigquery.JsonTextBigQueryInputFormat",
    "org.apache.hadoop.io.LongWritable",
    "com.google.gson.JsonObject",
    conf=conf)



# COMMAND ----------
#df= spark.read.csv("./GoogleCloudaPlatform/amzn.csv",header=True,sep="|");
#df = spark.read.csv("AMZN.csv", header=True, mode="DROPMALFORMED", schema=schema)
#df = (spark.read.option("header", "true").option("inferSchema", value=True).csv("amzn.csv"))

# COMMAND ----------

df.describe()

# COMMAND ----------

df.head(5)

# COMMAND ----------

df = df.select(df['Date'],df['Close'].cast(FloatType()))

# COMMAND ----------

df.head(5)

# COMMAND ----------

df.select(df.Date, df.Close).show()

# COMMAND ----------

df = df.toPandas()

# COMMAND ----------

df.describe()

# COMMAND ----------

df.head(5)

# COMMAND ----------

df.index =df['Date']

# COMMAND ----------

df.head()

# COMMAND ----------

df['Date'] = pd.to_datetime(df['Date'])

indexed_df = df.set_index('Date')

# COMMAND ----------

ts = indexed_df['Close']
ts.head()

# COMMAND ----------

# visualize the time series to see how AMZN stock Close price trends over time
plt.figure(figsize=(16,8))
plt.plot(ts)
plt.title('AMZN 3 years trend')
plt.xlabel('Year',fontsize=20)
plt.ylabel('Close Price',fontsize=20)

# COMMAND ----------

# resampling

ts = ts.resample('D').mean()
ts

# COMMAND ----------

ts = ts.dropna()

# COMMAND ----------

ts

# COMMAND ----------

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

# COMMAND ----------

test_stationarity(ts)

# COMMAND ----------

# MAGIC %md #### Because the test statistic is more than the 5% critical value and the p-value is larger than 0.05, the moving average is not constant over time and the null hypothesis of the Dickey-Fuller test cannot be rejected. This shows that the weekly time series is not stationary. Before you can apply ARIMA models for forecasting, you need to transform this time series into a stationary time series.

# COMMAND ----------

# apply a non linear log transform

ts_log = np.log(ts)

# COMMAND ----------

test_stationarity(ts_log)

# COMMAND ----------

ts.head()

# COMMAND ----------

# remove trend and seasonality with decomposition

decomposition = seasonal_decompose(ts_log, freq = 52)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(16,8))
plt.subplot(411)
plt.plot(ts_log[-80:], label='Original')
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

# COMMAND ----------

# remove trend and seasonality with differencing
ts_log_diff = ts_log - ts_log.shift()
plt.figure(figsize=(16,8))
plt.plot(ts_log_diff,color='green')
plt.legend(loc='best')
plt.xlabel('Time(Days)',fontsize=15)
plt.ylabel('$(Dollar)',fontsize=15)
plt.title('Seasonality')

# COMMAND ----------

ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)

# COMMAND ----------

# MAGIC %md ### The above graph shows how the rolling mean and rolling standard deviation are comparitively consistent over time after the time series transformation. We can proceed to use this transformed data for training our ARIMA model and forecasting the upcoming week's stock price.
# MAGIC 
# MAGIC ### ARIMA Model
# MAGIC 
# MAGIC * An ARIMA model is usually stated as ARIMA(p,d,q). This represents the order of the autoregressive components (p), the number of differencing operators (d), and the highest order of the moving average term. For example, ARIMA(2,1,1) means that you have a second order autoregressive model with a first order moving average component whose series has been differenced once to induce stationarity.

# COMMAND ----------

size = int(len(ts_log)*(0.7))
train, test = ts_log[0:size], ts_log[size:len(ts_log)]
history = [x for x in train]
predictions = list()

print('Printing Predicted vs Expected Values...')
print('\n')
for t in range(len(test)):
    model = ARIMA(history, order=(0,2,1)) #The order(p,d,q) of the model
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

# COMMAND ----------

predictions_series = pd.Series(predictions, index = test.index)

# COMMAND ----------

plt.figure(figsize=(16,8))
fig, ax = plt.subplots()
ax.set(title='Prediction (ARIMA)', xlabel='Date(weekly)', ylabel='$(Dollar)')
ax.plot(ts, label='observed', color='r')
ax.plot(np.exp(predictions_series), color='g', label='rolling one-step out-of-sample forecast')
legend = ax.legend(loc='upper left')
legend.get_frame().set_facecolor('w')

# COMMAND ----------

np.exp(predictions_series)
