# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:05:39 2019

@author: Dt
"""

# Importing Libraries
import numpy as np
import pandas as pd
import datetime as dt
from datetime import date
from datetime import datetime
import matplotlib.pyplot as plt
import fix_yahoo_finance as yf 
import pyspark as spark
from fbprophet import Prophet
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_list, struct
from pyspark.sql.types import FloatType, StructField, StructType, StringType, TimestampType
from sklearn.metrics import mean_squared_error

spark = SparkSession.builder.getOrCreate()

#**Get last 3 years of data from today**

start = datetime.now() # gets today's date
start = start.replace(year=start.year-3) # gets the date before 3 years
end = datetime.now() # gets today's date

#**Get Amazon stock data from yahoo finance**

def get_csv():
    amzn_data = yf.download('AMZN',start,end) # getting amazon stock from yahoo finance
    amzn_data.to_csv('AMZN.csv', index=True, header=True)
    amzn_data.Close.plot() # Plotting the close value of stocks
    plt.show()
    return;

get_csv()

#**Creating Spark DataFrame**

df = (spark.read
          .option("header", "true")
          .option("inferSchema", value=True)
          .csv("AMZN.csv"))

df.head(5)

#**Assign ds & y values**

data = df.select(
        df['date'].alias('ds'),
        df['Close'].cast(FloatType()).alias('y')
    )

data.head(5)

data.select(data.ds, data.y).show()

data = data.toPandas()

data.index = data.ds

data.tail(6)

length = len(data)

train = data[0:length-7]

train.head()

# Defining Amazon prime days
primedays = pd.DataFrame({
               'holiday':'primeday', 
               'ds' : pd.to_datetime(['2016-06-12', '2017-06-10','2018-06-20']),
               'lower_window': -1,
               'upper_window': 1,
})

# Defining Long weekends
longweekends = pd.DataFrame({
               'holiday':'longweekend', 
               'ds' : pd.to_datetime(['2016-05-10', '2017-03-15','2018-04-21']),  
               'lower_window': -1,
               'upper_window': 1,
})

primedays

#**Combining both prime and long weekend in to holidays dataframe**

holidays = pd.concat((primedays, longweekends))

holidays.index = holidays['ds']

def fb(i):
    length = len(data)
    train = data[0:length-i]
    m = Prophet(yearly_seasonality = True, weekly_seasonality= True, seasonality_prior_scale=0.1, 
            changepoint_prior_scale=0.95, n_changepoints=23, #changepoints=['2019-01-01'], 
            holidays= holidays
           ) # Creating the model 
    m.add_seasonality('monthly',period=30.5,fourier_order=15)
    m.add_country_holidays(country_name='US')
    m.fit(train) # fit data to model 
    future = m.make_future_dataframe(periods=i)#it creates  rows 
    forecast = m.predict(future) 
    m.plot(forecast)
    #p_length = len(forecast)
    #a_length = len(data)
    # s_length = length-i
    prediction = forecast['yhat'][length-i:length] #Get values till today
    test = data[length-i:length]
    prediction.index = test.index #set common index
    result = pd.concat([prediction, test.y], axis=1, ignore_index=False) #Concatinate actual and predicted
    result['date'] = result.index
    result.to_csv('out.csv', index=True, header=True) # export to csv
    print (result)
    return result;

fb(7)