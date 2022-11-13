import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import  adfuller
from math import *
from random import randint
from sklearn.metrics import mean_squared_error


def getXypairs(data, train_period, pred_period):
    data.drop(columns='time', inplace=True)
    n_observations = len(data)
    window_size = train_period + pred_period
    X = np.zeros([n_observations - window_size, train_period])
    y = np.zeros([n_observations - window_size, pred_period])

    for idx, x in enumerate(X):
        X[idx, :] = data.iloc[idx:idx + train_period].to_numpy()
        y[idx, :] = data.iloc[idx + train_period:idx + window_size].to_numpy()

    return X, y


def get_stationarity(timeseries):
    
    # rolling statistics
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()
    
    # rolling statistics plot
    original = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    # Dickey–Fuller test:
    result = adfuller(timeseries)
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))


#--------------------------------------------------------------------------------------------------------------------
# W_obs: number of old values used for prediction, W_pred: number of future values to predict  

W_obs = 24*4
W_pred = 24
n_runs = 1

data = pd.read_csv("demand_generation/energy_dataset_lininterp.csv")
ts = data["total load actual"]
ts = ts.diff()

X, y = getXypairs(ts,W_obs,W_pred)

MSEs = []
for i in range(0,n_runs):
    idx = randint(0,len(X))
    get_stationarity(pd.DataFrame(X[idx]))
    model = ARIMA(X[idx],order=(1,1,1))
    results = model.fit()
    train_predict = results.fittedvalues
    test_predict = results.forecast(steps=W_pred)
    mse = mean_squared_error(test_predict,y[idx])
    print(mse)
    MSEs.append(mse)
    plt.plot(X[idx])
    plt.plot(train_predict)
    plt.show()
    plt.plot(y[idx])
    plt.plot(test_predict)
    plt.show()

