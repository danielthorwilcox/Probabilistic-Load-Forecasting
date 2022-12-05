import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import  adfuller
from math import *
from random import randint
from sklearn.metrics import mean_squared_error
from pandas.plotting import autocorrelation_plot


def getXypairs(data, train_period, pred_period):

    # create a dataset of training and prediction windows from a given time series:
    # start with the first value as the first data point, the train_period-tht value
    # as the first value as the first prediction point (value to predict from corresponding training window),
    #  then shif this through the time series:
    #
    # _____data(n)______~~pred(n)~~************************** first data-pred pair
    # *********_____data(n)______~~pred(n)~~***************** n-th data-pred pair
    # **********_____data(n+1)____~pred(n+1)~**************** (n+1)-th data-pred pair
    #                               ...
    # *****************_____data(n+m)____~pred(n+m)~********* (n+m)-th data-pred pair
    #                               ...
    # **************************_____data(n+1)____~pred(n+1)~ last data-pred pair

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

W_obs = 24*2
W_pred = 12
n_runs = 100 #number of ARIMA models to fit and evaluate (and average MSEs over)
n_plots = 2 #number of example plots

data = pd.read_csv("sinedata.csv")
ts = data["value"]
ts_wss = ts.dropna() #find differencing order to make the series WSS


#print(ts.head())

#autocorrelation_plot(ts.iloc[:40])
#plt.show()

X, y = getXypairs(ts.dropna(),W_obs,W_pred)




MSEs = np.zeros(n_runs)
for i in range(0,n_runs):
    idx = randint(0,len(X)) # pick random training-test pair
    #get_stationarity(pd.DataFrame(X[idx]))
    model = ARIMA(X[idx],order=(24,1,0)) # train ARIMA(24,1,0) model on selected training-test pair
    results = model.fit(method='burg')
    #print(results.summary())
    train_predict = results.fittedvalues # get trainng predictions
    test_predict = results.forecast(steps=W_pred) # predict the next W_pred values form trained model
    mse = mean_squared_error(test_predict,y[idx]) # compute MSE of this specific train-test pair
    MSEs[i] = mse
    if(i%(n_runs/n_plots) == 0): # plot the training and test predictions vs the real values
        fig, ax = plt.subplots(1,2)
        ax[0].plot(X[idx])
        ax[0].plot(train_predict)
        ax[1].plot(y[idx])
        ax[1].plot(test_predict)
        ax[0].set_title("Training portion")
        ax[1].set_title("Test portion")
        fig.suptitle("Example prediction performance for one traing-test pair")
        ax[0].legend(["actual","predictions"])
        ax[1].legend(["actual","predictions"])
        
# print out the overall statistics over the n_runs predictors:
print("averaged MSE: ", MSEs.mean())
print("MSE variance: ", MSEs.var())
print("Max MSE: ", MSEs.max())
print("Min MSE: ", MSEs.min())
plt.show()
