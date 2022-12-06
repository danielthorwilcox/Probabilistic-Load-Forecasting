#-------------------------------------------------------------------------------------------------------------------
#linear regression predictor for energy data
#divides data into observation and prediction windows, then uses a linear regression model to predict the values in the
#prediction window based on the values in the corresponding observation window

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


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

#--------------------------------------------------------------------------------------------------------------------
# W_obs: number of old values used for prediction, W_pred: number of future values to predict  

W_obs = 164
W_pred = 24

load_data = pd.read_csv("sinedata.csv")
timeseries = load_data["value"]
timeseries = timeseries.diff().dropna()
print(timeseries.head())

X, y = getXypairs(timeseries,W_obs,W_pred)
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,shuffle=False)

lrmodel = LinearRegression()
lrmodel.fit(X_train,y_train)
predictions = lrmodel.predict(X_test)

mse = mean_squared_error(y_test,predictions)
r2 = r2_score(y_test,predictions)

print("Linear Regression predictor with MSE: ", mse, ", r2 score: ", r2)
#print("model coefficients: ", lrmodel.coef_)

#'''
#plot one of the perdiction windows vs the real values in that window
plt.plot(y_test[100,:])
plt.plot(predictions[100,:])
plt.legend(["true values", "predictions"])
plt.title("Example prediction of the Linear Regression model")
plt.show()
#'''
