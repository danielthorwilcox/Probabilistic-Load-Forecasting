import torch

from models import LSTMModel
import pandas as pd


data = pd.read_csv("demand_generation/energy_dataset_lininterp.csv")


def getXypairs(data, train_period, pred_period):
    data.drop(columns='time', inplace=True)
    (n_observations, n_features) = data.shape # number of timestamps
    window_size = train_period + pred_period
    X = torch.zeros([n_observations - window_size, train_period, n_features])
    y = torch.zeros([n_observations - window_size, pred_period, 1])

    for idx, x in enumerate(X):
        X[idx, :, :] = torch.tensor(data.iloc[idx:idx + train_period, :].to_numpy())
        y[idx, :, :] = torch.tensor(data['total load actual'].iloc[idx + train_period:idx + window_size].to_numpy())

    return X, y


X, y = getXypairs(data, 10, 1)
print(X)
