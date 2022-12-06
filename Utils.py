
# Utility functions that we may need at more than one places
import torch
import yaml
from os.path import join


def getXypairs(data, train_period, pred_period):
    data.drop(columns='time', inplace=True)
    print(data.shape)
    (n_observations, n_features) = data.shape # number of timestamps
    window_size = train_period + pred_period
    X = torch.zeros([n_observations - window_size, train_period, n_features])
    y = torch.zeros([n_observations - window_size, pred_period])

    for idx, x in enumerate(X):
        X[idx, :, :] = torch.tensor(data.iloc[idx:idx + train_period, :].to_numpy())
        y[idx, :] = torch.tensor(data['value'].iloc[idx + train_period:idx + window_size].to_numpy())

    return torch.diff(X,axis=0), torch.diff(y,axis=0), n_observations, n_features


def get_model_params(filepath):
    # get model parameters from config file
    with open(join(filepath, "config.yaml"), 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

    return config["parameters"]
