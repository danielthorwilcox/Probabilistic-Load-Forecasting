import numpy as np
import torch
from sklearn.model_selection import train_test_split
import models
import pandas as pd
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


def getXypairs(data, train_period, pred_period):
    data.drop(columns='time', inplace=True)
    (n_observations, n_features) = data.shape # number of timestamps
    window_size = train_period + pred_period
    X = torch.zeros([n_observations - window_size, train_period, n_features])
    y = torch.zeros([n_observations - window_size, pred_period])

    for idx, x in enumerate(X):
        X[idx, :, :] = torch.tensor(data.iloc[idx:idx + train_period, :].to_numpy())
        y[idx, :] = torch.tensor(data['total load actual'].iloc[idx + train_period:idx + window_size].to_numpy())

    return X, y, n_observations, n_features


train_period = 24  # observation window size
pred_period = 8  # prediction window size, should be 24 hours
data = pd.read_csv("demand_generation/energy_dataset_lininterp.csv")
X, y, n_observations, n_features = getXypairs(data, train_period=train_period, pred_period=pred_period)

X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.01, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, shuffle=False)

batch_size = 64

train_set = TensorDataset(X_train, y_train)
val_set = TensorDataset(X_val, y_val)
test_set = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, drop_last=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader_one = DataLoader(test_set, batch_size=1, shuffle=False, drop_last=True)

# Train network
input_dim = n_features
output_dim = pred_period
hidden_dim = 64
layer_dim = 3
batch_size = 64
dropout = 0.2
n_epochs = 12
learning_rate = 1e-3
weight_decay = 1e-6

model_params = {'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'layer_dim': layer_dim,
                'output_dim': output_dim,
                'dropout_prob': dropout}

model = models.get_model('bayesian_lstm', model_params)

loss_fn = nn.MSELoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

opt = models.Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)
opt.plot_losses()

predictions, true_values = opt.evaluate_with_ci(test_loader_one, batch_size=1, n_features=input_dim)

some_idx = 13
single_pred = predictions[some_idx, :, :]
ic_acc, ci_upper, ci_lower = models.get_confidence_intervals(single_pred, 1)

# Plot single prediction
plt.plot(np.squeeze(true_values[some_idx, :, :]))
plt.plot(np.squeeze(ic_acc))
plt.fill_between(x=np.arange(pred_period),
                 y1=np.squeeze(ci_upper),
                 y2=np.squeeze(ci_lower),
                 facecolor='green',
                 label="Confidence interval",
                 alpha=0.5)
plt.legend(["true values", "predictions", "ci"])
plt.show()
