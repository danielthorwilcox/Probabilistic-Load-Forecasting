import torch
from sklearn.model_selection import train_test_split
import models
import pandas as pd
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import yaml
from os.path import join
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# define filepath for config file and result data:
# for a new experiment, place a config file in a folder
# and set the filepath accordingly
filepath = "./network15"


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


# get model parameters from config file
with open(join(filepath,"config.yaml"),'r') as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc) 

params = config["parameters"]

train_period = params["train_period"]  # observation window size
pred_period = params["pred_period"]  # prediction window size, should be 24 hours
data = pd.read_csv("demand_generation/energy_dataset_lininterp.csv")
X, y, n_observations, n_features = getXypairs(data, train_period=train_period, pred_period=pred_period)

X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, shuffle=False)

batch_size = params["batch_size"]

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
hidden_dim = params["hidden_dim"]
layer_dim = params["layer_dim"]
dropout = params["dropout"]
n_epochs = params["n_epochs"]
learning_rate = params["learning_rate"]
weight_decay = params["weight_decay"]

model_params = {'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'layer_dim': layer_dim,
                'output_dim': output_dim,
                'dropout_prob': dropout}

model = models.get_model('lstm', model_params)

loss_fn = nn.MSELoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

opt = models.Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)
opt.plot_losses()

# save some metrics to disc
opt.save_losses(filepath=filepath)
predictions, values = opt.evaluate(test_loader_one, batch_size=1, n_features=input_dim)
mse = mean_squared_error(predictions.flatten(), values.flatten())
mae = mean_absolute_error(predictions.flatten(),values.flatten())
r2 = r2_score(predictions.flatten(),values.flatten())
with open(join(filepath,"test_loss.txt"),'w') as f:
    f.write("MSE: ")
    f.write(str(mse))
    f.write("\nMAE: ")
    f.write(str(mae))
    f.write("\nr2 score: ")
    f.write(str(r2))

# Plot single prediction
plt.plot(values[13, :, :])
plt.plot(predictions[13, :, :])
plt.legend(["true values", "predictions"])
plt.show()
