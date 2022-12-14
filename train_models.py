import numpy as np
import torch
from sklearn.model_selection import train_test_split
import models
import pandas as pd
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from os.path import join
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import Utils
from Utils import networkpath

# define filepath for config file and result data:
# for a new experiment, place a config file in a folder
# and set the filepath accordingly in Utils.py


params = Utils.get_model_params()

train_period = params["train_period"]  # observation window size
pred_period = params["pred_period"]  # prediction window size, should be 24 hours
data = pd.read_csv("demand_generation/timeseries.csv")
X, y, n_observations, n_features = Utils.getXypairs(data, train_period=train_period, pred_period=pred_period)

X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, shuffle=False)

batch_size = params["batch_size"]

train_set = TensorDataset(X_train, y_train)
val_set = TensorDataset(X_val, y_val)
test_set = TensorDataset(X_test, y_test)

shuffle = True
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, drop_last=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, drop_last=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, drop_last=True)
test_loader_one = DataLoader(test_set, batch_size=1, shuffle=shuffle, drop_last=True)

# Train network
input_dim = n_features
output_dim = pred_period
hidden_dim = params["hidden_dim"]
layer_dim = params["layer_dim"]
n_fc_layers = params["n_fc_layers"]
dropout = params["dropout"]
n_epochs = params["n_epochs"]
learning_rate = params["learning_rate"]
weight_decay = params["weight_decay"]

model_params = {'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'layer_dim': layer_dim,
                'output_dim': output_dim,
                'n_fc_layers': n_fc_layers,
                'dropout_prob': dropout}

model_name = params["model"]
model = models.get_model(model_name, model_params)
print(model)

loss_fn = nn.MSELoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

opt = models.Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)
opt.plot_losses()

# save some metrics to disc
opt.save_losses(filepath=Utils.networkpath)
torch.save(model, join(networkpath, "model"))

predictions, true_values = opt.evaluate(test_loader_one, model_name, batch_size=1, n_features=input_dim)
predictions_mean = np.mean(predictions, axis=1)  # mean of the bayesian outputs, if non-bayesian it has no effect
mse = mean_squared_error(predictions_mean.flatten(), true_values.flatten())
mae = mean_absolute_error(predictions_mean.flatten(), true_values.flatten())
r2 = r2_score(predictions_mean.flatten(), true_values.flatten())
with open(join(Utils.networkpath, "test_loss.txt"), 'w') as f:
    f.write("MSE: ")
    f.write(str(mse))
    f.write("\nMAE: ")
    f.write(str(mae))
    f.write("\nr2 score: ")
    f.write(str(r2))

some_idx = 13
some_idx_2 = 501
single_pred = predictions[some_idx, :, :]
single_pred_2 = predictions[some_idx_2, :, :]


if model_name == 'lstm':
    # Save single prediction
    df = pd.DataFrame(np.array([np.squeeze(single_pred),np.squeeze(true_values[some_idx,:,:]),np.squeeze(single_pred_2), np.squeeze(true_values[some_idx_2,:,:])]).T, columns=["prediction 1","true value 1","prediction 2", "true value 2"])
    df.to_csv(join(networkpath,"example_predictions.csv"))
    # Plot single prediction
    plt.plot(np.squeeze(true_values[some_idx, :, :]))
    plt.plot(np.squeeze(single_pred))
    plt.legend(["true values", "predictions"])
    plt.show()
elif model_name == 'bayesian_lstm':
    single_pred, ci_upper, ci_lower = models.get_confidence_intervals(single_pred, 2)
    single_pred_2, ci_upper_2, ci_lower_2 = models.get_confidence_intervals(single_pred_2, 2)
    df = pd.DataFrame(np.array([np.squeeze(single_pred),np.squeeze(true_values[some_idx,:,:]),np.squeeze(ci_lower),np.squeeze(ci_upper),np.squeeze(single_pred_2), np.squeeze(true_values[some_idx_2,:,:]),np.squeeze(ci_lower_2),np.squeeze(ci_upper_2)]).T, columns=["prediction 1","true value 1","ci lower 1","ci upper 1","prediction 2", "true value 2","ci lower 2","ci upper 2"])
    df.to_csv(join(networkpath,"example_predictions.csv"))
    # Plot single prediction
    plt.plot(np.squeeze(true_values[some_idx, :, :]))
    plt.plot(np.squeeze(single_pred))
    plt.fill_between(x=np.arange(pred_period),
                     y1=np.squeeze(ci_upper),
                     y2=np.squeeze(ci_lower),
                     facecolor='green',
                     label="Confidence interval",
                     alpha=0.5)
    plt.legend(["true values", "predictions", "ci"])
    plt.show()
