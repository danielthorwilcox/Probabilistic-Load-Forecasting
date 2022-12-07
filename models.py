import torch
from torch import nn
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join
from blitz.modules import BayesianLSTM, BayesianLinear
from blitz.utils import variational_estimator


def get_model(model, model_params):
    models = {
        "lstm": LSTMModel,
        "bayesian_lstm": BayesianLSTMModel,
    }
    return models.get(model.lower())(**model_params)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, n_fc_layers=1, dropout_prob=0):
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_fc_layers-1) if n_fc_layers > 1])
        self.out_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        for hidden_layer in self.fc:
            out = hidden_layer(out)
        out = self.out_layer(out)

        return out


@variational_estimator
class BayesianLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, n_fc_layers, dropout_prob=0):
        super(BayesianLSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.first_lstm = BayesianLSTM(input_dim, hidden_dim)
        self.lstms = nn.ModuleList([BayesianLSTM(hidden_dim, hidden_dim) for _ in range(layer_dim-1) if layer_dim > 1])

        # Fully connected layer
        self.fc = nn.ModuleList([BayesianLinear(hidden_dim, hidden_dim) for _ in range(n_fc_layers-1) if n_fc_layers > 1])
        self.out_layer = BayesianLinear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.input_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.input_dim, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.first_lstm(x)  # , (h0.detach(), c0.detach()))
        for lstm in self.lstms:
            out, (hn, cn) = lstm(out)

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        for hidden_layer in self.fc:
            out = hidden_layer(out)
        out = self.out_layer(out)

        return out


class Optimization:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

    def train_step(self, x, y):
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x)

        # Computes loss
        loss = self.loss_fn(y, yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def train(self, train_loader, val_loader, batch_size=64, n_epochs=50, n_features=1):
        model_path = f'models/{self.model}_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features])
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features])
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 50) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )

#        torch.save(self.model.state_dict(), model_path)

    def evaluate(self, test_loader, model_name, batch_size=1, n_features=1, n_samples=10):
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features])
                self.model.eval()
                values.append(y_test.detach().numpy())
                if model_name == 'lstm':
                    yhat = self.model(x_test)
                    predictions.append(yhat.detach().numpy())
                elif model_name == 'bayesian_lstm':
                    yhat = [self.model(x_test).detach().numpy() for _ in range(n_samples)]
                    predictions.append(yhat)

        return np.array(predictions), np.array(values)

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()

    def save_losses(self,filepath):
        epochs = range(1,len(self.train_losses)+1)
        df = pd.DataFrame(np.array([epochs,self.train_losses,self.val_losses]).T, columns=["epoch","training loss","validation loss"])
        df.to_csv(join(filepath,"losses.csv"))


def get_confidence_intervals(preds_test, ci_multiplier):
    # global scaler

    preds_test = torch.tensor(preds_test)

    pred_mean = preds_test.mean(dim=0).clone().detach()
    pred_std = preds_test.std(dim=0).clone().detach()

    # pred_std = torch.tensor(pred_std).clone().detach()

    upper_bound = pred_mean + (pred_std * ci_multiplier)
    lower_bound = pred_mean - (pred_std * ci_multiplier)
    # gather unscaled confidence intervals

    pred_mean_unscaled = pred_mean.unsqueeze(1).detach().cpu().numpy()
    # pred_mean_unscaled = scaler.inverse_transform(pred_mean_final)

    upper_bound_unscaled = upper_bound.unsqueeze(1).detach().cpu().numpy()
    # upper_bound_unscaled = scaler.inverse_transform(upper_bound_unscaled)

    lower_bound_unscaled = lower_bound.unsqueeze(1).detach().cpu().numpy()
    # lower_bound_unscaled = scaler.inverse_transform(lower_bound_unscaled)

    return pred_mean_unscaled, upper_bound_unscaled, lower_bound_unscaled
