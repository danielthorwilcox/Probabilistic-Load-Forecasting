import torch
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import models
import pandas as pd
from torchsummary import summary
from torch import nn, optim
from collections import OrderedDict
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import Utils

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def load_data(parameters, dataset):
    ##===========================================
    ## Loads the dataset and devides up the features
    ##===========================================
    train_period = parameters['train_period']
    pred_period = parameters['pred_period']
    data = dataset
    X, y, n_observations, n_features = Utils.getXypairs(data, train_period=train_period, pred_period=pred_period)

    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, shuffle=False)

    batch_size = parameters['batch_size']

    train_set = TensorDataset(X_train, y_train)
    val_set = TensorDataset(X_val, y_val)
    test_set = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader_one = DataLoader(test_set, batch_size=1, shuffle=True, drop_last=True)
    return train_loader, val_loader, test_loader, test_loader_one, n_features, n_observations

def create_model(dataset, parameters, type_of_model):
    ##===========================================
    ## Create the network model
    ##===========================================
    ## Network parameterss
    input_dim = dataset.shape[1]
    output_dim = parameters['pred_period'] # prediction window size
    model_params = {'input_dim': input_dim,
                    'hidden_dim': parameters['hidden_dim'],
                    'layer_dim': parameters['layer_dim'],
                    'output_dim': output_dim,
                    'dropout_prob': parameters['dropout'],
                    'n_fc_layers': parameters['n_fc_layers']}
    model = models.get_model(type_of_model, model_params)
    return model


class Client:
    ##===========================================
    ## The different clients feeding parameters to the global model
    ##===========================================
    def __init__(self, client_id, dataset, epochs_per_client, client_parameters, type_of_model):
        self.parameters = client_parameters
        self.client_id = client_id
        self.dataset = dataset   
        self.epochs = epochs_per_client 
        self.type_of_model = type_of_model
        self.learning_rate = self.parameters['learning_rate']
        self.batch_size = self.parameters['batch_size']
        self.weight_decay = self.parameters['weight_decay']
        self.train_period = self.parameters['train_period']
        self.pred_period = self.parameters['pred_period']
        self.train_loader, self.val_loader, self.test_loader, self.test_loader_one, self.n_features, self.n_observations = load_data(self.parameters, self.dataset)

    def train(self, global_parameters):
        ## Train the client model with the parameters from the global network
        model = to_device(create_model(self.dataset, self.parameters, self.type_of_model), device) # creates the model
        model.load_state_dict(global_parameters) # loads the parameters from the global network
        print(self.client_id)
        loss_fn = nn.MSELoss(reduction="mean")
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        opt = models.Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
        opt.train(self.train_loader, self.val_loader, batch_size=self.batch_size, n_epochs=self.epochs, n_features=self.n_features)
        return model.get_parameters()


def main():
    start = time.time()
    full_dataset = pd.read_csv("demand_generation/energy_dataset_lininterp.csv")
    params = Utils.get_model_params()
    type_of_model = params['model'] # 'bayesian_lstm' or 'lstm'
    ##=========================================================================
    # Parameters for the clients/agents
    parameters = {
        'train_period': params['train_period'],
        'pred_period': params['pred_period'],
        'hidden_dim': params['hidden_dim'],
        'layer_dim': params['layer_dim'],
        'dropout': params['dropout'],
        'learning_rate': params['learning_rate'],
        'weight_decay': params['weight_decay'],
        'batch_size': params['batch_size'], #keep this the same for both client and global
        'n_fc_layers': params['n_fc_layers']
    }

    # train_loader, val_loader, test_loader, test_loader_one, n_features, n_observations = load_data(global_parameters, full_dataset)
    ## FL settings
    num_clients = 3
    epochs_per_client = 5
    rounds = 5
    test_dataset = full_dataset.copy()[int(full_dataset.shape[0]*0.7):int(full_dataset.shape[0])]
    train_dataset = full_dataset.copy()[int(full_dataset.shape[0]*0):int(full_dataset.shape[0]*0.7)]
    # print(train_dataset.shape)
    # train_loader, val_loader, test_loader, test_loader_one, n_features, n_observations = load_data(global_parameters, test_dataset)

    # Settings above this line are to be changed
    #==================================================================
    # Splitting the dataset for the clients
    datasets = []
    for i in range(num_clients):
        datasets.append(test_dataset.copy()[int((i*test_dataset.shape[0])/num_clients) : int(((i+1)*test_dataset.shape[0])/num_clients)])
    clients = [Client('client_' + str(i), datasets[i], epochs_per_client, parameters, type_of_model) for i in range(num_clients)] ## creates the clients
    # ##==================================================================

    # ##==================================================================
    # # Creating the global network
    train_loader, val_loader, test_loader, test_loader_one, n_features, n_observations = load_data(parameters, test_dataset)
    global_model = create_model(test_dataset, parameters, type_of_model)
    global_model.to(device)

    ##====================================================================
    # Training the client models and updating parameters
    for i in range(rounds):
        print('Round: {}'.format(i+1))
        current_parameters = global_model.get_parameters() # takes the parameters from the original global model
        new_parameters = OrderedDict([(keys, torch.zeros(weight.size())) for keys, weight in current_parameters.items()]) ## creates new tensors of 0's to be aggregated by the client models
        for client in clients:
            client_parameters = client.train(current_parameters) ## training the client models
            new_parameters = OrderedDict([(key, new_parameters[key] + (client_parameters[key] / num_clients)) for key, values in new_parameters.items()]) ## adding the client models parameters to the new parameters that are going to be sent to the global model
        global_model.load_state_dict(new_parameters) ## loads the new parameters into the global model
    midtime = time.time()
    print('Time for training: '+ str(midtime-start))


    ##====================================================================
    # Global model evaluation
    input_dim = n_features
    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(global_model.parameters(), parameters['learning_rate'], weight_decay=parameters['weight_decay'])
    opt = models.Optimization(model=global_model, loss_fn=loss_fn, optimizer=optimizer)
    predictions, true_values = opt.evaluate(test_loader_one, type_of_model, batch_size=1, n_features=input_dim)
    predictions_mean = np.mean(predictions, axis=1)
    mse = mean_squared_error(predictions_mean.flatten(), true_values.flatten())
    mae = mean_absolute_error(predictions_mean.flatten(), true_values.flatten())
    r2 = r2_score(predictions_mean.flatten(), true_values.flatten())
    ## =============================================
    ## Save values to a file
    with open('bayseian_predictions_1.pickle', 'wb') as handle:
        pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('bayesian_values_1.pickle', 'wb') as handle:
        pickle.dump(true_values, handle, protocol=pickle.HIGHEST_PROTOCOL)
    ## ===================================================================
    end = time.time()
    print('Time for evaluate: '+ str(end-midtime))
    print('Total time running: '+ str(end-start))



    ##===============================
    pred_period = parameters['pred_period']
    some_idx = 13
    single_pred = predictions[some_idx, :, :]

    if type_of_model == 'lstm':
        # Plot single prediction
        plt.plot(np.squeeze(true_values[some_idx, :, :]))
        plt.plot(np.squeeze(single_pred))
        plt.legend(["true values", "predictions"])
        plt.show()
    elif type_of_model == 'bayesian_lstm':
        single_pred, ci_upper, ci_lower = models.get_confidence_intervals(single_pred, 2)
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
    print(predictions.shape)


if __name__ == "__main__":
    main()