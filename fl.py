import torch
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import fl_models
import pandas as pd
from torchsummary import summary
from torch import nn, optim
from collections import OrderedDict
import pickle

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def getXypairs(data, train_period, pred_period):
    if 'time' in data:
        data.drop(columns='time', inplace=True)
    (n_observations, n_features) = data.shape # number of timestamps
    window_size = train_period + pred_period
    X = torch.zeros([n_observations - window_size, train_period, n_features])
    y = torch.zeros([n_observations - window_size, pred_period])

    for idx, x in enumerate(X):
        X[idx, :, :] = torch.tensor(data.iloc[idx:idx + train_period, :].to_numpy())
        y[idx, :] = torch.tensor(data['total load actual'].iloc[idx + train_period:idx + window_size].to_numpy())

    return X, y, n_observations, n_features

def load_data(trainperiod, predperiod, dataset):
    ##===========================================
    ## Loads the dataset and devides up the features
    ##===========================================
    train_period = trainperiod  # observation window size
    pred_period = predperiod  # prediction window size, should be 24 hours
    # data = pd.read_csv("demand_generation/energy_dataset_lininterp.csv")
    data = dataset
    X, y, n_observations, n_features = getXypairs(data, train_period=train_period, pred_period=pred_period)

    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, shuffle=False)

    batch_size = 128

    train_set = TensorDataset(X_train, y_train)
    val_set = TensorDataset(X_val, y_val)
    test_set = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader_one = DataLoader(test_set, batch_size=1, shuffle=False, drop_last=True)
    return train_loader, val_loader, test_loader, test_loader_one, n_features, n_observations

def create_model(dataset):
    ##===========================================
    ## Create the network model
    ##===========================================
    ## Network parameters
    train_period = 168  # observation window size
    pred_period = 24  # prediction window size, should be 24 hours
    train_loader, val_loader, test_loader, test_loader_one, n_features, n_observations = load_data(train_period, pred_period, dataset)
    input_dim = n_features
    output_dim = pred_period
    hidden_dim = 64
    layer_dim = 3
    batch_size = 128
    dropout = 0.2
    n_epochs = 2
    learning_rate = 0.006
    weight_decay = 1e-6
    model_params = {'input_dim': input_dim,
                    'hidden_dim': hidden_dim,
                    'layer_dim': layer_dim,
                    'output_dim': output_dim,
                    'dropout_prob': dropout}
    model = fl_models.get_model('lstm', model_params)
    return model


class Client:
    ##===========================================
    ## The different clients feeding parameters to the global model
    ##===========================================
    def __init__(self, client_id, dataset, epochs_per_client, learning_rate, batch_size, train_period = 168, pred_period = 24):
        self.client_id = client_id
        self.dataset = dataset   
        self.epochs = epochs_per_client 
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = 1e-6
        self.train_period = train_period
        self.pred_period = pred_period
        self.train_loader, self.val_loader, self.test_loader, self.test_loader_one, self.n_features, self.n_observations = load_data(self.train_period, self.pred_period, self.dataset)

    def train(self, global_parameters):
        ## Train the client model with the parameters from the global network
        model = to_device(create_model(self.dataset), device) # creates the model
        model.load_state_dict(global_parameters) # loads the parameters from the global network
        # model.apply_parameters(parameters_dict)
        print(self.client_id)
        loss_fn = nn.MSELoss(reduction="mean")
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        opt = fl_models.Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
        opt.train(self.train_loader, self.val_loader, batch_size=self.batch_size, n_epochs=self.epochs, n_features=self.n_features)
        # train_history = model.fit(self.dataset, self.epochs, self.learning_rate, self.batch_size)
        # print('{}: Loss = {}, Accuracy = {}'.format(self.client_id, round(train_history[-1][0], 4), round(train_history[-1][1], 4)))
        return model.get_parameters() 




def main():
    full_dataset = pd.read_csv("demand_generation/energy_dataset_lininterp.csv")
    # print(full_dataset)

    ## FL settings
    num_clients = 3
    epochs_per_client = 10
    learning_rate = 0.006
    batch_size = 128
    rounds = 3
    datasets = []
    for i in range(num_clients):
        datasets.append(full_dataset.copy()[int((i*full_dataset.shape[0])/num_clients) : int(((i+1)*full_dataset.shape[0])/num_clients)])
    # print(datasets)
    clients = [Client('client_' + str(i), datasets[i], epochs_per_client, learning_rate, batch_size) for i in range(num_clients)] ## creates the clients
    #Network parameters
    global_model = create_model(full_dataset)
    global_parameters = global_model.state_dict()
    global_model.to(device)
    weight_decay = 1e-6
    for i in range(rounds):
        print('Round: {}'.format(i+1))
        current_parameters = global_model.get_parameters() # takes the parameters from the original global model
        new_parameters = OrderedDict([(keys, torch.zeros(weight.size())) for keys, weight in current_parameters.items()]) ## creates new tensors of 0's to be aggregated by the client models
        for client in clients:
            client_parameters = client.train(current_parameters) ## training the client models
            new_parameters = OrderedDict([(key, new_parameters[key] + (client_parameters[key] / num_clients)) for key, values in new_parameters.items()]) ## adding the client models parameters to the new parameters that are going to be sent to the global model
        global_model.load_state_dict(new_parameters) ## loads the new parameters into the global model


    train_period = 168
    pred_period = 24
    train_loader, val_loader, test_loader, test_loader_one, n_features, n_observations = load_data(train_period, pred_period, full_dataset)
    input_dim = n_features
    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(global_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    opt = fl_models.Optimization(model=global_model, loss_fn=loss_fn, optimizer=optimizer)
    predictions, values = opt.evaluate(test_loader_one, batch_size=1, n_features=input_dim)
    with open('predictions.pickle', 'wb') as handle:
        pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('values.pickle', 'wb') as handle:
        pickle.dump(values, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Plot single prediction
    # plt.plot(values[13, :, :])
    # plt.plot(predictions[13, :, :])
    # # plt.plot(values)
    # # plt.plot(predictions[:][0])
    # plt.legend(["true values", "predictions"])
    # plt.show()



if __name__ == "__main__":
    main()