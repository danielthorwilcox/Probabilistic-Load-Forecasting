#moving average predictor for energy data
#v0.1: Leo

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(df,target_index,reference_index):
    return {'mae' : mean_absolute_error(df[target_index], df[reference_index]),
            'rmse' : mean_squared_error(df[target_index], df[reference_index]) ** 0.5,
            'r2' : r2_score(df[target_index], df[reference_index])}

load_data = pd.read_csv("demand_generation/energy_dataset_lininterp.csv")
#print(load_data.tail())
load_data["SMA"] = load_data["total load actual"].rolling(12).mean() #moving average
load_data["SMA"] = load_data["SMA"].shift(1)
load_data["SMA"].fillna(0,inplace=True)
print(load_data.head())
load_data[["total load actual", "SMA"]].plot()
plt.show()
print(calculate_metrics(load_data,target_index="total load actual",reference_index="SMA"))