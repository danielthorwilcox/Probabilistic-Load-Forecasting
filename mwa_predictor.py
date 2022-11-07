#moving average predictor for energy data
#v0.1: Leo

import pandas as pd
import matplotlib.pyplot as plt

load_data = pd.read_csv("demand_generation/energy_dataset_lininterp.csv")
#print(load_data.tail())
load_data["SMA"] = load_data["total load actual"].rolling(12).mean()
print(load_data.tail())
load_data[["total load actual", "SMA"]].plot()
plt.show()