import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("demand_generation/energy_dataset.csv")
data.drop(["generation hydro pumped storage aggregated","forecast wind offshore eday ahead"],axis=1,inplace=True)
linear_interpolation = data.interpolate(method='linear')
print(linear_interpolation.isna().sum())
linear_interpolation.to_csv("demand_generation/energy_dataset_lininterp.csv")