#v1.0: Leo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#see https://towardsdatascience.com/imputing-missing-data-with-simple-and-advanced-techniques-f5c7b157fb87 for more techniques

data = pd.read_csv("demand_generation/energy_dataset.csv")
#drop empty columns:
data.drop(["generation hydro pumped storage aggregated","forecast wind offshore eday ahead"],axis=1,inplace=True)
data.interpolate(method='linear', inplace=True)
print(data.isna().sum())

for column in data:
    if column == 'time':
        continue
    elif data[column].std() == 0:
        data.drop(columns=column, inplace=True) #if cloumn is constant, remove it (no info)
    else:
        col = data[column]
        data[column] = (col - col.mean()) / col.std() #norm to std-dev 1

for column in data:
    if column == 'time':
        continue
    assert 1.0001 > data[column].std() > 0.9999 #check if normation worked

data.to_csv("demand_generation/energy_dataset_lininterp.csv", index=False)
