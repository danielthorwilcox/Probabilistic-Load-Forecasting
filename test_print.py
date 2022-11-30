import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

with open('predictions.pickle', 'rb') as handle:
    predictions = pickle.load(handle)
with open('values.pickle', 'rb') as handle:
    values = pickle.load(handle)


predictions_mean = np.mean(predictions, axis=1)
mse = mean_squared_error(predictions_mean.flatten(), values.flatten())
# print(predictions.shape)
# print(values.shape)
predictions = predictions.reshape(predictions.shape[0], predictions.shape[2])
values = values.reshape(values.shape[0], values.shape[2])
print(mse)

plt.plot(predictions[130,:])
plt.plot(values[130,:])
plt.show()