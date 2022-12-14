import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import models

with open('./results/fl_network41/fl_predictions.pickle', 'rb') as handle:
    predictions = pickle.load(handle)
with open('./results/fl_network41/fl_values.pickle', 'rb') as handle:
    true_values = pickle.load(handle)


# predictions_mean = np.mean(predictions, axis=1)
# mse = mean_squared_error(predictions_mean.flatten(), values.flatten())
# # print(predictions.shape)
# # print(values.shape)

# predictions = predictions.reshape(predictions.shape[0], predictions.shape[2])
# values = values.reshape(values.shape[0], values.shape[2])
# print(mse)

# plt.plot(predictions[130,:])
# plt.plot(values[130,:])
# plt.show()

predictions_mean = np.mean(predictions, axis=1)
mse = mean_squared_error(predictions_mean.flatten(), true_values.flatten())
mae = mean_absolute_error(predictions_mean.flatten(), true_values.flatten())
r2 = r2_score(predictions_mean.flatten(), true_values.flatten())
print(mse)
# print(predictions)


pred_period = 24
some_idx = 13
single_pred = predictions[some_idx, :, :]
print(predictions.shape)

single_pred, ci_upper, ci_lower = models.get_confidence_intervals(single_pred, 2)
# print(predictions[some_idx, 0, :])
# print('---------------')
# print(true_values[some_idx, 0, :])
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