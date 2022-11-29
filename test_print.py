import pickle
import matplotlib.pyplot as plt

with open('predictions.pickle', 'rb') as handle:
    predictions = pickle.load(handle)
with open('values.pickle', 'rb') as handle:
    values = pickle.load(handle)

predictions = predictions.reshape(predictions.shape[0], predictions.shape[2])
values = values.reshape(values.shape[0], values.shape[2])
print(predictions.shape)
print(values.shape)

plt.plot(predictions[1,:])
plt.plot(values[1,:])
plt.show()
# plt.show()