import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import *

n_samples = 35000 #number of hours to generate
T = 24 #period time in hours
sigma = 0.02 #noise power

X = np.arange(n_samples)
Y = np.sin((2*pi/T)*X)
n = np.random.normal(0,sigma,n_samples)
Y = Y + n
#plt.plot(X,Y)
#plt.show()

df = pd.DataFrame(Y)
df.to_csv("sinedata.csv")