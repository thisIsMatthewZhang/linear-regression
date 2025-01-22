import pandas as pd
import numpy as np
from custom_linear_regression_implementation import LinearRegression

data = pd.read_csv('data_for_lr.csv')
data = data.dropna()

# training dataset and labels
train_input = np.array(data.x[:500]).reshape(500, 1)
train_output = np.array(data.y[:500]).reshape(500, 1)

# valid dataset and labels
test_input = np.array(data.x[500:700]).reshape(199, 1)
test_output = np.array(data.y[500:700]).reshape(199, 1)

