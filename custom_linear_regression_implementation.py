# this module contains a re-implementation of the Linear Regression class found in scikit-learn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from matplotlib.animation import FuncAnimation

class LinearRegression:
    def __init__(self):
        self.parameters = {}

    def forward_propagation(self, train_input):
        pass

    def cost_function(self, predictions, train_output):
        pass

    def backward_propagation(self, train_input, train_output, predictions):
        pass

    def update_parameters(self, derivatives, learning_rate=0.0001):
        pass

    def train(self, train_input, train_output, learning_rate=0.0001, iters=20):
        pass

        def update():
            pass


