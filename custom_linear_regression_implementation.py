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
        """
        In forward propagation, linear regression function y = mx + c is applied by randomly
        assigning the values of parameters m and c
        """
        m = self.parameters['m']
        c = self.parameters['c']
        predictions = np.multiply(m, train_input) + c  # here, x is 'train_input'
        return predictions

    def cost_function(self, predictions, train_output):
        cost = np.mean((predictions-train_output)**2)
        return cost

    def backward_propagation(self, train_input, train_output, predictions):
        derivatives = {}
        df = predictions-train_output
        # dm = 2/n * mean of (predictions - actual) * input
        dm = 2 * np.mean(np.multiply(train_input, df))
        # dc = 2/n * mean of (predictions-actual)
        dc = 2 * np.mean(df)
        derivatives['dm'] = dm
        derivatives['dc'] = dc
        return derivatives


    def update_parameters(self, derivatives, learning_rate=0.0001):
        self.parameters['m'] = self.parameters['m'] - learning_rate * derivatives['dm']
        self.parameters['c'] = self.parameters['c'] - learning_rate * derivatives['dc']


    def train(self, train_input, train_output, learning_rate=0.0001, iters=20):
        # initialize random parameters
        self.parameters['m'] = np.random.uniform(0, 1) * -1
        self.parameters['c'] = np.random.uniform(0, 1) * -1

        # initialize loss
        self.loss = []

        # initialize figure and axis for animation
        fig, ax = plt.subplots()
        x_vals = np.linspace(min(train_input), max(train_input), 100)
        line, = ax.plot(x_vals, self.parameters['m'] * x_vals + self.parameters['c'], color='red', label='Regression Line')
        ax.scatter(train_input, train_output, marker='o', color='green', label='Training Data')

        # Set y-axis limits to exclude negative values
        ax.set_ylim(0, max(train_output) + 1)

        def update(frame):
            # Forward propagation
            predictions = self.forward_propagation(train_input)
            # Cost
            cost = self.cost_function(predictions, train_output)
            # Back propagation
            derivatives = self.backward_propagation(train_input, train_output, predictions)
            # update parameters
            self.update_parameters(derivatives, learning_rate)
            # update regression line
            line.set_ydata(self.parameters['m'] * x_vals + self.parameters['c'])
            # append loss and print
            self.loss.append(cost)
            print(f'Iteration = {frame + 1}, Loss = {cost}')

            return line,

        # Create animation
        anim = FuncAnimation(fig, update, frames=iters, interval=200, blit=True)

        # Save the animation as a video file
        anim.save('linear_regression_A.gif', writer='ffmpeg')

        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.title('Linear Regression')
        plt.legend()
        plt.show()

        return self.parameters, self.loss

