# fit linear regression model to dataset consisting of the age of random individuals (X) and their corresponding income in $1,000's (y)

# import necessary libraries
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
def main():
    try:
        X = [[25], [30], [35], [40], [45], [50], [55]]  # age of individuals
        y = [70000, 75000, 85000, 80000, 80000, 90000, 100000]  # income for each corresponding individual

        # fit a linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # print the coefficients
        print(f'Slope: {model.coef_}')  # slope of the line
        print(f'y-intercept: {model.intercept_}')  # y-intercept of the line

        # make predictions
        X_new = [[28], [33], [35], [60]]  # age of new individual
        y_pred = model.predict(X_new)  # predicted income of new individual
        print(f'predicted income: {y_pred}')

        # plot the data and the line of best fit
        plt.scatter(X, y, color='red')
        plt.plot(X, model.predict(X), color='blue')
        plt.show()

    except Exception as err:
        print(err)


if __name__ == "__main__":
    main()