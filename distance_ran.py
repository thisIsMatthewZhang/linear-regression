# fit linear regression model to dataset consisting of number of calories eaten (x) for each individual and the distance ran (y; in miles)

# import necessary libraries
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
def main():
    try:
        X = [[200], [250], [350], [500], [620], [890]]  # calories eaten by each individual
        y = [3, 3.5, 3.7, 4.5, 5.5, 6]  # miles ran

        # fit LR model to data
        model = LinearRegression()
        model.fit(X, y)

        print(f'Slope: {model.coef_}')
        print(f'y-intercept: {model.intercept_}')

        X_new = [[100], [200], [630], [700], [950]]
        y_pred = model.predict(X_new)
        print(f'Predicted distances ran: {y_pred}')

        plt.scatter(X, y, color='red')
        plt.plot(X, model.predict(X), color='blue')
        plt.show()

    except Exception as err:
        print(err)


if __name__ == '__main__':
    main()