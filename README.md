# linear-regression
Linear Regression is a statistical algorithm used to model the relationship between a dependent and one or more independent variables. It is also a supervised machine learning algorithm that learns from labeled datasets and maps the data points with the most optimized linear functions. It models the linear relationship between a dependent variable and one or more independent variable by fitting a linear equation with observed data. 

Simplicity is one of the greatest strengths of linear regression-it displays clear coefficients which illustrate the influence of each independent variable on the dependent variable. 

Our primary objective with linear regression is to locate the line of best fit, which implies that the error between predicted and actual values should be kept to a minimum. 

<img width="568" alt="Screenshot 2025-01-20 at 9 34 42 PM" src="https://github.com/user-attachments/assets/274fc61a-df7a-493f-a73d-19c5b326b7ed" />

Here Y is called a dependent or target variable and X is the independent or predictor of Y. X may be a single or multiple features. 

Hypothesis function in Linear Regression
Assumptions:
  - Linearity: It assumes that there is a linear relationship between the independent and dependent variables. This means that changes in the independent variable lead to proportional changes in the dependent variable.
  - Independence: The observations should be independent from each other that is the errors from one observation should not influence other.
<img width="648" alt="Screenshot 2025-01-20 at 10 37 19 PM" src="https://github.com/user-attachments/assets/8169ceed-0f02-4a82-bece-23dba2545f42" />

How to update θ1 and θ2 values to get the best-fit line? 
  - To achieve the best-fit regression line, the model aims to predict the target value such that the error difference between the predicted value and the true value is minimum. 

Types of Linear Regression
  1)  Simple Linear Regression: involves only one independent and dependent variable. Equation is of the from y=β0 + β1X
      - Assumptions: Linear, Independence, Homoscedasticity (constant variance of error across all levels of independent variable(s)), Normality
  2) Multiple Linear Regression: more than one independent variable and one dependent variable. Equation is of the form y = β0 + β1X1 + β2X2 + ……… βnXn
      - Assumptions: No multicollinearity (check correlation matrix and Variance Inflation Factor), Additivity (no interaction between variables in their effects on the dependent variable), Feature Selection, Overfitting

Cost function for Linear Regression: difference between the predicted value and the true value.
  - It is not easy to get the line of best fit in real life scenarios, so we need a way to calculate the errors that affect it.
  - In linear regression, the Mean Squared Error (MSE) is the most commonly used cost function. It calculates the average of the squared errors between the predicted values and actual values.
    <img width="312" alt="Screenshot 2025-01-21 at 1 12 29 PM" src="https://github.com/user-attachments/assets/4051bcf4-d505-42dd-a350-18c3d3e6b0c1" />
  - Utilizing MSE, gradient descent is applied to update θ1 and θ2. This ensures that the MSE value converges to a global minima, signifying the most accurate fit of the linear regression line to the dataset. This process involves continuously adjusting the parameters θ1 and θ2 based on the gradients calculated from MSE. The end result is a regression line that minimizes the error between predicted and actual values.
  - A gradient is a derivative that defines the effects on outputs of the function given minor variations in inputs.
  - By moving in the direction of the MSE negative gradient w.r.t. the coefficients, the coefficients can be changed.
<img width="414" alt="Screenshot 2025-01-21 at 1 34 29 PM" src="https://github.com/user-attachments/assets/aedc04d0-447b-442d-b07d-576f37b339e2" />

Evaluation Metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Coefficient of Determination (R-squared), Adjusted R-squared Error

Regularization Techniques for Linear Models
  1) Lasso Regression (L1 Regularization): adds a penalty term to the linear regression objective function to prevent overfitting
  2) Ridge Regression (L2 Regularization): adds a regularization term ti the standard linear objective. The goal is once again to prevent overfitting by penalizing large coefficients in the equation. It is useful when the dataset has multicollinearity.
  3) Elastic Net Regression: hybrid technique that combines L1 and L2 in linear regression objective.  








