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

Cost function for Linear Regression

    

