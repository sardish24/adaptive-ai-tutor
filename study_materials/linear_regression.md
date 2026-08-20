# Linear Regression Study Notes

## 1. Overview
Linear regression is a foundational supervised learning algorithm used to model the relationship between a dependent variable ($Y$) and one or more independent variables ($X$).

## 2. Mathematical Formulation
The simple linear regression equation is given by:
$$Y = \beta_0 + \beta_1 X + \epsilon$$

Where:
- $\beta_0$ is the y-intercept (the expected value of $Y$ when $X=0$).
- $\beta_1$ is the slope coefficient (the change in $Y$ per unit change in $X$).
- $\epsilon$ is the error term / residual.

## 3. Cost Function & Optimization
We define the Ordinary Least Squares (OLS) Mean Squared Error (MSE) loss function:
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

Gradient Descent updates parameter weights iteratively:
$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$$

## 4. Key Assumptions
1. **Linearity**: The relationship between $X$ and the mean of $Y$ is linear.
2. **Homoscedasticity**: The variance of residual errors is constant across all levels of the independent variable.
3. **Independence**: Observations and residuals are independent.
4. **Normality**: For any fixed value of $X$, $Y$ is normally distributed around the regression line.
