#!/usr/bin/env python

# Title: Hands on ML: Chapter 4
# Description: Example code from Hands on Machine Learning
# Author: Pat Underwood
# Date: 11/24/2018

#%%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import (LinearRegression, SGDRegressor, 
        Ridge, Lasso, ElasticNet, LogisticRegression)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn import datasets

#%%
# OLS Regression
# Generate some Gaussian noise 
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Parameter estimation via OLS
X_b = np.c_[np.ones((100, 1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Predictions with Theta hat
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)
y_predict

plt.plot(X_new, y_predict, 'r-')
plt.plot(X, y, 'b.')
plt.axis([0, 2, 0, 15])
plt.show()

#%%
# OLS with SciKit Learn
lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_
lin_reg.predict(X_new)

# LinearRegression class inherits from numppy's lstsq class:
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
# This calculates Theta hat with the pseudoinverse of X
# This method is useful if the var/cov matrix is singular (or close to it)
# We can also do directly:
np.linalg.pinv(X_b).dot(y)

#%%
# Linear regression with gradient descent
# Batch
eta = 0.1
n_iterations = 1000
m = 100

theta = np.random.randn(2, 1) # random start

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

#%%
# Stochastic
n_epochs = 50
t0, t1 = 5, 50 # learning schedule hyperparameters

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2, 1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index + 1]
        yi = y[random_index:random_index + 1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients

#%%
# Stochastic gradient descent linear regression with SKLearn
sgd_reg = SGDRegressor(n_iter=50, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())
sgd_reg.intercept_, sgd_reg.coef_

#%%
# Polynomial regression
# Generate some data
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

# Generate x^2 as a separate variable
# NOTE: By default PolynomialFeatures() calculates all possible 
# degree combinations of the regressors you give it.
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X) #Returns [x, x^2])

# Fit the model
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_

#%%
# Plot learning curves to diagnose over/underfitting


def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
        plt.plot(np.sqrt(train_errors), 'r-+', linewidth=.75, label='train')
        plt.plot(np.sqrt(val_errors), 'b-', linewidth=.75, label='val')

lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)
plt.show()

polynomial_regression = Pipeline([
        ('poly_features', PolynomialFeatures(degree=10, include_bias=False)),
        ('lin_reg', LinearRegression()),
])
plot_learning_curves(polynomial_regression, X, y)
plt.show()

#%%
# Regularization
# Ridge Regression with SKLearn
ridge_reg = Ridge(alpha=1, solver='cholesky')
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])

sgd_reg = SGDRegressor(penalty='l2') #i.e. l2 norm
sgd_reg.fit(X, y.ravel())
sgd_reg.predict([[1.5]])

#%%
# LASSO Regression with SKLearn
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_reg.predict([[1.5]])

sgd_reg = SGDRegressor(penalty='l1')
sgd_reg.fit(X, y)
sgd_reg.predict([[1.5]])

#%%
# Elastic Net with SKLearn
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
elastic_net.predict([[1.5]])

#%%
# Early stopping
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
poly_scaler = Pipeline([
        ('poly_feature', PolynomialFeatures(degree=90, include_bias=False)),
        ('std_scaler', StandardScaler())
])
X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_val_poly_scaled = poly_scaler.transform(X_val)

# warm_start=True means the optimizer will begin with 
# parameter values reached in the previous iteration
sgd_reg = SGDRegressor(n_iter=1, warm_start=True, penalty=None,
        learning_rate='constant', eta0=0.0005)

minimum_val_error = float('inf')
best_epoch = None
best_model = None

for epoch in range(1000):
        sgd_reg.fit(X_train_poly_scaled, y_train.ravel())
        y_val_predict = sgd_reg.predict(X_val_poly_scaled)
        val_error = mean_squared_error(y_val, y_val_predict)
        if val_error < minimum_val_error:
                minimm_val_error = val_error
                best_epoch = epoch
                best_model = clone(sgd_reg)

#%%
# Logistic regression with SKLearn
iris = datasets.load_iris()
list(iris.keys())
X = iris['data'][:, 3:] # petal width
y = (iris['target'] == 2).astype(np.int) # 1 if == Virginica

log_reg = LogisticRegression()
log_reg.fit(X, y)

# np.linspace(min, max, n) generates n evenly spaced numbers between min and max
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
plt.plot(X_new, y_proba[:, 1], 'g-', label='Iris-Virginica')
plt.plot(X_new, y_proba[:, 0], 'b--', label='Other')

log_reg.predict([[1.7], [1.5]])

#%%
# Multinomial regression (Softmax) with SKLearn
X = iris['data'][:, (2, 3)]
y = iris['target']

softmax_reg = LogisticRegression(multi_class='multinomial', 
        solver='lbfgs', C=10) # C regulates the l2 penalty
softmax_reg.fit(X, y)
softmax_reg.predict([[5, 2]])
softmax_reg.predict_proba([[5, 2]])