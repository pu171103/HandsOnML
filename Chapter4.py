#!/usr/bin/env python

# Title: Hands on ML: Chapter 4
# Description: Example code from Hands on Machine Learning
# Author: Pat Underwood
# Date: 11/24/2018

#%%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor

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