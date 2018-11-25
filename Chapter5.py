#!/usr/bin/env python

# Title: Hands On ML Chapter 5
# Description: Example code for chapter 5 of Hands On Machine Learning
# Author: Pat Underwood
# Date: 11/25/2018

#%%
import numpy as np
from numpy.random import randint
from sklearn import datasets
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR

#%%
# Soft margin SVM with SKLearn
iris = datasets.load_iris()
X = iris['data'][:, (2, 3)]
y = (iris['target'] == 2).astype(np.float64)

svm_clf = Pipeline([
    ('Scaler', StandardScaler()),
    ('linear_svc', LinearSVC(C=1, loss='hinge')),
])
svm_clf.fit(X, y)
svm_clf.predict([[5.5, 1.7]])

#%%
# SVMs with nonlinear covariates
# SKLearn's make_moons() is a neat little function 
# that generates useful synthetic data for clustering and classification
polynomial_svm_clf = Pipeline([
    ('poly_features', PolynomialFeatures(degree=3)),
    ('scaler', StandardScaler()),
    ('svm_clf', LinearSVC(C=10, loss='hinge'))
])
X, y = make_moons()
polynomial_svm_clf.fit(X, y)
polynomial_svm_clf.predict(X[:10, :])

#%%
# The Kernel Trick
poly_kernel_svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm_clf', SVC(kernel='poly', degree=3, coef0=1, C=5))
])
X, y = make_moons()
poly_kernel_svm_clf.fit(X, y)

#%% 
# The 'landmark trick,' via the kernel trick
# The Gaussian Radial Basis Function provides a metric of distance of a 
# data point from some reference point in the parameter space.
# We will use that as a mapping function with the kernel trick.
rbf_kernel_svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm_clf', SVC(kernel='rbf', gamma=5, C=0.001))
])
rbf_kernel_svm_clf.fit(X, y)

#%%
# Support Vector Regression
# Sort of the 'opposite' of SVM- find a margin that contains most data points
X = randint(0, 100, 10000).reshape(-1, 1)
y = 25 + X**2
svm_reg = LinearSVR(epsilon=1.5)
svm_reg.fit(X, y)
svm_reg.predict(X[:10, :])

svm_poly_reg = SVR(kernel='poly', degree=2, C=100, epsilon=0.1)
svm_poly_reg.fit(X, y)
