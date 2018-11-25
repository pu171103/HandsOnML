#!/usr/bin/env python

# Title: Hands On ML Chapter 5
# Description: Example code for chapter 5 of Hands On Machine Learning
# Author: Pat Underwood
# Date: 11/25/2018

#%%
import numpy as np
from sklearn import datasets
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVC, SVC

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

