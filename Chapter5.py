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
from sklearn.svm import LinearSVC

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
