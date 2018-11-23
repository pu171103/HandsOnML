#!/usr/bin/env python

# Title: Hands On ML, Chapter 3
# Description: Examples and exercises from Chapter 3 of Hands-On Machine Learning
# Author: Patrick Underwood
# Date: 11/23/2018

#%%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.base import clone
from scipy.io.matlab import loadmat

# Use helper function to download MNIST data
mnist = fetch_mldata('MNIST Original')

# Read from disk (Matlab format)
mnist = loadmat('.\\datasets\\mnist\\mnist-original.mat')

# Predictor matrix (pixels here) stored under 'data' key
# Outcome vector stored under 'label' key ('target' if using helper function)
# Need to transpose covariate arrays to conform to book's example code.
X, y = mnist['data'].T, mnist['label'].T

# Each IV is one pixel position, 28X28 pixels so 784 IVs total (70k images)
X.shape
y.shape

# Reshape one observation to 28x28 grid and visualize as bitmap
some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')
plt.axis('off')
plt.show()
y[36000]

# The outcome categories aren't distributed randomly in the dataset, 
# so we need to randomize the row order in the training set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

#%%
# Binary classification
# y=5 => 1/True; y!=5 => 0/False
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# Convert outcome vector to a 1D array to work with SGDClassifier
y_train_5 = y_train_5.ravel()

sgd_clf = SGDClassifier()
sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict([some_digit])

#%%
# Aside example: custom cross validation
# Use StratifiedKFold() to generate the validation sets, then loop
# over each pair and get pct correctly predicted
# SKlearn provides clone() to copy model objects
skfolds = StratifiedKFold(n_splits=3, random_state=42)
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))

#%%
# Cross validation
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')

# Approx 10% of observations are 5s, so we need a better 
# metric of accuracy than just PCP (we could get 90% accuracy
# by just guessing 'not-5' every time.)

#%%
# Confusion matrix and related metrics
# cross_val_predict() will return 
# each observation's estimate from when it was in a test fold
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
confusion_matrix(y_train_5, y_train_pred)
precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)
f1_score(y_train_5, y_train_pred)

# See the (continuous) outcome the SGD classifier uses to assign (binary) predictions
y_scores = sgd_clf.decision_function(X)
# SGDClassifier() restricts decision threshold = 0 so:
threshold = 0
y_scores > threshold 
