#!/usr/bin/env python

# Title: Hands On ML, Chapter 3
# Description: Examples and exercises from Chapter 3 of Hands-On Machine Learning
# Author: Patrick Underwood
# Date: 11/23/2018

# %%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (cross_val_predict, cross_val_score, 
    StratifiedKFold)
from sklearn.metrics import (precision_score, recall_score, f1_score, 
    confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score)
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Use helper function to download MNIST data
mnist = fetch_mldata('MNIST Original')

# Predictor matrix (pixels here) stored under 'data' key
# Outcome vector stored under 'target' key.
X, y = mnist['data'], mnist['target']

# Each IV is one pixel position, 28X28 pixels so 784 IVs total (70k images)
X.shape
y.shape

# Reshape one observation to 28x28 grid and visualize as bitmap
some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
           interpolation='nearest')
plt.axis('off')
plt.show()
y[36000]

# The outcome categories aren't distributed randomly in the dataset,
# so we need to randomize the row order in the training set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# %%
# Binary classification
# y=5 => 1/True; y!=5 => 0/False
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(random_state=42, max_iter=5, tol=None)
sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict([some_digit])

# %%
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

# %%
# Cross validation
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')

# Approx 10% of observations are 5s, so we need a better
# metric of accuracy than just PCP (we could get 90% accuracy
# by just guessing 'not-5' every time.)

# %%
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

# %%
# Plot precision and recall as a function of decision threshold
# Get decision scores to empirically estimate the decision function
#
# We can also use this function to set a decision threshold such that it
# yields a classifier of desired precision or recall

# Plotting precision against recall can give you a good sense of
# how high you can crank recall before precision takes a nosedive

y_scores = cross_val_predict(
    sgd_clf, X_train, y_train_5, method='decision_function', cv=3)
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

# Plot the empirical curves


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    """Plot empirical estimates of decision function.

    Arguments:
        precisions {nparray} -- Array of precisions generated from precision_recall_curve
        recalls {nparray} -- Array of recalls generated from precision_recall_curve()
        thresholds {nparray} -- Array of counterfactual decision thresholds.
    """
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.xlabel('Threshold')
    plt.legend(loc='center left')
    plt.ylim([0, 1])


plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

# Say we want a classifier with 90% precision
# Empirical precision curve hits 90% at threshold ~ 70k so
y_train_pred_90 = (y_scores > 70000)
precision_score(y_train_5, y_train_pred_90)
recall_score(y_train_5, y_train_pred_90)

# %%
# ROC Curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


plot_roc_curve(fpr, tpr)
plt.show()

# %%
# Area under the (ROC) curve
forest_clf = RandomForestClassifier(random_state=42)
# Get probability of classification in each category
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method='predict_proba')
# ROC needs binary outcomes, so set probability of being a 5 as correct outcome
y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(
    y_train_5, y_scores_forest)

plt.plot(fpr, tpr, 'r:', label='SGD')
plot_roc_curve(fpr_forest, tpr_forest, 'Random Forest')
plt.legend(loc='lower right')
plt.show()

#%%
# Multinomial classification
# SKLearn does OvA by default if you pass a multi class var to a binary classifier
sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])
some_digit_scores = sgd_clf.decision_function([some_digit])
np.argmax(some_digit_scores)
sgd_clf.classes_

# Can force OvO or OvA by importing OneVsOne(Rest)Classifier
# and pass a binary classifier to its constructor
ovo_clf = OneVsOneClassifier(SGDClassifier(
    random_state=42, max_iter=5, tol=None))
ovo_clf.fit(X_train, y_train)
ovo_clf.predict([some_digit])
len(ovo_clf.estimators_)

#%%
from sklearn.preprocessing import StandardScaler
# Multinomial random forest
forest_clf.fit(X_train, y_train)
forest_clf.predict([some_digit])

# Get class probabilities for a prediction
forest_clf.predict_proba([some_digit])

# Cross Validation
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring='accuracy')

# Can potentially boost performance by scaling
scaler=StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring='accuracy')

# Diagnostics
# Confusion matrix
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
# We can just treat the matrix as a bitmap and visualize it
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

#%%
# Transform confusion matrix to error matrix by dividing 
# off diagnoals by row sums (wrong classification / class total) 
# and zero fill main diagonal
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show() 

#%%
# Using KNN to do multilabel classification
# Make two different outcome classes for each case
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 ==1)
y_multilabel = np.c_[y_train_large, y_train_odd]
