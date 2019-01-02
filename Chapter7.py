#!/usr/bin/env python

# Title: Hands On ML: Chapter 7
# Description: Example code for Chapter 7 of Hands On Machine Learning
# Author: Pat Underwood
# Date: 11/25/2018

#%%
from sklearn.ensemble import (
    RandomForestClassifier, VotingClassifier, BaggingClassifier,
    AdaBoostClassifier, GradientBoostingRegressor)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_moons, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error


#%% 
# Specifying a voting classifier with SKLearn
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()
X, y = make_moons(1000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='hard'
)
m0 = voting_clf.fit(X_train, y_train)

#%% 
# Check each classifier's accuracy
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    m0 = clf.fit(X_train, y_train)
    y_pred = m0.predict(X_test)
    print(m0.__class__.__name__, accuracy_score(y_test, y_pred))

#%%
# Bagging with SciKit Learn
# 500 decision trees (n_estimators) on 4 cores (n_jobs)
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators = 500, 
    max_samples = 100, bootstrap=True, n_jobs=4)
bag_fit = bag_clf.fit(X_train, y_train)
y_pred = bag_fit.predict(X_test)

#%%
# Cross validate on out of bag observations
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    bootstrap=True, n_jobs=4, oob_score=True
)
bag_fit = bag_clf.fit(X_train, y_train)
bag_fit.oob_score_
bag_fit.oob_decision_function_

#%%
# Specify a random forest with SciKit Learn
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=4)
rnd_fit = rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_fit.predict(X_test)

# Same model built with BaggingClassifier()
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(
        splitter='random', max_leaf_nodes=16
    ),
    n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=4
)

#%%
# Variable importance features
iris = load_iris()
rnd_clf =  RandomForestClassifier(n_estimators=500, n_jobs=4)
rnd_clf.fit(iris['data'], iris['target'])

for name, score in zip(iris['feature_names'], rnd_clf.feature_importances_):
    print(name, score)

#%%
# Boosting
# AdaBoost Method
# SAMME.R uses class probabilities, SAMME uses classifications
# Want to keep individual model complexity low to keep down variance
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), 
    n_estimators=200, algorithm='SAMME.R', learning_rate=0.5
)
ada_clf.fit(X_train, y_train)

# Gradient Boosting, Step by Step
# Train 3 models, m1, and m2 predicting previous model's errors
tree_reg1 = DecisionTreeRegressor(max_depth=2)
tree_reg1.fit(X_train, y_train)

y2 = y_train - tree_reg1.predict(X_train)
tree_reg2 = DecisionTreeRegressor(max_depth=2)
tree_reg2.fit(X_train, y2)

y3 = y2 - tree_reg2.predict(X_train)
tree_reg3 = DecisionTreeRegressor(max_depth=2)
tree_reg3.fit(X_train, y3)

y_pred = sum(tree.predict(X_test) for tree in 
    (tree_reg1, tree_reg2, tree_reg3)
)

# Same as above, but using SciKit Learn's GradientBOostingRegressor()
gbrt = GradientBoostingRegressor(
    max_depth=2, n_estimators=3, learning_rate=1.0
)
gbrt.fit(X_train, y_train)

# Lowering learning rate can allow the ensemble to more 
# methodically approach a good fit, but too many trees 
# tend to lead to overfitting, so we usually employ early stopping. 
X_train, X_test, y_train, y_test = train_test_split(X, y)
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120)
gbrt.fit(X_train, y_train)
y_pred = gbrt.predict(X_test)
errors = [mean_squared_error(y_test, y_pred)]