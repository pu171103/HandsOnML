#!/usr/bin/env python

# Title: Hands On ML: Chapter 7
# Description: Example code for Chapter 7 of Hands On Machine Learning
# Author: Pat Underwood
# Date: 11/25/2018

#%%
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

