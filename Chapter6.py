#!/usr/bin/env python

# Title: Hands On ML Chapter 6
# Description: Example code for Chapter 6 of Hands On Machine Learning
# Author: Pat Underwood
# Date: 11/25/2018

#%%
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# Specifying a decision tree with SKLearn
iris = load_iris()
X = iris['data'][:, 2:]
y = iris['target']

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

#%%
# Export tree to a format GraphViz can ingest
export_graphviz(
    tree_clf, 
    out_file='iris_tree.dot',
    feature_names=iris['feature_names'][2:],
    class_names=iris['target_names'],
    rounded=True,
    filled=True
)