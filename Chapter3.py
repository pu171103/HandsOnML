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


