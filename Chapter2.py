#!/usr/bin/env python

# Title: Hands-On ML, Chapter 2
# Description: Examples from Hands-On Machine Leaning, Chapter 2
# Author: Pat Underwood
# Date: 11/15/2018

#%%
import os
import tarfile
import pandas as pd
import matplotlib.pyplot as plt
from six.moves import urllib

# Grab example CA housing data
download_root = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
housing_path  = os.path.join('datasets', 'housing')
housing_url   = download_root + "datasets/housing/housing.tgz"

#%%
# Shell controll with os package
# Create a directory
os.makedirs(housing_path)

# Pull tarball from remote and unpack
tgz_path = os.path.join(housing_path, "housing.tgz")
urllib.request.urlretrieve(housing_url, tgz_path)
housing_tgz = tarfile.open(name=tgz_path)
housing_tgz.extractall(path=housing_path)
housing_tgz.close()
os.remove('.\\datasets\\housing\\housing.tgz')

#%%
# Load data into Pandas dataframe
csv_path = os.path.join(housing_path, "housing.csv")
housing = pd.read_csv(csv_path)
housing.head()
housing.info()

# Distinct counts of a categorical var
housing['ocean_proximity'].value_counts()

# Get summary statistics
housing.describe()

# Easy histogram matrix
housing.hist(bins=50, figsize=(20,15))
plt.show()

# Test/Train split function

