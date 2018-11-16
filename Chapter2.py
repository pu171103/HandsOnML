#!/usr/bin/env python

# Title: Hands-On ML, Chapter 2
# Description: Examples from Hands-On Machine Leaning, Chapter 2
# Author: Pat Underwood
# Date: 11/15/2018

# %%
from pandas.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import os
import tarfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from six.moves import urllib
from zlib import crc32

# Grab example CA housing data
download_root = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
housing_path = os.path.join('datasets', 'housing')
housing_url = download_root + "datasets/housing/housing.tgz"

# %%
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


# %%
# Load data into Pandas dataframe
csv_path = os.path.join(housing_path, "housing.csv")
housing = pd.read_csv(csv_path)
housing.head()
housing.info()

# %%
# Distinct counts of a categorical var
housing['ocean_proximity'].value_counts()

# %%
# Get summary statistics
housing.describe()

# %%
# Easy histogram matrix
housing.hist(bins=50, figsize=(20, 15))


# %%
# Test/Train split function


def split_train_test(data, test_ratio):
    """Split data into training and test sets.

    Arguments:
        data {dataframe} -- Data in Pandas dataframe.
        test_ratio {float} -- Proportion of data for test set.
    """

    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]


train_set, test_set = split_train_test(housing, 0.2)

# One way to pick and retrieve the same random rows for a test set
# Hash some ID and select rows where hash < 20% max hash value


def test_set_check(identifier, test_ratio):
    """Select rows with hash of ID column less than test 
    ratio proportion of max hash value

    Arguments:
        identifier {array} -- A column of a Pandas data frame
        test_ratio {float} -- Proportion of data for test set
    """
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32


def split_train_test_by_id(data, test_ratio, id_column):
    """Get test set by hashing ID values.

    Arguments:
        data {dataframe} -- A Pandas dataframe to be split
        test_ratio {float} -- Proportion of data for test set
        id_column {column} -- Pandas dataframe column to be hashed
    """
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# Designate an ID column
# We can just use the row index
housing_with_id = housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'index')

# Scikit-Learn provides its own splitting function
# Param random_state sets the seed
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# %%
# Convert median income to a (new) categorical variable
housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)  # New var
# Return true when income_cat < 5, 5.0 with false, modify DF in place
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)
housing['income_cat'].hist()

# Do stratified sampling with SciKit Learn
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Drop the categorical variable, but keep the stratified splits
for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)

# %%
# Exploratory Visualization
hosuing = strat_train_set.copy()  # New object, not a pointer
housing.plot(kind='scatter', x='longitude', y='latitude')
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=.1)

# %%
# Params: s-> size dimension, c-> color dimension
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=.1,
             s=housing['population'] / 100,
             label='Population', figsize=(10, 7),
             c='median_house_value',
             cmap=plt.get_cmap('jet'), colorbar=True)
plt.legend()  # Initialize legend constructor

# %%
# Pearson's R
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)

# %%
# Scatterplot matrix function from Pandas
attributes = ['median_house_value', 'median_income',
              'total_rooms', 'housing_median_age']
scatter_matrix(housing[attributes], figsize=(12, 8))

# %%
# Derrived variables
housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms'] / \
    housing['total_rooms']
hosuing['population_per_household'] = housing['population'] / \
    housing['households']
#%%
housing.head()
