#!/usr/bin/env python

# Title: Hands-On ML, Chapter 2
# Description: Examples from Hands-On Machine Leaning, Chapter 2
# Author: Pat Underwood
# Date: 11/15/2018

# %%
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

# Distinct counts of a categorical var
housing['ocean_proximity'].value_counts()

# Get summary statistics
housing.describe()

# Easy histogram matrix
housing.hist(bins=50, figsize=(20, 15))
plt.show()

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
