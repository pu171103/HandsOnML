#!/usr/bin/env python

# Title: Hands-On ML, Chapter 2
# Description: Examples from Hands-On Machine Leaning, Chapter 2
# Author: Pat Underwood
# Date: 11/15/2018

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
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
housing['population_per_household'] = housing['population'] / \
    housing['households']
housing.head()
corr_matrix = housing.corr()
corr_matrix

# Variable transforms
# %%
# Get copy of only IVs (.drop() creates copy by default)
housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()

# %%
# Clean missing
# Row-wise deletion
# housing.dropna(subset=['total_bedrooms'], axis=1)
# Variable deletion
# housing.drop('total_bedrooms', axis=1)
# Median replacement
# housing['total_bedrooms'].fillna(housing['total_bedrooms'].medain, inPlace=True)

# Median replacement with SciKit Learn
imputer = Imputer(strategy='median')

# Cat vars don't have medians
housing_num = housing.drop('ocean_proximity', axis=1)
# Initializer a (median replacement) imputer
imputer.fit(housing_num)
imputer.statistics_
housing_num.median().values
# Call the imputer
# Note: fit_transform() combines these steps
X = imputer.transform(housing_num)
# Imputer's output is a numpy array so:
housing_tr = pd.DataFrame(X, columns=housing_num.columns)

# %%
# Convert cat var to (numeric) factor
housing_cat = housing['ocean_proximity']
housing_cat_encoded, housing_categories = housing_cat.factorize()
housing_cat_encoded[:10]
housing_categories

# %%
# Dummy coding with SciKit Learn
# SKLearn calls dummy coding 'one hot encoding'
encoder = OneHotEncoder()
# Reshape because OneHotEncoder() takes a 2-D array
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
# Returns a sparse array
housing_cat_1hot

# %%
# Creating custom SKLearn transformers
# Requires fit(), transform(), and fit_transform()
# Derrive from BaseEstimator to get useful get/set params methods
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """Custom attribute addition transformer."""

    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        """Implementation of transformer fit() method.

        Arguments:
            X {dataframe} -- A Pandas dataframe

        Keyword Arguments:
            y {array} -- Optional outcome vector (default: {None})
        """
        return self  # Nothing else to do

    def transform(self, X, y=None):
        """Implementation of transformer transform() method.
        Returns values from input dataframe.

        Arguments:
            X {dataframe} -- A Pandas dataframe
        """
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

# %%
# Scikit Learn provides a convenient Pipeline class for, well, a pipeline

# Initialize the pipeline
num_pipeline = Pipeline([
    ('imputer', Imputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),  # Z-score scaling
])

# Call the pipeline on the data
# All steps except last must have tranform() methods
housing_num_tr = num_pipeline.fit_transform(housing_num)

# %%
# Need to write our own transformer to send Pandas dataframe through a pipeline


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """Custom transformer to work with Pandas dataframes 
    in SKLearn pipelines.
    Keep desired variables and pass along as array."""

    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        """Implementation of transformer fit() method.

        Arguments:
            X {dataframe} -- A Pandas dataframe

        Keyword Arguments:
            y {array} -- Optional outcome vector (default: {None})
        """
        return self

    def transform(self, X):
        """Implementation of transformer transform() method.
        Returns values from input dataframe.

        Arguments:
            X {dataframe} -- A Pandas dataframe
        """
        return X[self.attribute_names].values


#%%
# Run multiple pipelines (concurrently) and concatenate results
from sklearn.pipeline import FeatureUnion

num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('cat_encoder', OneHotEncoder(housing_cat_encoded, sparse=False))
])

full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline),
])

# Now call the unified pipeline
housing_prepared = full_pipeline.fit_transform(housing)
