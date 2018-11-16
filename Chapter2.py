#!/usr/bin/env python

# Title: Hands-On ML, Chapter 2
# Description: Examples from Hands-On Machine Leaning, Chapter 2
# Author: Pat Underwood
# Date: 11/15/2018

import os
import tarfile
from six.moves import urllib

# Grab example CA housing data
download_root = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
housing_path  = os.path.join('datasets', 'housing')

# Shell controll with os package
# Create a directory
os.makedirs(housing_path)

# Pull tarball from remote and unpack
tgz_path = os.path.join(housing_path, "housing.tgz")
