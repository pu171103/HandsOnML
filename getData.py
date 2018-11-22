#!/usr/bin/env python

# Title: Data collection
# Description: Download the datasets utilized in 'Hands On ML'
# Author: Patrick Underwood
# Date: 11/22/2018

import requests
import os


def GetData(url, save_dir, file_name):
    """Convenience function to download data from URLs

    Arguments:
        url {string} -- A fully qualified URL
        save_dir {string} -- Path to the save directory
        file_name {string} -- FIle name, with extension
    """

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    full_path = os.path.join(save_dir, file_name)
    r = requests.get(url, allow_redirects=True)
    open(full_path, 'wb').write(r.content)

    print('File written to', full_path)
    r.close()

    return None


# Globals
root_dir = 'C:\\Users\\Pat\\PythonProjects\\Tutorials\\HandsOnML\\datasets\\'

# Inception data
url = 'https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/inception/imagenet_class_names.txt'
save_dir = os.path.join(root_dir, 'inception')
file_name = 'imagenet_class_names.txt'
GetData(url, save_dir, file_name)

# Life satisfaction data
# GDP percapita set
url = 'https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/lifesat/gdp_per_capita.csv'
save_dir = os.path.join(root_dir, 'lifesat')
file_name = 'gdp_per_capita.csv'
GetData(url, save_dir, file_name)

# OECD data
url = 'https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/lifesat/oecd_bli_2015.csv'
file_name = 'oecd_blie_2015.csv'
GetData(url, save_dir, file_name)

# Readme
url = 'https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/lifesat/README.md'
file_name = 'README.md'
GetData(url, save_dir, file_name)
