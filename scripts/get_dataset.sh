#!/bin/bash

# before you run this:
# ensure kaggle API key is stored as env variable:
# e.g. export KAGGLE_USERNAME=my_username && export KAGGLE_KEY=xxxxxxxxxxxxxx 

# download dataset from kaggle
pip install kaggle
kaggle datasets download -d mbornoe/lisa-traffic-light-dataset

# unzip dataset
unzip -qq lisa-traffic-light-dataset -d lisa-traffic-light-dataset

# delete zip
rm lisa-traffic-light-dataset.zip

# convert dataset to easier format for training
python merge_dataset.py -d lisa-traffic-light-dataset

# get ax for hyperparameter tuning
pip install ax-platform