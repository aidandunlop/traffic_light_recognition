#!/bin/bash

# if you are wanting to download the dataset from kaggle:
# ensure kaggle API key is stored as env variable:
# e.g. export KAGGLE_USERNAME=my_username && export KAGGLE_KEY=xxxxxxxxxxxxxx 

set -e

# check here if KAGGLE envs provided
if [[ "$1" == "--download" && (-z $KAGGLE_USERNAME || -z $KAGGLE_KEY )]]; then
    printf 'The LISA Dataset is hosted on Kaggle. Please ensure KAGGLE_USERNAME and KAGGLE_KEY env variables are set.\nSee https://www.kaggle.com/docs/api#authentication for more details.\n'
    exit;
fi

# # install dependencies
printf 'Installing dependencies...\n'
pip install -r requirements.txt

# download dataset from kaggle if needed
if [[ "$1" == "--download" || ! -d "lisa-traffic-light-dataset" ]]
then
    printf 'Downloading dataset from kaggle... (4GB, may take a while)'
    kaggle datasets download -d mbornoe/lisa-traffic-light-dataset
    printf 'Unzipping dataset...\n'
    unzip -qq lisa-traffic-light-dataset -d lisa-traffic-light-dataset
    printf 'Deleting zip file...\n'
    rm lisa-traffic-light-dataset.zip
    # convert dataset to easier format for training
    printf 'Merging dataset...\n'
    python traffic_lights/data/merge_dataset.py -d lisa-traffic-light-dataset
fi


printf 'Everything set up. You are now ready to begin training or inference.\n'
printf 'To train, use `python traffic_lights train`\n'
printf 'To predict, use `python traffic_lights predict`\n'