#!/bin/bash
pwd
ls
echo $1 $2
# before you run this:
# ensure kaggle API key is stored as env variable:
# e.g. export KAGGLE_USERNAME=my_username && export KAGGLE_KEY=xxxxxxxxxxxxxx 
source get_dataset.sh

# install papermill to execute headless notebook whilst training
pip install papermill

cd prototype
touch $2
papermill --no-progress-bar --log-output $1 $2

cp $2 /artifacts/$2

echo 'done'
