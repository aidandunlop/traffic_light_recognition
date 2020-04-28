# traffic-light-recognition

With the potential to improve road safety, reduce congestion and free up time for human drivers, the impact that the self driving car might have is huge, and needs to be taken seriously. 
[McKinsey & Co. consulting firm reported](https://www.wsj.com/articles/self-driving-cars-could-cut-down-on-accidents-study-says-1425567905) that widespread embrace of self-driving vehicles could eliminate 90% of all auto accidents in the U.S.

In order for self driving cars to become a reality, the car needs to be able to identify it’s environment in order to navigate around it safely. 
One clear example of this is the detection and classification of traffic light signals. 
The car will need to obey the signals, and act accordingly based on the state of the traffic light. 
A system will have to be developed that, based on the cameras or sensors on the car, determines whether there are any traffic lights near by and what to do when approaching them.

In this project, I’m going to attempt to use Deep Learning techniques, such as Convolutional Neural Networks, on the [LISA traffic light dataset](https://www.kaggle.com/mbornoe/lisa-traffic-light-dataset). 
This is a supervised classification problem - first I will need to determine whether or not a traffic light is in a frame, and then determine what state a traffic light is in within that frame. 
The dataset has labelled data which can be used to learn a model of traffic light images: each frame in the dataset has the pixel coordinates of each visible traffic light, and the state of the traffic light (e.g. “Stop”). 

To download the dataset, ensure you have pip and kaggle installed (and authenticated) and run `./download_dataset.sh`. 

## Data exploration
I've started exploring the LISA dataset, and you can view my adventures in [`Data_Exploration.ipynb`](https://github.com/aidandunlop/traffic-lights/blob/master/Data_Exploration.ipynb). 
It's also on [Kaggle](https://www.kaggle.com/aidandunlop/capstone-project-data-exploration).
