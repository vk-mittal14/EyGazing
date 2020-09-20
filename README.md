# EyMazing: Predicting future gaze locations
2nd Place Solution for [Facebook OpenEDS Gaze Prediction Challenge](https://research.fb.com/programs/openeds-2020-challenge/) by Vivek Mittal

![model](Drawing.png "Model Overview")


## About
In this challenge, our goal was to understand the **spatial-temporal** movement of eye gaze direction. Specifically, we need to predict the gaze direction for future time-steps based on the previously estimated gaze vectors.  To this end, we use a two-stage approach in which we use: a ResNet18 backbone for predicting the gaze direction for an image and an LSTM model to capture the temporal dependency between the gaze direction and predict the future gaze direction.

To know more about the challenge visit:
* Workshop: https://openeyes-workshop.github.io/ 
* Facebook Webpage: https://research.fb.com/programs/openeds-2020-challenge/
