#**Behavioral Cloning** 

This is a wonderful project for me to work on preprocessing the dataset, constructing a network, then training the model and testing it.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py training the network
* drive.py for driving the car in autonomous mode
* model.h5 containing weights
* model.json saving model architecture
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network inspired by NVIDIA's paper (model.py lines 191-260)

It is also end to end learning with raw image as input and steering angle for output.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 238-247).

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model was compiled using adam optimizer, so the learning rate was not tuned manually (model.py line 262), and loss was computed by mean squared error.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road to augment the dataset.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to make the network more scalable and perform faster.

My first step was to use a convolution neural network model similar to the commaai one, but there are many parameters and weights are too large to be scalable. The NVIDIA based model was tried because weights are smaller, easier to train.

Then I modified the model so that 5 convolutional layers, 3 dropout layers, and 4 dense layers are included.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle was close to the edge of the track, but it recovered quickly. To improve the driving behavior in these cases, I increased the steering angle correction for side camera images.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 191-260) consisted of a convolution neural network with the following layers and layer sizes:

Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
convolution2d_1 (Convolution2D)  (None, 38, 158, 24)   1824        convolution2d_input_1[0][0]
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 38, 158, 24)   0           convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 17, 77, 36)    21636       activation_1[0][0]
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 17, 77, 36)    0           convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 7, 37, 48)     43248       activation_2[0][0]
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 7, 37, 48)     0           convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 4, 19, 64)     27712       activation_3[0][0]
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 4, 19, 64)     0           convolution2d_4[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 9, 64)      36928       activation_4[0][0]
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 1, 9, 64)      0           convolution2d_5[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 576)           0           activation_5[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 80)            46160       flatten_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 80)            0           dense_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 40)            3240        dropout_1[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 40)            0           dense_2[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 16)            656         dropout_2[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 16)            0           dense_3[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            170         dropout_3[0][0]
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             11          dense_4[0][0]
====================================================================================================

####3. Creation of the Training Set & Training Process

Using just Udacity's data, rejecting samples with steering angle less than 0.05 can remove data with small angle, then dividing the rest into 3 lists: going left <= -0.15 < going ahead < 0.15 <= going right.

To balance data, find sample differences between going ahead and going left, going ahead and going right, then generate random indices for side camera images: for negative angle, add right images and minus steering correction; for positive angle, add left images and plus steering correction.

After the collection process, I had 3558 number of data points. I then preprocessed this data by cropping out the sky and car deck.

I finally randomly shuffled the data set and put 10% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 25 as evidenced by successful laps around track one without leaving the road in most cases. I used an adam optimizer so that manually training the learning rate wasn't necessary.
