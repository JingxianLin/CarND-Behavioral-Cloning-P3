"""
First preprocess the dataset, then train a model.
"""
import pickle
import math
import json
import os
import csv
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import pandas
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, ELU
from keras.activations import relu, softmax
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import backend as K

# Path to label
label_path = "data/driving_log.csv"

# Import the dataset
data = []
with open(label_path) as F:
	reader = csv.reader(F)
	for i in reader:
		data.append(i)
print("data loaded")

# Empty generators for features and labels
features = ()
labels = ()
ahead = ()
y_ahead = ()
left = ()
y_left = ()
right = ()
y_right = ()

# The helper function to resize images
def load_image(data_line, j):
	img = plt.imread(data_line[j].strip())[60:140,:,:]
	return img.flatten().tolist()

data = data[1:]
for i in tqdm(range(len(data)), unit='items'):
	steering = float(data[i][3])
	if steering < -0.05 or steering > 0.05:
		if -0.15 < steering < 0.15:
			ahead += (load_image(data[i], 0),)
			y_ahead += (steering,)
		elif steering < 0:
			left += (load_image(data[i], 0),)
			y_left += (steering,)
		else:
			right += (load_image(data[i], 0),)
			y_right += (steering,)

# Balance sample differences
a_size, l_size, r_size = len(ahead), len(left), len(right)
l_more, r_more = a_size - l_size, a_size - r_size
index_l, index_r = random.sample(range(len(data)), l_more), random.sample(range(len(data)), r_more)
for i in index_l:
	angle = float(data[i][3])
	if angle < 0:
		left += (load_image(data[i], 2),)
		y_left += (angle - 0.27,)
for i in index_r:
	angle = float(data[i][3])
	if angle > 0:
		right += (load_image(data[i], 1),)
		y_right += (angle + 0.27,)

# Combine three data
features = ahead + left + right
labels = y_ahead + y_left + y_right

item_size = len(features)
print("features size", item_size)
features = np.array(features).reshape(item_size, 80, 320, 3)
labels = np.array(labels)
print("features shape", features.shape, "labels size", labels.shape)

X_train = features
y_train = labels
from sklearn.cross_validation import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(
	X_train,
	y_train,
	test_size = 0.10,
	random_state = 832289)
item_num = len(X_valid)
print("validation size", item_num)
print("validation shape", X_valid.shape, "validation size", y_valid.shape)

train_size = X_train.shape[0]
valid_size = X_valid.shape[0]

input_shape = X_train.shape[1:]
features_count = input_shape[0]*input_shape[1]*input_shape[2]

print("train size", train_size)
print("valid size", valid_size)

print("input shape", input_shape)
print("features count", features_count)

# Save the dataset
pickle_file = 'camera.pickle'
try:
	with open(pickle_file, 'wb') as pfile:
		pickle.dump(
			{'train_dataset': X_train,
			 'train_labels': y_train,
			 'valid_dataset': X_valid,
			 'valid_labels': y_valid,
			}, pfile, pickle.HIGHEST_PROTOCOL)
except Exception as e:
	print('Not saving', pickle_file, ':', e)
	raise

print('Data saved')

# Reload the data
pickle_file = 'camera.pickle'
with open(pickle_file, 'rb') as f:
    pickle_data = pickle.load(f)
    X_train = pickle_data['train_dataset']
    y_train = pickle_data['train_labels']
    X_valid = pickle_data['valid_dataset']
    y_valid = pickle_data['valid_labels']
    del pickle_data  # Free up memory

# Print shapes of arrays that are imported
print('Data and modules loaded.')
print("train_features size:", X_train.shape)
print("train_labels size:", y_train.shape)
print("valid_features size:", X_valid.shape)
print("valid_labels size:", y_valid.shape)

# the data, shuffled and split between train and test sets
X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_train /= 255
X_valid /= 255
X_train -= 0.5
X_valid -= 0.5

# This is the shape of the image
input_shape = X_train.shape[1:]
print(input_shape, 'input shape')

# Set the parameters and print out the summary of the model
np.random.seed(1337)  # for reproducibility

batch_size = 128 # The lower the better
nb_classes = 1 # The output is a single digit: a steering angle
nb_epoch = 25 # The higher the better

# import model and wieghts if exists
try:
	with open('model.json', 'r') as jfile:
	    model = model_from_json(json.load(jfile))

	# Use adam and mean squared error for training
	model.compile("adam", "mse")

	# import weights
	model.load_weights('model.h5')

	print("Model and weights imported")

# If the model and weights do not exist, create a new model
except:
	# If model and weights do not exist in the local folder,
	# initiate a model

	# number of convolutional filters to use
	nb_filters1 = 24
	nb_filters2 = 36
	nb_filters3 = 48
	nb_filters4 = 64

	# size of pooling area for max pooling
	pool_size = (2, 2)

	# convolution kernel size
	kernel_size = (5, 5)
	kernel_numb = (3, 3)

	# Initiating the model
	model = Sequential()

	# Starting with the convolutional layer
	# The first layer will turn 3 channel into nb_filters1 channels
	model.add(Convolution2D(nb_filters1, kernel_size[0], kernel_size[1],
	                        border_mode='valid', subsample=(2,2), W_regularizer=l2(0.001),
	                        input_shape=input_shape))
	# Applying ReLU
	model.add(Activation('relu'))
	# The second conv layer will convert nb_filters1 channels into nb_filters2 channels
	model.add(Convolution2D(nb_filters2, kernel_size[0], kernel_size[1],
							border_mode='valid', subsample=(2,2), W_regularizer=l2(0.001)))
	# Applying ReLU
	model.add(Activation('relu'))
	# The third conv layer will convert nb_filters2 channels into nb_filters3 channels
	model.add(Convolution2D(nb_filters3, kernel_size[0], kernel_size[1],
							border_mode='valid', subsample=(2,2), W_regularizer=l2(0.001)))
	# Applying ReLU
	model.add(Activation('relu'))
	# The fourth conv layer will convert nb_filters3 channels into nb_filters4 channels
	model.add(Convolution2D(nb_filters4, kernel_numb[0], kernel_numb[1],
							border_mode='same', subsample=(2,2), W_regularizer=l2(0.001)))
	# Applying ReLU
	model.add(Activation('relu'))
	# The last conv layer
	model.add(Convolution2D(nb_filters4, kernel_numb[0], kernel_numb[1],
							border_mode='valid', subsample=(2,2), W_regularizer=l2(0.001)))
	# Applying ReLU
	model.add(Activation('relu'))

	# Flatten the matrix.
	model.add(Flatten())
	# Adding Dense
	model.add(Dense(80, W_regularizer=l2(0.001)))
	# Applying Dropout
	model.add(Dropout(0.5))
	# Input 80 Output 40
	model.add(Dense(40, W_regularizer=l2(0.001)))
	# Applying Dropout
	model.add(Dropout(0.5))
	# Input 40 Output 16
	model.add(Dense(16, W_regularizer=l2(0.001)))
	# Applying Dropout
	model.add(Dropout(0.5))
	# Adding Dense
	model.add(Dense(10, W_regularizer=l2(0.001)))
	# Input 10 Output 1
	model.add(Dense(nb_classes, W_regularizer=l2(0.001)))

# Print out summary of the model
model.summary()

# Compile model using Adam optimizer
# and loss computed by mean squared error

#optimizer = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=0.0001),
              metrics=['accuracy'])

### Model training
history = model.fit(X_train, y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_valid, y_valid))

# Save the model.
# If the model.json file already exists in the local file,
# warn the user to make sure if user wants to overwrite the model.
if 'model.json' in os.listdir():
	print("The file already exists")
	print("Want to overwite? y or n")
	user_input = input()

	if user_input == "y":
		# Save model as json file
		json_string = model.to_json()

		with open('model.json', 'w') as outfile:
			json.dump(json_string, outfile)

			# save weights
			model.save_weights('./model.h5')
			print("Overwrite Successful")
	else:
		print("the model is not saved")
else:
	# Save model as json file
	json_string = model.to_json()

	with open('model.json', 'w') as outfile:
		json.dump(json_string, outfile)

		# save weights
		model.save_weights('./model.h5')
		print("Saved")
