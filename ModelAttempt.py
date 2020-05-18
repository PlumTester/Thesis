import numpy as np 
import tensorflow as tf
from matplotlib import pyplot as plt

from keras.models import Sequential
from keyas.layers import Dense, Dropout, Activation, Flatten, Merge
from keras.layers import Convolution2D, MaxPooling2D

from tf.keras.preprocessing import image_dataset_from_directory

from keras.utils import np_utils

# define parameters
filter_size = 8
max_pool_dim = (2, 2)
optimizer = 'Adadelta'
a_func = 'relu'
epochs = 200
batch_size = 32
dropout_rate =  0.3

# read data from graphs directory
# Up and Down directory should infer labels
# shuffle false because its time series
imgData, labelData = image_dataset_from_directory('Graphs_Normalised/', labels='inferred', label_mode='binary_crossentropy', color_mode = 'rgb', batch_size=batch_size, imagesize=(256,256), shuffle=False, validation_split=0.3, subset='training')

imgData.shape()

# define model architecture

branch_1 = Sequential()
branch_1.add(Convolution2D(32, 3, 3, activation=a_func, input_shape(3, 28, 28)))
branch_1.add(MaxPooling(pool_size=(2,2)))
branch_1.add(Dropout(dropout_rate))

branch_2 = Sequential()
branch_2.add(Convolution2D(32, 5, 5, activation=a_func, input_shape(3, 28, 28)))
branch_2.add(MaxPooling(pool_size=(2,2)))
branch_1.add(Dropout(dropout_rate))

model = Sequential()

#hidden layers
model.add(Merge([branch_1, branch_2], mode='concat'))
branch_1.add(Dropout(dropout_rate))

model.add(Convolution2D(32, 3, 3, activation=a_func, input_shape(3, 28, 28)))
model.add(MaxPooling(pool_size=(2,2)))
model.add(Dense(128, activation=a_func, branch_1.add(Dropout(dropout_rate))
#change 128 to be dimensionality of data ?

# output
model.add(Dense(1, activation='softmax'))

# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metriocs=['accuracy'])

# fit to data
model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=epochs, verbose=1)

# evaluation
score = model.evaluate(x_test, y_test, verbose=0)


