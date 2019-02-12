#Based on example-code provided by keras

import os

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import numpy as np
import matplotlib.pyplot as pyplot

#Suppress warnings from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

#Specify some variables
batch_size = 128
num_classes = 10
epochs = 3

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
# this is something one typically would have to handle oneself, but keras does this for us in this case
# if you have to do it, scikit-learn has a variety of functions that does it for you
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# one hot encoding
# 1 => 0 1 0 0 0 0 0 0 0 0
# 4 => 0 0 0 0 1 0 0 0 0 0
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Example-structure:
# Conv - pooling - conv - pooling - dense - dropout - dense
# Remember to specify the input- and output-shape
# Also remember to flatten the output of the last convolutional layer before feeding it to the dense layers

# Some tips:
# Frequently used number of kernels: 16, 32, 64, 96
# Normal kernel-sizes: (3, 3), (5, 5), (7, 7)

model = Sequential()
# TODO:
# Complete the model architecture
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation="relu",
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.summary()

# TODO:
# Train the model and save the training-history in a variable called history
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
          
x_axis = list(range(1, len(history.history['acc']) + 1))
pyplot.plot(x_axis, history.history['acc'], x_axis, history.history['val_acc'])
pyplot.legend(("Training accuracy", "Validation accuracy"))
pyplot.show()
