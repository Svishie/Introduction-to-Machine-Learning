import os

import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import numpy as np
import matplotlib.pyplot as pyplot

#Suppress warnings from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

#Specify some variables
num_classes = 10

#These can, and should, be tuned
batch_size = 128
epochs = 3

# input image dimensions
img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#TODO
#Complete the implementation
#Use the model you implemented for the MNIST-dataset as inspiration
#This classifier will require more tuning, and maybe some adjustements to the architecture to achieve higher accuracy
