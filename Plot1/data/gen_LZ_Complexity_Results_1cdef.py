#!/usr/bin/env python
import os, sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import math
import operator
import random
import scipy
import os
from ..complexities import K_lemp_ziv
from tensorflow.keras import layers
from scipy.special import softmax
from tensorflow.keras import backend as K
from collections import OrderedDict as OD
from collections import Counter
from tqdm import tqdm as tqdm
from itertools import product
from itertools import repeat
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend
from tensorflow.keras.losses import binary_crossentropy
from tqdm import tqdm
import pandas as pd
import pickle

# Define Step function which converts output to a binary string
def step(x):
    if x <= 0.5:
        return 0
    else:
        return 1
step_vec = np.vectorize(step)

# Function to convert binary array in to the decimal number it represents

def decimal(x):
    n = len(x)
    output = 0
    for i in range(len(x)):
        output += x[i]*2**(n-1-i)
    return output

def binary(x):
    output = np.zeros(7)
    b = str(int(bin(int(x))[2:]))
    c = []
    n = len(b)
    for digit in b:
        c.append(int(digit))
    for i in range(n):
        output[7-n+i] = np.array(c)[i]
    return output

# Generate all points of the n dimensional boolean hypercube as inputs
n=7
from itertools import product
inputs = np.array(list(product([0, 1], repeat=n)))
np.random.shuffle(inputs)
split_x = np.split(inputs,2)


def sampling(m):
     # Defina training and testing inputs as 7 bit stings
     train_x = split_x[0]
     test_x = split_x[1]

     data_y = np.ones((128,2))
     data_y[:,0] =0

     for i in random.sample(range(128), m):
        data_y[i,:] = [1,0]

     split_y = np.split(data_y,2)
     train_y = split_y[0]
     test_y = split_y[1]

     # Dictionary holding decimal numbers and their associated binary arrays
     train_dict = {}
     for i in range(64):
        train_dict[decimal(train_x[i])] = train_y[i]

     #Define error dictionary used to record error on training elements
     error_dict = {}
     for i in range(64):
        error_dict[decimal(train_x[i])] = [i]


     return train_x, train_y, test_x, test_y, train_dict, error_dict, data_y[:,0]



# Define Neural Network
def NeuralNet(weight_sd, bias_sd, n, active_func, m, outformat = 'gen_bias'):

    # Define the width specification of the network
    widthspec = [7,40,40,2]


    # Set up a fully connected (dense) neural network with width specification
    model = keras.Sequential()
    model.add(keras.layers.Dense(widthspec[1], input_dim=n, activation = active_func))

    for i in range(2,len(widthspec)-1):
        model.add(keras.layers.Dense(widthspec[i], activation = active_func))


    model.add(keras.layers.Dense(widthspec[-1], activation = "softmax"))

    # Initialise the weights and biases of the neural network
    for i in range(len(widthspec)-1):
        w = np.random.normal(loc = 0.0, scale = 1.0, size =[widthspec[i], widthspec[i+1]])*(weight_sd/np.sqrt(widthspec[i]))
        b = np.random.normal(loc = 0.0, scale = 1.0, size =[widthspec[i+1],])*bias_sd
        model.layers[i].set_weights([w, b])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    input_data = sampling(m)
    train_x = input_data[0]
    train_y = input_data[1]
    test_x = input_data[2]
    test_y = input_data[3]
    train_dict = input_data[4]
    error_dict = input_data[5]
    inputs = input_data[6]

    input_complexity = K_lemp_ziv(inputs)

    # For first mini batch we just randomly sample 10 elements from the training set
    mini_batch_x = np.zeros((10,7))
    mini_batch_y = np.zeros((10,2))
    sample = random.sample(range(64), 10)
    for i in range(10):
        mini_batch_x[i] = train_x[sample[i]]
        mini_batch_y[i] = train_y[sample[i]]

    error_array = np.ones((64,4))*0.5
    counter_array = np.zeros(64)
    weight_std = np.zeros(1000)
    train_epoch = 0
    n_epochs = 100
    weight_before = np.array(model.layers[1].get_weights()[0])
    train_err=0
    for epoch in range(4000):
        train_err = float(model.evaluate(train_x, train_y, batch_size=32, verbose=0)[1])

        # Traing the model over n epochs
        if train_err==1:
            break

        train_epoch += 1

        layer_weights_std = np.std(np.array(model.layers[1].get_weights()[0])) * np.sqrt(40)
        model.train_on_batch(mini_batch_x, mini_batch_y)
        # Updating the errors of the individual elements within the mini batch
        model.predict(mini_batch_x)
        err = backend.eval(binary_crossentropy([mini_batch_y], [model.predict(mini_batch_x)]))

        for j in range(len(mini_batch_x)):

            error_array[error_dict[decimal(mini_batch_x[j])], int(counter_array[error_dict[decimal(mini_batch_x[j])]])] = err[0][j]
            error_array[error_dict[decimal(mini_batch_x[j])], 3] = np.sum(error_array[error_dict[decimal(mini_batch_x[j])], 0:3])
            counter_array[error_dict[decimal(mini_batch_x[j])]] = (counter_array[error_dict[decimal(mini_batch_x[j])]] + 1) % 3

        # Using updated errors, choose a new mini batch
        errors = error_array[:,3]
        keys = np.fromiter(error_dict.keys(), dtype=float)
        softmax_err = scipy.special.softmax(errors)
        batch = np.random.choice(keys, size = 10, replace= False, p=softmax_err)

        for i in range(len(batch)):
            mini_batch_x[i] = binary(batch[i])

        for i in range(len(batch)):
            mini_batch_y[i] = train_dict[batch[i]]


    weight_after = np.array(model.layers[1].get_weights()[0])
    weight_distance = weight_after - weight_before
    output_complexity = K_lemp_ziv(np.concatenate([step_vec(model.predict(train_x))[:,0],step_vec(model.predict(test_x))[:,0]]))
    training = model.evaluate(train_x,  train_y, batch_size=64, verbose=0)
    testing = model.evaluate(test_x,  test_y, batch_size=64, verbose=0)
    generalisation_error = training[1] - testing[1]

    if outformat == 'gen_bias':
       return generalisation_error, input_complexity
    elif outformat == 'weight_std':
       return weight_std
    elif outformat == 'lz_complexity':
       return input_complexity, output_complexity
    elif outformat == 'weight_distance':
        return weight_distance
    else:
        raise NotImplementedError('unknown outformat')

its = 10000

gen_error_tanh8 = np.zeros((its,2))
gen_error_tanh1 = np.zeros((its,2))


results = {}

for j in range(its):
    print(j, 'of ', its)
    gen_error_tanh1[j] = NeuralNet(0.1, 0.0, 7, 'tanh', np.random.randint(0,128), outformat = 'gen_bias')
    gen_error_tanh8[j] = NeuralNet(8.0, 0.0, 7, 'tanh', np.random.randint(0,128), outformat = 'gen_bias')

    results["tanh8"] = gen_error_tanh8
    results["tanh1"] = gen_error_tanh1

    save_file_path = "./lz_results/"

    with open(save_file_path, 'wb') as f:
        pickle.dump(results, f)
