import pandas as pd
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
from tensorflow.keras.initializers import RandomNormal
import numpy as np
import sys
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import History 
history = History()

global function_false
function_false=0
def sample_function(X_train,X_test,y_train,y_test,layer_sizes,num_layer,batchsize,optimiser,loss_function,overtrain_true,dataset,arch,input_size,pool,number_of_test_examples, sigma_w):
    global function_false
    max_epochs=5000
    def binary_accuracy(y_true,y_pred):
        return keras.backend.mean(tf.cast(tf.equal(tf.math.sign(y_pred),y_true), tf.float32))

    class EarlyStoppingByAccVal(Callback):
        global function_false
        def __init__(self, monitor='accuracy', value=0.00001, verbose=0):
            global function_false
            super(Callback, self).__init__()
            self.monitor = monitor
            self.value = value
            self.verbose = verbose
            self.i = 0
        def on_epoch_end(self, epoch, logs={}):
            global function_false
            if overtrain_true!=0:
                current = logs.get(self.monitor)
                if current >= self.value:
                    if self.i==0:
                        function_false = np.argmax(model.predict(X_test))
                    self.i += 1
                if self.i > overtrain_true:
                    self.model.stop_training = True
            elif overtrain_true==0:
                current = logs.get(self.monitor)
                loss, acc = self.model.evaluate(X_train, y_train, verbose=0)
                if acc >= self.value:
                    if self.verbose > 10:
                        print("Epoch %05d: early stopping THR" % epoch)
                    self.model.stop_training = True

    model = Sequential()
    n = layer_sizes

    if arch=='FCN':
        RN = RandomNormal(mean=0.0, stddev=sigma_w/np.sqrt(input_size))
        RN2 = RandomNormal(mean=0.0, stddev=0.2)

        model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))
        model.add(Dense(n, activation='tanh', input_shape=(input_size,), kernel_initializer=RN, bias_initializer=RN2))
        for i in range(num_layer-1):
            RN = RandomNormal(mean=0.0, stddev=sigma_w/np.sqrt(n))
            RN2 = RandomNormal(mean=0.0, stddev=0.2)
            model.add(Dense(n, activation='tanh', kernel_initializer = RN, bias_initializer=RN2))
        if loss_function=='cross_entropy':
            RN = RandomNormal(mean=0.0, stddev=sigma_w/np.sqrt(n))
            RN2 = RandomNormal(mean=0.0, stddev=0.2)
            model.add(Dense(10, activation="softmax", kernel_initializer = RN, bias_initializer=RN2))
            if optimiser=="sgd":
                model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=1e-3), metrics=['accuracy'])
            else:
                model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
            stopper = EarlyStoppingByAccVal(monitor='accuracy', value=1.0, verbose=0)

        elif loss_function=='mse':
            model.add(Dense(1, activation="linear", kernel_initializer = RN, bias_initializer=RN))
            if optimiser=='sgd':
                model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(lr=1e-2), metrics=[binary_accuracy])
            else:
                model.compile(loss='mean_squared_error', optimizer=optimiser, metrics=[binary_accuracy])
            stopper = EarlyStoppingByAccVal(monitor='val_binary_accuracy', value=1.0, verbose=0)

        hist = model.fit(X_train, y_train, epochs=max_epochs, batch_size=batchsize, callbacks=[stopper, history], verbose=0)
        acc_hist = hist.history['accuracy']

        function = np.argmax(model.predict(X_test))
        _,train_eval=model.evaluate(X_train,y_train)
        _,test_eval=model.evaluate(X_test,y_test)
        model=0
        print('TEST EVAL: ',test_eval)

    if overtrain_true==0:
        return test_eval, acc_hist
    elif overtrain_true!=0:
        print('RETURNING')
        print(function_false)
        return test_eval, acc_hist


cifar10 = tf.keras.datasets.cifar10

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

x_train = X_train[0:50000,:,:,:]
x_test = X_test[0:10000,:,:,:]
y_train = Y_train[0:50000]
y_test = Y_test[0:10000]


from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
filename=rank

# Test Layers
layer_no = [int(sys.argv[1])]


#Test sigma_w
sw = np.linspace(1,2.5,31)

#test accuracy
test_acc = np.zeros((len(layer_no), len(sw)))

#Results
results = {}

#Epoch History
epoch_hist = []


for i in range(len(layer_no)):
    for j in range(len(sw)):

        test_acc[i,j] = sample_function(x_train, x_test, y_train, y_test, layer_sizes = 200, num_layer = layer_no[i], batchsize = 32, optimiser = "sgd", loss_function = 'cross_entropy', overtrain_true = 20 , dataset = 'CIFAR', arch = 'FCN', input_size = 3072, pool ='None', number_of_test_examples = 10000, sigma_w = sw[j])[0]

        results["CIFAR_bias"] = test_acc
        save_file_path = "New_CIFAR_Results/CIFAR_trainsize_50k_width_200_layer_" + str(sys.argv[1]) + "_run_" + str(filename)

        with open(save_file_path, 'wb') as f:
            pickle.dump(results, f)

        print(j)
