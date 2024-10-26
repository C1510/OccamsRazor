import pandas as pd
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import tqdm
from tqdm import tqdm as tqdm
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
from tensorflow.keras.initializers import RandomNormal
import numpy as np
import sys
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

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
                        print(function_false)
                    self.i += 1
                if self.i > overtrain_true:
                    print(epoch)
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

        model.add(keras.layers.Flatten(input_shape=(28, 28)))
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

        print(model.summary())

        model.fit(X_train, y_train, epochs=max_epochs, batch_size=batchsize, callbacks=[stopper], verbose=0)

        function = np.argmax(model.predict(X_test))
        _,train_eval=model.evaluate(X_train,y_train)
        _,test_eval=model.evaluate(X_test,y_test)
        print('TEST EVAL: ',test_eval)

    def preprocess(image):
        image = tf.cast(image, tf.float32)
        return image

    loss_object = tf.keras.losses.CategoricalCrossentropy()

    def create_adversarial_pattern(input_image, input_label):
        with tf.GradientTape() as tape:
            tape.watch(input_image)
            prediction = model(input_image)
            loss = loss_object(input_label, prediction)

        gradient = tape.gradient(loss, input_image)

        signed_grad = tf.sign(gradient)
        return signed_grad

    def crit_samp_ratio(train_ex, train_ex_label, model, alpha = 0.5/255, beta = 0.5/255, r = 0.01):

        def hamming_distance(a, b):
            r = (1 << np.arange(8))[:,None]
            return np.count_nonzero( (a & r) != (b & r) )

        x = train_ex
        x_tilda = x
        x_hat = 0

        label = labels[train_ex_label]
        image = preprocess(train_ex.reshape(1,28,28))
        perturbations = create_adversarial_pattern(image, label)

        n_iter = 2
        iter_counter = 0

        converged = False
        while (converged == False) and (iter_counter < n_iter):

            iter_counter += 1

            delta = alpha*perturbations[0] + beta*np.random.normal(loc = 0.0, scale = 1.0, size =[28, 28])
            x_tilda = np.array(x_tilda + delta)

            for i in range(28):
                for j in range(28):
                    if abs(255*x[i,j] - 255*x_tilda[i,j]) > 1:
                        x_tilda[i,j] = x[i,j] + r*np.sign(x_tilda[i,j] - x[i,j])
                    else:
                        x_tilda[i,j] = x_tilda[i,j]

                    if x_tilda[i,j] > 255:
                        x_tilda[i,j] = 255
                    elif x_tilda[i,j] < 0:
                        x_tilda[i,j] = 0

            if np.argmax(model.predict(image)) != np.argmax(model.predict(tf.reshape(x_tilda, (1,28,28)))):
                converged = True
                x_hat = 1
        return x_hat

    crit_ratio = 0

    if overtrain_true==0:
        return test_eval
    elif overtrain_true!=0:
        print('RETURNING')
        print(function_false)
        return test_eval, crit_ratio

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[0:1000,:,:]
x_test = x_test[0:1000,:,:]
y_train = y_train[0:1000]
y_test = y_test[0:1000]

c=0

np.random.seed(10)
for i in range(int(len(y_train)*c)):
    y_train[i] = np.random.randint(0,10)

labels = tf.one_hot(y_test.reshape(1000,1), 10)

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
filename=rank

n_sample = 100

sw = np.linspace(1,2.5,31)

test_acc = {2:np.zeros((31,2)),5:np.zeros((31,2)),10:np.zeros((31,2))}

results = {2:{},5:{},10:{}}
n_sample = 1
for i in range(n_sample):
    for count, sw_ in enumerate(sw):
        for nl in [2,5,10]:
            test_acc[nl][count] = sample_function(x_train, x_test, y_train, y_test, layer_sizes = 200, num_layer = nl, batchsize = 32, optimiser = "sgd", loss_function = 'cross_entropy', overtrain_true = 0 , dataset = 'MNIST', arch = 'FCN', input_size = 784, pool ='None', number_of_test_examples = 10000, sigma_w = sw_)
            results[nl]["csr"] = test_acc
            save_file_path = f"Acc_CSR_Results/CSR_MNIST_1K_layer_{nl}_sigma_{sw_}_corrupt_{c}_run_{filename}"

            with open(save_file_path, 'wb') as f:
                pickle.dump(results[nl], f)
