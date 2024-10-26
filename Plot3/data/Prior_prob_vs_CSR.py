import pandas as pd
import pickle
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import tqdm
from tqdm import tqdm as tqdm



# Define neural network
def sample_function(x_train,y_train,layer_sizes,num_layer,batchsize,optimiser,loss_function,arch,input_size,sigma_w):

    model = Sequential()
    n = layer_sizes

    # Set up a fully connected neural network with random initialisation of weights and biases
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

    def preprocess(image):
        image = tf.cast(image, tf.float32)
        return image

    loss_object = tf.keras.losses.CategoricalCrossentropy()

    def create_adversarial_pattern(input_image, input_label):
        with tf.GradientTape() as tape:
            tape.watch(input_image)
            prediction = model(input_image)
            loss = loss_object(input_label, prediction)

        # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, input_image)
        # Get the sign of the gradients to create the perturbation
        signed_grad = tf.sign(gradient)
        return signed_grad


    def crit_samp_ratio(train_ex, train_ex_label, model, alpha = 0.005, beta = 0.005, r = 0.01):

        def hamming_distance(a, b):
            r = (1 << np.arange(8))[:,None]
            return np.count_nonzero( (a & r) != (b & r) )

        # train_ex is a 28x28 MNIST image
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

    count = np.zeros(1000)
    for k in range(1000):
        count[k] = crit_samp_ratio(x_train[k], k, model)
    crit_samp_number = sum(count)
    crit_ratio = crit_samp_number / 1000

    return crit_ratio

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    filename=rank
except:
    pass

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[0:1000,:,:]
x_test = x_test[0:1000,:,:]
y_train = y_train[0:1000]
y_test = y_test[0:1000]

labels = tf.one_hot(y_test.reshape(1000,1), 10)

results = {}

layers = 10
s_w = np.float32(sys.argv[1])

n_samples = 100

crit_ratios = np.ones(n_samples)

for i in range(n_samples):
    crit_ratios[i] = sample_function(x_train, y_train, layer_sizes = 200, num_layer = layers, batchsize = 32, optimiser = "sgd", loss_function = 'cross_entropy', arch = 'FCN', input_size = 784, sigma_w = s_w)
    if i%100 == 0:
        print(i)
    results["csr"] = crit_ratios
    save_file_path = "Apriori_CSR_Results/Apriori_CSR_1K_layer_"  + str(sys.argv[1]) + "_sigma_" + str(sys.argv[2]) +  "_run_" + str(filename)

    with open(save_file_path, 'wb') as f:
        pickle.dump(results, f)
