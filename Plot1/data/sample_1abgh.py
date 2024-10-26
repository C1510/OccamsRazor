import numpy as np
import itertools, math
from ..complexities import bool_complexity, K_lemp_ziv
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except:
    rank, size = 0, 1

def importdata(in_size):
    inputs = [[0,1] for _ in range(in_size)]
    inputs = itertools.product(*inputs)
    inputs = [i for i in inputs]
    inputs = np.array(inputs)
    return inputs


class DNN():
    def __init__(self, layer_width=1024, nl=2, sw=1, act = 'tanh', in_size = 7):
        self.nl, self.sw = nl, sw
        self.bias_magnitude = args.sigmab*sw #0.2
        self.layers = {}
        self.biases = {}
        self.layers[0] = np.random.normal(loc=0.0, scale=1.0, size=(in_size, layer_width))*(sw/math.sqrt(7))
        self.biases[0] = np.random.normal(loc=0.0, scale=1.0, size=layer_width)*(self.bias_magnitude / in_size)
        for i in range(nl - 1):
            j = i + 1
            self.layers[j] = np.random.normal(loc=0.0, scale=1.0, size=(layer_width, layer_width))*(sw/np.sqrt(layer_width))
            self.biases[j] = np.random.normal(loc=0.0, scale=1.0, size=layer_width)*(self.bias_magnitude / layer_width)
        self.layers[nl] = np.random.normal(loc=0.0, scale=1.0, size=(layer_width, 1))*(sw/np.sqrt(layer_width))
        self.biases[nl] = np.random.normal(loc=0.0, scale=1.0, size=1)*(self.bias_magnitude / layer_width)
        self.layer_width = layer_width
        self.act = act
        print('HELP', len(self.layers))

    def reset_weights(self):
        for k, v in self.layers.items():
            self.layers[k] = np.random.normal(loc=0.0, scale=1.0, size=(v.shape[0], v.shape[1]))*(self.sw/np.sqrt(v.shape[0]))
        for k, v in self.biases.items():
            self.biases[k] = np.random.normal(loc=0.0, scale=1.0, size=v.shape[0])*(self.bias_magnitude/v.shape[0])

    def multiply(self, output):
        for i in range(nl + 1):
            output = np.matmul(output, self.layers[i])
            output += self.biases[i]
            if self.act == 'tanh':
                output = np.tanh(output*args.tanhscale)/args.tanhscale
            elif self.act == 'relu':
                output = np.maximum(output, 0)
            elif self.act == 'linear':
                output = output
        return (output > 0).astype(int)

    def get_train_acc(self, inputt, y):
        ypred = self.multiply(inputt)
        return np.count_nonzero(y - ypred)



import argparse, sys, os, random

fullCmdArguments = sys.argv
argumentList = fullCmdArguments[1:]
parser = argparse.ArgumentParser()
parser.add_argument('-nf', '--filenumber', help="Which filenumber?", type=int, default=1)
parser.add_argument('-nl', '--number_layers', help="Number of layers", type=int, default=10)
parser.add_argument('-sw', '--sigmaw', help="Sigmaw", type=float, default=1)
parser.add_argument('-in', '--indim', help="Number of iterations", type=int, default=7)
parser.add_argument('-t', '--times', help="Number of iterations", type=int, default=1000000)
parser.add_argument('-act', '--activation', help="Number of iterations", type=str, default='tanh')
parser.add_argument('-sb', '--sigmab', help="Number of iterations", type=float, default=0.2)
parser.add_argument('-ta', '--tanhscale', help="Number of iterations", type=float, default=1.0)

args = parser.parse_args()

nf, nl, sw, times, act, indim = args.filenumber, args.number_layers, args.sigmaw, args.times, args.activation, args.indim
print(nl)
nf=rank
nf = random.randint(1,1e11)
sb = args.sigmab

print(sb)
filename_ = f'./{nl}_{sw}_{sb}_{indim}_{act}_bool_{args.tanhscale}/'
print(filename_)
try:
    if not os.path.exists(filename_):
        os.mkdir(filename_)
except:
    pass

X_all = importdata(in_size=args.indim)
filename = filename_ + f'/{nl}_{sw}_{nf}.txt'
dnn = DNN(layer_width=64, nl=nl, sw=sw, act = act, in_size = indim)

f_list = []

for i in range(times):
    dnn.reset_weights()
    f = dnn.multiply(X_all)
    f = ''.join([str(int(im)) for im in f])
    kc = K_lemp_ziv(f)
    kc = round(kc*2)/2
    kc_bool = bool_complexity(f)
    f_list.append(f+' ' + str(kc)+ ' '+str(kc_bool)+' '+str(sum([1 if i_=='1' else 0 for i_ in f])))
    #print(i)
    if True: # (i + 1) % 1 == 0:
        file = open(filename, 'a+')
        for j in f_list:
            file.write(j + '\n')
        file.close()
        f_list = []
        print('File number ' + str(nf) + ' has done '  + str(i+1) + ' done of ' + str(times))
