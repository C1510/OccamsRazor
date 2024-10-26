import numpy as np
import itertools
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import copy, random
from complexities import K_lemp_ziv

get_K = K_lemp_ziv

global_folder = 'train_runs_1c2'

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except:
    rank, size = 0, 1

import argparse, sys, os

fullCmdArguments = sys.argv
argumentList = fullCmdArguments[1:]
parser = argparse.ArgumentParser()
parser.add_argument('-nf', '--filenumber', help="Which filenumber?", type=int, default=1)
parser.add_argument('-nl', '--nl', help="Number of layers", type=int, default=10)
parser.add_argument('-sw', '--sw', help="sw", type=float, default=1)
parser.add_argument('-t', '--times', help="Number of iterations", type=int, default=100000)
parser.add_argument('-in', '--indim', help="Number of iterations", type=int, default=13)
parser.add_argument('-act', '--activation', help="Number of iterations", type=str, default='tanh')
parser.add_argument('-f', '--function', help="Number of iterations", type=int, default=0)
parser.add_argument('-nte', '--nte', help="Number of iterations", type=int, default=100)
parser.add_argument('-r', '--random', help="How to randomise", type=str, default='True')
parser.add_argument('-lo', '--loss', help="How to randomise", type=str, default='mse')
args = parser.parse_args()
args.random = True if args.random=='True' else False

nl, sw = args.nl, args.sw

def importdata(dim = 7):
    inputs = [[0,1] for _ in range(dim)]
    inputs = itertools.product(*inputs)
    inputs = [i for i in inputs]
    inputs = np.array(inputs)
    return inputs

def make_data(y, train_size = 100, args = None):
    y = torch.Tensor([int(i) for i in y])
    x = torch.Tensor(importdata(dim = args.indim)) 

    traindataset = TensorDataset(x[:train_size,:],y[:train_size])
    traindataloader = DataLoader(traindataset, batch_size=min(train_size,256))
    testdataset = TensorDataset(x[train_size:, :], y[train_size:])
    testdataloader = DataLoader(testdataset, batch_size=x[train_size:].shape[0])
    return traindataset, testdataset, traindataloader, testdataloader

def make_data_random(y,  train_indices, test_indices, train_size = 100, args = None):
    y = torch.Tensor([int(i) for i in y])
    x = torch.Tensor(importdata(dim = args.indim)) 

    totaldataset = TensorDataset(x,y)
    traindataset = TensorDataset(x[train_indices,:],y[train_indices])
    traindataloader = DataLoader(traindataset, batch_size=min(train_size,256))
    testdataset = TensorDataset(x[test_indices, :], y[test_indices])
    testdataloader = DataLoader(testdataset, batch_size=x[train_size:].shape[0])
    return traindataset, testdataset, traindataloader, testdataloader, totaldataset

class Net_FCN(nn.Module):
    def __init__(self, args):
        super(Net_FCN, self).__init__()
        self.args = args
        self.out_size = 1
        ilw = 256
        self.fc1 = nn.Linear(args.indim, ilw)
        with torch.no_grad():
            self.fc1.weight = nn.Parameter(self.fc1.weight*args.sw)
        try:
            self.fcs = nn.ModuleList([copy.deepcopy(nn.Linear(ilw, ilw)) for _ in range(args.nl-2)])
            with torch.no_grad():
                for i in range(len(self.fcs)):
                    self.fcs[i].weight = nn.Parameter(self.fcs[i].weight*args.sw)
        except:
            self.fcs = nn.ModuleList()
        self.fc3 = nn.Linear(ilw, self.out_size)
        with torch.no_grad():
            self.fc3.weight = nn.Parameter(self.fc3.weight*args.sw)
        if args.activation.casefold() == 'relu':
            self.m = F.relu
        elif args.activation.casefold() == 'tanh':
            self.m = F.tanh

    def forward(self, x):
        bs = x.shape[0]
        x = self.m(self.fc1(x))
        if len(self.fcs) > 0:
            for layer in self.fcs:
                x = layer(x)
        x = self.fc3(x)
        return x

def get_model_acc(model, x, y, loss):
    model.eval()
    y_pred = (model(x).cpu().detach())
    y_pred_loss = loss(y_pred.reshape(-1).cpu(), y.reshape(-1).cpu()).cpu().detach().item()
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred < 0.5] = 0

    y, y_pred = y.reshape(-1), y_pred.reshape(-1)
    error = np.sum(((y - y_pred) != 0).numpy().astype(int))

    ystring = "".join([str(int(i.item())) for i in y_pred])
    return float(error), y_pred_loss, ystring

def train(y, m, args):
    args.indim = int(np.log2(len(y)))
    max_epochs = 100000
    if args.random:
        train_indices = random.sample([i for i in range(len(y))], m)
        test_indices = list(set([i for i in range(len(y))]) - set(train_indices))
    else:
        train_indices = [i for i in range(len(y) // 2)]
        test_indices = list(set([i for i in range(len(y))]) - set(train_indices))
    traindataset, testdataset, traindataloader, testdataloader, totaldataset = make_data_random(y, train_indices,
                                                                                                test_indices,
                                                                                                train_size=m, args=args)
    model = Net_FCN(args)
    optimiser = torch.optim.Adam(model.parameters(),lr=0.0005)

    if args.loss == 'mse':
        loss = nn.MSELoss()
    else:
        loss = nn.BCEWithLogitsLoss()
    for epoch in range(max_epochs):
        for x_, y_ in traindataloader:
            optimiser.zero_grad()
            x_ = model(x_)
            l = loss(x_.reshape(-1),y_.reshape(-1))
            l.backward()
            optimiser.step()
        error, y_pred_loss, ystring = get_model_acc(model, traindataset.tensors[0], traindataset.tensors[1], loss)
        testerror, test_y_pred_loss, test_ystring = get_model_acc(model, testdataset.tensors[0], testdataset.tensors[1], loss)
        totalerror, total_y_pred_loss, total_ystring = get_model_acc(model, totaldataset.tensors[0], totaldataset.tensors[1],loss)
        if error == 0:
            break
    return error, y_pred_loss, ystring, testerror, test_y_pred_loss, test_ystring, totalerror, total_y_pred_loss, total_ystring

PAC_BAYES = True
num_file = 3
m = 64
import matplotlib.pyplot as plt
import pandas as pd
fig, ax = plt.subplots()
dfs = {}
colours = {1:'#1f77b4',2:'#de6610',4:'#551887',8:'#d00000', 'relu':'#1d9641', 1.5:'#ffc90e'}
for idxs in ['rand']:
    alpha, alpha2 = 0.1, 0.5
    file = f'data_1c/{1}_10.csv'
    dfs2 = pd.read_csv(file, names=['kc', 'mean', 'std'], index_col=None, delimiter=',')
    x = np.array(list(dfs2['kc']))
    y = np.array([0.5]*len(x))
    ystd = np.array([4/64]*len(x))
    plt.fill_between(x, y - ystd, y + ystd, color='#006f00', alpha=alpha, capstyle='round')
    plt.scatter(x, y, color='#006f00', label='unbiased', alpha=alpha2, s=8)
for idxs in [8,1]:
    alpha = 0.15 if idxs==8.0 else 0.25
    alpha2 = 0.75 if idxs==8.0 else 1
    file = f'data_1c/{idxs}_10.csv'
    dff = pd.read_csv(file,names = ['kc','mean','std'], index_col=None, delimiter=',')
    dff.sort_values(by=['kc'],inplace=True)
    dfs[idxs]=dff
    x = dfs[idxs]['kc']
    y = dfs[idxs]['mean']
    ystd = dfs[idxs]['std']
    plt.fill_between(x, y - ystd, y + ystd, color = colours[idxs],alpha=alpha, capstyle='round')
    plt.scatter(x, y, color=colours[idxs], label =str(idxs)+' |'+' 10',alpha = alpha2, s=8)
    if idxs == 1:
        xbound = [i for i in range(7,160)]
        ybound = [1-np.exp(-1*(np.log(128) + min(i, 128)*np.log(2) )/(m)) for i in range(153)]
        if PAC_BAYES:
            ax.plot(xbound, ybound, linestyle=':', color = colours[idxs])
    if idxs == 8:
        xbound = [i for i in range(7,160)]
        yscale = [-14+14*i/128 for i in range(128)]+[0 for i in range(128,160)]
        yscale = [np.log(10**i) for i in yscale]
        ybound = [1-np.exp(-1*(-1*yscale[i] + min(i, 128)*np.log(2) )/(m)) for i in range(153)]
        if PAC_BAYES:
            ax.plot(xbound, ybound, linestyle=':', color = colours[idxs])

xbound = [i for i in range(7,160)]
ybound = [1-np.exp((np.log(2**(-128)))/(m)) for _ in xbound]
if PAC_BAYES:
    ax.plot(xbound, ybound, linestyle=':', color = '#006f00')

ax.set_xlim([0,160])
ax.set_xticks([0,40,80,120,160])

ax.set_ylabel(r'Generalisation Error')
ax.set_xlabel(r'LZ complexity, target function')
ax.legend(fontsize='x-small',title_fontsize='x-small',title=r'$\sigma_w\mid N_l$')

fig.set_size_inches(3, 2.4)
plt.tight_layout()
plt.savefig(f'plots/1c_{m}_pb-{PAC_BAYES}.pdf', dpi=300)