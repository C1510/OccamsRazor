import itertools, pickle
from math import comb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics, sys
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

fig, ax = plt.subplots()
ax.margins(0.05)

sw = np.linspace(1,2.5,31).reshape(-1)
colours = {2:'#1f77b4',5:'#de6610',9:'#551887',8:'#d00000', 10:'#1d9641', 1.5:'#ffc90e'}
alpha = 0.7
means = []
stds = []
runs = []
for run in range(10):
    with open(f'data/Acc_CSR_Results/CSR_MNIST_1K_layer_{10}_sigma_{sw[-1]}_corrupt_0_run_{run}', 'rb') as f:
        x = pickle.load(f)['csr']
    runs.append(x)

for nl in [2,5,10]:
    x_ = [i[nl][:,1] for i in runs]
    x = np.stack(x_,axis = 0)
    mean = np.mean(x,axis = 0).reshape(-1)
    std = np.std(x, axis=0).reshape(-1)
    means.append(mean)
    stds.append(std)
    mean = (1-mean)
    ax.plot(sw,mean, color = colours[nl],label = 'tanh $\mid$'+ str(nl))
    ax.fill_between(sw, mean - std, mean + std, color=colours[nl], alpha=alpha, capstyle='round')


fig.set_size_inches(3,2.5)
ax.set_ylim([0.3,0.9])
ax.set_yticks([0.3,0.5,0.7,0.9])
ax.set_ylim([0.1,0.7])
ax.set_yticks([0.1,0.2,0.3,0.5,0.7])
ax.set_xticks([1.0,1.5,2.0,2.5])
ax.set_xlim([1.0,2.5])
ax.set_xlabel(r'$\sigma_w$')
ax.set_ylabel('Generalisation Error')
ax.legend(fontsize='small', title_fontsize='small', title=r'$\phi\mid N_l$')
plt.savefig('plots/3a.pdf',dpi=300,bbox_inches='tight')