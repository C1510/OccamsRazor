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
for nl in [2,5,10]:
    xs = []
    for run in range(10):
        with open(f'data/New_CIFAR_Results/CIFAR_width_200_layer_{nl}_run_{run}', 'rb') as f:
            x = pickle.load(f)['CIFAR_bias']
            xs.append(x)
    x = np.concatenate(xs,axis = 0)
    mean = np.mean(x,axis = 0).reshape(-1)
    std = np.std(x, axis=0).reshape(-1)
    means.append(mean)
    stds.append(std)
    mean = 1-mean
    ax.plot(sw,mean, color = colours[nl],label = 'tanh $\mid$'+ str(nl))
    ax.fill_between(sw, mean - std, mean + std, color=colours[nl], alpha=alpha, capstyle='round')


fig.set_size_inches(3,2.5)
ax.set_ylim([0.56,0.68])
ax.set_yticks([0.56,0.60,0.64,0.68])
ax.set_xticks([1.0,1.5,2.0,2.5])
ax.set_xlim([1.0,2.5])
ax.set_xlabel(r'$\sigma_w$')
ax.set_ylabel('Generalisation Error')
ax.legend(fontsize='small', title_fontsize='small', title=r'$\phi\mid N_l$')
plt.savefig('plots/3b.pdf',dpi=300,bbox_inches='tight')