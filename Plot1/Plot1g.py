colours = {1:'#1f77b4',2:'#de6610',4:'#551887',8:'#d00000', 'relu':'#1d9641', 1.5:'#ffc90e','ms':'#006f00'}

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np
fig, ax = plt.subplots()

ax.margins(0.05)

fname = '10_8.0_7_kc.txt'

def plot_manual(fig,ax):
    plt.fill_between([i for i in range(1,128)],[2**(i-128) for i in range(1,128)], alpha = 0.25, color = 'red', label = 'theoretical')
    ax.set_yscale('log')
    return fig, ax

fit, ax = plot_manual(fig, ax)

def main(fname, fig,ax,inlab,pltnum, colour, norm = False):
    if norm:
        fact1 = 6
        fact2 = 128/173
    else:
        fact1,fact2=0,1
    df = pd.read_csv(fname, sep = ' ',names=['kc','freq','count'])
    df.sort_values(by=['kc'], inplace = True)

    df.reset_index(inplace=True, drop=True)
    df['kc']=(df['kc']-fact1)*fact2 

    df2 = df[:-10]
    new_list = list(np.log10(df2['freq']))
    fit = np.polyfit(list(df2['kc']), new_list, 1)
    minkc = int(min(list(df['kc'])))
    if fname == 'ms_kc.txt':
        x1,y1,x2,y2 = 7, np.log10(2*(2**-128)), 109, np.log10(7e-8)

        fit = np.polyfit([x1,x2], [y1,y2], 1)

    if fname != 'ms_kc.txt' and '8' in fname:
        fitstart = -10 
        fit = np.polyfit(list(df2['kc'])[:fitstart], new_list[:fitstart], 1)
        xs = [i for i in range(0,minkc+1,1)]
        ys = [10**(fit[0]*i+fit[1]) for i in xs]
        l = ax[pltnum].plot(xs,ys, color = colour,linestyle='-.')
        ax[pltnum].fill_between(xs, ys, alpha=0.15, color=colour,label = '__no__label__')

    if fname =='ms_kc.txt':
        xs = [i for i in range(0,minkc+1,1)]
        ys = [10 **(fit[0]*i+fit[1]) for i in xs]
        l = ax[pltnum].plot(xs,ys, color = colour, linestyle = '-.')
        ax[pltnum].fill_between(xs, ys, alpha=0.15, color=colour, label='__no__label__')
        ax[pltnum].scatter([1],[10**y1],color = colour, marker = 'x')

    if norm:
        ax[pltnum].plot([128,128], [1e-45,1], color='grey')

    ax[pltnum].fill_between(df['kc'], df['freq'], label = inlab, alpha = 0.5, color = colour)
    summ=0
    kcc=list(df['kc'])
    freqq = list(df['freq'])
    idx=0
    while summ<0.9:
        summ+=list(reversed(freqq))[idx]
        currlz = list(reversed(kcc))[idx]
        idx+=1
    ax[pltnum].plot([currlz,currlz], [1e-45,1], color=colour,linestyle=':')

    ax[pltnum].set_yscale('log')
    if norm:
        ax[pltnum].set_xlim([0,140])
    else:
        ax[pltnum].set_xlim([0, 180])
    ax[pltnum].set_ylim([2*2**-128,1])
    if norm:
        ax[pltnum].set_ylabel(r"$P({K'})$")
        ax[pltnum].set_xlabel(r"Normalised LZ Complexity, ${K'}$")
    elif not norm:
        ax[pltnum].set_ylabel(r"$P({K})$")
        ax[pltnum].set_xlabel(r"LZ Complexity, ${K}$")

    ax[pltnum].legend(title=r'$\sigma_w \; \mid \; N_L$', fontsize = 'x-small', title_fontsize = 'x-small', loc = 'upper left')

    ax[pltnum].xaxis.set_major_locator(MultipleLocator(40))
    ax[pltnum].xaxis.set_minor_locator(MultipleLocator(20))
    return fig, ax

mode = '1s'
if mode == '3s':

    norm = False
    fig, ax = plt.subplots(1,3)
    fname = '10_1.0_0.2_7_tanh_kc.txt' 
    fig, ax = main(fname, fig, ax, '1|10',0,colour=colours[1], norm = norm)
    fname = '10_8.0_0.1_7_tanh_kc.txt' 
    fig, ax = main(fname, fig, ax,'8|10',1,colour=colours[8], norm = norm)
    fname = 'ms_kc.txt'
    fig, ax = main(fname, fig, ax, 'unbiased',2,colour=colours['ms'], norm = norm)

    if True:
        ax[2].plot([i for i in range(1, 128)], [2 ** (i - 128) for i in range(1, 128)], alpha=1.0, color='maroon',
                         label='theoretical')
        ax[2].plot([128,128], [2*2**-128,1], alpha=1.0, color='maroon',
                         label='__')
        ax[2].fill_between([i for i in range(1, 128)], [2 ** (i - 128) for i in range(1, 128)],
                           alpha=0.15, color='maroon', label='__no__label__')

        ax[2].legend(title=r'$\sigma_w \; \mid \; N_L$', fontsize='x-small', title_fontsize='x-small', loc='lower right')
        ax[2].plot([124, 124], [1e-45, 1], linestyle=':', color='maroon')

    for i in range(3):
        ax[i].set_yticks([10**-i for i in [0,5,10,15,20,25,30,35]])
        scale = 179/128 if not norm else 1
        ax[i].set_aspect(3.5*scale)
        if i!=0:
            ax[i].set_ylabel('')

    fig.set_size_inches(11,3)

    plt.savefig(f'extended_renorm_{norm}.png', bbox_inches='tight', dpi=300)

else:

    norm = False
    fname = 'data/data1gh/ms_kc.txt'
    fig, ax = main(fname, fig, [ax], 'unbiased', 0, colour=colours['ms'], norm=norm)
    ax = ax[0]

    if True:
        ax.plot([i for i in range(1, 128)], [2 ** (i - 128) for i in range(1, 128)], alpha=1.0, color='maroon',
                         label='__theoretical')
        ax.plot([128,128], [2*2**-128,1], alpha=1.0, color='maroon',
                         label='__')
        ax.fill_between([i for i in range(1, 128)], [2 ** (i - 128) for i in range(1, 128)],
                           alpha=0.15, color='maroon', label='__no__label__')

        ax.legend(fontsize='x-small', title_fontsize='x-small', loc='upper left')
        ax.plot([124, 124], [1e-45, 1], linestyle=':', color='maroon')
        ax.plot([7, 109], [2*2**-(128), 8*1e-8], linestyle='-.', color=colours['ms'])

    for i in range(1):
        ax.set_yticks([10**-i for i in [0,10,20,30]])
        scale = 179/128 if not norm else 1

        ax.set_xlim([0,160])

    fig.set_size_inches(3,2.4)
    plt.savefig(f'plots/1g.pdf', bbox_inches='tight', dpi=300)