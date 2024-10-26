import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

fname = '10_1.0_7_kc.txt'

fig, ax = plt.subplots()
alpha = 0.75
colours = {1: '#1f77b4', 2: '#de6610', 4: '#551887', 8: '#d00000', 'ms': '#006f00', 1.5: '#ffc90e'}

def make(fname, n=5, sw = 1):

    ax.margins(0.1)
    colours = {1:'#1f77b4',2:'#de6610',4:'#551887',8:'#d00000', 'ms':'#006f00', 1.5:'#ffc90e'}
    df = pd.read_csv('data/data1gh/'+fname, sep = ' ',names=['kc','freq','count'])
    df['kc']=[i if i!=14 else 7 for i in df['kc']]
    df['freq'] 

    plt.bar(df['kc'], df['freq'],width=3.6, label = str(float(sw))+f'$\mid 10$' if sw!='ms' else 'random sampling', color=colours[sw],alpha=alpha)

    ax.set_yscale('log')
    if n == 7:
        ax.set_xlim([0,180])
        ax.set_xticks([0,40,80,120,160])
        ax.set_ylim([1e-8,1])
    elif n == 5:
        ax.set_xlim([0, 60])
        ax.set_ylim([1e-10, 1])
    elif n == 5:
        ax.set_xlim([0, 60])
        ax.set_ylim([1e-10, 1])
    ax.set_ylabel(r'$P(K)$')
    ax.set_xlabel(r'LZ Complexity, $K$')

    ax.xaxis.set_major_locator(MultipleLocator(40))
    ax.xaxis.set_minor_locator(MultipleLocator(20))

    locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8,1.0),numticks=12)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    locmax = matplotlib.ticker.LogLocator(base=10.0,numticks=1)

    fig.set_size_inches(3,3)

make('10_1.0_0.2_7_tanh_kc.txt',n=7,sw=1)
make('10_8.0_0.1_7_tanh_kc.txt',n=7,sw=8)

fig.set_size_inches(3,2.4)
ax.set_xlim([0,160])
ax.legend(title=r'$\sigma_w \; \mid \; N_L$', fontsize = 'x-small', title_fontsize = 'x-small', loc = 'lower left')

plt.savefig(f'plots/1h.pdf', bbox_inches='tight', dpi=300)

