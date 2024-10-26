import itertools
from math import comb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics, sys
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import pandas as pd
import collections
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

def main(nte, sw, f_idx, mm, ax):
    gpblue, gpred = '#1f76b6', '#ff350c'

    filename = f'data/abc/epsilon_plot_m_{f_idx}.npy'

    for nte in [nte]:
        kcf = 31.5 if f_idx in [0, 2] else (66.5 if f_idx in [1, 10] else 101.5)
        arr = np.load(filename, allow_pickle=True)
        arr = pd.DataFrame(data = arr, index = None, columns = ['f', 'kc', 'err'])
        arr.loc[len(arr)] = [0, kcf, 1]
        arr['combs'] = [(comb(128 - int((1 - i) * 128), nte)) for i in arr['err']]
        arr['new_error'] = ((arr['err']) ** nte) #*arr['combs']
        arr.drop(columns=['f','err'],inplace=True)
        arr = arr.groupby(['kc'])['new_error'].apply(lambda x: statistics.mean(sorted(x)[-mm:]))
        arr = arr.reset_index()

    kc_freqs = []
    kc_freq = pd.read_csv(f'data/kc/10_1.0_7_kc.txt', sep=' ', names=['kc', 'freq', 'count'])
    kc_freq = pd.read_csv(f'data/kc/10_1.0_1.0_7_tanh_kc.txt', sep=' ', names=['kc', 'freq', 'count'])
    kc_freq.sort_values(by=['kc'], inplace=True)
    kc_freq.drop(columns = ['count'], inplace=True)
    kc_freqs.append(kc_freq)
    kc_freq = pd.read_csv(f'data/kc/10_8.0_0.1_7_tanh_kc.txt', sep=' ', names=['kc', 'freq', 'count'])
    kc_freq = pd.read_csv(f'data/kc/10_8.0_0.1_7_tanh_kc.txt', sep=' ', names=['kc', 'freq', 'count'])

    kc_freq.sort_values(by=['kc'], inplace=True)
    kc_freq.drop(columns = ['count'], inplace=True)

    new_list = list(np.log10(kc_freq['freq']))
    fit = np.polyfit(list(kc_freq['kc'])[:-20], new_list[:-20], 1)
    xs = [i/2 for i in range(0, int(2*min(list(kc_freq['kc']))))]
    ys = [10 ** (fit[0] * i + fit[1]) for i in xs]
    xydict = {'kc':xs, 'freq':ys}
    ext_eight = pd.DataFrame.from_dict(xydict)

    kc_freqs.append(kc_freq)

    for count, kc_freq in enumerate(kc_freqs):

        df = pd.merge(arr, kc_freq, on = ['kc'],how='outer')
        print('AAAAA',df)
        df.dropna(inplace=True)

        df['y']=df['new_error']*df['freq']
        sw = 1 if count == 0 else 8
        print(f'P(S), {sw}, {f_idx}, {nte},',sum(list(df['y'])))
        sumy = sum(list(df['y']))
        df['y']/=sum(list(df['y']))

        if nte == 32:
            label1 = r'$1\mid 10, Bayes$'
            label2 = r'$8\mid 10, Bayes$'
        else:
            label1, label2 = None, None

        if count == 0:
            ax.bar(df['kc'], df['y'], width = 3.0, color = gpblue, label = label1, alpha = 0.75) # '#07EBDB', label = label1)
        elif f_idx!=2 or nte !=100:
            ax.bar(df['kc'], df['y'], width=3.0, color = gpred, label = label2, alpha = 0.75) #'#DE6610', label = label2)

        if nte == 100 and count!=0 and f_idx==2:
            ext = pd.merge(arr, ext_eight, on=['kc'], how='outer')
            ext.dropna(inplace=True)

            ext['y']=ext['new_error']*ext['freq']
            sumext = sum(list(ext['y']))
            ext['y']/=(sumy+sumext)
            df['y'] *= sumy/(sumy + sumext)
            ax.bar(ext['kc'], ext['y'], width=3.0, color = '#ff7d63', label = '__nolabel__', alpha = 0.75) #'#DE6610', label = label2)
            ax.bar(df['kc'], df['y'], width=3.0, color=gpred, label=label2, alpha = 0.75)

        ax.set_xlim([0,160])

    ax.set_ylim([0, 1.1 * max(df['y'])])
    k_func = 31.5 if f_idx in [0, 2] else (66.5 if f_idx in [1, 10] else 101.5)
    ax.plot([k_func, k_func],[0,1], linestyle=':', color = 'dimgrey', alpha = 0.75)
    return ax

gpnn = 'nn'
folder = '10_1_0_True_train'
base_folder = f'train_runs/{gpnn}/{folder}/'

folder = '10_1_0_1_train'

def main2(file, ax, ccc, colour, label='_nolabel_'):
    files = ['data/train_runs_henry/'+file+'.csv']

    df_list = []
    for f in files:
        df_list.append(pd.read_csv(f'{f}', sep=' ', names=['kct', 'kc','e']))
    df = pd.concat(df_list)
    print(np.mean(df['e']))

    freqs = list(df['kc'])
    freqs = collections.Counter(freqs)
    x, y = [], []
    summ = 0
    for key, value in freqs.items():
        y.append(key)
        x.append(value)
        summ+=value
    x = [i/summ for i in x]
    ddd = {'y':x, 'kc':y}
    df = pd.DataFrame(ddd)
    ax.bar(y,x,width = 3, color=colour, label = label, alpha=0.75)
    k_func = complexities_[ccc]
    ax.plot([k_func, k_func], [0, 1], linestyle=':', color='dimgrey', alpha=0.75)
    ax.set_ylim([0,1.1*max(x)])

    return ax

complexities_ = {2:31.5, 5:45.5, 10:66.5, 15:87.5, 21:94.5, 30:101.5, 40:119, 50:129.5, 60:143.5}
f_idxs = [2,10, 21]

for f_idx in f_idxs:
    nte = 32
    sw = 8.0
    mm = 10
    fig, axs = plt.subplots(6,1,sharex=True)

    axs[0] = main(32, sw, f_idx,mm, axs[0])
    axs[2] = main(64, sw, f_idx,mm, axs[2])
    axs[4] = main(100, sw, f_idx,mm, axs[4])

    blue, red = '#1f78b4', '#f70200'
    blue, red =  '#0001fb', '#d00000'
    f_idx1 = f_idx if f_idx!=3 else 2
    axs[1] = main2(f'{f_idx1}_1_32', axs[1], f_idx1, blue, label=r'$1\mid 10$')
    axs[1] = main2(f'{f_idx1}_8_32', axs[1], f_idx1,red, label=r'$8\mid 10$')
    axs[3] = main2(f'{f_idx1}_8_64', axs[3], f_idx1,red)
    axs[3] = main2(f'{f_idx1}_1_64', axs[3], f_idx1,blue)
    axs[5] = main2(f'{f_idx1}_8_85', axs[5],f_idx1, red)
    axs[5] = main2(f'{f_idx1}_1_85', axs[5], f_idx1,blue)

    axs[0].set_ylabel(r' ' + '\n' + r'$m=32$')
    axs[2].set_ylabel(r'$\langle P(K\mid S)\rangle_m$'+' & ' +r'$\langle P_{SGD}(K\mid S)\rangle_m$'+ '\n' +r'$m=64$')
    axs[4].set_ylabel(r' ' + '\n' + r'$m=100$')

    if f_idx==0:
        axs[0].legend(fontsize = 'x-small', ncol = 2)
        axs[1].legend(fontsize = 'x-small', ncol = 2)
    fig.set_size_inches(3, 7)

    for i in range(3):
        a1 = axs[2*i].get_ylim()
        a2 = axs[2*i+1].get_ylim()
        ffs = max(a1[1], a2[1])
        ffs = np.ceil(20*ffs) / 20
        if i ==0 and f_idx==2:
            ffs = 0.55
        ffs2 = round(20*0.75*ffs) / 20
        af = [0,ffs]
        axs[2 * i].set_ylim(af)
        axs[2 * i + 1].set_ylim(af)
        axs[2 * i].yaxis.set_major_locator(MultipleLocator(ffs2))
        axs[2 * i +1].yaxis.set_major_locator(MultipleLocator(ffs2))


    for i in range(6):
        axs[i].margins(0.05)
        axs[i].set_xlim([0, 160])
        axs[i].set_xticks([0,40,80,120,160])

    for i in range(6):
        axs[i].yaxis.set_minor_locator(MultipleLocator(0.05))

    for i in range(2):
        axs[i].legend(fontsize='x-small', title=r'$\sigma_w\mid N_L$', title_fontsize='x-small',ncol=1)

    axs[-1].set_xlabel('LZ Complexity')

    plt.savefig(f'plots/2_{f_idx}.pdf', bbox_inches='tight')
