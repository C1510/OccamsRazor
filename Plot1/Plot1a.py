import sys
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from scipy.optimize import curve_fit

def make_zipf(filename, fig, ax, label=None, fit=False):
    print('loading')
    df = pd.read_csv(filename, names=['x', 'kc'], delimiter=' ', usecols=[1,2])
    print('loaded')
    colours = {
        1.0: '#1f77b4',
        1.5: '#ffc90e',
        2.0: '#de6610',
        4.0: '#551887',
        8.0: '#d00000',
        'relu': 'midnightblue',
        'relu8': '#111145'
    }
    data_name = filename.split('/')[-1]
    sigmaw = float(data_name.split("_")[1])
    colour = colours.get(sigmaw, 'black')
    if 'relu8' in data_name:
        colour = colours['relu8']
    elif 'relu' in data_name:
        colour = colours['relu']

    if df['x'].sum() == 0:
        return fig, ax

    df_backup = df.copy()
    ax.margins(0.05)

    def func(x, a, b):
        return a - b * x

    df = df_backup.copy()
    
    df.reset_index(drop=True, inplace=True)
    df.reset_index(drop=False, inplace=True)
    df.rename(columns={'index': 'rank'}, inplace=True)
    df['rank'] += 1  # Adjust rank to start from 1

    sumdf = df['x'].sum()
    col_line = colour

    df1 = df.drop_duplicates(subset='x', keep="last")
    df2 = df.drop_duplicates(subset='x', keep="first")
    df = pd.concat([df1,df2])
    df.sort_values(by=['rank'],inplace=True)
    df['x'] = df['x'] / sumdf
    ax.plot(df['rank'], df['x'], '-', color=col_line, label=label)


    # Fit the data if requested
    if fit:
        popt_cons, _ = curve_fit(
            func,
            np.log10(df['rank'])[3:],  # Exclude first few points if necessary
            np.log10(df['x'] / sumdf)[3:]
        )
        x_fit = np.array([1, 1e8])
        y_fit = 10 ** (popt_cons[0] - popt_cons[1] * np.log10(x_fit))
        ax.plot(x_fit, y_fit, linestyle='dotted', color=col_line)
    else:
        # Use a default fit line if fitting is not performed
        popt_cons = [-np.log10(128 / np.log2(np.exp(1))), 1]
        x_fit = np.array([1, 1e8])
        y_fit = 10 ** (popt_cons[0] - popt_cons[1] * np.log10(x_fit))
        ax.plot(x_fit, y_fit, linestyle='dotted', color=col_line)

    ax.set_xlim([1, 1e8])
    ax.set_ylim([1e-8, 1])
    ax.set_xlabel('Rank(f)')
    ax.set_ylabel('P(f)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(loc='upper right', title='m', fontsize='small', ncol=2)

    return fig, ax

def main():
    dirlist = [f'10_{i}_0.1_7_tanh.txt' for i in [1.0, 1.5, 2.0, 4.0, 8.0]] + ['10_1.0_0.025_7_relu.txt']

    fig, ax = plt.subplots()
    ax.loglog([1e-8, 1e-8], [1, 1e8], alpha=0.8, color='#1d9641', label='random')

    for filename in dirlist:
        m = 128
        sigma_w = filename.split('_')[1]
        activation = 'relu' if 'relu' in filename else 'tanh'
        label = f"{sigma_w} | 10{', relu' if activation == 'relu' else ''}"
        fig, ax = make_zipf('data/data_1ab/'+filename, fig=fig, ax=ax, label=label, fit=False)

    ax.legend(loc='upper right', fontsize='x-small', title_fontsize='x-small', title=r'$\sigma_w \; \mid \; N_L$')
    ax.set_xlim([1, 1e8])
    ax.set_ylim([1e-8, 1])
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim([0.8, 1.2e8])
    ax.set_ylim([0.8e-8, 1.2])
    ax.set_yticks([10 ** (-2 * i) for i in range(5)])
    ax.set_xticks([10 ** (2 * i) for i in range(5)])
    fig.set_size_inches(3, 2.4)
    plt.tight_layout()
    plt.savefig('plots/1a.pdf', bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    main()
