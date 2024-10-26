import copy
import sys, os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import random
from complexities import K_lemp_ziv
import argparse, sys
from statistics import mean, stdev
from utils import sampling

fullCmdArguments = sys.argv
argumentList = fullCmdArguments[1:]
parser = argparse.ArgumentParser()
parser.add_argument('-f_idx', '--f_idx', help="Number of layers", type=int, default=4)
parser.add_argument('-s', '--seed', help="Number of layers", type=int, default=20)
parser.add_argument('-num_layers', '--num_layers', help="Number of layers", type=int, default=10)
parser.add_argument('-mode', '--mode', help="Number of layers", type=str, default='norm')
parser.add_argument('-mode_main', '--mode_main', help="Number of layers", type=str, default='train')
parser.add_argument('-n', '--n', help="Number of layers", type=int, default=7)
parser.add_argument('-train_size', '--train_size', help="Number of layers", type=int, default=100)
parser.add_argument('-collect', '--collect', help="Number of layers", type=str, default='False')
args = parser.parse_args()

if args.f_idx==2:
    counts_rand = [143, 2354, 23490, 164315, 900492, 3850306, 13148477, 35770419, 78170403, 137368447, 197400509, 240670420,
              261662564, 267542392, 268385327, 268434254, 268435456]
elif args.f_idx == 1:
    counts_rand = [2, 8, 20, 87, 293, 1009, 3050, 9165, 25938, 69373, 176812, 426190, 976559, 2114650, 4331930, 8402117,
              15374812, 26574488, 43213652, 66212269, 95240574, 128774209, 163776031, 196600335, 224000077, 244165408,
              257162516, 264192311, 267195389, 268167684, 268397604, 268432995, 268435408, 268435456]
elif args.f_idx == 3:
    counts_rand = [2, 9, 28, 135, 433, 1361, 4247, 12190, 35029, 93872, 243691, 599727, 1402150, 3109658, 6527379,
                   12928892, 24082573, 42070692, 68528341, 103544539, 144518090, 185738447, 220880012, 245712699,
                   259826090, 265968632, 267948485, 268380201, 268431735, 268435364, 268435456]

elif args.f_idx == 5:
    counts_rand = [ 5, 70, 607, 3749, 18768, 74528, 252721, 762309, 2085843, 5208865, 11937680, 24910084, 47277045,
                    81164335,125494135, 174333456, 217480979, 246846449, 261564570, 266846821, 268182151, 268408257,
                    268434105, 268435446, 268435456]

elif args.f_idx == 4:
    counts_rand = [2, 3, 62, 150, 482, 1872, 5324, 16017, 45719, 120864, 306121, 730539, 1643734, 3490904, 6962967,
                   13086255,23116489, 38347482, 59751151, 87246350, 119617178, 154180066, 187536497, 216422099,
                   238597903, 253545098, 262216954, 266367767,
                   267927181, 268349871, 268426638, 268434900, 268435440, 268435456]

nn_pac_bayes = [0,-53.968863239092386,-85.44661940441168,-55.478370905919924,-54.4572454295059,-111.83228634229718]

if args.mode == 'rand':
    args.seed = -1
if args.collect == 'False':
    args.collect = False
else:
    args.collect = True

def main(f_idx=0, T_size = 64,layer_no = 10, mode = 'norm', seed = 20):

    y_true, train_idxs, test_idxs = sampling(m=f_idx, n=7, train_size=T_size, seed=seed, mode_ = 'not')
    kc_ = K_lemp_ziv(y_true)
    sigma = [1, 8]

    def gen_random_data(y_true, file):
        with open(f'data/{file}', 'rb') as handle:
            b = pickle.load(handle)
        return b

    def gen_random_data_2(y_true, file):
        arr_rand = []
        f = y_true
        train_string = "".join([f[i] for i in train_idxs])
        test_string = "".join([f[i] for i in test_idxs])

        for count in range(10000):
            a_ = np.random.randint(0,5)
            f_ = "".join([str(int(np.random.randint(0,2))) for _ in range(2**a_)]) * (2**(7-a_))
            f_train = "".join([f_[i] for i in train_idxs])
            f_test = "".join([f_[i] for i in train_idxs])
            train_err = sum([1 if i != j else 0 for i, j in zip(f_train, train_string)])
            err = sum([1 if i != j else 0 for i, j in zip(f_test, test_string)])
            kc_f_ = K_lemp_ziv(f_)
            if kc_f_<kc_:
                arr_rand.append([kc_, kc_f_, err / (128 - T_size),train_err/T_size])

        for count in range(10000):
            ran_idx__ = random.sample([i for i in range(128)], int(random.randint(0,5)))
            f_ = ['0' if y_true.count('1')<=y_true.count('0') else '1']*128
            if y_true.count('1')== y_true.count('0'):
                if np.random.uniform(0,1)>0.5:
                    f_ = ['1'] * 128
            for i in ran_idx__:
                f_[i] = str(int(random.randint(0, 1)))
            f_ = "".join(f_)
            f_train = "".join([f_[i] for i in train_idxs])
            f_test = "".join([f_[i] for i in train_idxs])
            train_err = sum([1 if i != j else 0 for i, j in zip(f_train, train_string)])
            err = sum([1 if i != j else 0 for i, j in zip(f_test, test_string)])
            kc_f_ = K_lemp_ziv(f_)
            if kc_f_<kc_:
                arr_rand.append([kc_, kc_f_, err / (128 - T_size),train_err/T_size])

        for count in range(10000):
            ran_idx__ = random.sample([i for i in range(128)], int(random.randint(0,5)))
            f_ = copy.deepcopy(f)
            f_ = list(f_)
            for i in ran_idx__:
                f_[i] = str(int(random.randint(0, 1)))
            f_ = "".join(f_)
            f_train = "".join([f_[i] for i in train_idxs])
            f_test = "".join([f_[i] for i in test_idxs])
            train_err = sum([1 if i != j else 0 for i, j in zip(f_train, train_string)])
            err = sum([1 if i != j else 0 for i, j in zip(f_test, test_string)])
            kc_f_ = K_lemp_ziv(f_)
            if kc_f_<21:
                arr_rand.append([kc_, kc_f_, err / (128 - T_size),train_err/T_size])

        arr_rand = np.array(arr_rand)
        kcs = list(set(list(arr_rand[:,1])))
        kcs.sort()
        min_tes_list = []
        for kc in kcs:
            min_tes_list.append(min(list((arr_rand[arr_rand[:,1]==kc])[:,3])))
        min_tes_list = [min(min_tes_list[:i+1]) for i in range(len(min_tes_list))]
        min_tes = {i:j for i,j in zip(kcs, min_tes_list)}

        arr_rand_2 = []
        for kc, min_train in min_tes.items():
            test_errs = (arr_rand[arr_rand[:,1]==kc])
            test_errs = test_errs[test_errs[:,3]==min_train]
            arr_rand_2.append(test_errs)
        arr_rand_2 = np.concatenate(arr_rand_2)
        return arr_rand_2, train_string, test_string

    arr_rand = gen_random_data(y_true, f'n7/{f_idx}_{seed}_{T_size}')
    arr_rand2, _, _ = gen_random_data_2(y_true, f'{f_idx}_{T_size}_{layer_no}')

    def lz_data_dict(file, seed=None):
        import glob
        import pandas as pd
        files = glob.glob('data/LZ_Complexity_Results_bias_variance/*')
        files = [i for i in files if file in i]
        if seed is not None:
            files = [i for i in files if f'yay_{seed}' in i]
        arrlist = []
        for f in files:
            arr = np.load(f, allow_pickle=True)['lz']
            arrlist.append(arr)
        arr = np.concatenate(arrlist)
        arr = arr[arr[:,0] == kc_]
        return arr

    labels = [str(sigma[0]) + ' | ' + str(layer_no), str(sigma[1]) + ' | ' + str(layer_no)]
    input_data1 = lz_data_dict('Train' + str(T_size) + '_lz_l' + str(layer_no) + '_w' + str(sigma[0]) + '_m' + str(f_idx)+'_s'+str(seed))
    input_data2 = lz_data_dict('Train' + str(T_size) + '_lz_l' + str(layer_no) + '_w' + str(sigma[1]) + '_m' + str(f_idx)+'_s'+str(seed))
    input_data3 = lz_data_dict('ES' + str(T_size) + '_lz_l' + str(layer_no) + '_w' + str(sigma[0]) + '_m' + str(f_idx)+'_s'+str(seed))

    fig, ax = plt.subplots()
    ax.margins(0.5)

    plt.axvline(x=kc_, linestyle='dashed', color='grey')

    def to_dict(item, idd =0):
        unique_kc = list(set(list(item[:, 1])))
        unique_kc.sort()
        xs, ys, stds, counts = [], [], [], []
        for max_k in unique_kc:

            item_ = item[item[:, 1] <= max_k]
            if idd == 1 and args.f_idx!=2:
                minte = min(item_[:, 3])
                item_ = item_[item_[:, 3] == minte]
            ym = mean(list(item_[:, 2]))
            try:
                std = stdev(item_[:, 2])
            except:
                std = 0
            xs.append(max_k)
            ys.append(ym)
            stds.append(std)
            counts.append(len(list(item_[:, 2])))
        return unique_kc, xs, ys, stds, counts

    for idd, item in enumerate([input_data1,input_data3, input_data2, arr_rand, arr_rand2]):
        colours = {0:'#1f77b4',2:'#d00000', 3:'#1d9641', 4:'#73d190',1:'#6d97b5'}

        if idd != 3:
            unique_kc, xs, ys, stds, counts = to_dict(item, idd)
            if idd == 0:
                minkc = min(unique_kc)
                idx_xs = xs.index(minkc)
                for_input_data3 = [minkc,ys[idx_xs],stds[idx_xs]]
        else:
            xs, ys, stds = item['ks'], item['eg'], item['egstd']
            unique_kc =xs
            min_kcc, max_kcc = min(xs), max(xs)
            min_eg = min(ys)
            for_later = xs
            idx_xs = xs.index(min_kcc)
            for_much_layer = [min_kcc, ys[idx_xs], stds[idx_xs]]

        unique_kc = [k if k!=14 else 7 for k in unique_kc]
        xs = [k if k != 14 else 7 for k in xs]

        if idd == 4:
            xs.append(for_much_layer[0])
            ys.append(for_much_layer[1])
            stds.append(for_much_layer[2])

        if max(unique_kc)<160:
            xs2 = [max(unique_kc)+3.5*i for i in range(50)]
            xs2 = [i for i in xs2 if i<160]
            ys2 = [ys[-1] for _ in range(len(xs2))]
            stds2 = [stds[-1] for _ in range(len(xs2))]

        if args.f_idx == 3 and idd==2:
            stds[0]=0.7*stds[1]

        if idd == 1:
            idx4 = [i for i in range(len(xs)) if xs[i]<for_input_data3[0]]
            xs = [xs[i] for i in range(len(xs)) if i in idx4]
            unique_kc = [unique_kc[i] for i in range(len(unique_kc)) if i in idx4]
            ys = [ys[i] for i in range(len(ys)) if i in idx4]
            stds = [stds[i] for i in range(len(stds)) if i in idx4]
            xs = xs+[for_input_data3[0]]
            ys = ys+[for_input_data3[1]]
            stds = stds+[for_input_data3[2]]

        label = r'$\epsilon_G\mid 10\mid 1$' if idd == 0 else (r'$\epsilon_G\mid 10\mid 8$' if idd == 2 else ('$\epsilon_G$ unbiased' if idd == 3 else '__'))
        alpha = 0.5 if idd not in [1] else 0.25
        ax.fill_between(xs,[i + j for i,j in zip(ys, stds)],[i - j for i,j in zip(ys, stds)], alpha = alpha, color = colours[idd])
        ax.scatter(xs, ys, alpha=1, color=colours[idd], label = label, s=10)
        ax.set_xlim([0,80])
        ax.set_ylim([0,0.6])
        ax.set_xticks([0,40,80,120,160])

        if mode == 'norm' and idd not in [1,4]:
            try:
               ax.fill_between(xs2, [i + j for i, j in zip(ys2, stds2)], [i - j for i, j in zip(ys2, stds2)], alpha=0.25,
                                color=colours[idd])
            except:
                pass
        if idd == 4:
            xs, ys = list(item[:,1]), list(item[:,3])
            xs += [min_kcc]
            ys += [0]
            xs = [i if i!=14 else 7 for i in xs]
            ax.plot(xs, ys, linestyle = ':', color = colours[idd])
            ax.plot([min_kcc, max_kcc], [0,0], linestyle=':', color=colours[3], label=r'$\epsilon_S\mid$ unbiased')
            ax.scatter([min_kcc], [min_eg], alpha=1, color=colours[3], label=label, s=10)

    ax.set_ylim([-0.025, 0.675])
    x_pac = for_later

    df=pd.read_csv('data/data1gh/ms_kc.txt',names=['k','p','f'],delimiter=' ')
    df=df.sort_values(by=['k'])

    df.reset_index(drop=True,inplace=True)
    kdf = list(df['k'])
    pdf = list(df['p'])

    df={}
    for i in range(len(kdf)):
        df[kdf[i]]=sum(pdf[:i+1])

    x_pac2 = [(k-7)*np.log(2) if k not in df else 128*np.log(2)+np.log(df[k]) for k in x_pac]

    y_pac = [1 - np.e ** ((-1 * k+np.log(2*100/100)) / (100)) for k in x_pac2]
    y_pac_bayes = [1 - np.e ** ((np.log(tc) -1 * k+np.log(2*100/100)) / (100)) for tc,k in zip(counts_rand,x_pac2)]

    ax.plot(x_pac, y_pac_bayes, label='PAC-Bayes', color='#1d9641')
    ax.plot(x_pac, y_pac, label='PAC', color='#2A5738')

    fig.set_size_inches(3,2.4)
    ax.set_xlabel('Cutoff LZ Complexity (Model Capacity)')
    ax.set_ylabel('Error')
    ax.set_ylim([-0.025,0.725])
    ax.set_ylim([-0.025, 0.625])

    ax.legend(loc='lower left',
              bbox_to_anchor= (-0.175, 1.01),
              fontsize='x-small',
              title_fontsize='x-small',
              ncol=3,
              frameon=True)
    if f_idx == 1:
        plt.savefig(f'plots/1i.pdf',dpi=300,bbox_inches='tight')

if __name__=='__main__':
    args.mode = 'rand' if args.seed == -1 else 'norm'
    for i in range(1,6):
        args.f_idx = i
        if i != 1:
            continue

        if args.f_idx == 2:
            counts_rand = [143, 2354, 23490, 164315, 900492, 3850306, 13148477, 35770419, 78170403, 137368447,
                           197400509, 240670420,
                           261662564, 267542392, 268385327, 268434254, 268435456]
        elif args.f_idx == 1:
            counts_rand = [2, 8, 20, 87, 293, 1009, 3050, 9165, 25938, 69373, 176812, 426190, 976559, 2114650, 4331930,
                           8402117,
                           15374812, 26574488, 43213652, 66212269, 95240574, 128774209, 163776031, 196600335, 224000077,
                           244165408,
                           257162516, 264192311, 267195389, 268167684, 268397604, 268432995, 268435408, 268435456]
        elif args.f_idx == 3:
            counts_rand = [2, 9, 28, 135, 433, 1361, 4247, 12190, 35029, 93872, 243691, 599727, 1402150, 3109658,
                           6527379,
                           12928892, 24082573, 42070692, 68528341, 103544539, 144518090, 185738447, 220880012,
                           245712699,
                           259826090, 265968632, 267948485, 268380201, 268431735, 268435364, 268435456]

        elif args.f_idx == 5:
            counts_rand = [5, 70, 607, 3749, 18768, 74528, 252721, 762309, 2085843, 5208865, 11937680, 24910084,
                           47277045,
                           81164335, 125494135, 174333456, 217480979, 246846449, 261564570, 266846821, 268182151,
                           268408257,
                           268434105, 268435446, 268435456]

        elif args.f_idx == 4:
            counts_rand = [2, 3, 62, 150, 482, 1872, 5324, 16017, 45719, 120864, 306121, 730539, 1643734, 3490904,
                           6962967,
                           13086255, 23116489, 38347482, 59751151, 87246350, 119617178, 154180066, 187536497, 216422099,
                           238597903, 253545098, 262216954, 266367767,
                           267927181, 268349871, 268426638, 268434900, 268435440, 268435456]

        nn_pac_bayes = [0, -53.968863239092386, -85.44661940441168, -55.478370905919924, -54.4572454295059,
                        -111.83228634229718]

        if args.mode == 'rand':
            args.seed = -1
        if args.collect == 'False':
            args.collect = False
        else:
            args.collect = True

        main(args.f_idx, args.train_size, args.num_layers, args.mode, args.seed)