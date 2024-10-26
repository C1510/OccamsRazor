import copy

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import random, sys
from complexities import K_lemp_ziv

m1 = [2,10,21]
kc_1 = [31.5,66.5,101.5]
i=int(sys.argv[1])
m, kc_ = m1[i],kc_1[i]

T_size = 64
layer_no = 10
plot_number = 1
sigma = [1, 8]

def gen_random_data():
    arr_rand = []
    f = '11111111111010111111111111111111111111111111111111111111111111111111111111001111111111111101101111111111111111111111101011101011'
    for count in range(1000):
        ran_idx = random.sample(range(128), 64)
        f_ = copy.deepcopy(f)
        f_ = list(f_)
        for i in ran_idx:
            f_[i]=str(int(random.randint(0,1)))
        f_ = "".join(f_)
        err = sum([1 if i!=j else 0 for i,j in zip(f,f_)])
        arr_rand.append([kc_, K_lemp_ziv(f_),err/64])
    return np.array(arr_rand)

arr_rand = gen_random_data()

def set_target(no_ones=10, special_case='n'):
    if special_case == 'n':
        np.random.seed(10)
        target = [0] * 128
        index = np.random.choice(128, size=no_ones)
        for i in index:
            target[i] = 1
        target_attribute = K_lemp_ziv(target)
        return np.array(target), target_attribute, index
    elif special_case == 'y':
        target = [0, 1] * 64
        target_attribute = K_lemp_ziv(target)
        return np.array(target), target_attribute, index

def lz_data_dict(file):
    import glob
    import pandas as pd
    files = glob.glob('./data/LZ_Complexity_Results/*')
    files = [i for i in files if file in i]
    arrlist = []
    for f in files:
        arr = np.load(f, allow_pickle=True)['lz']
        arrlist.append(arr)
    arr = np.concatenate(arrlist)

    arr = arr[arr[:,0] == kc_]
    return arr

def mode_rows(arr):
    df = pd.DataFrame(arr,index=None, columns=['k1','k','e'])
    df = df.groupby(df.columns.tolist(),as_index=False).size()
    max_num = max(list(df['size']))
    iddf = df[df['size']==max_num]
    iddf.reset_index(drop=True, inplace=True)
    cross = [iddf['k'][0], iddf['e'][0]]
    return cross

lz = kc_ 

labels = [str(sigma[0]) + ' | ' + str(layer_no), str(sigma[1]) + ' | ' + str(layer_no)]

input_data1 = lz_data_dict('Train' + str(T_size) + '_lz_l' + str(layer_no) + '_w' + str(sigma[0]) + '_m' + str(m))[
              0:1000]
input_data2 = lz_data_dict('Train' + str(T_size) + '_lz_l' + str(layer_no) + '_w' + str(sigma[1]) + '_m' + str(m))[
              0:1000]

mean = np.mean(input_data1[:, 1])

cross1 = mode_rows(input_data1)
cross2 = mode_rows(input_data2)
corss_rand = mode_rows(arr_rand)

fig = plt.figure(figsize=(3, 3), dpi=300)
grid = plt.GridSpec(6, 6, hspace=0, wspace=0)

ax_main = fig.add_subplot(grid[1:, :-1])
ax_right = fig.add_subplot(grid[1:, -1], xticklabels=[], yticklabels=[])
ax_top = fig.add_subplot(grid[0, 0:-1], xticklabels=[], yticklabels=[])

ax_main.axhline(y=0.5, color='grey', linestyle=':', linewidth=0.75, alpha = 0.8)

s = 2
ax_main.scatter(arr_rand[:, 1], arr_rand[:, 2], color='green', label='random', alpha=0.3, s=s)
ax_main.scatter(corss_rand[0], corss_rand[1], color='black', marker='x', s=8)

ax_main.scatter(input_data1[:, 1], input_data1[:, 2], color='blue', label=labels[0], alpha=0.3, s=s)
ax_main.scatter(cross1[0], cross1[1], color='black', marker='x', s=8)
ax_main.scatter(input_data2[:, 1], input_data2[:, 2], color='red', label=labels[1], alpha=0.3, s=s)
ax_main.scatter(cross2[0], cross2[1], color='black', s=8, marker='x')
ax_main.set_ylim(-0.01, 0.6)
ax_main.set_xlim(0, 160)
ax_main.set_xticks([0,40,80,120,160])
ax_main.set_aspect(160/0.6)

ax_top.hist(arr_rand[:, 1], np.linspace(0, 160, 93), histtype='stepfilled', orientation='vertical', color='green')

ax_top.hist(input_data1[:, 1], np.linspace(0, 160, 93), histtype='stepfilled', orientation='vertical', color='blue')
ax_top.hist(input_data2[:, 1], np.linspace(0, 160, 93), histtype='stepfilled', orientation='vertical', color='red')
ax_top.tick_params(axis='both', which='both', length=0)
ax_top.axis('off')
ax_top.set_xlim(0, 161)

ax_right.hist(arr_rand[:, 2], np.linspace(0, 1, (129 - T_size) * 2), histtype='stepfilled', orientation='horizontal',
              color='green')
ax_right.hist(input_data1[:, 2], np.linspace(0, 1, (129 - T_size) * 2), histtype='stepfilled', orientation='horizontal',
              color='blue')
ax_right.hist(input_data2[:, 2], np.linspace(0, 1, (129 - T_size) * 2), histtype='stepfilled', orientation='horizontal',
              color='red')
ax_right.tick_params(axis='both', which='both', length=0)
ax_right.axis('off')
ax_right.set_ylim(-0.01, 0.6)

legend = ax_main.legend(title=u'$\sigma_w$ | $N_l$', loc='upper left', fontsize='x-small',title_fontsize='x-small')

ax_main.set(xlabel='Output LZ Complexity', ylabel='Generalisation Error')

xlabels = ax_main.get_xticks().tolist()
ax_main.set_xticklabels(xlabels)
ax_main.axvline(x=lz, color='black', linestyle='--', linewidth=0.75)

handles, labels = ax_main.get_legend_handles_labels()
order = [1,2,0]
ax_main.legend([handles[idx] for idx in order],[labels[idx] for idx in order], title=u'$\sigma_w$ | $N_l$', loc='upper left', fontsize='x-small',title_fontsize='x-small')

fig.set_size_inches(3,3)

i_to_fig = {0:'d', 1:'e', 2:'f'}
fig.savefig(f"plots/1{i_to_fig[i]}.pdf", bbox_inches='tight', pad_inches=0.1, dpi=1000)