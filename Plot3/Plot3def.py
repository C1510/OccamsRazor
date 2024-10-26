import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# MNIST data
filepath = "data/Acc_CSR_Results_2/"


layers = [10]
sigmas = [1,2]
corrupt = [0,1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,60,70,80,90,100]
corrupt=[0,25,50]
Acc_CSR_dict = {}
#xaxis = np.linspace(1,2.5,31)

for layer in layers:
    for sigma in sigmas:
        for c in corrupt:
            data_array = np.zeros((1000,2))
            for file in range(10):
                file_to_open_path = str(filepath) + "CSR_MNIST_1K_layer_" + str(layer) + "_sigma_" + str(sigma) + "_corrupt_" + str(c) + "_run_" + str(file)
                opened_file = pd.read_pickle(file_to_open_path)
                data_array[file*100:file*100+100] = opened_file['csr']
            Acc_CSR_dict['layer' + str(layer) + "sigma" + str(sigma) + "corrupt" + str(c)] = data_array

print(Acc_CSR_dict.keys())

# Change these variable to change the plot
T_size = 1000
layer_no = 10
for c in [0, 25, 50]:

    labels = ['1 | 10', '2 | 10']

    input_data1 = Acc_CSR_dict['layer' + str(layer_no) + "sigma1" + "corrupt" + str(c)]
    input_data2 = Acc_CSR_dict['layer' + str(layer_no) + "sigma2" + "corrupt" + str(c)]

    mean1 = np.mean(input_data1[:, 1])
    mean2 = np.mean(input_data2[:, 1])

    # Create Fig and gridspec
    fig = plt.figure(dpi=300)
    grid = plt.GridSpec(6, 6, hspace=0, wspace=0)

    # Define the axes
    ax_main = fig.add_subplot(grid[1:, :-1])
    ax_right = fig.add_subplot(grid[1:, -1], xticklabels=[], yticklabels=[])
    ax_top = fig.add_subplot(grid[0, 0:-1], xticklabels=[], yticklabels=[])

    # Scatterplot on main ax
    s = 2
    ax_main.scatter(input_data1[:, 1], 1 - input_data1[:, 0], color='blue', label=labels[0], alpha=0.3, s=s)
    ax_main.scatter(input_data2[:, 1], 1 - input_data2[:, 0], color='red', label=labels[1], alpha=0.3, s=s)
    ax_main.set_ylim(0, 1)
    ax_main.set_xlim(0, 0.3)
    ax_main.set_xticks([0,0.1,0.2,0.3])


    # histogram on the bottom
    ax_top.hist(input_data1[:, 1], np.linspace(0, 1, 2000), histtype='stepfilled', orientation='vertical', color='blue')
    ax_top.hist(input_data2[:, 1], np.linspace(0, 1, 2000), histtype='stepfilled', orientation='vertical', color='red')
    ax_top.tick_params(axis='both', which='both', length=0)
    ax_top.axis('off')
    ax_top.set_xlim(0, 0.5)

    # histogram in the right
    ax_right.hist(1 - input_data1[:, 0], np.linspace(0, 1, (2000 - T_size) * 2), histtype='stepfilled',
                  orientation='horizontal', color='blue')
    ax_right.hist(1 - input_data2[:, 0], np.linspace(0, 1, (2000 - T_size) * 2), histtype='stepfilled',
                  orientation='horizontal', color='red')
    ax_right.tick_params(axis='both', which='both', length=0)
    ax_right.axis('off')
    ax_right.set_ylim(0, 1)

    # Decorations
    ax_main.legend(title=u'$\sigma_w$ | N$^o$ Layers', fontsize='x-small', title_fontsize='x-small')
    ax_main.set(xlabel='Critical Sample Ratio', ylabel='Generalisation Error')

    ax_main.axvline(x=mean1, color='blue', linestyle='-.', linewidth=0.75)
    ax_main.axvline(x=mean2, color='red', linestyle='-.', linewidth=0.75)
    fig.set_size_inches(3,2.4)
    corrupt_dict = {0:'d', 25:'e', 50:'f'}
    fig.savefig(f'plots/3{corrupt_dict[c]}.pdf', bbox_inches='tight', dpi=300)