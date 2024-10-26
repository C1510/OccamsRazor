import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
filepath = "data/Apriori_CSR_Results/"


layers = [2,5,10]
sigmas = [1,2,4,8]
Apriori_CSR_dict = {}
file_size = 100
n_file = 200

for layer in layers:
    for sigma in sigmas:
        data_array = np.zeros(n_file*file_size)
        for file in range(n_file):
            file_to_open_path = str(filepath) + "Apriori_CSR_1K_layer_" + str(layer) + "_sigma_" + str(sigma) + "_run_" + str(file)
            opened_file = pd.read_pickle(file_to_open_path)
        
            '''
            if file < 100:
                edited_file = opened_file['csr']
                for i in range(file_size):
                    if edited_file[file_size-1-i] == 0:
                        edited_file[file_size-1-i] = 1
                    else:
                        break
            else:
                edited_file = opened_file['csr']
            '''    
                
            edited_file = opened_file['csr']    
            data_array[file*file_size:file*file_size+file_size] = edited_file
        Apriori_CSR_dict['layer' + str(layer) + "sigma" + str(sigma)] = data_array[data_array != 1]


def hist_to_bar(input_data, layers, sigmas):
    fig, axs = plt.subplots(len(sigmas), len(layers), dpi=200, sharey=True)
    label = [str(sigma) + ' | 10 ']
    custom_xlim = (0, 1)
    custom_ylim = (0, 1e-8)
    n_bins = 101
    n_range = (0.0, 1)
    plt.setp(axs, xlim=custom_xlim, yscale='log')

    for i, w in enumerate(sigmas):
        for j, l in enumerate(layers):
            hist, bin_edges = np.histogram(input_data['layer' + str(l) + 'sigma' + str(w)], bins=n_bins, density=True,
                                           range=n_range)
            axs[i].bar((bin_edges[1:] + bin_edges[:-1]) * .5, hist / 20000, width=(bin_edges[1] - bin_edges[0]),
                          label=label[0], color='red' if w==2 else 'blue')

            axs[0].set_ylabel('$P(CSR)$')
            axs[0].yaxis.set_label_coords(-0.2, -0.1)
            axs[i].set_yticks([1e-7, 1e-5, 1e-3])
            axs[i].set_ylim([1e-7, 1e-2])



    for ax in axs.flat:
        ax.set(xlabel='Critical Sample Ratio (CSR)')

    for ax in axs.flat:
        ax.label_outer()

    fig.set_size_inches(3,2.5)
    fig.savefig("plots/3c.pdf",
                bbox_inches='tight')


hist_to_bar(Apriori_CSR_dict, [10], [1,2])