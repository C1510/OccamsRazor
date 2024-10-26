import numpy as np
import matplotlib.pyplot as plt
from complexities import K_lemp_ziv
from tqdm import tqdm
import sys
import os
import heapq



for no_ones in [2, 10, 21]:
    # Define a decimal convertion function
    def decimal(x):
        n = len(x)
        output = 0
        for i in range(len(x)):
            output += x[i]*2**(n-1-i)
        return output

    # Set the Target function    
    def set_target(no_ones = 10, special_case = 'n'):
        if special_case == 'n':
            np.random.seed(10)
            target = [0]*128
            index = np.random.choice(128, size = no_ones)
            for i in index:
                target[i] = 1
            target_attribute = K_lemp_ziv(target)
            return np.array(target), target_attribute, index
        elif special_case == 'y':
            target = [0,1]*64
            target_attribute = K_lemp_ziv(target)
            return np.array(target), target_attribute, index

    # Search in the region of function space around the target
    def target_function_search(target, iters=10000, no_bit_flips=20, search_type = 'balanced'):
        data_set = set()
        target_func = target[0]
        np.random.seed(None)
        if search_type == 'balanced':
            for i in tqdm(range(iters)):
                test_func = target_func.copy()
                for j in range((i%no_bit_flips)+1):
                    bit_flip = np.random.randint(128)
                    if target_func[bit_flip] == 1:
                        test_func[bit_flip] = 0
                    else:
                        test_func[bit_flip] = 1
                test_func_complexity = K_lemp_ziv(test_func)
                test_func_error = 1 - np.sum(np.abs(test_func - target_func))/2**7
                data_set.add((decimal(test_func), test_func_complexity, test_func_error))
            return np.array(list(data_set))
        elif search_type == 'biased':
            flipped_bits = target[2]
            for k in tqdm(range(100)):
                zero_bit = np.random.choice(flipped_bits, size = np.random.choice((len(flipped_bits)+1)), replace = False)
                test_func = target_func.copy()
                for l in range(len(zero_bit)):
                    test_func[zero_bit[l]] = 0
                    for i in range(int(iters/100)):
                        test_func_complexity = K_lemp_ziv(test_func)
                        test_func_error = 1 - np.sum(np.abs(test_func - target_func))/2**7
                        data_set.add((decimal(test_func), test_func_complexity, test_func_error))
            return np.array(list(data_set))
                    
                    
                

    def random_function_search(target, prob=0.8, iters=10000):
        target_func = target[0]
        data_set = set()
        for i in tqdm(range(iters)):
            q = np.random.normal(prob, scale=1.0)
            if q > 1:
                q = prob
            elif q < 0:
                q = prob
            test_func = np.random.choice(np.arange(2), p=[q, 1-q], size = 128)
            test_func_complexity = K_lemp_ziv(test_func)
            test_func_error = 1 - np.sum(np.abs(test_func - target_func))/2**7
            data_set.add((decimal(test_func), test_func_complexity, test_func_error))
        return np.array(list(data_set))


    def epsilon_sorter(file, complexity, num_large_val):
        input_file = np.load(file)
        epsilon_dict = {}

        # Set the LZ Values
        lz_values = [7]
        for m in range(5,48):
            lz_values.append(7/2*m)

        largest_vals_dict = {}
        for i in tqdm(range(len(lz_values))):
            if lz_values[i] == complexity:
                error_list = [0.5]*(num_large_val)
                error_list[0] = 1.0
                largest_vals_dict[str(lz_values[i])] = error_list
            else:    
                largest_vals_dict[str(lz_values[i])] = [0.5]*(num_large_val)
            for j in range(len(input_file)):
                if input_file[j,1] == lz_values[i]:
                    for k in range(num_large_val):
                        if input_file[j,2] > largest_vals_dict[str(lz_values[i])][k]:
                            largest_vals_dict[str(lz_values[i])][k] = input_file[j,2]
                            break
                
                #heapq.nlargest(num_large_val, epsilon_list, key=None)[0]
        return largest_vals_dict



    fig = plt.figure(dpi =200)
    # Set these parameters
    lz = set_target(no_ones)[1]
    input_file = epsilon_sorter('data/abc/epsilon_plot_m_' + str(no_ones) + '.npy', complexity=lz, num_large_val=5)

    # Converts dictionary input into numpy array
    output_array = np.zeros((44, 2))
    for i, key in enumerate(input_file.keys()):
        output_array[i,0] = key
    # Choose what the error of the target function is (Should it be 1.0 or average of the top performin cases)
        if output_array[i,0] <= 25:
            output_array[i,1] = np.max(input_file[key])
        else:
            output_array[i,1] = np.mean(input_file[key])

    np.save('data/abc/epsilon_m'+str(no_ones)+'_mean_values.npy', output_array)

    labels = ['1','10','20','40','60','80','100']

    for i in range(len(labels)):
        m = int(labels[i])
        plt.plot(output_array[:,0], output_array[:,1]**m, label = labels[i])

    plt.xlim(7,170)
    plt.ylabel(r'$\langle 1 - \epsilon(f) \rangle ^ m$')
    plt.xlabel('Lempel-Ziv Complexity')

    plt.legend(title = 'm', loc = 'upper right', fontsize='x-small', title_fontsize='x-small')
    fig.set_size_inches(3,2)

    ones_dict = {2:'a', 10:'b', 21:'c'}
    fig.savefig(f"plots/2{ones_dict[no_ones]}.pdf", bbox_inches='tight')
