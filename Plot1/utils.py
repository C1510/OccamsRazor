import numpy as np
from itertools import product
import math

def lz_complexity(s):
    i, k, l = 0, 1, 1
    k_max = 1
    n = len(s) - 1
    c = 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k = k + 1
            if l + k >= n - 1:
                c = c + 1
                break
        else:
            if k > k_max:
               k_max = k
            i = i + 1
            if i == l:
                c = c + 1
                l = l + k_max
                if l + 1 > n:
                    break
                else:
                    i = 0
                    k = 1
                    k_max = 1
            else:
                k = 1
    return c

def K_lemp_ziv(sequence):

    if (np.sum(sequence == 0) == len(sequence)) or (np.sum(sequence == 1) == len(sequence)) :

        out = math.log2(len(sequence))

    else:
        forward = sequence
        backward = sequence[::-1]

        out = math.log2(len(sequence))*(lz_complexity(forward) + lz_complexity(backward))/2

    return out

def decimal(x):
    n = len(x)
    output = 0
    for i in range(len(x)):
        output += x[i]*2**(n-1-i)
    return output

def sampling(m, n, train_size, seed, mode_ = 'norm'):
    input_y = np.ones((2 ** n, 2))
    input_y[:, 0] = 0
    input_x = np.array(list(product([0, 1], repeat=n)))

    getbinary = lambda x, n: format(x, 'b').zfill(n)

    if n == 7:
        funcs_list = ['0001' * 32,
                      '10011010' * 16,
                      '00110111101001110100101110010111000111001011101110111010111011110000110001100011111111100111011001111010010101011101111001110001',
                      '10010111000101011001010100010111' * 4,
                      '1001'*32,
                      ''.join([str(getbinary(i,n).count('1')%2) for i in range(128)])
                        ]
        f = funcs_list[m]
        for i in range(128):
            if f[i] == '1':
                input_y[i, :] = [1, 0]

    elif n == 5:
        funcs_list = ['01010101010101010101010101010101',
                      '01001111000111111110101111110100',
                      '1001' * 8,
                      '0011' * 8,
                      '00110011001100110011001100110011',
                      ''.join([str(getbinary(i,n).count('1')%2) for i in range(32)])
                      ]
        f = funcs_list[m]
        for i in range(32):
            if f[i] == '1':
                input_y[i,:] = [1,0]

    if seed == -1 or seed == -2:
        np.random.seed()
    else:
        np.random.seed(seed)

    data_x = input_x
    data_y = input_y
    complete_data =  np.concatenate((data_x, data_y), axis=1)
    if seed != -2:
        np.random.shuffle(complete_data)
    split_data = np.vsplit(complete_data, np.array([train_size,2**n]))
    split_data_x = np.hsplit(split_data[0], np.array([n,n+2]))
    train_x = split_data_x[0]
    train_y = split_data_x[1]
    split_data_y = np.hsplit(split_data[1], np.array([n,n+2]))
    test_x = split_data_y[0]
    test_y = split_data_y[1]


    # Dictionary holding decimal numbers and their associated binary arrays
    train_dict = {}
    for i in range(train_size):
        train_dict[decimal(train_x[i])] = train_y[i]

    # Define error dictionary used to record error on training elements
    error_dict = {}
    for i in range(train_size):
        error_dict[decimal(train_x[i])] = [i]
    #print(error_dict.keys())
    np.random.seed()

    train_list = [int(i) for i in list(error_dict.keys())]
    test_list = [i for i in range(2**n)]
    for i in train_list:
        test_list.remove(i)
    train_list.sort()
    test_list.sort()
    f_train = "".join([f[i] for i in train_list])
    f_test = "".join([f[i] for i in test_list])
    #print(f, f_train, f_test)
    if mode_ == 'norm':
        #print([int(i) for i in train_dict.keys()])
        return train_x, train_y, test_x, test_y, train_dict, error_dict, data_y[:,0]
    else:
        return f, train_list, test_list



if __name__ == '__main__':
    sampling(m=4, n=7, train_size=16, seed=-2)
