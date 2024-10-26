import numpy as np
import random
import pickle
from ..complexities import K_lemp_ziv
from itertools import product
import argparse
import sys
from statistics import mean, stdev
import glob

def decimal(x):
    return int("".join(map(str, x)), 2)

def sampling(m, n, train_size, seed, mode_='norm'):
    num_samples = 2 ** n
    input_x = np.array(list(product([0, 1], repeat=n)))
    input_y = np.zeros((num_samples, 2))
    input_y[:, 1] = 1  

    if n == 7:
        funcs_list = [
            '0001' * 32,
            '10011010' * 16,
            '0011011110100111010010111001011100011100101110111011101011101111'
            '0000110001100011111111100111011001111010010101011101111001110001',
            '10010111000101011001010100010111' * 4,
            '1001' * 32,
            ''.join([str(bin(i).count('1') % 2) for i in range(128)])
        ]
    elif n == 5:
        funcs_list = [
            '01010101010101010101010101010101',
            '01001111000111111110101111110100',
            '1001' * 8,
            '0011' * 8,
            '00110011001100110011001100110011',
            ''.join([str(bin(i).count('1') % 2) for i in range(32)])
        ]
    else:
        raise ValueError("Unsupported value of n.")

    f = funcs_list[m]

    for i, val in enumerate(f):
        if val == '1':
            input_y[i, :] = [1, 0]

    if seed == -1 or seed == -2:
        np.random.seed()
    else:
        np.random.seed(seed)

    complete_data = np.hstack((input_x, input_y))

    if seed != -2:
        np.random.shuffle(complete_data)

    train_data = complete_data[:train_size]
    test_data = complete_data[train_size:]

    train_x = train_data[:, :n]
    train_y = train_data[:, n:]
    test_x = test_data[:, :n]
    test_y = test_data[:, n:]

    train_dict = {decimal(train_x[i]): train_y[i] for i in range(train_size)}
    error_dict = {decimal(train_x[i]): [i] for i in range(train_size)}

    all_indices = set(range(num_samples))
    train_indices = set(map(decimal, train_x))
    test_indices = sorted(all_indices - train_indices)
    train_indices = sorted(train_indices)

    if mode_ == 'norm':
        return train_x, train_y, test_x, test_y, train_dict, error_dict, input_y[:, 0]
    else:
        return f, train_indices, test_indices

parser = argparse.ArgumentParser()
parser.add_argument('-f_idx', '--f_idx', type=int, default=1)
parser.add_argument('-s', '--seed', type=int, default=20)
parser.add_argument('-mode', '--mode', type=str, default='norm')
parser.add_argument('-mode_main', '--mode_main', type=str, default='train')
parser.add_argument('-n', '--n', type=int, default=7)
parser.add_argument('-train_size', '--train_size', type=int, default=100)
parser.add_argument('-collect', '--collect', type=str, default='False')

args = parser.parse_args()
if args.mode == 'rand':
    args.seed = -1
args.collect = args.collect.lower() == 'true'

def generate(rank):
    start = int(rank * (2 ** 24))
    end = int((rank + 1) * (2 ** 24))
    y_true, train_idxs, test_idxs = sampling(
        m=args.f_idx, n=args.n, train_size=args.train_size, seed=args.seed, mode_='not'
    )
    kc_y_true = K_lemp_ziv(y_true)
    test_size = len(test_idxs)
    arr_rand = []

    print(f"Processing counts from {start} to {end}")

    for count in range(start, end):
        str_temp = format(count, "032b")
        if str_temp[:4] != '0000':
            continue

        if args.mode == 'rand':
            train_idxs = random.sample(range(2 ** args.n), args.train_size)
            test_idxs = [i for i in range(2 ** args.n) if i not in train_idxs]

        f_ = list(y_true)
        for idx_t, i in enumerate(test_idxs):
            f_[i] = str_temp[4 + idx_t]

        f_modified = "".join(f_)
        err = sum(a != b for a, b in zip(y_true, f_modified))
        kc_f_modified = K_lemp_ziv(f_modified)
        error_rate = err / test_size

        arr_rand.append([kc_y_true, kc_f_modified, error_rate])

    output_filename = f'n7/temp/{args.f_idx}_{args.seed}_{args.train_size}_{rank}'
    with open(output_filename, 'wb') as handle:
        pickle.dump(np.array(arr_rand), handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    except ImportError:
        rank = 0

    if not args.collect:
        generate(rank)
    else:
        files = glob.glob('n7/temp/*')
        files = [i for i in files if f'{args.f_idx}_{args.seed}' in i]
        pickles = []
        for file_path in files:
            with open(file_path, 'rb') as handle:
                try:
                    data = pickle.load(handle)
                    if data.ndim > 1:
                        pickles.append(data)
                except Exception as e:
                    print(f"Failed to load {file_path}: {e}")
                    continue
        if pickles:
            combined_data = np.concatenate(pickles)
            unique_kc = np.unique(combined_data[:, 1])
            xs, ys, stds = [], [], []
            for max_k in sorted(unique_kc):
                subset = combined_data[combined_data[:, 1] <= max_k]
                error_rates = subset[:, 2]
                xs.append(max_k)
                ys.append(np.mean(error_rates))
                stds.append(np.std(error_rates))

            result = {'ks': xs, 'eg': ys, 'egstd': stds}
            output_filename = f'n7/{args.f_idx}_{args.seed}_{args.train_size}'
            with open(output_filename, 'wb') as handle:
                pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print("No data files loaded.")