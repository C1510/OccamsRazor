import math
import numpy as np
from math import log

#######################

# Old LZ 76

def KC_LZ(string):
    n=len(string)
    s = '0'+string
    c=1
    l=1
    i=0
    k=1
    k_max=1
    stop=0

    while stop==0:
        if s[i+k] != s[l+k]:
            if k>k_max:
                k_max=k # k_max stores the length of the longest pattern in the LA that has been matched somewhere in the SB

            i=i+1 # we increase i while the bit doesn't match, looking for a previous occurence of a pattern. s[i+k] is scanning the "search buffer" (SB)

            if i==l: # we stop looking when i catches up with the first bit of the "look-ahead" (LA) part.
                c=c+1 # If we were actually compressing, we would add the new token here. here we just count recounstruction STEPs
                l=l+k_max # we move the beginning of the LA to the end of the newly matched pattern.

                if l+1>n: # if the new LA is beyond the ending of the string, then we stop.
                    stop=1

                else: #after STEP,
                    i=0 # we reset the searching index to beginning of SB (beginning of string)
                    k=1 # we reset pattern matching index. Note that we are actually matching against the first bit of the string, because we added an extra 0 above, so i+k is the first bit of the string.
                    k_max=1 # and we reset max lenght of matched pattern to k.
            else:
                k=1 #we've finished matching a pattern in the SB, and we reset the matched pattern length counter.

        else: # I increase k as long as the pattern matches, i.e. as long as s[l+k] bit string can be reconstructed by s[i+k] bit string. Note that the matched pattern can "run over" l because the pattern starts copying itself (see LZ 76 paper). This is just what happens when you apply the cloning tool on photoshop to a region where you've already cloned...
            k=k+1

            if l+k>n: # if we reach the end of the string while matching, we need to add that to the tokens, and stop.
                c=c+1
                stop=1

    return c


def calc_KC(s):
    L = len(s)
    if s == '0'*L or s == '1'*L:
        return np.log2(L)
    else:
        return np.log2(L)*(KC_LZ(s)+KC_LZ(s[::-1]))/2.0


def log2(x):
    return log(x)/log(2.0)

def entropy(f):
    n0=0
    n=len(f)
    for char in f:
        if char=='0':
            n0+=1
    n1=n-n0
    if n1 > 0 and n0 > 0:
        return log2(n) - (1.0/n)*(n0*log2(n0)+n1*log2(n1))
    else:
        return 0

########################



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

def decimal(x):
    n = len(x)
    output = 0
    for i in range(len(x)):
        output += x[i]*2**(n-1-i)
    return output

def K_lemp_ziv(sequence):
    if (np.sum(sequence == 0) == len(sequence)) or (np.sum(sequence == 1) == len(sequence)) :

        out = math.log2(len(sequence))
    else:
        forward = sequence
        backward = sequence[::-1]

        out = math.log2(len(sequence))*(lz_complexity(forward) + lz_complexity(backward))/2

    if out == 14.0:
        return 7.0
    return out


if __name__=='__main__':
    print('0001'*32)
    print(len('00010101000101010001010100010101' * 4))
    lz = K_lemp_ziv('10010111000101011001010100010111' * 4)
    print(lz)
