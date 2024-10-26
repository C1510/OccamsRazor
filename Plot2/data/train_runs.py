import itertools
import pickle
import glob

import pandas as pd

def run(sw, layer, func, nte):
    filelist = glob.glob('./LZ_Complexity_Results/*')
    filelist = [i for i in filelist if (f'Train{nte}' in i and f'w{sw}' in i and f'l{layer}' in i and f'm{func}_' in i)]
    filelist = [i for i in filelist if str(func) in i]
    df = None
    for i in filelist:
        with open(i, "rb") as output_file:
            e = pickle.load(output_file)['lz']
            if df is None:
                df = pd.DataFrame(e,columns=['kc_t','kc','err'])
            else:
                dftemp = pd.DataFrame(e,columns=['kc_t','kc','err'])
                if dftemp['kc_t'][0]==31.5:
                    df = dftemp
                    break
                df = pd.concat((df,dftemp), ignore_index=True)

    df = df.drop(df[df['kc'] ==0].index)
    df.to_csv(f'../train_runs_henry/{func}_{sw}_{nte}.csv',index=None,header=False, sep = ' ')

sw = 8
layer = 10
func = 2
nte = 32

for sw, layer, func, nte in itertools.product([1,8],[10],[2,10,21],[32,64,85]):
    try:
        run(sw,layer,func,nte)
    except:
        print('failed',sw,layer,func,nte)