import numpy as np
from symbols import *

def read_features(f):
    lines = open(f,'r').readlines()
    return np.asarray([float(i) for i in lines])

table = read_features(f'Density.table')
dics  = {sym: float(num) for sym, num in zip(symbols, table)}

print(dics['Fe'])
print(dics['0'])
