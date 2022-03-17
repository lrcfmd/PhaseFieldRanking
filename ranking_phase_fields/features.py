import numpy as np
from ranking_phase_fields.symbols import *
from ranking_phase_fields.numberedsymbols import *
#from numberedsymbols import *

def read_features(f):
    lines = open(f,'r').readlines()
    return np.asarray([float(i) for i in lines])

def pad(phases, M=None):
    """ Padding phases with 0s to make them all of equal length """
    if not M:
        M = max([len(phase) for phase in phases])
    for phase in phases:
        d = M - len(phase)
        if d != 0:
            phase += [0 for i in range(d)]
    return phases, M

def sym2num(data, features):
    n = len(features)
    dics = [ {} for i in range(n)]
    #create n new descriptors from the tables
    for i in range(n):
       table = read_features(f'TABLES/{features[i]}.table')
       dics[i]  = {sym: float(num) for sym, num in zip(symbols, table)}

    vectors = []
    for vector in data:
        numbers = []
        for el in vector:
            for i in range(n): 
                numbers.append(float(dics[i][el]))
        vectors.append(numbers)

    return np.asarray(vectors)

def num2sym(number, feature):
    numbers = [str(int(num)) for num in read_features(f'TABLES/{feature}.table')]
    dic = {num: sym for num, sym in zip(numbers, symbols)}
    return dic[str(int(number))]

def unpet(vector):
    vector = sorted(list(map(int, vector)))
    # remove 0 from the padded ends
    vector = [i for i in list(numberedsymbols[vector]) if i != '0'] 
    return ' '.join(vector)

#print(unpet(np.array([3.0, 4, 3.0])))
