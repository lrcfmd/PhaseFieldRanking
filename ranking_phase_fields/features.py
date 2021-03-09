import numpy as np
from ranking_phase_fields.symbols import *

def read_features(f):
    lines = open(f,'r').readlines()
    return np.asarray([float(i) for i in lines])

def pad(phases):
    """ Padding phases with 0s to make them all of equal length """
    M = max([len(phase) for phase in phases])
    for phase in phases:
        d = M - len(phase)
        if d != 0:
            phase += [0 for i in range(d)]
    return phases

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

    return vectors

def num2sym(number, feature):
    numbers = [str(int(num)) for num in read_features(f'TABLES/{feature}.table')]
    dic = {num: sym for num, sym in zip(numbers, symbols)}
    return dic[str(int(number))]
