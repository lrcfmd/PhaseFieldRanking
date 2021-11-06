import numpy as np
from ranking_phase_fields.symbols import *
#from symbols import *

def read_features(f):
    lines = open(f,'r').readlines()
    return np.asarray([float(i) for i in lines])

def atom2vec_dic(features):
    n = len(features)
    dics = [ {} for i in range(n)]
    #create n new descriptors from the tables
    for i in range(n):
       table = read_features(f'TABLES/{features[i]}.table')
       dics[i]  = {sym: float(num) for sym, num in zip(symbols, table)}
    return dics

def sym2num(data, features):
    n = len(features)
    dics = atom2vec_dic(features)

    vectors = []
    for vector in data:
        numbers = []
        for el in vector:
            for i in range(n): 
                numbers.append(float(dics[i][el]))
        vectors.append(numbers)
    return np.array(vectors)

def num2sym(number, feature):
    numbers = [str(int(num)) for num in read_features(f'TABLES/{feature}.table')]
    dic = {num: sym for num, sym in zip(numbers, symbols)}
    return dic[str(int(number))]

if __name__ == "__main__":
    import pickle
    features = open('TABLES/features.dat', 'r').readline().strip().split(', ')
    for f in features: print(f)
    dics = atom2vec_dic(features)
    
    atom2vec = {s: [dic[s] for dic in dics] for s in symbols}

    with open('atom2vec_dic.pickle', 'wb') as handle:
        pickle.dump(atom2vec, handle, protocol=pickle.HIGHEST_PROTOCOL)
