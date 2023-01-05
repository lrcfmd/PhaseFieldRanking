import numpy as np
import pandas as pd
from ranking_phase_fields.symbols import *
#from symbols import *

# ORIGINALLY, features are given as a list of magpie features stored as 'tables'
# -----------------------------------------------------------------------------

def read_features(f):
    """ This function works for the original submission, 
    where features are given as a list of magpie features stored as 'tables' """
    lines = open(f,'r').readlines()
    return np.asarray([float(i) for i in lines])

def atom2vec_dic(features):
    """ This function works for the original submission, 
    where features are given as a list of magpie features stored as 'tables' """
    n = len(features)
    dics = [ {} for i in range(n)]
    #create n new descriptors from the tables
    for i in range(n):
       table = read_features(f'TABLES/{features[i]}.table')
       dics[i]  = {sym: float(num) for sym, num in zip(symbols, table)}
    return dics

def sym2num_original(data, features):
    """ This function works for the original submission, 
    where features are given as a list of magpie features stored as 'tables' """
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

def num2sym_original(number, feature):
    """ This function works for the original submission, 
    where features are given as a list of magpie features stored as 'tables' """
    numbers = [str(int(num)) for num in read_features(f'TABLES/{feature}.table')]
    dic = {num: sym for num, sym in zip(numbers, symbols)}
    return dic[str(int(number))]

# EXPERIMENTALLY, in the features branch to study effects on train - test distributions, 
# different sets of features are read from the .csv files in ranking_phase_fields/elemental_features 
# --------------------------------------------------------------------------------------------------

def sym2num(data, features):
    """ getting features from .csv dataframe """
    df = pd.read_csv(f'elemental_features/{features}.csv')
    vectors = []
    for vector in data:
        numbervector = []
        for element in vector:
            bf = df.loc[df['element'] == element]
            try:
                numbervector += list(bf.values[0][1:])
            except:
                print(vector, 'is problematic')
                break
        if len(numbervector) == len(vector) * (len(df.columns) - 1):
            vectors.append(numbervector)
    return np.array(vectors)

def num2sym(data, features):
    df = pd.read_csv(f'elemental_features/{features}.csv')
    nfeatures = len(df.columns) - 1
    positions = np.arange(0,len(data[0]), nfeatures)
    feature = df.columns[1]
    phases = []
    for vector in data:
        numerical = [vector[int(position)] for position in positions]
        idx = df.index[df[feature].isin(numerical)].to_list()
        phases.append(sorted(df['element'][idx]))
    return phases


if __name__ == "__main__":
# create magpie_37.csv from TABLES features
#------------------------------------------ 
#   features = open('TABLES/features.dat', 'r').readline().strip().split(', ')
#   df = pd.DataFrame({'element': symbols})
#   for feature in features:
#       table = read_features(f'TABLES/{feature}.table')
#       df[feature] = list(table)
#   df.to_csv('magpie_37.csv', index=False)

#------------------------------------------ 
# check new sym2num and num2sym

    data = [['O', 'Cm', 'Cl', 'B'],['Te', 'S', 'Ga', 'Bi']]
    vectors = sym2num(data, 'megnet16')
    phases = num2sym(vectors, 'megnet16')
    print(data)
    print(len(vectors))
    print(phases)
