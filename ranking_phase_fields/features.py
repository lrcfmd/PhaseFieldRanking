import numpy as np
import pandas as pd
from ranking_phase_fields.symbols import *

# In the features branch to study effects on train - test distributions, 
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

def num2sym_dic(features):
    """create numbers to symbols dictionary a dictionary"""
    elfeatures = pd.read_csv(f'elemental_features/{features}.csv')
    dic = {}
    nelements, nfeatures = elfeatures.shape
    for i, element in zip(range(nelements), elfeatures['element']):
        dic[tuple(elfeatures.loc[i][1:])] = element
    return dic, nfeatures - 1 

def num2sym(data, features):
    """ transform numerical vectors for phases into alphabetic representation """
    eldic, nfeatures = num2sym_dic(features)
    nelements = int(len(data[0]) / nfeatures)

    phases_alpha = []
    for vector in data:
        tupvector = [tuple(element) for element in np.array_split(vector, nelements)]
        alpha = [eldic[tup] for tup in tupvector]
        phases_alpha.append(sorted(alpha))

    return phases_alpha 


if __name__ == "__main__":
#------------------------------------------ 
# check new sym2num and num2sym

    data = [['O', 'Cm', 'Cl', 'B'],['Te', 'S', 'Ga', 'Bi']]
    vectors = sym2num(data, 'megnet16')
    phases = num2sym(vectors, 'megnet16')
    print(data)
    print(len(vectors))
    print(phases)
