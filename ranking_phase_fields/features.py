import numpy as np
import pandas as pd
from pathlib import Path

def sym2num(data, features):
    """Convert symbolic elements to numeric vectors based on feature CSV."""
    features_dir = Path(__file__).parent.parent / "data" / "elemental_features"
    features_csv = features_dir / f"{features}.csv"
    df = pd.read_csv(features_csv)
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
    """Create a dictionary mapping feature tuples to element symbols."""
    features_dir = Path(__file__).parent.parent / "data" / "elemental_features"
    features_csv = features_dir / f"{features}.csv"
    elfeatures = pd.read_csv(features_csv)
    dic = {}
    for i, element in zip(range(len(elfeatures)), elfeatures['element']):
        dic[tuple(elfeatures.loc[i][1:])] = element
    return dic, elfeatures.shape[1] - 1

def num2sym(data, features):
    """Convert numerical vectors back to symbolic element lists."""
    eldic, nfeatures = num2sym_dic(features)
    nelements = int(len(data[0]) / nfeatures)

    phases_alpha = []
    for vector in data:
        tupvector = [tuple(element) for element in np.array_split(vector, nelements)]
        alpha = [eldic[tup] for tup in tupvector]
        phases_alpha.append(sorted(alpha))
    return phases_alpha

def vec2name(ndes, natom):
    """Get feature index for each atom."""
    return [ndes * i for i in range(natom)]

