""" Takes an original list of scores and extracts a subset list for an element of interest """

import pandas as pd
import sys

# original list of scores
df = pd.read_csv('quaternary_VAE_test_scores.csv')
# element to extract a subset for
el = 'B'

#exctract
df['n_overlap'] = df.phases.apply(lambda x: len({el}.intersection(set(x.split()))))
df['rank_original'] = df.index
ex = df[df.n_overlap >= 1]
ex = ex[['phases','scores','rank_original']]

ex.to_csv("quaternary_LiBAA_VAE_scores.csv",index=False)
print(ex.head())
