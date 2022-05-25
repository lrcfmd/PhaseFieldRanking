import sys
import pickle
import pandas as pd
from generate_study import permute 

df = pd.read_csv('icsd2021_phases.csv')
training = list(map(permute,df['phases']))
