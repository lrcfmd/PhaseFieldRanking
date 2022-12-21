import sys
from scipy.stats import kendalltau as tau
import pandas as pd
import matplotlib.pyplot as plt

def get_list(n, check=None):
    f = 'quaternary_AE_test_scores.csv'
    if check:
       df = pd.read_csv(check)
    else:
       df = pd.read_csv(f'features_{n}_mg/{f}')
    df['rank'] = df.index
    df = df.sort_values(by=['Phase fields'])
    return df['rank'].values

base = get_list(37)



taus = []
for i in range(1,37):
    r, _ = tau(base, get_list(i))
    taus.append(r)

plt.scatter([i for i in range(1,37)], taus) 
plt.xlabel('N features', fontsize=14)
plt.ylabel("Kendall's tau", fontsize=14)
plt.show()
