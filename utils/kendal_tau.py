import sys
from scipy.stats import kendalltau as tau
import pandas as pd
import matplotlib.pyplot as plt

def get_list(f='quaternary_VAE_random_200_test_scores.csv'):
    df = pd.read_csv(f'{f}')
    df = df.sort_values(by=['Norm. score'])
    df = df.reset_index()
    df['rank'] = df.index
    df = df.sort_values(by=['Phase fields'])
    return df['rank'].values


def get_table(scores):
    table = []
    for f1 in scores:
        table_i = []
        for f2 in scores:
#            if f1 == f2: continue
            t, _ = tau(get_list(f1), get_list(f2))
            table_i.append(round(t,2))
        table.append(table_i)
    return table


scores = [
        'quaternary_VAE_random_200_test_scores.csv',
        'quaternary_VAE_megnet16_test_scores.csv',
        'quaternary_VAE_magpie_test_scores.csv',
        'original_VAE_magpie37_test_scores.csv',
        'quaternary_VAE_LEAF_L_test_scores.csv',
        'quaternary_VAE_mat2vec_test_scores.csv',
        ]


table = get_table(scores)
for t in table:
    print(t)

sys.exit(0)

# -- plot taus againts the base ranking ---
base = get_list()
taus = []
for i in scores:
    r, _ = tau(base, get_list(i))
    taus.append(r)

#labels=['Megnet16', 'Magpie22', 'LEAF36', 'Mat2Vec200', 'Random200']
labels=['Megnet16', 'Magpie22', 'Magpie37', 'LEAF36', 'Mat2Vec200']
n_features = [16, 22, 37, 36, 200]

for i in range(len(labels)):
    plt.scatter([n_features[i]], [taus[i]], label=labels[i])

#plt.scatter(n_features, taus)

#fig, ax = plt.scatter(n_features, taus)
#for i, label in enumerate(labels):
#    ax.annotate(label, (n_features[i], taus[i])) 

plt.xlabel('N features', fontsize=14)
plt.ylabel("Kendall's tau", fontsize=14)
plt.legend()
plt.show()
