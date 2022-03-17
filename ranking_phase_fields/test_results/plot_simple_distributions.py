import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def select_elements(testing):
    Se, S, Bi, P, O, Zn, Li, Br, Cl = [],[],[],[],[],[],[],[],[]
    for phase, norm in zip(testing.phases, testing['norm.scores'].values):
        if 'Se' in phase.split(): Se.append(norm)
        if 'Bi' in phase.split(): Bi.append(norm)
        if 'S' in phase.split(): S.append(norm)
        if 'P' in phase.split(): P.append(norm)
        if 'O' in phase.split(): O.append(norm)
        if 'Br' in phase.split(): Br.append(norm)
        if 'Zn' in phase.split(): Zn.append(norm)
        if 'Li' in phase.split(): Li.append(norm)
        if 'Cl' in phase.split(): Cl.append(norm)

def plot_hists(training, testing, test_name):
    plt.hist(training['scores'].values, 120, label=f'Training, {len(training["norm.scores"].values)}')
    plt.hist(testing['scores'].values, 10, alpha=1, label=f'Testing {test_name}, {len(testing["norm.scores"].values)}')
    #pplt.hist(Br,  40, alpha=0.5, label=f'Br, {len(Br)}')
    plt.xlabel('Ranking, RE')
    plt.ylabel('Number of phase fields')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    try:
        ftr, fte = sys.argv[1:3]
    except Exception:
        print('Provide training and test sets file')
        sys.exit(0)

    test_name = 'Zr/Hf-M-Bi'

    training = pd.read_csv(ftr)
    testing = pd.read_csv(fte)
    training.columns = ['phases','scores','norm.scores','']
    testing.columns = ['phases','scores','norm.scores','']

    plot_hists(training, testing, test_name)
