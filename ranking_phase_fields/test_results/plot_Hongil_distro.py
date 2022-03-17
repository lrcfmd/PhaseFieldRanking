import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

training = pd.read_csv('quaternary_Hongil_VAE_training_scores.csv')
testing = pd.read_csv('quaternary_Hongil_VAE_test_scores.csv')

training.columns = ['phases','scores','norm.scores','']
testing.columns = ['phases','scores','norm.scores','']

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


#plt.hist(training['norm.scores'].values, 120, label='Training, 783')
#plt.hist(training['norm.scores'].values, 80, label='Training, 783')
#plt.hist(testing['norm.scores'].values, 40, alpha=0.5, label='Testing, 81')
plt.hist(testing['norm.scores'].values, 20, alpha=1, label='Testing, 81')
#plt.hist(Li, 20, color='tab:green', alpha=1, label=f'Li, {len(Li)}')
#plt.hist(Bi, 20, color='tab:red', alpha=0.5, label=f'Bi, {len(Bi)}')
#plt.hist(Zn, 20, alpha=1., label=f'Zn, {len(Zn)}')
#lt.hist(S,  40, color='tab:brown', alpha=1., label=f'S, {len(S)}')
#lt.hist(Se, 40, color='tab:pink', alpha=0.6, label=f'Se, {len(Se)}')
#lt.hist(P,  40, alpha=0.6, label=f'P, {len(P)}')
plt.hist(O,  40, alpha=1., label=f'O, {len(O)}')
plt.hist(Br,  40, alpha=0.5, label=f'Br, {len(Br)}')
plt.hist(Cl,  40, alpha=0.5, label=f'Cl, {len(Cl)}')
plt.xlabel('Norm. RE')
plt.ylabel('Number of phase fields')
plt.legend()
plt.show()
