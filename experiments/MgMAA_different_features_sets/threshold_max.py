import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
import sys

def threshold_fraction(train, test):
    ftrain, ftest = [], []
    fthresh = []
    for i,thr in enumerate(train):
        ftr = len(train[train < thr])/len(train)
        fte = len(test[test < thr])/len(test)
        ftrain.append(ftr)
        ftest.append(fte)
        #if fte - ftr:
        if ftr - fte > 0:
            fthresh.append((ftr - fte) / (ftr + fte))
        else:
            fthresh.append(0)
    return ftrain, ftest, fthresh

training, test = sys.argv[1:]

traindata = pd.read_csv(training)
testdata= pd.read_csv(test)

score = 'Norm. score'
#score = 'scores'
train = traindata.sort_values([f'{score}'])[f'{score}'].values
test = testdata.sort_values([ f'{score}'])[f'{score}'].values

ftrain, ftest, fthresh = threshold_fraction(train, test)
# print peak threshold
arg = np.argmax(fthresh)
print('Peak threshold:', train[arg])


# plotting
plt.plot(train,ftrain, label='Training set')
plt.plot(train,ftest, label='Test set')
plt.plot(train,fthresh,label='Threshold fractional score')
plt.ylabel("Fraction of 'outliers' phase fields")
plt.xlabel('Threshold RE')
plt.legend()
plt.show()
