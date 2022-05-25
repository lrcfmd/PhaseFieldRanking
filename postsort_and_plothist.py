import pandas as pd 
from matplotlib import pyplot as plt
import sys


training, test = sys.argv[1:]

traindata = pd.read_csv(training)
testdata= pd.read_csv(test)

traindata = traindata.sort_values(by=['scaled'])
testdata = testdata.sort_values(by=['scaled'])


plt.hist(traindata['scores'].values, 100, label="Training M-M'-M''-A-A'")
#plt.hist(testdata['scores'].values, 19, label='Test Li-Si-S-I-X')
plt.hist(testdata['scores'].values, 24, label='Test Li-X-X-S-I')
plt.xlabel('VAE RE')
plt.ylabel('Number of phase fields')
plt.legend()
plt.show()


traindata.to_csv(training, index=False)
testdata.to_csv(test, index=False)
