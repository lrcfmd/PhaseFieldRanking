import pandas as pd 
from matplotlib import pyplot as plt
import sys


training, test = sys.argv[1:]

traindata = pd.read_csv(training)
testdata= pd.read_csv(test)

#traindata = traindata.sort_values(by=['scaled'])
#testdata = testdata.sort_values(by=['scaled'])


#plt.hist(traindata['scores'].values, 100, label="Training M-M'-M''-A")
#plt.hist(traindata['scores'].values, 24, label="Test Mg-M-M'-A outliers in training")
plt.hist(testdata['scores'].values, 100, color='tab:orange', label="Test Mg-M-M'-A") 
plt.xlabel('VAE RE')
plt.ylabel('Number of phase fields')
plt.legend()
plt.show()


#traindata.to_csv(training, index=False)
#testdata.to_csv(test, index=False)
