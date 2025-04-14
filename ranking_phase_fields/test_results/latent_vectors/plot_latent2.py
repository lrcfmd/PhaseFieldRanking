import sys
import pandas as pd
#import matplotlib.pyplot as plt
import plotly.express as plt

df = pd.read_pickle('quaternary_VAE_encoder_training_latent2.pkl')

print(df.head())

def frac(data):
    a, b = [], []
    for i in data:
       a.append(i[0] / (i[0] + i[1]))
       b.append(i[2] / (i[2] + i[3]))
      # b.append(i[3])
    return a, b

a, b = frac(df['latent'].values)

bf = pd.DataFrame({'phases':df['phases'], 'a':a, 'b':b, 'scores':df['scores']})

# matplotlib
#plt.scatter(a, b, c=scores['scores'].values)
#plt.colorbar()
#plt.show()

# plotly
fig = plt.scatter(bf, x='a', y='b', hover_name='phases', color='scores')
fig.show()
