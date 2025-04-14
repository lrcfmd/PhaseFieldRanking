import plotly.express as plt
import plotly.graph_objs as go

#import matplotlib.pyplot as plt
import pandas as pd

def plotdata(file):
    df = pd.read_csv(file)
    fig= plt.scatter(df, x='leaf', y='mat2vec', text='phase fields')
#   plt.xlabel('LEAF',fontsize=14)
#   plt.ylabel('Mat2Vec',fontsize=14)
    fig.show()


file ='ranking_MgMMA_above_thresholds.csv'
plotdata(file)
