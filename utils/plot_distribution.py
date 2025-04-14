#plot subset distribution on full distribution

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans

def kde(data,xr):
    """ Kernel density """
    kd = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(data[:,np.newaxis])
    return kd.score_samples(xr)

def cluster(data, color):
    kmeans = KMeans(n_clusters=2).fit(data[:,np.newaxis])
    centers = kmeans.cluster_centers_
    c = np.average(centers)
    # or where cluster 1 ends:
    labels = kmeans.labels_
    c1 = len(labels[labels==0])
    c = data[c1] 

    plt.annotate('inliers centre', xy=(centers[0],50), color='tab:green', \
            xytext=(centers[0],60), fontsize=10, arrowprops=dict(arrowstyle="->", color='teal'))
    plt.annotate('outliers centre', xy=(centers[1],50), color='tab:red', \
            xytext=(centers[1],60), fontsize=10, arrowprops=dict(arrowstyle="->", color='teal'))
    plt.axvline(x=centers[0], c=color, ls='--', lw=2)
    plt.axvline(x=centers[1], c=color, ls='--', lw=2)
   #plt.axvline(x=c, c=color, ls='--', label='Separation between 2 clusters')
    return c

def plot_kernel(plot, data, label, c):
    xmin = min(data)
    xmax = max(data)
    xr = np.linspace(xmin,xmax,len(data))[:,np.newaxis]

    kernel = kde(data, xr)
    newx = xr[:,0]
    newy = np.exp(kernel)

    plot.plot(newx, newy, label=label)
    plot.ylabel('Gaussian kernel distribution estimate')
    plot.xlabel('Raw VAE score')
    plot.axvline(x=np.mean(data), color=c, ls='--')

def plot_mean(data, c):
    plt.axvline(x=np.mean(data), c=c, ls='--', label='mean')

def plot(data, label, nbins=10, score='raw', alpha=1, separation=None):
    if score == 'norm':
        xlab = 'Norm. scores'
    else: # raw score
        xlab = 'Raw VAE scores'

    n, bins, patches = plt.hist(data, nbins, alpha=alpha, label=label)
    print(n)
   
    if separation:
        for p in patches: 
            if p.get_x() < separation:
                p.set_facecolor('tab:green')
            else:
                p.set_facecolor('tab:red')

    plt.xlabel(xlab)
    plt.ylabel('Number of phase fields')

if __name__=="__main__":
    ftr, fte  = sys.argv[1:3]
    ftr = pd.read_csv(ftr).values[:,-1]
    fte = pd.read_csv(fte).values[:,-1]

    # MMAA kernel
    plot_kernel(plt, ftr, "M-M'-M''-A training", 'tab:blue')
    plot_kernel(plt, fte, "unexplored Mg-M-M'-A testing", 'tab:orange')

    # MEAN diff
    D = np.mean(ftr) - np.mean(fte)
    print('Difference between mean RE for training, and mean for testing:', D)
    
    #plt.hist(ftr, 40) # label='original VAE')
    #plt.axvline(x=np.mean(fte), color='tab:orange', ls='--')
    #plt.hist(fte, 60, alpha=0.5) #, label='with anion-quaternaries in training')
    #plt.axvline(x=np.mean(ftr), color='tab:blue', ls='--')

    #plt.xlim([0,30])
    plt.legend()
    plt.show()
