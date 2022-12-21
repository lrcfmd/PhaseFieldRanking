#plot subset distribution on full distribution

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mycolorpy import colorlist as mcp
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

def plot_kernel(plot, nf, data, label, ls, c):
    xmin = min(data)
    xmax = max(data)
    xr = np.linspace(xmin,xmax,len(data))[:,np.newaxis]
    kernel = kde(data, xr)
    newy = np.exp(kernel)
    if label:
        #plot.plot(xr[:,0], np.exp(kernel), ls=ls, label=label, color=c)
        plot.fill_between(xr[:,0], np.exp(kernel), label=label, color=c)
    plot.plot(xr[:,0], np.exp(kernel), ls=ls, color=c)

    #plot.axvline(x=np.mean(data), color=c, ls='-')
    return plt

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
    # files for train and test
    N = 16
    #fig, ax = plt.subplots(N, 2)

    colors = mcp.gen_color(cmap='tab10', n=10)

#    for i,n in enumerate(range(3,38,7)):
    for i,n in enumerate([3,9,17,27,37]):
        folder = f'features_{n}_mg'
        ftr = f'{folder}/quaternary_AE_training_scores.csv'
        fte = f'{folder}/quaternary_AE_test_scores.csv'
        ftr = pd.read_csv(ftr).values[:,-3]
        fte = pd.read_csv(fte).values[:,-3]

        # MEAN difference
        delta = round(-np.mean(ftr)+np.mean(fte),2)
     
        # MMAA kernel
        plt = plot_kernel(plt,  n, ftr, f"{n} f: D = {delta}", '-', colors[i-1])
        plt = plot_kernel(plt,  n, fte, None, '--', colors[i-1])
        plt.legend()
    plt.xlabel('VAE raw score', fontsize=14)
    plt.ylabel('GKD', fontsize=14)
    plt.show()

    plt.savefig(f'{n}.png')
