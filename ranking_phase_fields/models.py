import numpy as np
from symbols import *
from features import *
# Import all models
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.vae import VAE
from pyod.models.abod import ABOD
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.sos import SOS
from pyod.models.lscp import LSCP
from pyod.models.cof import COF
from pyod.models.cblof import CBLOF
from pyod.models.sod import SOD
from pyod.models.loci import LOCI
from pyod.models.mcd import MCD

def vec2name(ndes, natom):
    if natom == 2:
        return [0, ndes]
    elif natom == 3:
        return [0, ndes, 2*ndes]
    else:
        return [0, ndes, 2*ndes, 3*ndes]

def scale(data, var):
    r = max(data) - min(data)
    if r != 0:
        return (data - min(data)) / r, var / r 
    else:
        return data, var

def mse(instance, aver):
    """ calculate RMSE for scores """
    er = [(i-aver)**2 for i in instance]
    s = np.zeros(len(aver))
    for i in er:
        s += i
    return np.sqrt(s)/aver

def getout(natom, data, var, scores, nnet, fname):
    dout = []
    for i in range(len(data)):
        v = [num2sym(data[i][n]) for n in nnet] + [scores[i],var[i]]
        dout.append(v)
    dout = np.asarray(dout)
    dout = dout[scores.argsort()]
    print("Phase fields   av. scores        variance over runs", file=open(fname,'a'))
    for i in range(len(dout)):
        print(f"{' '.join(dout[i,:natom]):14} {dout[i,-2]:17} {dout[i,-1]}", file=open(fname,'a'))

def rank(phase_fields, x_train, x_test, model, ndes, natom, average=1, scaling=False):
    """ train a model on x_train and predict x_test """
    # build hidden layers for AE and VAE:
    if natom == 4 and ndes == 34:
        nnet = [136,102,68,40,20,4,20,40,68,102,136]
    else:
        nnet = [ndes*natom, int(ndes*natom*.75), int(ndes*natom*.5), int(ndes*natom*.25), \
                natom, int(ndes*natom*.25),  int(ndes*natom*.5), int(ndes*natom*.75), ndes*natom ] 

    # models
    clfs = {
    'AE'             : AutoEncoder(hidden_neurons=nnet, contamination=0.1, epochs=15),
    'VAE'            : VAE(encoder_neurons=nnet[:6], decoder_neurons=nnet[6:], contamination=0.1, epochs=15),
    'ABOD'           : ABOD(),
    'FeatureBagging' : FeatureBagging(),
    'HBOS'           : HBOS(),
    'IForest'        : IForest(),
    'KNN'            : KNN(),
    'LOF'            : LOF(),
    'OCSVM'          : OCSVM(),
    'PCA'            : PCA(),
    'SOS'            : SOS(),
    'COF'            : COF(),
    'CBLOF'          : CBLOF(),
    'SOD'            : SOD(),
    'LOCI'           : LOCI(),
    'MCD'            : MCD()
    }

    clf = clfs[model]
    y_test_scores = np.zeros(len(x_test))
    y_train_scores = np.zeros(len(x_train))
    print(f"Training a {model} model on {len(x_train)} {phase_fields} phase fields in ICSD")
    print(f'Run {model} {average} times to average the scores')

    # run {average} time for scores averaging 
    print(f"Prediciting the similarity (proximity in terms of reconstruction error for VAE) of {len(x_test)} unexplored phase fields to ICSD data")
    for i in range(average):
        print(f"{model} RUN: {i+1}")
        clf.fit(x_train)

        # get the prediction labels and outlier scores of the training data
        y_train_pred = clf.labels_               # binary labels (0: inliers, 1: outliers)
        lastt = np.asarray(clf.decision_scores_) # raw outlier scores
        y_train_scores += lastt
     
        # get the prediction on the test data
        y_test_pred = clf.predict(x_test)              # outlier labels (0 or 1)
        last = np.asarray(clf.decision_function(x_test))  # outlier scores
        y_test_scores += last
     
        if i == 0:
            trstack = lastt
            tstack = last
        else:
            trstack = np.vstack([trstack,lastt])
            tstack = np.vstack([tstack,last])

    # results out
    vart =mse(trstack, y_train_scores/average)
    var = mse(tstack, y_test_scores/average)

    if scaling:
        y_train_scores, vart = scale(y_train_scores/average, vart)
        y_test_scores, var = scale(y_test_scores/average, var)
    else:
        y_train_scores /= average
        y_test_scores /= average

    print(f"Writing scores to {phase_fields}_{model}_train_scores.dat")
    getout(natom, x_train, vart, y_train_scores, vec2name(ndes, natom), f'{phase_fields}_{model}_train_scores.dat')
    print(f"Writing scores to {phase_fields}_{model}_test_scores.dat")
    getout(natom, x_test, var,  y_test_scores, vec2name(ndes, natom), f'{phase_fields}_{model}_test_scores.dat')
