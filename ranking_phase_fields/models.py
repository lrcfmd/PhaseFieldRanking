import numpy as np
from ranking_phase_fields.symbols import *
from ranking_phase_fields.features import *
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
    v = []
    for i in range(natom):
        v += [ndes * i]
    return v 

def scale(data):
    r = max(data) - min(data)
    return (data - min(data)) / r if r != 0 else data

def mse(instance, aver):
    """ calculate RMSE for scores """
    er = [(i-aver)**2 for i in instance]
    s = np.zeros(len(aver))
    for i in er:
        s += i
    return np.sqrt(s)/aver

def getout(natom, data, feature, var, scores, nnet, fname, mode):
    """ Make human readeable and print the results """
    results = {}
    for i in range(len(data)):
        name = ' '.join(sorted([num2sym(data[i][n], feature) for n in nnet]))
        if name not in results:
            results[name] = np.array([scores[i], var[i]])
        else:
            results[name] += np.array([scores[i], var[i]])
    n = np.math.factorial(natom)
    results = {k:v/n for k,v in sorted(results.items(), key=lambda i : i[1][0])} 

    print(f"Phase fields     scores        {mode}", file=open(fname,'a'))
    for name, score in results.items():
        print(f"{name:16} {score[0]:17} {score[1]}", file=open(fname,'a'))

def rank(phase_fields, features, x_train, x_test, model, natom, average=1):
    """ train a model on x_train and predict x_test """
    ndes = len(features)
    nnet = [ndes*natom, int(ndes*natom*.75), int(ndes*natom*.5), int(ndes*natom*.25), \
                natom, int(ndes*natom*.25),  int(ndes*natom*.5), int(ndes*natom*.75), ndes*natom ] 
    nnet = [136, 102, 68, 34, 4, 34, 68, 102, 136]

    # models
    clfs = {
    'AE'             : AutoEncoder(hidden_neurons=nnet, contamination=0.1, epochs=15),
    'VAE'            : VAE(encoder_neurons=nnet[:5], decoder_neurons=nnet[4:], contamination=0.1, epochs=15),
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
    net = vec2name(ndes, natom)    
    if average > 1:
        vart =mse(trstack, y_train_scores/average)
        var = mse(tstack, y_test_scores/average)
        y_train_scores /= average
        y_test_scores /= average
        getout(natom, x_train, features[0], vart, y_train_scores, net, f'{phase_fields}_{model}_train_scores.dat', 'variance from av. score')
        getout(natom, x_test, features[0], var,  y_test_scores, net, f'{phase_fields}_{model}_test_scores.dat', 'variance from av. score')
    else:
        y_train_scaled  = scale(y_train_scores)
        y_test_scaled = scale(y_test_scores)

    print(f"Writing scores to {phase_fields}_{model}_train_scores.dat")
    getout(natom, x_train, features[0], y_train_scaled,  y_train_scores, net, f'{phase_fields}_{model}_train_scores.dat', 'Norm. score')
    print(f"Writing scores to {phase_fields}_{model}_test_scores.dat")
    getout(natom, x_test, features[0], y_test_scaled,  y_test_scores, net, f'{phase_fields}_{model}_test_scores.dat', 'Norm. score')
