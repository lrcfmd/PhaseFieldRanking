import numpy as np
from ranking_phase_fields.symbols import *
from ranking_phase_fields.features import *
from ranking_phase_fields.train_and_validate import *

def vec2name(ndes, natom):
    """ get index of the first feature for each element """
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

def average_permutations(natom, data, feature, scores, var, nnet):
    """ average scores for permutations """
    results = {}
    for i in range(len(data)):
        name = ' '.join(sorted([num2sym(data[i][n], feature) for n in nnet]))
        if name not in results.keys():
            results[name] = np.array([scores[i], var[i]])
        else:
            results[name] += np.array([scores[i], var[i]])
    n = np.math.factorial(natom)
    return {k:v/n for k,v in sorted(results.items(), key=lambda i : i[1][0])} 

def get_samples(data, feature, scores, var, nnet):
    """ get one sample of permutations """
    results = {}
    for i in range(len(data)):
        name = ' '.join(sorted([num2sym(data[i][n], feature) for n in nnet]))
        results[name] = [scores[i], var[i]]
    return {k:v for k,v in sorted(results.items(), key=lambda i: i[1][0])}

def getout(results, fname, mode):
    """ print the results """
    print(f"Phase fields,     scores,     {mode},", file=open(fname,'a'))
    for name, score in results.items():
        print(f"{name:16}, {round(score[0],3):6}, {round(score[1],3):8},", file=open(fname,'a'))

def rank(clft, phase_fields, features, x_train, x_test, model, natom, average=1):
    """ train a model on x_train and predict x_test """
    ndes = len(features)
    nnet = [int(ndes*natom/2), int(ndes*natom/4), int(ndes*natom/8), int(ndes*natom/16), \
            natom, int(ndes*natom/16),  int(ndes*natom/8), int(ndes*natom/4), int(ndes*natom/2)]
    net = vec2name(ndes, natom)   

    if average == 1:
        y_test_scores = clft.decision_function(x_test)
        y_test_scaled = scale(y_test_scores)
        print(f"Writing scores to {phase_fields}_{model}_test_scores.csv")
        results = average_permutations(natom, x_test, features[0], y_test_scores, y_test_scaled, net)
        getout(results, f'{phase_fields}_{model}_test_scores.csv', 'Norm. score')

    else:
        y_test_scores = np.zeros(len(x_test))
        y_train_scores = np.zeros(len(x_train))
        print(f"Training a {model} model on {len(x_train)} {phase_fields} phase fields in ICSD")
        print(f'Run {model} {average} times to average the scores')
     
        # run {average} times for scores averaging 
        print(f"Prediciting the similarity (proximity in terms of reconstruction error for VAE) of {len(x_test)} unexplored phase fields to ICSD data")
        for i in range(average):
            print(f"{model} RUN: {i+1}")
            clft.fit(x_train)
     
            # get the prediction outlier scores of the training data
            lastt = np.asarray(clft.decision_scores_) # raw outlier scores
            y_train_scores += lastt
         
            # get the prediction on the test data
            last = np.asarray(clft.decision_function(x_test))  # outlier scores
            y_test_scores += last
         
            if i == 0:
                trstack = lastt
                tstack = last
            else:
                trstack = np.vstack([trstack,lastt])
                tstack = np.vstack([tstack,last])
     
        # results out: averaging over runs, calculate variance 
        vart =mse(trstack, y_train_scores/average)
        var = mse(tstack, y_test_scores/average)
        y_train_scores /= average
        y_test_scores /= average
        print(f"Writing scores to {phase_fields}_{model}_train_scores.csv")
        results_train = average_permutations(natom, x_train, features[0], y_train_scores, vart, net)
        getout(results_train, f'{phase_fields}_{model}_train_scores.csv', 'variance from av. score')
        print(f"Writing scores to {phase_fields}_{model}_test_scores.csv")
        results = average_permutations(natom, x_test, features[0], y_test_scores, var, net)
        getout(results, f'{phase_fields}_{model}_test_scores.csv', 'variance from av. score')
