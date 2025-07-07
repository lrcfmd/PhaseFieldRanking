from math import factorial
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_array
from ranking_phase_fields.features import num2sym

def scale(data):
    """Normalize scores to [0, 1]."""
    r = max(data) - min(data)
    return (data - min(data)) / r if r != 0 else data

def mse(instance, aver):
    """Compute RMSE for scores."""
    er = [(i - aver) ** 2 for i in instance]
    s = np.zeros(len(aver))
    for e in er:
        s += e
    return np.sqrt(s) / aver

def average_permutations(natom, data, features, scores, var):
    """Average scores across permutations."""
    results = {}
    phases = num2sym(data, features)
    for i, phase in enumerate(phases):
        name = ' '.join(phase)
        if name not in results:
            results[name] = np.array([scores[i], var[i]])
        else:
            results[name] += np.array([scores[i], var[i]])
    n = factorial(natom)
    return {k: v / n for k, v in results.items()}

def get_samples(data, feature, scores, var, nnet):
    """Get one sample permutation scores."""
    results = {}
    for i in range(len(data)):
        name = ' '.join(sorted([num2sym(data[i][n], feature) for n in nnet]))
        results[name] = [scores[i], var[i]]
    return {k: v for k, v in sorted(results.items(), key=lambda x: x[1][0])}

def latent_file(trained_results, latent, fname):
    """Save latent representations to pickle."""
    p = list(trained_results.keys())
    s = [v[0] for v in trained_results.values()]
    ns = [v[1] for v in trained_results.values()]
    results = {'phases': p, 'scores': s, 'norm.scores': ns}
    df = pd.DataFrame(results)
    df['latent'] = [np.array(l) for l in latent]
    df.to_pickle(fname)

def getout(results, fname, mode):
    """Write results to CSV."""
    with open(fname, 'a') as f:
        f.write(f"Phase fields,scores,{mode}\n")
        for name, score in results.items():
            f.write(f"{name:16}, {round(score[0], 3):6}, {round(score[1], 3):8}\n")

def pairwise_distances_no_broadcast(X, Y):
    """Utility function to calculate row-wise euclidean distance of two matrix.
    Different from pair-wise calculation, this function would not broadcast.
    For instance, X and Y are both (4,3) matrices, the function would return
    a distance vector with shape (4,), instead of (4,4).
    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        First input samples
    Y : array of shape (n_samples, n_features)
        Second input samples
    Returns
    -------
    distance : array of shape (n_samples,)
        Row-wise euclidean distance of X and Y
    """
    X = check_array(X)
    Y = check_array(Y)

    if X.shape[0] != Y.shape[0] or X.shape[1] != Y.shape[1]:
        raise ValueError("pairwise_distances_no_broadcast function receive"
                         "matrix with different shapes {0} and {1}".format(
            X.shape, Y.shape))

    euclidean_sq = np.square(Y - X)
    return np.sqrt(np.sum(euclidean_sq, axis=1)).ravel()
