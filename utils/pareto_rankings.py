# find pareto front for different rankings (based on different elemental features)

import sys
import pandas as pd
from typing import List, Tuple

def pareto_front(points: List[Tuple[float, float, float, float, float]]) -> List[Tuple[float, float, float, float, float]]:
    # Sort the points in ascending order based on the first variable - the smaller the closer to the front
    points = sorted(points, key=lambda x: x[0]) # switch to reverse=True for non-ascending order
    
    # Initialize the Pareto front with the first point
    pareto_front = [points[0]]
    
    # Iterate through the rest of the points and add them to the Pareto front if they dominate
    # any of the points in the front
    for point in points[1:]:
        dominated = False
        for front_point in pareto_front:
            if any(front_point[i] <= point[i] for i in range(1,len(point))):
                dominated = True
                break
        if not dominated:
            pareto_front.append(point)
    
    return pareto_front

def get_scores(f):
    df = pd.read_csv(f)
    return df.sort_values(by=['Phase fields'])


def create_tuples(phases, scores):
    """ Collect scores as tuples for each phase field
    and create tuple - phase dictionary """
    tuple_scores = []
    tuple2phase = {}

    for i, phase in enumerate(phases):
        tup = tuple([score[i] for score in scores])
        tuple_scores.append(tup)
        tuple2phase[tup] = phase

    return tuple_scores, tuple2phase


if __name__ == "__main__":
   # Example usage
   #points = [(1, 2, 3, 4, 5), (5, 4, 3, 2, 1), (2, 3, 4, 5, 1), (4, 5, 1, 2, 3), (3, 1, 5, 4, 2)]
   #pareto_front = pareto_front(points)
   #print(pareto_front)  # Output: [(5, 4, 3, 2, 1), (4, 5, 1, 2, 3), (3, 1, 5, 4, 2)]
   #
    files = [
        'quaternary_VAE_magpie_37_test_scores.csv',
#        'original_VAE_magpie37_test_scores.csv',
        'quaternary_VAE_random_200_test_scores.csv',
        'quaternary_VAE_megnet16_test_scores.csv',
        #'quaternary_VAE_magpie_test_scores.csv',
        'quaternary_VAE_LEAF_L_test_scores.csv',
        'quaternary_VAE_mat2vec_test_scores.csv',
        ]


    # Collect all scores for phase fields
    scores = []
    for f in files:
        df = get_scores(f)
        print(df.head())
        scores.append(df['scores'].values)

    tuple_scores, tuple2phase = create_tuples(df['Phase fields'], scores)

    # get pareto front scores and phase fields
    pareto_scores = pareto_front(tuple_scores)
    pareto_phases = [tuple2phase[tup] for tup in pareto_scores]
    
    # get FIVE separate lists:
    rand, megnet, magpie, leaf, mat2vec = [], [], [], [], []
    for tup in pareto_scores:
        ma, r, me, l, m2 = tup
        rand.append(r)
        megnet.append(me)
        magpie.append(ma)
        leaf.append(l)
        mat2vec.append(m2)

    newdf = pd.DataFrame({'phase fields': pareto_phases,
                          'magpie37': magpie,
                          'random200': rand,
                          'megnet16': megnet,
                          'leaf36': leaf,
                          'mat2vec':mat2vec
                          })
    newdf.to_csv('pareto_MgMMA.csv', index=False)
    #newdf.to_csv('pareto_MgMAA.csv', index=False)

