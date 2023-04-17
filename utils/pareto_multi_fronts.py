# find pareto front for different rankings (based on different elemental features)

import sys
import pandas as pd
from typing import List, Tuple

def pareto_front(data, fronts=None):
    # Sort the data in ascending order
    if fronts is None:
        fronts = [[]]
        data = sorted(data, key=lambda x: x[0])

    # Initialize the current Pareto front
    current_front = [data[0]]

    for point in data[1:]:
        dominated = False
        for front_point in current_front:
            if any(front_point[i] <= point[i] for i in range(1,len(point))):
                dominated = True
                break
            if not dominated:
                current_front.append(point)

    fronts.append(current_front)
#    allfronts = [point for front in fronts for point in front]
#    remaining_points = [point for point in data if point not in allfronts]

    # Check if there are points left
    remaining_points = [point for point in data if point not in current_front]
    if remaining_points:
        print('Remaining points not in the current front: YES', len(remaining_points), 'How many fronts filled:', len(fronts))
        # Calculate the next Pareto front
        pareto_front(remaining_points,fronts)
    return fronts

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
#        'quaternary_VAE_magpie_37_test_scores.csv',
#        'original_VAE_magpie37_test_scores.csv',
#        'quaternary_VAE_random_200_test_scores.csv',
#        'quaternary_VAE_megnet16_test_scores.csv',
        #'quaternary_VAE_magpie_test_scores.csv',
        'quaternary_VAE_LEAF_L_test_scores.csv',
        'quaternary_VAE_mat2vec_test_scores.csv',
        ]


    # Collect all scores for phase fields
    scores = []
    for f in files:
        df = get_scores(f)
        print(df.head())
        #scores.append(df['scores'].values)
        scores.append(df['Norm. score'].values)

    tuple_scores, tuple2phase = create_tuples(df['Phase fields'], scores)

    # get pareto front scores and phase fields
    pareto_fronts = pareto_front(tuple_scores)
    pareto_scores = [score for front in pareto_fronts for score in front]
    pareto_phases = [tuple2phase[tup] for tup in pareto_scores]
    fronts = [[i for k in range(len(front))] for i,front in enumerate(pareto_fronts)]
    fronts = [i for sublist in fronts for i in sublist]
    
    # get FIVE separate lists:
    #rand, megnet, magpie, leaf, mat2vec = [], [], [], [], []
    leaf, mat2vec = [], []
    for tup in pareto_scores:
       #ma, r, me, l, m2 = tup
       #rand.append(r)
       #megnet.append(me)
       #magpie.append(ma)
        l, m2 = tup
        leaf.append(l)
        mat2vec.append(m2)

    newdf = pd.DataFrame({'phase fields': pareto_phases,
                          'pareto front': fronts,
#                         'magpie37': magpie,
#                         'random200': rand,
#                          'megnet16': megnet,
                          'leaf': leaf,
                          'mat2vec':mat2vec
                          })
    newdf = newdf.drop_duplicates(['phase fields'])
    #newdf.to_csv('pareto_fronts_MgMMA.csv', index=False)
    newdf.to_csv('pareto_MgMAA.csv', index=False)

