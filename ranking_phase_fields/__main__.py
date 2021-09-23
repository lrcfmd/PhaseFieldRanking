# Ranking unexplored phase fields.
# 13.08.2020
# Andrij Vasylenko 
# and.vasylenko@gmail.com
#
# This programme 
# 1) reads the list of compositions in ICSD,
# where oxidation states are specified
# 2) and extracts a training set of phase fields:
#    binaries
#    ternaries
#    quaternaries
#    quinaries
#    etc. with the specified ox. states
# 3) AE, VAE and other models can be trained on the training set
# for detection of the outliers in the training and testing sets
# and ranking the unexplored phase fields of choice in terms of the degree
# of 'outlying' (degree = reconstruction error). That is hypothesised to 
# correlate with similarity with the 'known' phase fields, and hence,
# with the likelihood of forming a stable composition within a phase field.

import sys
import numpy as np
import pandas as pd
from ranking_phase_fields.parse_icsd import *
from ranking_phase_fields.generate_study  import *
from ranking_phase_fields.features import *
from ranking_phase_fields.models import *
#from ranking_phase_fields.validation import *
from ranking_phase_fields.train_and_validate import *


def main(input_file='rpp.input'):
    ''' Main routine '''
    print("========================================================")
    print("RANKING OF THE PHASE FIELDS BY LIKELIHOOD WITH ICSD DATA \n")
    print("Similar phase fields to those found in ICSD are believed to yield stable compostions.")
    print("The similarity is measured via encoding and decoding vectorized phase fields with VAE")
    print("\n Andrij Vasylenko 13.08.2020")
    print("=========================================================")
    
    # parce input file
    print("Parsing input...")
    params = parse_input(input_file)    
    # exctract training and generate testing set
    training = parse_icsd(params['phase_fields'], params['train_fields'], params['icsd_file'])
    testing = generate_study(params['phase_fields'], params['elements_test'], training)

    # vectorise phase fields with features
    print("==============================================")
    print(f"Representing each element with {len(params['features'])} features.")
    print(f"This represents each phase fields with {numatoms(params['phase_fields'])} x {len(params['features'])} - dimensional vector.")
    print("==============================================")

    # featurize phase fields
    training = sym2num(training, params['features'])
    testing = sym2num(testing, params['features'])

    print('Augmenting data by permutations ...')
    # augment data by permutations:
    training = list(map(permute,training))
    print('summing augmented data')
    training = pd.DataFrame({'phases':training})
    training['phases'] = training['phases'].sum()
    print("Saving augmented training data")
    training.to_csv('icsd2021_permuted.csv', index=None)
    testing = permute_all(testing)

    print('padding phases with 0 ...')
    # padd ends with zeros
    training, natom = pad(training['phases'].values)
    testing, _ = pad(testing, natom)

    # model training: 
    trained, clft, threshold, nnet = train_model(training, natom, params['method'], params['average_runs'])

    # 5-fold cross validation:
    if params['cross-validate'] == 'True':
        validate(params['phase_fields'], params['features'], training, params['method'], threshold, nnet)

    #print(f"Training set: {len(trained)} {params['phase_fields']} phase fields")
    #print(f"Testing set: {len(testing)} unexplored {params['phase_fields']} phase fields")
    
    # predict based on the trained model:
    rank(clft, params['phase_fields'], params['features'], trained, testing, params['method'], \
            natom, params['average_runs'])
    
    print("Finalising and exiting.")

if __name__ == "__main__":
    try:
        ff = sys.argv[1]
    except:
        rpp.input   

    main(ff)
