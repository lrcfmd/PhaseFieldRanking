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
#    etc. with the specified ox. states
# 3) AE, VAE and other models can be trained on the training set
# for detection of the outliers in the training and testing sets
# and ranking the unexplored phase fields of choice in terms of the degree
# of 'outlying' (degree = reconstruction error). That is hypothesised to 
# correlate with similarity with the 'known' phase fields, and hence,
# with the likelihood of forming a stable composition within a phase field.

import sys
import numpy as np
from ranking_phase_fields.parse_icsd import *
from ranking_phase_fields.generate_study  import *
from ranking_phase_fields.features import *
from ranking_phase_fields.models import *
from ranking_phase_fields.validation import *


def main(input_file):
    ''' Main routine '''
    print("========================================================")
    print("RANKING OF THE PHASE FIELDS BY LIKELIHOOD WITH ICSD DATA \n")
    print("Similar phase fields to those found in ICSD are believed to yield stable compostions.")
    print("The similarity is measured via encoding and decoding vectorized phase fields with VAE")
    print("\n Andrij Vasylenko 13.08.2020")
    print("=========================================================")
    
    # parce input file
    params = parse_input(input_file)    
    # exctract training and generate testing set
    training = parse_icsd(params['phase_fields'], params['anions_train'], \
            params['nanions_train'], params['cations_train'], params['icsd_file'])
    
    testing = generate_study(params['phase_fields'], params['elements_test'], training)
   
    # 5-fold cross validation:
    validate(params['phase_fields'], params['features'], training, params['method'], \
        numatoms(params['phase_fields']), params['average_runs'])

    # data augmentation by permutation
    print("Augmenting data by elemental permutations:")
    training = permute(training)
    testing = permute(testing)
    print(f"Training set: {len(training)} {params['phase_fields']} phase fields")
    print(f"Testing set: {len(testing)} unexplored {params['phase_fields']} phase fields")
    print("==============================================")
    
    # vectorise phase fields with features
    print(f"Representing each element with {len(params['features'])} features.")
    print(f"This represents each phase fields with {numatoms(params['phase_fields'])} x {len(params['features'])} - dimensional vector.")
    print("==============================================")
    training = sym2num(training, params['features'])
    testing  = sym2num(testing, params['features'])
    
    # train a model and predict
    rank(params['phase_fields'], params['features'], training, testing, params['method'], \
            numatoms(params['phase_fields']), params['average_runs'])

if __name__ == "__main__":
    main()
