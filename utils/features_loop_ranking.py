from ranking_phase_fields.__main__ import main
import itertools as it

import sys
import os

if __name__=="__main__":
    features = open('features.dat', 'r').readlines()
    features = features[0].split(',')
    features = [f.strip() for f in features]

    i = "mg"
    for n_features in range(21, len(features)+1):
        os.system(f'cp null.input features_{n_features}_{i}.input')
        os.system(f'mkdir features_{n_features}_{i}')
        ffile = open(f'features_{n_features}_{i}.input', 'a')
        string = 'features       : '
        for f in features[:n_features]:
            string += f'{f}, '
        ffile.write(string[:-2])
        ffile.close()

        main(f'features_{n_features}_{i}.input')
        os.system(f'mv features_{n_features}_{i}.input features_{n_features}_{i}/.')
        os.system(f'mv quaternary_AE_test_scores.csv features_{n_features}_{i}/.')
        os.system(f'mv quaternary_AE_training_scores.csv features_{n_features}_{i}/.')
