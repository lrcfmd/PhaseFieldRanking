""" From the lists of elements of interest
(e.g. for cations and anions) generate a list of phase fields
(binary, ternary or quaternary) that have not yet been explored
(don't have associated compounds reported in ICSD)
"""
import os, sys
from ranking_phase_fields.parse_icsd import *
from itertools import permutations as pmts

def permute(vectors):
    permuted = []
    for v in vectors:
        p = [list(i) for i in pmts(v)]
        permuted += p 
    return permuted

def augment(chain, vec):
    p = map(lambda val: chain + [val] if type(chain) is list else [chain] + [val], vec)
    return [i for i in p]

def generate_study(phase_fields, lists, training):
    print (f'Creating testing data for unexplored {phase_fields} phase fields:')

    field = lists[0]

    # aument fields with elements from lists recursively
    for i in range(1, len(lists)):
        field = [ r for s in map(lambda el: augment(el, lists[i]), field) for r in s]

    study = []
    for f in field:
        # check elements are unique
        if len(set(f)) == len(lists) and sorted(f) not in training and sorted(f) not in study:
            study.append(sorted(f))

    if os.path.isfile(f'{phase_fields}_testing.dat'):
        print(f'Rewriting {len(study)} testing phase fields to {phase_fields}_testing.dat')
        os.remove(f'{phase_fields}_testing.dat')
    else:
        print(f'Writing {len(study)} testing phase fields to {phase_fields}_testing.dat')

    for i in study:
        print (' '.join(map(str, i)), file=open(f'{phase_fields}_testing.dat','a'))

    print("==============================================")
    return study

if __name__ == "__main__":
    try:
        ffile = sys.argv[1]
    except:
        print('Provide list of elements of interest in the input file. Usage: python generate_study.py <input_file>')
        print('Reading default parameters from rpp.input')
    params = parse_input()
    training = parse_icsd(params['phase_fields'], params['anions_train'], \
            params['nanions_train'], params['cations_train'], params['icsd_file'])
    testing = generate_study(params['phase_fields'], params['elements_test'], training) 
