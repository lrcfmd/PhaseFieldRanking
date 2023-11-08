""" From the lists of elements of interest
(e.g. for cations and anions) generate a list of phase fields
(binary, ternary or quaternary) that have not yet been explored
(don't have associated compounds reported in ICSD)
"""
import os, sys
#from functools import lru_cache
from ranking_phase_fields.parse_icsd import *
#from parse_icsd import *
from itertools import permutations as pmts

def permute(vectors):
    permuted = []
    for v in vectors:
        p = [list(i) for i in pmts(v)]
        permuted += p 
    return permuted

#@lru_cache(maxsize=None) -- lists are unhashable
def augment(chain, vec):
    p = map(lambda val: chain + [val] if type(chain) is list else [chain] + [val], vec)
    return [i for i in p]

def generate_study(phase_fields, lists, training):
    print (f'Creating testing data for unexplored {phase_fields} phase fields:')

    # if elements for all positions in test phase fiels are the same:
    if lists[0] == lists[1] == lists[2]:
        print('Phase fields AAAA are built from the same elements A')
        natoms = numatoms(phase_fields)
        study = [list(field) for field in pmts(lists[0], natoms) if field not in training] 

    else:
        field = lists[0]
        # aument fields with elements from lists recursively
        for i in range(1, len(lists)):
            field = [ r for s in map(lambda el: augment(el, lists[i]), field) for r in s]

        study = []
        for f in field:
            # check elements are unique
            if len(set(f)) == len(lists) and sorted(f) not in training and sorted(f) not in study:
                study.append(sorted(f))

 #   if os.path.isfile(f'{phase_fields}_testing.dat'):
 #       print(f'Rewriting {len(study)} testing phase fields to {phase_fields}_testing.dat')
 #       os.remove(f'{phase_fields}_testing.dat')
 #   else:
 #       print(f'Writing {len(study)} testing phase fields to {phase_fields}_testing.dat')
 #
 #   for i in study:
 #       print (' '.join(map(str, i)), file=open(f'{phase_fields}_testing.dat','a'))

    print("==============================================")
    return study

if __name__ == "__main__":
    try:
        ffile = sys.argv[1]
    except:
        print('Provide list of elements of interest in the input file. Usage: python generate_study.py <input_file>')
        print('Reading default parameters from rpp.input')
        ffile='rpp.input'
    #params = parse_input(ffile)
    #training = parse_icsd(params['phase_fields'], params['anions_train'], \
    #        params['nanions_train'], params['cations_train'], params['icsd_file'])
    #testing = generate_study(params['phase_fields'], params['elements_test'], training) 
    M1 =  ['Li']
    M2 =  ['B', 'Mg', 'Al', 'Si', 'P', 'K', 'Ca', 'Zn', 'Sr', 'Y', 'Zr', 'Sn', 'Ba', 'Ta', 'La']
    A1 = ['S','O','Cl','Br','I','N','F']
    A2 = ['S','O','Cl','Br','I','N','F']
    A3 = ['S','O','Cl','Br','I','N','F']
    A1 = 'Li,Be,Na,Mg,Al,K,Ca,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Ga,Rb,Sr,Y,Zr,Nb,Mo,Ag,Cd,In,Sn,Cs,Ba,Hf,Ta,W,Os,Ir,Pt,Au,Hg,Tl,Pb,Bi'.split(',')
    A2 = 'Li,Be,Na,Mg,Al,K,Ca,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Ga,Rb,Sr,Y,Zr,Nb,Mo,Ag,Cd,In,Sn,Cs,Ba,Hf,Ta,W,Os,Ir,Pt,Au,Hg,Tl,Pb,Bi'.split(',')
    A3 = 'Li,Be,Na,Mg,Al,K,Ca,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Ga,Rb,Sr,Y,Zr,Nb,Mo,Ag,Cd,In,Sn,Cs,Ba,Hf,Ta,W,Os,Ir,Pt,Au,Hg,Tl,Pb,Bi'.split(',')
    #lists = [M1, M2, A1, A2, A3]
    lists = [A1, A2, A3]
    testing = generate_study('ternary', lists, [])
    print(len(testing), testing[:3])
