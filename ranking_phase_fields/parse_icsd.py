# Ranking phase fields
# Andrij Vasylenko
# and.vasylenko@gmail.com
# 13.08.2020
# 
# Parse input to read requirements for the training and testing sets,
# to extract the training set from ICSD, and to form a testing set 

import os
import sys
import pandas as pd
from itertools import permutations as pt
from ranking_phase_fields.symbols import *
#from symbols import *

def assign_params(params, input_params, name):
    # if in input file
    if name in params:
        parameters = str(params[name]).split(',')
        if len(parameters) == 1 and parameters[0].strip() in symbols:
            input_params[name] = [parameters[0].strip()]
        elif len(parameters) == 1:
            p = parameters[0].strip()
            if p.isdigit():
                p = int(p)
            input_params[name] = p
        else:
            input_params[name] = [i.strip() for i in parameters]
    # default values
    else: 
        print (f'{name} is not specified in the input file. \
                Will use a default value from rpp.input')
        if 'cation' or 'anion' in name and 'nani' not in name:
            input_params[name] = 'all'
        elif 'nani' in name:
            input_params[name] =  0
        elif name == 'method':
            input_params[name] ='VAE'
        elif name == 'phase_fields':
            input_params[name] = 'quaternary'
        elif name == 'icsd_file':
            input_params[name] = 'icsd2017'
        elif name == 'features':
            input_params[name] ='magpie_37'
        elif name == 'average':
            input_params[name] = 1
        elif name == 'cross-validate':
            input_params[name] = False
    
    # default for 'all' elements
    if input_params[name] == 'all':
        input_params[name] = symbols   


def parse_input(inputfile='rpp.input'):
    print(f'Reading input file {inputfile}')
    print('Input parameters:')
    print("==============================================")
    lines = open(inputfile, 'r+').readlines()
    params = {}
    for l in lines:
        l = l.strip().split(':')
        params[str(l[0]).strip()] = l[-1]

    input_params = {} 
    assign_params(params, input_params, 'icsd_file')
    assign_params(params, input_params, 'phase_fields')
    assign_params(params, input_params, 'anions_train')
    assign_params(params, input_params, 'nanions_train')
    assign_params(params, input_params, 'cations_train')
    assign_params(params, input_params, 'cation1_test')
    assign_params(params, input_params, 'cation2_test')
    assign_params(params, input_params, 'anion1_test')
    assign_params(params, input_params, 'anion2_test')
    assign_params(params, input_params, 'method')
    assign_params(params, input_params, 'cross-validate')
    assign_params(params, input_params, 'average_runs')
    assign_params(params, input_params, 'features')

    for k,v in input_params.items():
        print(f'{k:15} : {v}')

#   castomise testing elements for phase fields cases:
    if input_params['phase_fields'] == 'binary':
        input_params['elements_test'] = [input_params['cation1_test'], input_params['anion1_test']]
    elif input_params['phase_fields'] == 'ternary' and input_params['nanions_train'] in [0, 2]:
        input_params['elements_test'] = [input_params['cation1_test'], input_params['anion1_test'], input_params['anion2_test']]
    elif input_params['phase_fields'] == 'ternary' and input_params['nanions_train'] == 1:
        input_params['elements_test'] = [input_params['cation1_test'], input_params['cation2_test'], input_params['anion1_test']]
    else:
        input_params['elements_test'] = [input_params['cation1_test'], input_params['cation2_test'], input_params['anion1_test'], input_params['anion2_test']]

    input_params.pop('cation1_test') 
    input_params.pop('cation2_test') 
    input_params.pop('anion1_test') 
    input_params.pop('anion2_test') 
    return input_params

def numatoms(phase_fields):
    nums = {'binary': 2, 'ternary': 3, 'quaternary': 4, 'quinary': 5}
    return nums[phase_fields]

def justify(field, maxatom):
    """ the vectors will be justified by the largest phase field
    in the training data by zero-padding
    Here, a symbol 'X' is appended for missing elements
    X is featurized in elemental features """
    diff =  maxatom - len(field)
    if diff:
        field += ['X' for a in range(diff)]
    return field

def parse_icsd(phase_fields, anions_train, nanions_train, cations_train, icsd, return_dic=False):
    print("==============================================")
    if cations_train == 'all':
        cations_train = symbols
    if anions_train == 'all':
        anions_train = symbols
    anions = [a+'-' for a in anions_train]

    print(f'Reading training DATA: {icsd}')
    if 'icsd' or 'ICSD' not in icsd:
        print(f'Custom training data in {icsd} will be treated as a list of phase fields')
        df = pd.read_csv(f'{icsd}')
        df['fields'] = df.iloc[:,0].apply(lambda field: sorted(field.split()))
        # prune the list of phase fields to only include cations_train
        df['cations'] = df['fields'].apply(lambda field: len(set(field).difference(set(cations_train)))) 
        print(f'original size:' ,len(df))
        df = df[df['cations']==0]
        print(f'with cations train size:' ,len(df))

        # justify by the largest field
        df['natoms'] = df['fields'].apply(lambda field: len(field))
        maxatom = df['natoms'].max()
        df['fields'] = df['fields'].apply(lambda field: justify(field, maxatom)) 
        fields = df['fields'].to_list()
#        fields = list(set(fields))

    else:
        print(f'for {phase_fields} phase fields with {nanions_train} anions...')
        lines = open(icsd, 'r+').readlines()
        nanions = int(nanions_train)
        fields = []
        for i in lines:
            oxi,field = [],[]
     
            # check the composition belongs to the chosen phase fields types:
            if len(i.split()) != numatoms(phase_fields) + 1:
                continue
     
            # read elements of a composition
            for n in range(1, len(i.split())):
                el = list(i.split()[n])
                sym = el[0]
                if not el[1].isdigit(): 
                    sym += el[1]
                ox = sym + el[-1]
                if sym not in cations_train + anions_train:
                    break
                field.append(sym)
                oxi.append(ox)
     
            #check if the elements / cations are right:
            if len(field) != numatoms(phase_fields):
                continue
     
            # check there is a right number of anions in a composition:
            if nanions != 0:
                field = sorted(field)
                if len(set(oxi) & set(anions)) >= nanions and field not in fields:
                    fields.append(field)
     
            elif sorted(field) not in fields:
                fields.append(sorted(field))
    
  # if os.path.isfile(f"{phase_fields}_training_set.dat"):
  #     print(f"Rewriting {len(fields)} {phase_fields} phase fields to {phase_fields}_training_set.dat")
  #     os.remove(f"{phase_fields}_training_set.dat")
  # else:
  #     print(f"Writing {len(fields)} {phase_fields} phase fields to {phase_fields}_training_set.dat")
  # for f in fields:
  #     print(' '.join(f), file=open(f"{phase_fields}_training_set.dat", 'a'))

    print("==============================================")
    return fields

if __name__ == "__main__":
    try: 
        ffile = sys.argv[1]
    except:
        print('Provide list of elements of interest in the input file. Usage: python parse_icsd.py <input_file>')
        print('Reading default parameters from rpp.input')
        ffile = 'rpp.input'
    params = parse_input(inputfile=ffile)
    training = parse_icsd(params['phase_fields'], params['anions_train'], \
            params['nanions_train'], params['cations_train'], params['icsd_file'], True)
    print('Example phases:', training[:10])
