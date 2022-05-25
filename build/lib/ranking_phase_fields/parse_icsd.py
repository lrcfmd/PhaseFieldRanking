# Ranking phase fields
# Andrij Vasylenko
# and.vasylenko@gmail.com
# 13.08.2020
# 
# Parse input to read requirements for the training and testing sets,
# to extract the training set from ICSD, and to form a testing set 

import os
from itertools import permutations as pt
try:
    from ranking_phase_fields.symbols import *
except:
    from symbols import *


def assign_params(params, input_params, name):
    # if in input file
    if name in params:
        parameters = str(params[name]).split(',')
        if len(parameters) == 1 and (name == 'features' or parameters[0].strip() in symbols):
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
        elif name == 'train_fields':
            input_params[name] = 'all fields'
        elif name == 'icsd_file':
            input_params[name] = 'icsd2021_phases.csv'
        elif name == 'features':
            input_params[name] ='Pettifor'
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
    assign_params(params, input_params, 'cation3_test')
    assign_params(params, input_params, 'anion1_test')
    assign_params(params, input_params, 'anion2_test')
    assign_params(params, input_params, 'method')
    assign_params(params, input_params, 'cross-validate')
    assign_params(params, input_params, 'average_runs')
    assign_params(params, input_params, 'features')
    assign_params(params, input_params, 'train_fields')

    for k,v in input_params.items():
        print(f'{k:15} : {v}')

    #   customise testing elements for phase fields cases:
    if input_params['phase_fields'] == 'binary':
        input_params['elements_test'] = [input_params['cation1_test'], input_params['anion1_test']]
    elif input_params['phase_fields'] == 'ternary' and input_params['nanions_train'] == 2:
        input_params['elements_test'] = [input_params['cation1_test'], input_params['anion1_test'], input_params['anion2_test']]
    elif input_params['phase_fields'] == 'ternary' and input_params['nanions_train'] == 1:
        input_params['elements_test'] = [input_params['cation1_test'], input_params['cation2_test'], input_params['anion1_test']]
    elif input_params['phase_fields'] == 'quaternary':
        input_params['elements_test'] = [input_params['cation1_test'], input_params['cation2_test'], input_params['anion1_test'], input_params['anion2_test']]
    elif input_params['phase_fields'] == 'quinary':
        input_params['elements_test'] = [input_params['cation1_test'], input_params['cation2_test'], input_params['cation3_test'],
                                         input_params['anion1_test'], input_params['anion2_test']]

    input_params.pop('cation1_test') 
    input_params.pop('cation2_test') 
    input_params.pop('cation3_test') 
    input_params.pop('anion1_test') 
    input_params.pop('anion2_test') 
    return input_params

def numatoms(phase_fields):
    nums = {'binary': 2, 'ternary': 3, 'quaternary': 4, 'quinary': 5}
    return nums[phase_fields]

def field_conditions(field, fields, anions_train):
    if not set(field).difference(set(symbols)) and \
      field not in fields and len(set(field)) == 5 and \
      len(set(field).intersection(set(anions_train))) >= 2:
        return True
    else: 
        return False

def parse_icsd(phase_fields, train_fields, icsd, anions_train):
    print("==============================================")
    print(f'Reading ICSD list {icsd}')
    print(f'for {phase_fields} phase fields...')
    lines = open(icsd, 'r+').readlines()
    
    fields = []

    for i in lines:
        field = i.split()[1:]
        field = [''.join([a for a in i if a.isalpha()]) for i in field]
        field = sorted(field)
        # check the composition belongs to the chosen phase fields types
        #if train_fields != 'all fields' and len(fields) != numatoms(phase_fields):
        if len(field) != numatoms(phase_fields):
            continue
        # read elements of a composition
        #print(set(field).difference(set(symbols)))

        if field_conditions(field, fields, anions_train):
            fields.append(field)
    
    if os.path.isfile(f"{phase_fields}_training_set.dat"):
        print(f"Rewriting {len(fields)} {phase_fields} phase fields to {phase_fields}_training_set.dat")
        os.remove(f"{phase_fields}_training_set.dat")
    else:
        print(f"Writing {len(fields)} {phase_fields} phase fields to {phase_fields}_training_set.dat")
    for f in fields:
        print(' '.join(f), file=open(f"{phase_fields}_training_set.dat", 'a'))

    print("==============================================")
    return fields 

if __name__ == "__main__":
    import sys
    from symbols import *
    try: 
        ffile = sys.argv[1]
    except:
        print('Provide list of elements of interest in the input file. Usage: python parse_icsd.py <input_file>')
        print('Reading default parameters from rpp.input')
        ffile = 'rpp.input'
        pass
   
    params = parse_input(ffile)
    training = parse_icsd(params['phase_fields'], params['train_fields'],  params['icsd_file'], params['anions_train'])
    os.remove(f"quinary_training_set.dat")
    for t in training:
        for f in pt(t):
            print(' '.join(list(f)), file=open(f"quinary_training_set.dat", 'a'))
