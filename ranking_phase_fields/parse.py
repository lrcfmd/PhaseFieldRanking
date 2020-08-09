# Ranking phase fields
# Andrij Vasylenko
# and.vasylenko@gmail.com
# 13.08.2020
# 
# Parse input to read requirements for the training and testing sets,
# to extract the training set from ICSD, and to form a testing set 

from itertools import permutations as pt
from symbols import *
import os

def assign_params(params, input_params, name):
    if name in params and name == 'scaling':
        if str(params[name]).strip() == 'True':
            input_params[name] = True
        else:
            input_params[name] = False
    elif name in params:
        parameters = str(params[name]).split(',')
        if len(parameters) == 1 and name == 'features':
            input_params[name] = [parameters[0].strip()]
        elif len(parameters) == 1:
            p = parameters[0].strip()
            if p.isdigit():
                p = int(p)
            input_params[name] = p
        else:
            input_params[name] = [i.strip() for i in parameters]
    else: 
        print (f'{name} is not specified in the input file. \
                Will use a default value from rpp.input')
        if 'ion' in name and 'nani' not in name:
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
            input_params[name] ='Pettifor'
        elif name == 'average':
            input_params[name] = 1
        elif name == 'scaling':
            input_params[name] = False

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
    assign_params(params, input_params,'icsd_file')
    assign_params(params, input_params,'phase_fields')
    assign_params(params, input_params,'anions_train')
    assign_params(params, input_params,'nanions_train')
    assign_params(params, input_params,'cations_train')
    assign_params(params, input_params,'cation1_test')
    assign_params(params, input_params,'cation2_test')
    assign_params(params, input_params,'anion1_test')
    assign_params(params, input_params,'anion2_test')
    assign_params(params, input_params,'method')
    assign_params(params, input_params,'average_runs')
    assign_params(params, input_params,'scaling')
    assign_params(params, input_params,'features')

    for k,v in input_params.items():
        print(f'{k:15} : {v}')
    
    return input_params

def numatoms(phase_fields):
    nums = {'binary': 2, 'ternary': 3, 'quaternary': 4}
    return nums[phase_fields]

def parse_icsd(phase_fields, anions_train, nanions_train, cations_train, icsd):
    print("==============================================")
    print(f'Reading ICSD list {icsd}')
    print(f'for {phase_fields} phase fields with {nanions_train} anions...')
    lines = open(icsd, 'r+').readlines()
    nanions = int(nanions_train)
    if cations_train == 'all':
        cations_train = symbols
    if anions_train == 'all':
        anions_train = symbols
    anions = [a+'-' for a in anions_train]
    fields = []

    for i in lines:
        #print('Processing line {}'.format(i.strip()))
        oxi,field = [],[]
        # check the composition belongs to the chosen phase fields types:
        if len(i.split()) != numatoms(phase_fields) + 1:
            continue
        # read elements of a composition
        for n in range(1,len(i.split())):
            el = list(i.split()[n])
            sym = el[0]
            if not el[1].isdigit(): 
                sym += el[1]
            ox = sym + el[-1]
            if sym not in cations_train + anions_train:
                break
            field.append(sym)
            oxi.append(ox)
        i#check if the elements / cations are right:
        if len(field) != numatoms(phase_fields):
            continue

        # check there is a right number of anions in a composition:
        if nanions != 0:
            if len(set(oxi) & set(anions)) == nanions and field not in fields:
                fields.append(field)
        elif field not in fields:
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

