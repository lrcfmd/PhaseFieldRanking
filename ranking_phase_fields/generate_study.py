import os
from itertools import permutations as pt
from symbols_noradio import *

def all2sym(elements):
    if elements == 'all':
        return symbols
    elif isinstance(elements,str):
        return [elements]
    elif isinstance(elements,list):
        return elements

def permute(sets):
    lists = []
    for s in sets:
        p = [list(i) for i in pt(s)]
        for i in p:
            lists.append(i)
    return lists

def binary(c1, a1, training):
    print("Element1: ", c1)
    print("Element2: ", a1)
    fields = []
    for a in c1:
        for b in a1:
            if a != b and {a,b} not in fields \
                    and {a,b} not in training:
                    fields.append({a,b})
    return fields

def loop3(c1,a1,a2, training):
    fields = []
    for c in c1:
        for a in a1:
            for b in a2:
                f = {a,b,c}
                if f not in fields and f not in training \
                        and len(f) == 3:
                    fields.append(f)
    return fields

def ternary(c1, c2, a1, a2, nanions_train, training):
    if nanions_train == '2':
        print("Cations: ", c1)
        print("Anion1: ", a1)
        print("Anion2: ", a2)
        return loop3(c1,a1,a2, training)
    else:
        print("Cation1: ", c1)
        print("Cation2: ", c2)
        print("Anions: ", a1)
    return loop3(c1,c2,a1, training)

def quaternary(c1,c2,c3,c4, training):
    fields = []
    print("Cation1: ", c1) 
    print("Cation2: ", c2)
    print("Anion1: ", c3)
    print("Anion2: ", c4)
    for c in c1:
        for d in c2:
            for a in c3:
                for b in c3:
                    f = {c,d,a,b}
                    if f not in fields and f not in training \
                        and len(f) == 4:
                        fields.append(f)
    return fields

def generate_test(phase_fields, nanions_train, c1, c2, a1, a2, training):

    print (f'Creating testing data for unexplored {phase_fields} phase fields:')
    training = [set(i) for i in training]
    c1 = all2sym(c1)
    c2 = all2sym(c2)
    a1 = all2sym(a1)
    a2 = all2sym(a2)
    if phase_fields == 'binary':
        testing = binary(c1,a1, training)
    elif phase_fields == 'ternary':
        testing = ternary(c1, c2, a1, a2, nanions_train, training)
    else:
        testing = quaternary(c1, c2, a1, a2, training)

    if os.path.isfile(f'{phase_fields}_testing.dat'):
        print(f'Rewriting {len(testing)} testing phase fields to {phase_fields}_testing.dat')
        os.remove(f'{phase_fields}_testing.dat')
    else:
        print(f'Writing {len(testing)} testing phase fields to {phase_fields}_testing.dat')

    for i in testing:
        print (' '.join(map(str, i)), file=open(f'{phase_fields}_testing.dat','a'))

    print("==============================================")
    return testing

