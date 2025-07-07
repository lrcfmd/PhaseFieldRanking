import logging
from itertools import permutations as pmts
from ranking_phase_fields.parse_icsd import numatoms
from ranking_phase_fields.logger import get_logger

logger = get_logger(__name__)

def permute(vectors):
    """Return all permutations of each vector in the list, flatten the results."""
    permuted = []
    for v in vectors:
        p = [list(i) for i in pmts(v)]
        permuted.extend(p)
    return permuted

def augment(chain, vec):
    """
    Append each element in vec to the chain.
    If chain is a list, append; if not, convert to list first.
    """
    if not isinstance(chain, list):
        chain = [chain]
    return [chain + [val] for val in vec]

def generate_study(phase_fields, lists, training):
    logger.info(f'Creating testing data for unexplored {phase_fields} phase fields:')

    # If all positions have the same elements (e.g., ['A'], ['A'], ['A'])
    if all(lst == lists[0] for lst in lists):
        logger.info('Phase fields are homogenous (all positions same elements).')
        natoms = numatoms(phase_fields)
        study = [list(field) for field in pmts(lists[0], natoms) if list(field) not in training]
    else:
        # Recursively augment fields with elements from lists
        field = lists[0]
        for i in range(1, len(lists)):
            field = [r for s in map(lambda el: augment(el, lists[i]), field) for r in s]

        study = []
        for f in field:
            if len(set(f)) == len(lists) and sorted(f) not in training and sorted(f) not in study:
                study.append(sorted(f))

    logger.info(f'Generated {len(study)} new phase fields for study.')
    return study
