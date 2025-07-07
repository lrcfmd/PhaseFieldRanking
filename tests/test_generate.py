import logging
from ranking_phase_fields.generate_study import generate_study

def test_generate_study_ternary_nonhomogenous_lists():
    # Distinct element sets per position
    A1 = ['Li', 'Be', 'Na']
    A2 = ['Mg', 'Al', 'K']
    A3 = ['S', 'O', 'Cl']
    
    lists = [A1, A2, A3]
    training = []

    logging.getLogger('ranking_phase_fields.generate_study').setLevel(logging.INFO)

    result = generate_study('ternary', lists, training)

    # Each output item should be length 3 (one element from each list)
    assert all(len(item) == 3 for item in result)

    # The elements should be unique inside each item
    assert all(len(set(item)) == 3 for item in result)

    # The items should be sorted unique lists 
    for item in result:
        assert item == sorted(item)

    # Expected number of unique combinations is all distinct picks with unique elements:
    # Since lists have no overlapping elements, total combinations = len(A1)*len(A2)*len(A3)
    expected_count = len(A1) * len(A2) * len(A3)

    assert len(result) == expected_count

    print(f"Test passed: generated {len(result)} unique ternary phase fields.")

