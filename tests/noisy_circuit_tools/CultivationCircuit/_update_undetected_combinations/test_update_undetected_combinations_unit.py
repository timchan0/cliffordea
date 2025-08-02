from collections import defaultdict, Counter

import stim

from cliffordep.combinators import fault_count

def test_duplicate_source(dummy_fault_count, update_undetected_combinations):
    orig_fault_count = fault_count
    try:
        globals()['fault_count'] = dummy_fault_count

        undetected_combinations = defaultdict(Counter)
        effect = "ZZ"
        x00 = (0, "X_ERROR", (stim.GateTarget(0),))
        candidate = [ (x00, 1), (x00, 2) ]
        update_undetected_combinations(undetected_combinations, effect, candidate)
        assert undetected_combinations == defaultdict(Counter)
    finally:
        globals()['fault_count'] = orig_fault_count

def test_2_source_types(dummy_fault_count, update_undetected_combinations):
    orig_fault_count = fault_count
    try:
        globals()['fault_count'] = dummy_fault_count

        undetected_combinations = defaultdict(Counter)
        effect = "XY"
        candidate = [
            ( (0, "DEPOLARIZE1", (stim.GateTarget(0),)), 1 ),
            ( (1, "Z_ERROR", (stim.GateTarget(1),)), 2 ),
        ]
        update_undetected_combinations(undetected_combinations, effect, candidate)
        assert undetected_combinations == defaultdict(Counter, {
            effect: Counter({3: 2})
        })
    finally:
        globals()['fault_count'] = orig_fault_count

def test_3_source_types(dummy_fault_count, update_undetected_combinations):
    orig_fault_count = fault_count
    try:
        globals()['fault_count'] = dummy_fault_count

        undetected_combinations = defaultdict(Counter)
        effect = "XY"
        candidate = [
            ( (0, "DEPOLARIZE1", (stim.GateTarget(0),)), 1 ),
            ( (1, "Z_ERROR", (stim.GateTarget(1),)), 2 ),
            ( (2, "DEPOLARIZE2", (stim.target_x(0), stim.target_y(1))), 5 ),
        ]
        update_undetected_combinations(undetected_combinations, effect, candidate)
        assert undetected_combinations == defaultdict(Counter, {
            effect: Counter({3*15: 10})
        })
    finally:
        globals()['fault_count'] = orig_fault_count