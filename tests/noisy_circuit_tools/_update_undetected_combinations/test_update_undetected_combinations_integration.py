from collections import defaultdict, Counter
import itertools

import stim

from cliffordep.noisy_circuit_tools import _update_undetected_combinations, fault_count

def test_1_effect(dummy_fault_count):
    # Patch fault_count to use our dummy version
    orig_fault_count = fault_count
    try:
        globals()['fault_count'] = dummy_fault_count
        undetected_combinations = defaultdict(Counter)
        effect = "XI"
        candidate_1 = [ ( (0, "X_ERROR", (stim.GateTarget(0),)), 2 ), ( (1, "Z_ERROR", (stim.GateTarget(1),)), 3 ) ]
        candidate_2 = [ ( (0, "X_ERROR", (stim.GateTarget(0),)), 1 ), ( (1, "X_ERROR", (stim.GateTarget(1),)), 1 ) ]
        _update_undetected_combinations(undetected_combinations, effect, candidate_1)
        _update_undetected_combinations(undetected_combinations, effect, candidate_2)
        assert undetected_combinations == defaultdict(Counter, {
            effect: Counter({(1, 1): 7})
        })
    finally:
        globals()['fault_count'] = orig_fault_count

def test_multiple_effects(dummy_fault_count):
    orig_fault_count = fault_count
    try:
        globals()['fault_count'] = dummy_fault_count

        undetected_combinations = defaultdict(Counter)
        effect_1 = "XI"
        effect_2 = "IZ"
        effect_3 = "YY"
        candidate_1 = [ ( (0, "X_ERROR", (stim.GateTarget(0),)), 1 ), ( (1, "Z_ERROR", (stim.GateTarget(1),)), 1 ) ]
        candidate_2 = [ ( (2, "DEPOLARIZE2", (stim.target_x(0), stim.target_y(1))), 2 ), ( (3, "Y_ERROR", (stim.GateTarget(1),)), 1 ) ]
        x00 = (0, "X_ERROR", (stim.GateTarget(0),))
        candidate_3 = [ (x00, 1), (x00, 2) ]
        _update_undetected_combinations(undetected_combinations, effect_1, candidate_1)
        _update_undetected_combinations(undetected_combinations, effect_2, candidate_2)
        _update_undetected_combinations(undetected_combinations, effect_3, candidate_3)
        assert undetected_combinations == defaultdict(Counter, {
            effect_1: Counter({(1, 1): 1}),
            effect_2: Counter({(1, 15): 2}),
        })
    finally:
        globals()['fault_count'] = orig_fault_count

def test_length_2_candidates(dummy_fault_count):
    orig_fault_count = fault_count
    effect = "XY"
    try:
        globals()['fault_count'] = dummy_fault_count

        options = [
            (
                ((0, "X_ERROR", (stim.GateTarget(0),)), 1),
                ((1, "Y_ERROR", (stim.GateTarget(1),)), 1),
                ((2, "Z_ERROR", (stim.GateTarget(1),)), 1),
            ),
            (
                ((0, "X_ERROR", (stim.GateTarget(0),)), 1),
                ((3, "DEPOLARIZE2", (stim.target_x(0), stim.target_y(1))), 7),
            ),
        ]
        for candidate in itertools.product(*options):
            undetected_combinations = defaultdict(Counter)
            _update_undetected_combinations(undetected_combinations, effect, candidate)

            goal = defaultdict(Counter)
            (source_1, count_1), (source_2, count_2) = candidate
            if source_1 != source_2:
                denominators = tuple(sorted([
                                fault_count(source_1[1]),
                                fault_count(source_2[1]),
                            ]))
                goal[effect][denominators] += count_1 * count_2

            assert undetected_combinations == goal
    finally:
        globals()['fault_count'] = orig_fault_count