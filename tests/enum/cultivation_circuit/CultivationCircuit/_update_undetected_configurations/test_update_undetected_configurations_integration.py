from collections import defaultdict, Counter
import itertools
import math

import stim

from cliffordea.enum.combinators._base import error_event_count


def test_1_effect(dummy_error_event_count, update_undetected_configurations):
    # Patch error_event_count to use our dummy version
    orig_error_event_count = error_event_count
    try:
        globals()['error_event_count'] = dummy_error_event_count
        undetected_configurations = defaultdict(Counter)
        effect = "XI"
        candidate_1 = [ ( (0, "X_ERROR", (stim.GateTarget(0),)), 2 ), ( (1, "Z_ERROR", (stim.GateTarget(1),)), 3 ) ]
        candidate_2 = [ ( (0, "X_ERROR", (stim.GateTarget(0),)), 1 ), ( (1, "X_ERROR", (stim.GateTarget(1),)), 1 ) ]
        update_undetected_configurations(undetected_configurations, effect, candidate_1)
        update_undetected_configurations(undetected_configurations, effect, candidate_2)
        assert undetected_configurations == defaultdict(Counter, {
            effect: Counter({1: 7})
        })
    finally:
        globals()['error_event_count'] = orig_error_event_count

def test_multiple_effects(dummy_error_event_count, update_undetected_configurations):
    orig_fault_count = error_event_count
    try:
        globals()['error_event_count'] = dummy_error_event_count

        undetected_configurations = defaultdict(Counter)
        effect_1 = "XI"
        effect_2 = "IZ"
        effect_3 = "YY"
        candidate_1 = [ ( (0, "X_ERROR", (stim.GateTarget(0),)), 1 ), ( (1, "Z_ERROR", (stim.GateTarget(1),)), 1 ) ]
        candidate_2 = [ ( (2, "DEPOLARIZE2", (stim.target_x(0), stim.target_y(1))), 2 ), ( (3, "Y_ERROR", (stim.GateTarget(1),)), 1 ) ]
        x00 = (0, "X_ERROR", (stim.GateTarget(0),))
        candidate_3 = [ (x00, 1), (x00, 2) ]
        update_undetected_configurations(undetected_configurations, effect_1, candidate_1)
        update_undetected_configurations(undetected_configurations, effect_2, candidate_2)
        update_undetected_configurations(undetected_configurations, effect_3, candidate_3)
        assert undetected_configurations == defaultdict(Counter, {
            effect_1: Counter({1: 1}),
            effect_2: Counter({15: 2}),
        })
    finally:
        globals()['error_event_count'] = orig_fault_count

def test_length_2_candidates(dummy_error_event_count, update_undetected_configurations):
    orig_fault_count = error_event_count
    effect = "XY"
    try:
        globals()['error_event_count'] = dummy_error_event_count

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
            undetected_configurations = defaultdict(Counter)
            update_undetected_configurations(undetected_configurations, effect, candidate)

            goal = defaultdict(Counter)
            (source_1, count_1), (source_2, count_2) = candidate
            if source_1 != source_2:
                denominators = error_event_count(source_1[1]) * error_event_count(source_2[1])
                goal[effect][denominators] += count_1 * count_2

            assert undetected_configurations == goal
    finally:
        globals()['error_event_count'] = orig_fault_count