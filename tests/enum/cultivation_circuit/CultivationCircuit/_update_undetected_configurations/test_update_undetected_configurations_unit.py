from collections import defaultdict, Counter

import stim

from cliffordea.enum.combinators._base import error_event_count

def test_duplicate_source(dummy_error_event_count, update_undetected_configurations):
    original_error_event_count = error_event_count
    try:
        globals()['error_event_count'] = dummy_error_event_count

        undetected_configurations = defaultdict(Counter)
        effect = "ZZ"
        x00 = (0, "X_ERROR", (stim.GateTarget(0),))
        candidate = [ (x00, 1), (x00, 2) ]
        update_undetected_configurations(undetected_configurations, effect, candidate)
        assert undetected_configurations == defaultdict(Counter)
    finally:
        globals()['error_event_count'] = original_error_event_count

def test_2_source_types(dummy_error_event_count, update_undetected_configurations):
    original_error_event_count = error_event_count
    try:
        globals()['error_event_count'] = dummy_error_event_count

        undetected_configurations = defaultdict(Counter)
        effect = "XY"
        candidate = [
            ( (0, "DEPOLARIZE1", (stim.GateTarget(0),)), 1 ),
            ( (1, "Z_ERROR", (stim.GateTarget(1),)), 2 ),
        ]
        update_undetected_configurations(undetected_configurations, effect, candidate)
        assert undetected_configurations == defaultdict(Counter, {
            effect: Counter({3: 2})
        })
    finally:
        globals()['error_event_count'] = original_error_event_count

def test_3_source_types(dummy_error_event_count, update_undetected_configurations):
    orig_error_event_count = error_event_count
    try:
        globals()['error_event_count'] = dummy_error_event_count

        undetected_configurations = defaultdict(Counter)
        effect = "XY"
        candidate = [
            ( (0, "DEPOLARIZE1", (stim.GateTarget(0),)), 1 ),
            ( (1, "Z_ERROR", (stim.GateTarget(1),)), 2 ),
            ( (2, "DEPOLARIZE2", (stim.target_x(0), stim.target_y(1))), 5 ),
        ]
        update_undetected_configurations(undetected_configurations, effect, candidate)
        assert undetected_configurations == defaultdict(Counter, {
            effect: Counter({3*15: 10})
        })
    finally:
        globals()['error_event_count'] = orig_error_event_count