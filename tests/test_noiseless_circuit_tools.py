import stim

from cliffordea import noiseless_circuit_tools
from cliffordea.type_aliases import ErrorEvent


def test_batched_error_event_insertion_matches_sequential_insertion():
    """Cached locations preserve sequential insertion across measurement groups."""
    circuit = stim.Circuit("""
        M 0 1
        MX 2
        TICK
        H 0 1 2
        TICK
        MRX 0 1
        MRY 2
    """)
    original_circuit = circuit.copy()
    event_groups: tuple[tuple[tuple[ErrorEvent, ...], float], ...] = (
        (
            (
                (0, 'M', (stim.GateTarget(0),)),
                (0, 'MX', (stim.GateTarget(2),)),
                (1, 'X_ERROR', (stim.GateTarget(1),)),
                (2, 'MRX', (stim.GateTarget(0),)),
                (2, 'MRY', (stim.GateTarget(2),)),
            ),
            0.125,
        ),
        (
            (
                (0, 'M', (stim.GateTarget(1),)),
                (1, 'E', (stim.target_x(0), stim.target_z(1))),
                (2, 'MRX', (stim.GateTarget(1),)),
            ),
            0.375,
        ),
    )

    sequential = circuit
    for events, probability in event_groups:
        sequential = noiseless_circuit_tools.insert_error_events(
            circuit=sequential,
            weighted_error_events=(
                (error_event, probability) for error_event in events
            ),
        )

    slices = noiseless_circuit_tools.split_by_ticks(circuit)
    locations = noiseless_circuit_tools.measurement_locations_by_event(slices)
    assert locations == {
        (0, (0,)): (0, 0),
        (0, (1,)): (0, 1),
        (0, (2,)): (1, 0),
        (2, (0,)): (0, 0),
        (2, (1,)): (0, 1),
        (2, (2,)): (1, 0),
    }
    batched = noiseless_circuit_tools.insert_error_events(
        circuit=circuit,
        weighted_error_events=(
            (error_event, probability)
            for events, probability in event_groups
            for error_event in events
        ),
        measurement_locations=locations,
    )

    assert batched == sequential
    assert circuit == original_circuit
