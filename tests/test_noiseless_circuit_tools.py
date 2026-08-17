import stim

from cliffordep import noiseless_circuit_tools
from cliffordep.type_aliases import ErrorEvent


def test_batched_error_event_insertion_matches_sequential_insertion():
    """Multiple event groups should require only one split and composition.

    The event groups exercise Pauli errors and repeated splitting of
    multi-target measurement and measurement-reset instructions.
    """
    circuit = stim.Circuit("""
        M 0 1
        TICK
        H 0 1
        TICK
        MRX 0 1
    """)
    original_circuit = circuit.copy()
    event_groups: tuple[tuple[tuple[ErrorEvent, ...], float], ...] = (
        (
            (
                (0, 'M', (stim.GateTarget(0),)),
                (1, 'X_ERROR', (stim.GateTarget(1),)),
                (2, 'MRX', (stim.GateTarget(0),)),
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
            error_events=events,
            probability=probability,
        )

    slices = noiseless_circuit_tools.split_by_ticks(circuit)
    for events, probability in event_groups:
        noiseless_circuit_tools._insert_error_events_into_slices(
            circuits=slices,
            error_events=events,
            probability=probability,
        )
    batched = noiseless_circuit_tools.compose_slices(slices)

    assert batched == sequential
    assert circuit == original_circuit
