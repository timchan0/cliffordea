"""Module for inserting faults into circuits."""

from typing import Iterable

import stim

from cliffordep.type_aliases import ErrorEvent


def split_by_ticks(circuit: stim.Circuit):
    """Split a circuit into a list of circuits, one for each timeslice.

    Input:
    * `circuit` a stim.Circuit object.
    
    Output:
    * `circuits` a list of stim.Circuit objects, one for each timeslice.
    Its length is `circuit.num_ticks + 1`.
    """
    circuits: list[stim.Circuit] = []
    current_circuit = stim.Circuit()
    for op in circuit:
        if op.name == "TICK":
            circuits.append(current_circuit)
            current_circuit = stim.Circuit()
        else:
            current_circuit.append(op)
    circuits.append(current_circuit)
    return circuits

def compose_slices(circuits: list[stim.Circuit]):
    """Inverse of `split_by_ticks`.
    
    Input:
    * `circuits` a list of circuits, one for each timeslice.

    Output:
    * `composed` the sequential composition of the circuits in `circuits`.
    """
    composed = stim.Circuit()
    for circuit in circuits:
        composed += circuit
        composed.append("TICK") # type: ignore
    composed.pop()
    return composed

def insert_error_events(
        circuit: stim.Circuit,
        error_events: Iterable[ErrorEvent],
        probability: float = 1,
):
    """Insert error events into stim circuit.
    
    Input:
    * `circuit` a stim.Circuit object.
    * `error_events` an iterable of error events to insert.
    * `probability` the probability of the error events.

    Output:
    * The circuit with the error events inserted.
    """
    circuits = split_by_ticks(circuit)
    for timeslice, name, targets in error_events:
        circuits[timeslice].append(name, targets, probability)
    return compose_slices(circuits)