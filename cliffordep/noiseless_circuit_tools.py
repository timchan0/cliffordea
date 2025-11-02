"""Module for inserting faults into circuits."""

from typing import Iterable

import stim

from cliffordep.type_aliases import ErrorEvent


def split_by_ticks(circuit: stim.Circuit):
    """Split a circuit into a list of circuits, one for each timeslice.

    Input:
    * `circuit` a stim.Circuit object.
    
    Output:
    * `circuits` a list of new stim.Circuit objects, one for each timeslice.
    Its length is `circuit.num_ticks + 1`.
    Does not modify the input circuit.
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
    Does not modify the input circuits.
    """
    composed = stim.Circuit()
    for circuit in circuits:
        composed += circuit
        composed.append("TICK") # type: ignore
    composed.pop()  # remove last TICK
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
    * A new circuit with the error events inserted.
    Does not modify the input circuit.
    The tags are not preserved in the output circuit.
    """
    circuits = split_by_ticks(circuit)
    for timeslice, name, targets in error_events:
        if name.startswith('M'):
            target, = targets
            # find the index of the instruction containing the erroneous measurement
            for instruction_index, instruction in enumerate(circuits[timeslice]):
                if isinstance(instruction, stim.CircuitRepeatBlock):
                    raise ValueError("There is a REPEAT block in the circuit.")
                if instruction.num_measurements:
                    # TODO: store measurement index in `ErrorEvent` to avoid below search
                    targets_copy: list[stim.GateTarget] = instruction.targets_copy()
                    if target in targets_copy:
                        target_index = targets_copy.index(target)
                        insertand = stim.Circuit()
                        insertand.append(stim.CircuitInstruction(name, targets_copy[:target_index]))
                        insertand.append(stim.CircuitInstruction(name, [target], [probability]))
                        insertand.append(stim.CircuitInstruction(name, targets_copy[target_index+1:]))
                        circuits[timeslice].pop(instruction_index)
                        circuits[timeslice].insert(instruction_index, insertand)
                        break
        else:
            circuits[timeslice].append(name, targets, probability)
    return compose_slices(circuits)