"""Module for inserting faults into circuits."""

from typing import Iterable

import stim

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

def insert_fault(
        circuit: stim.Circuit,
        timeslice: int,
        name: str,
        targets: int | stim.GateTarget | Iterable[int | stim.GateTarget],
        probability: float = 1,
):
    """Insert a fault on given qubits at the end of a give timeslice.
    
    Input:
    * `circuit` a stim.Circuit object.
    * `timeslice` which timeslice to insert the fault after.
    * `name` the name of the fault. Possible values are
    "X_ERROR", "Y_ERROR", "Z_ERROR", "E", "DEPOLARIZE1", "DEPOLARIZE2".
    * `targets` the qubits to apply the fault to. This can be a single integer,
      a stim.GateTarget, or an iterable of integers or stim.GateTargets.
      If an iterable is provided, the fault will be applied to all targets in the iterable.
    * `probability` the probability of the fault.

    Output:
    * The circuit with the fault inserted after the specified timeslice.
    """
    circuits = split_by_ticks(circuit)
    circuits[timeslice].append(name, targets, probability)
    return compose_slices(circuits)