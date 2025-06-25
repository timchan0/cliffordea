"""Deprecated due to being able to do:
```python
noise_model = cliffordep.NoiseModel.uniform_depolarizing(noise_strength)
noisy_circuit = noise_model.noisy_circuit_skipping_mpp_boundaries(circuit)
```
"""

from collections.abc import Sequence

import stim

from cliffordep.noiseless_circuit_tools import split_by_ticks, compose_slices

def depolarize_circuit_slice(
        circuit_slice: stim.Circuit,
        qubits: Sequence[int],
        probability: float = 1,
):
    """Add noise to a circuit slice by inserting depolarizing errors.

    Input:
    * `circuit_slice` a stim.Circuit object representing a single timeslice
    i.e. contains no TICKs.
    * `qubits` which qubits to insert errors on.
    * `probability` the probability of the error.

    Output:
    * The circuit slice with depolarizing errors inserted after all operations.
    """
    unaccounted_for: set[int] = set(qubits)
    for instruction in circuit_slice:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            raise ValueError("There is a REPEAT block in the circuit.")
        if instruction.name == 'CX':
            circuit_slice.append('DEPOLARIZE2', instruction.targets_copy(), probability)
            unaccounted_for.difference_update(gate_target.qubit_value for gate_target in instruction.targets_copy())
    circuit_slice.append('DEPOLARIZE1', unaccounted_for, probability)
    return circuit_slice

def depolarize_circuit(
        circuit: stim.Circuit,
        timeslices: None | Sequence[int] = None,
        qubits: None | Sequence[int] = None,
        probability: float = 1,
):
    """Add noise to a circuit by inserting depolarizing errors.
    
    Input:
    * `circuit` a stim.Circuit object.
    * `timeslices` which timeslices to insert errors after.
    * `qubits` which qubits to insert errors on.
    * `probability` the probability of the error.

    Output:
    * The circuit with depolarizing errors after each specified timeslice.
    """
    if timeslices is None:
        timeslices = range(circuit.num_ticks + 1)
    if qubits is None:
        qubits = range(circuit.num_qubits)
    circuits = split_by_ticks(circuit)
    for timeslice in timeslices:
        circuits[timeslice] = depolarize_circuit_slice(circuits[timeslice], qubits, probability)
    return compose_slices(circuits)