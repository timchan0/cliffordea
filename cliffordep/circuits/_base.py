import stim


def find_data_indices(inner_circuit: stim.Circuit) -> tuple[int, ...]:
    """Return the indices of the data qubits in ascending order."""
    for instruction in inner_circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            raise ValueError("Repeat blocks not supported.")
        if instruction.name == "MPP":
            return tuple(sorted(
                target.qubit_value for target in instruction.targets_copy()
                if target.qubit_value is not None
            ))
    raise ValueError("No MPP instruction found in inner circuit, so could not determine data indices.")


def find_stabilizer_generators(circuit: stim.Circuit) -> tuple[tuple[int, ...], ...]:
    """Return the stabilizer generators using qubit indices from `circuit`.
    
    :param circuit: A stim circuit containing a MPP instructions
        indicating the state's stabilizer generators.
    :return stabilizer_generators: A tuple of tuples,
        where each inner tuple contains the indices of qubits
        that are part of a stabilizer generator.
        Each inner tuple is sorted in ascending order,
        but the outer tuple is not sorted.
    """
    result: set[tuple[int, ...]] = set()
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            raise ValueError("Repeat blocks not supported.")
        if instruction.name == "MPP":
            if any(target.pauli_type=='Y' for target in instruction.targets_copy()):
                continue
            for target_group in instruction.target_groups():
                result.add(tuple(sorted(
                    target.qubit_value for target in target_group
                    if target.qubit_value is not None
                )))
    return tuple(result)