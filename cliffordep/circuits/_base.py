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


def find_logical_s_gate(circuit: stim.Circuit) -> stim.Circuit:
    """Return the logical S gate for a double-check circuit.

    :param circuit: A logical-Clifford-measurement circuit containing
        a logical S dagger,
        an X-parity measurement,
        and a logical S,
        among other instructions.
    :return logical_s: A stim circuit representing the logical S gate.
    """
    result = stim.Circuit()
    instructions_containing_s = []
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            raise ValueError("Repeat blocks not supported.")
        if instruction.name in {"S", "S_DAG"}:
            instructions_containing_s.append(instruction)
    if len(instructions_containing_s) == 4:
        relevant_indices = (2, 3)
    elif len(instructions_containing_s) == 2:
        relevant_indices = (1,)
    else:
        raise ValueError(
            "Expected either 2 or 4 S/S_DAG instructions in the circuit, "
            f"but found {len(instructions_containing_s)}."
        )
    for index in relevant_indices:
        result.append(instructions_containing_s[index])
    return result