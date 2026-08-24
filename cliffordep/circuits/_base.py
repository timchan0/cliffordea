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