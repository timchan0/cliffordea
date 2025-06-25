import stim

from cliffordep.noise._noise import NoiseModel


def uniformly_depolarize(circuit_without_noise: stim.Circuit, noise_level: float):
    """Near-standard circuit depolarizing noise.

    Everything has the same parameter p.
    Single qubit clifford gates get single qubit depolarization.
    Two qubit clifford gates get single qubit depolarization.
    Dissipative gates have their result probabilistically bit flipped (or phase flipped if appropriate).

    Non-demolition measurement is treated a bit unusually in that it is the result that is flipped instead of
    the input qubit. The input qubit is depolarized.

    Only the qubits that are acted upon at least once are subject to depolarization.
    """
    system_qubit_indices: set[int] = set()
    for instruction in circuit_without_noise:
        if isinstance(instruction, stim.CircuitInstruction):
            for target in instruction.targets_copy():
                if (val:=target.qubit_value) is not None:
                    system_qubit_indices.add(val)
    return NoiseModel.uniform_depolarizing(noise_level).noisy_circuit(
        circuit_without_noise,
        system_qubit_indices=system_qubit_indices,
    )