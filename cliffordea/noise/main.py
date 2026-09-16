from collections.abc import Container

import stim

from cliffordea.noise._noise import NoiseModel, NoiseRule
from cliffordea.noiseless_circuit_tools import split_by_ticks, compose_slices


def uniformly_depolarize(
        noiseless_circuit: stim.Circuit,
        noise_level: float,
        noisy_timeslices: None | Container[int] = None,
        system_qubit_indices: None | set[int] = None,
        immune_qubit_indices: None | set[int] = None,
):
    """Near-standard circuit depolarizing noise.

    Everything has the same noise level parameter p.
    Single qubit clifford gates get 1-qubit depolarization.
    Two qubit clifford gates get 2-qubit depolarization.
    Dissipative gates have their result probabilistically bit flipped (or phase flipped if appropriate).

    Non-demolition measurement is treated a bit unusually in that it is the result that is flipped instead of
    the input qubit. The input qubit is depolarized.

    Only the qubits that are acted upon at least once are subject to depolarization.

    :param noiseless_circuit: A stim.Circuit without noise,
        which may optionally contain one MPP instruction that defines where the data qubits enter.
        The data qubits are noiseless before this introductory MPP.
    :param noise_level: A float in [0, 1].
    :param noisy_timeslices: An optional iterable of timeslice indices to apply noise to.
        If unspecified, noise is applied to all timeslices, *except the first and last*.
    :param system_qubit_indices: An optional set of qubit indices to be considered system qubits.
        If unspecified, all qubits acted upon at least once in the circuit are considered system qubits.
    :param immune_qubit_indices: An optional set of qubit indices to be considered immune to noise,
        even if they are operated on.
    """
    if noisy_timeslices is None:
        noisy_timeslices = range(1, noiseless_circuit.num_ticks)

    data_qubits: set[int] = set()
    if system_qubit_indices is None:
        system_qubit_indices = set()
        for instruction in noiseless_circuit:
            if isinstance(instruction, stim.CircuitInstruction):
                for target in instruction.targets_copy():
                    if (val:=target.qubit_value) is not None:
                        system_qubit_indices.add(val)
                if instruction.name == "MPP":
                    for target in instruction.targets_copy():
                        if (val:=target.qubit_value) is not None:
                            data_qubits.add(val)
    system_qubit_indices.difference_update(data_qubits)

    model = NoiseModel(
        idle_depolarization=noise_level,
        any_clifford_1q_rule=NoiseRule(after={"DEPOLARIZE1": noise_level}),
        any_clifford_2q_rule=NoiseRule(after={"DEPOLARIZE2": noise_level}),
        gate_rules={
            "RX": NoiseRule(after={"Z_ERROR": noise_level}),
            "RY": NoiseRule(after={"X_ERROR": noise_level}),
            "R": NoiseRule(after={"X_ERROR": noise_level}),
            "MX": NoiseRule(flip_result=noise_level),
            "MY": NoiseRule(flip_result=noise_level),
            "M": NoiseRule(flip_result=noise_level),
            "MRX": NoiseRule(after={"Z_ERROR": noise_level}, flip_result=noise_level),
            "MRY": NoiseRule(after={"X_ERROR": noise_level}, flip_result=noise_level),
            "MR": NoiseRule(after={"X_ERROR": noise_level}, flip_result=noise_level),
            "MPP": NoiseRule(),
        },
    )
    layers = split_by_ticks(noiseless_circuit)
    noisy_layers: list[stim.Circuit] = []
    for timeslice, layer in enumerate(layers):
        noisy_layer = model.noisy_circuit(
                layer,
                system_qubit_indices=system_qubit_indices,
                immune_qubit_indices=immune_qubit_indices,
            ) if timeslice in noisy_timeslices else layer
        noisy_layers.append(noisy_layer)
        
        for instruction in layer:
            if isinstance(instruction, stim.CircuitInstruction):
                if instruction.name in {"MX", "MY", "M"}:
                    for target in instruction.targets_copy():
                        if (val:=target.qubit_value) is not None:
                            system_qubit_indices.remove(val)
                elif instruction.name == "MPP":
                    system_qubit_indices.update(data_qubits)

    return compose_slices(noisy_layers)
