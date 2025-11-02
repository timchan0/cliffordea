from collections.abc import Container

import stim

from cliffordep.noise._noise import NoiseModel, NoiseRule
from cliffordep.noiseless_circuit_tools import split_by_ticks, compose_slices


def uniformly_depolarize(
        noiseless_circuit: stim.Circuit,
        noise_level: float,
        noisy_timeslices: None | Container[int] = None,
):
    """Near-standard circuit depolarizing noise.

    Everything has the same parameter p.
    Single qubit clifford gates get single qubit depolarization.
    Two qubit clifford gates get single qubit depolarization.
    Dissipative gates have their result probabilistically bit flipped (or phase flipped if appropriate).

    Non-demolition measurement is treated a bit unusually in that it is the result that is flipped instead of
    the input qubit. The input qubit is depolarized.

    Only the qubits that are acted upon at least once are subject to depolarization.

    Input:
    * `noiseless_circuit` a stim.Circuit without noise.
    * `noise_level` a float in [0, 1].
    * `noisy_timeslices` an optional iterable of timeslice indices to apply noise to.
    If unspecified, noise is applied to all timeslices, *except the first and last*.
    """
    if noisy_timeslices is None:
        noisy_timeslices = range(1, noiseless_circuit.num_ticks)
    system_qubit_indices: set[int] = set()
    for instruction in noiseless_circuit:
        if isinstance(instruction, stim.CircuitInstruction):
            for target in instruction.targets_copy():
                if (val:=target.qubit_value) is not None:
                    system_qubit_indices.add(val)
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
        },
    )
    layers = split_by_ticks(noiseless_circuit)
    noisy_layers: list[stim.Circuit] = []
    for timeslice, layer in enumerate(layers):
        noisy_layer = model.noisy_circuit(
                layer,
                system_qubit_indices=system_qubit_indices,
            ) if timeslice in noisy_timeslices else layer
        noisy_layers.append(noisy_layer)
        
        for instruction in layer:
            if isinstance(instruction, stim.CircuitInstruction) and (instruction.name in {"MX", "MY", "M"}):
                for target in instruction.targets_copy():
                    if (val:=target.qubit_value) is not None:
                        system_qubit_indices.remove(val)

    return compose_slices(noisy_layers)