from collections.abc import Container

import stim

from cliffordep.noise._noise import NoiseModel
from cliffordep.noiseless_circuit_tools import split_by_ticks, compose_slices


def uniformly_depolarize(
        circuit_without_noise: stim.Circuit,
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
    * `circuit_without_noise` a stim.Circuit without noise.
    * `noise_level` a float in [0, 1].
    * `noisy_timeslices` an optional iterable of timeslice indices to apply noise to.
    If unspecified, noise is applied to all timeslices.
    """
    system_qubit_indices: set[int] = set()
    for instruction in circuit_without_noise:
        if isinstance(instruction, stim.CircuitInstruction):
            for target in instruction.targets_copy():
                if (val:=target.qubit_value) is not None:
                    system_qubit_indices.add(val)
    model = NoiseModel.uniform_depolarizing(noise_level)
    if noisy_timeslices is None:
        return model.noisy_circuit(
            circuit_without_noise,
            system_qubit_indices=system_qubit_indices,
        )
    layers = split_by_ticks(circuit_without_noise)
    noisy_layers: list[stim.Circuit] = []
    for timeslice, layer in enumerate(layers):
        noisy_layer = model.noisy_circuit(
                    layer,
                    system_qubit_indices=system_qubit_indices,
                ) if timeslice in noisy_timeslices else layer
        noisy_layers.append(noisy_layer)
    return compose_slices(noisy_layers)