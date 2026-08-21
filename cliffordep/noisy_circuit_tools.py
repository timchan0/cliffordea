"""Module for enumerating faults in noisy stim circuits."""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import cast

import numpy as np
import numpy.typing as npt
import stim

from cliffordep.constants import (
    DEPOLARIZE2_ERROR_EVENTS,
    ONE_QUBIT_ERROR_EVENTS,
)
from cliffordep.noiseless_circuit_tools import (
    measurement_event_key,
    measurement_locations_by_event,
    split_by_ticks,
)
from cliffordep.type_aliases import (
    ErrorEvent,
    ErrorLocation,
    MeasurementEventKey,
    MeasurementLocation,
    PauliMask,
    SyndromeMask,
)


ResponseMask = int
"""Packed syndrome mask followed by a packed final Pauli effect."""

GeneratorResponses = tuple[
    tuple[ResponseMask, ...],
    tuple[ResponseMask, ...],
]
"""X-generator responses followed by Z-generator responses."""

@dataclass(frozen=True, slots=True)
class _ReversePropagationData:
    """Store the results of one reverse circuit sweep.

    :param responses_by_timeslice: X/Z generator responses immediately after
        each circuit timeslice.
    :param measurement_index_by_event: Global measurement indices keyed by
        their timeslice and canonical target group.
    :param measurement_detector_masks: Packed detector responses in global
        measurement order.
    """

    responses_by_timeslice: tuple[GeneratorResponses, ...]
    """Generator responses immediately after every timeslice."""
    measurement_index_by_event: dict[MeasurementEventKey, int]
    """Global measurement indices keyed without storing Stim targets."""
    measurement_detector_masks: tuple[SyndromeMask, ...]
    """Packed detector masks in global measurement order."""
def _response_for_pauli(
        pauli: stim.PauliString,
        x_responses: Sequence[ResponseMask],
        z_responses: Sequence[ResponseMask],
) -> ResponseMask:
    """Combine generator responses into the response of one Pauli.

    :param pauli: Integer Pauli values in qubit order.
    :param x_responses: Responses of X generators in the same qubit frame.
    :param z_responses: Responses of Z generators in the same qubit frame.
    :return response: The XOR-combined response of ``pauli``.
    """
    response = 0
    for qubit in range(len(pauli)):
        pauli_value = pauli[qubit]
        if pauli_value in (1, 2):
            response ^= x_responses[qubit]
        if pauli_value in (2, 3):
            response ^= z_responses[qubit]
    return response


def _mask_to_bool_array(
        mask: int,
        width: int,
) -> npt.NDArray[np.bool_]:
    """Decode a little-endian integer mask into a boolean array.

    :param mask: The integer mask to decode.
    :param width: The number of output bits.
    :return values: A boolean array ordered from least- to most-significant bit.
    """
    return np.fromiter(
        (bool(mask >> index & 1) for index in range(width)),
        dtype=bool,
        count=width,
    )


def mask_to_unsigned_pauli(mask: PauliMask, qubit_count: int) -> str:
    """Decode a packed X/Z mask into an unsigned Pauli string.

    :param mask: X support in the lower bits followed by Z support.
    :param qubit_count: The number of qubits in each support mask.
    :return unsigned_pauli: The decoded Pauli string using ``_`` for identity.
    """
    support_mask = (1 << qubit_count) - 1
    x_mask = mask & support_mask
    z_mask = mask >> qubit_count & support_mask
    return ''.join(
        'Y' if x_mask >> qubit & 1 and z_mask >> qubit & 1
        else 'X' if x_mask >> qubit & 1
        else 'Z' if z_mask >> qubit & 1
        else '_'
        for qubit in range(qubit_count)
    )


def _measurement_pauli_value(
        instruction_name: str,
        target: stim.GateTarget,
) -> int:
    """Return Stim's integer Pauli value for a measured target.

    :param instruction_name: The measurement instruction name.
    :param target: A target in one measurement group.
    :return pauli_value: ``1``, ``2``, or ``3`` for X, Y, or Z.
    """
    if target.pauli_type != 'I':
        return {'X': 1, 'Y': 2, 'Z': 3}[target.pauli_type]
    if 'X' in instruction_name:
        return 1
    if 'Y' in instruction_name:
        return 2
    return 3


def _apply_tableau_responses(
        tableau: stim.Tableau,
        qubits: tuple[int, ...],
        x_responses: list[ResponseMask],
        z_responses: list[ResponseMask],
) -> None:
    """Pull generator responses backward through one local Clifford tableau.

    :param tableau: The local Clifford action in forward-conjugation form.
    :param qubits: Circuit qubits corresponding to the tableau axes.
    :param x_responses: Mutable responses of the current X generators.
    :param z_responses: Mutable responses of the current Z generators.
    :return: None.
    """
    old_x_responses = tuple(x_responses[qubit] for qubit in qubits)
    old_z_responses = tuple(z_responses[qubit] for qubit in qubits)
    for local_qubit, qubit in enumerate(qubits):
        x_responses[qubit] = _response_for_pauli(
            tableau.x_output(local_qubit),
            old_x_responses,
            old_z_responses,
        )
        z_responses[qubit] = _response_for_pauli(
            tableau.z_output(local_qubit),
            old_x_responses,
            old_z_responses,
        )


def _apply_stim_conjugation_responses(
        instruction: stim.CircuitInstruction,
        qubits: tuple[int, ...],
        x_responses: list[ResponseMask],
        z_responses: list[ResponseMask],
        qubit_count: int,
) -> None:
    """Pull responses through a Clifford lacking a local tableau property.

    :param instruction: The unitary Clifford instruction to cross backward.
    :param qubits: Qubits acted on by the instruction.
    :param x_responses: Mutable responses of the current X generators.
    :param z_responses: Mutable responses of the current Z generators.
    :param qubit_count: The full circuit width used for Stim conjugation.
    :return: None.
    """
    old_x_responses = tuple(x_responses)
    old_z_responses = tuple(z_responses)
    for qubit in qubits:
        x_generator = stim.PauliString(qubit_count)
        x_generator[qubit] = 'X'
        z_generator = stim.PauliString(qubit_count)
        z_generator[qubit] = 'Z'
        x_responses[qubit] = _response_for_pauli(
            x_generator.after(instruction),
            old_x_responses,
            old_z_responses,
        )
        z_responses[qubit] = _response_for_pauli(
            z_generator.after(instruction),
            old_x_responses,
            old_z_responses,
        )


class CultivationCircuit:
    """A class representing a noisy stim circuit with methods to analyze faults."""
    
    def __init__(self, noisy_circuit: stim.Circuit) -> None:
        tagged_circuit = stim.Circuit()
        """Each measurement instruction is tagged with the index of its first measurement."""
        measurement_index = 0
        for instruction in noisy_circuit:
            if isinstance(instruction, stim.CircuitRepeatBlock):
                raise ValueError
            tagged_circuit.append(
                name=instruction.name,
                targets=instruction.targets_copy(),
                arg=instruction.gate_args_copy(),
                tag=str(measurement_index) if (measurement_count:=instruction.num_measurements) else '',
            )
            measurement_index += measurement_count
        self.noisy_circuit = tagged_circuit
        """The `stim.Circuit` annotated with noise."""


    def __repr__(self):
        return f"""{self.__class__.__name__}(noisy_circuit={self.noisy_circuit})"""


    def get_syndrome_and_effect(
            self,
            error_event: ErrorEvent,
    ) -> tuple[npt.NDArray[np.bool_], str]:
        """Get the syndrome and the resultant Pauli string after inserting an error event.

        :param self: The circuit in which the event occurs.
        :param error_event: The error event to analyze.

        Require:
        * No qubit is noisily measured more than once per tick in `self.noisy_circuit`.

        :return syndrome: A tuple of booleans representing the syndrome, where each boolean
            indicates whether the corresponding detector has been flipped.
        :return effect: The effect of the error event when propagated to the end of the circuit,
            as an unsigned Pauli string.
        """
        syndrome_mask, effect_mask = self._get_syndrome_and_effect_masks(
            error_event,
        )
        return (
            _mask_to_bool_array(
                syndrome_mask,
                self.noisy_circuit.num_detectors,
            ),
            mask_to_unsigned_pauli(
                effect_mask,
                self.noisy_circuit.num_qubits,
            ),
        )


    def _get_syndrome_and_effect_masks(
            self,
            error_event: ErrorEvent,
    ) -> tuple[SyndromeMask, PauliMask]:
        """Analyze one error event directly in the packed response domain.

        :param self: The circuit in which the event occurs.
        :param error_event: The measurement or Pauli error event to analyze.
        :return syndrome_mask: The packed detector syndrome.
        :return effect_mask: The packed unsigned final Pauli effect.
        """
        timeslice, name, targets = error_event
        reverse_data = self._reverse_propagation_data
        if name.startswith('M'):
            measurement_index = reverse_data.measurement_index_by_event[
                measurement_event_key(timeslice, targets)
            ]
            return reverse_data.measurement_detector_masks[measurement_index], 0

        x_responses, z_responses = reverse_data.responses_by_timeslice[timeslice]
        response = 0
        if name == 'E':
            pauli_components = (
                (target.value, target.pauli_type) for target in targets)
        else:
            target, = targets
            pauli_components = ((target.value, name[0]),)
        for qubit, pauli_type in pauli_components:
            if pauli_type in ('X', 'Y'):
                response ^= x_responses[qubit]
            if pauli_type in ('Y', 'Z'):
                response ^= z_responses[qubit]

        detector_count = self.noisy_circuit.num_detectors
        syndrome_mask = response & (1 << detector_count) - 1
        effect_mask = response >> detector_count
        return syndrome_mask, effect_mask


    def group_error_events_by_location(self):
        """Group all possible error events in a noisy circuit by their error location.

        :return: A map from a `ErrorLocation` to a list of `ErrorEvent`s.
            Each list is sorted consistently e.g. the order of error events for DEPOLARIZE2 is
            II, IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ
            for targets A and B, where A < B.
        """
        _events: defaultdict[ErrorLocation, list[ErrorEvent]] = defaultdict(list)
        for timeslice, layer in enumerate(split_by_ticks(self.noisy_circuit)):
            for instruction in layer:
                if isinstance(instruction, stim.CircuitRepeatBlock):
                    raise ValueError("There is a REPEAT block in the circuit.")
                name = instruction.name
                if name in ONE_QUBIT_ERROR_EVENTS:
                    for target in instruction.targets_copy():
                        event = (timeslice, name, (target,))
                        _events[event].append(event)
                elif name == 'DEPOLARIZE1':
                    for target in instruction.targets_copy():
                        for basis in ('X', 'Y', 'Z'):
                            _events[timeslice, instruction.name, (target,)].append(
                                (timeslice, f'{basis}_ERROR', (target,)))
                elif name == 'DEPOLARIZE2':
                    for targets in instruction.target_groups():
                        target_1, target_2 = sorted(targets, key=lambda t: t.value)
                        group = _events[timeslice, name, (target_1, target_2)]
                        for event in DEPOLARIZE2_ERROR_EVENTS:
                            match event:
                                case ('I', basis):
                                    group.append((timeslice, f'{basis}_ERROR', (target_2,)))
                                case (basis, 'I'):
                                    group.append((timeslice, f'{basis}_ERROR', (target_1,)))
                                case (basis_1, basis_2):
                                    group.append((timeslice, 'E', (
                                        stim.target_pauli(target_1.value, basis_1),
                                        stim.target_pauli(target_2.value, basis_2),
                                    )))
        return dict(_events)


    @cached_property
    def noiseless_circuit(self):
        return self.noisy_circuit.without_noise()


    @cached_property
    def _measurement_to_detectors(self) -> defaultdict[int, set[int]]:
        """A map from each measurement index in `self.noisy_circuit`
        to a set of indices of the detectors it flips.
        """
        measurement_to_detectors: defaultdict[int, set[int]] = defaultdict(set)
        total_measurement_count = 0
        detector_count: int = 0
        for instruction in self.noisy_circuit:
            if isinstance(instruction, stim.CircuitRepeatBlock):
                raise ValueError("REPEAT blocks not handled in this mapping.")
            if instruction.name == "DETECTOR":
                for target in instruction.targets_copy():
                    measurement_to_detectors[total_measurement_count+target.value].add(detector_count)
                detector_count += 1
            total_measurement_count += instruction.num_measurements
        return measurement_to_detectors


    @cached_property
    def _reverse_propagation_data(self) -> _ReversePropagationData:
        """Build packed generator responses with one reverse circuit sweep.

        :param self: The circuit whose fault responses are precomputed.
        :return reverse_data: Generator responses, measurement lookup, and
            detector masks required for mask-native event analysis.
        """
        qubit_count = self.noisy_circuit.num_qubits
        detector_count = self.noisy_circuit.num_detectors
        measurement_detector_masks = tuple(sum(
            1 << detector
            for detector in self._measurement_to_detectors[measurement_index]
        ) for measurement_index in range(self.noisy_circuit.num_measurements))
        x_responses = [
            1 << (detector_count + qubit)
            for qubit in range(qubit_count)
        ]
        z_responses = [
            1 << (detector_count + qubit_count + qubit)
            for qubit in range(qubit_count)
        ]
        responses_in_reverse_order: list[GeneratorResponses] = []
        measurement_index_by_event: dict[MeasurementEventKey, int] = {}

        for timeslice in reversed(range(len(self._noiseless_layers))):
            responses_in_reverse_order.append((
                tuple(x_responses),
                tuple(z_responses),
            ))
            layer = self._noiseless_layers[timeslice]
            for instruction_index in reversed(range(len(layer))):
                instruction = layer[instruction_index]
                if isinstance(instruction, stim.CircuitRepeatBlock):
                    raise ValueError("There is a REPEAT block in the circuit.")
                data = stim.gate_data(instruction.name)
                if data.is_unitary:
                    if data.is_single_qubit_gate or data.is_two_qubit_gate:
                        tableau = cast(stim.Tableau, data.tableau)
                        for target_group in reversed(instruction.target_groups()):
                            _apply_tableau_responses(
                                tableau,
                                tuple(target.value for target in target_group),
                                x_responses,
                                z_responses,
                            )
                    else:
                        targets = instruction.targets_copy()
                        qubits = tuple(sorted({
                            target.value
                            for target in targets
                            if target.is_qubit_target
                            or target.pauli_type != 'I'
                        }))
                        _apply_stim_conjugation_responses(
                            instruction,
                            qubits,
                            x_responses,
                            z_responses,
                            qubit_count,
                        )
                elif data.produces_measurements:
                    for measurement_index, target_group in enumerate(
                            instruction.target_groups(),
                            start=int(instruction.tag),
                    ):
                        measurement_index_by_event[measurement_event_key(
                            timeslice,
                            target_group,
                        )] = measurement_index
                        detector_response = measurement_detector_masks[
                            measurement_index
                        ]
                        for target in target_group:
                            qubit = target.value
                            measured_pauli = _measurement_pauli_value(
                                instruction.name,
                                target,
                            )
                            x_anticommutes = measured_pauli in (2, 3)
                            z_anticommutes = measured_pauli in (1, 2)
                            if data.is_reset:
                                x_responses[qubit] = (
                                    detector_response
                                    if x_anticommutes else 0
                                )
                                z_responses[qubit] = (
                                    detector_response
                                    if z_anticommutes else 0
                                )
                            else:
                                if x_anticommutes:
                                    x_responses[qubit] ^= detector_response
                                if z_anticommutes:
                                    z_responses[qubit] ^= detector_response
                elif data.is_reset:
                    for target in instruction.targets_copy():
                        x_responses[target.value] = 0
                        z_responses[target.value] = 0

        return _ReversePropagationData(
            responses_by_timeslice=tuple(reversed(responses_in_reverse_order)),
            measurement_index_by_event=measurement_index_by_event,
            measurement_detector_masks=measurement_detector_masks,
        )


    @cached_property
    def _noiseless_layers(self) -> list[stim.Circuit]:
        """A list of stim.Circuit objects, one for each timeslice,
        representing the noiseless layers of the circuit.
        """
        return split_by_ticks(self.noisy_circuit.without_noise())


    @cached_property
    def _measurement_locations_by_event(
            self,
    ) -> dict[MeasurementEventKey, MeasurementLocation]:
        """Lazily index measurement insertion locations for visualization.

        :param self: The circuit whose noiseless layers are indexed.
        :return locations: Instruction and target-group indices keyed by
            measurement event.
        """
        return measurement_locations_by_event(self._noiseless_layers)
