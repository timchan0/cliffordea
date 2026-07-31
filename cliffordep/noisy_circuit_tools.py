"""Module for enumerating faults in noisy stim circuits."""

from collections import defaultdict
from functools import cached_property

import numpy as np
import numpy.typing as npt
import stim

from cliffordep.constants import (
    DEPOLARIZE2_ERROR_EVENTS,
    ONE_QUBIT_ERROR_EVENTS,
)
from cliffordep.noiseless_circuit_tools import split_by_ticks
from cliffordep.type_aliases import ErrorEvent, ErrorLocation
from cliffordep.pauli_string_tools import forget_sign


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


    def get_syndrome_and_effect(self, error_event: ErrorEvent):
        """Get the syndrome and the resultant Pauli string after inserting an error event.
        
        :param error_event: The error event to analyze.

        Require:
        * No qubit is noisily measured more than once per tick in `self.noisy_circuit`.

        :return syndrome: A tuple of booleans representing the syndrome, where each boolean
            indicates whether the corresponding detector has been flipped.
        :return effect: The effect of the error event when propagated to the end of the circuit,
            as an unsigned Pauli string.
        """
        timeslice, name, targets = error_event
        pauli_string = self._error_event_to_pauli_string(name=name, targets=targets)
        syndrome: npt.NDArray[np.bool_] = np.zeros(self.noisy_circuit.num_detectors, dtype=bool)
        if name.startswith('M'):
            # pauli_string is identity
            for instruction in self._noiseless_layers[timeslice]:
                if isinstance(instruction, stim.CircuitRepeatBlock):
                    raise ValueError("There is a REPEAT block in the circuit.")
                if instruction.num_measurements:
                    # TODO: store measurement index in `ErrorEvent` to avoid below search
                    for measurement_index, target_group in enumerate(
                        instruction.target_groups(),
                        start=int(instruction.tag),
                    ):
                        if set(target_group) == set(targets):
                            for detector in self._measurement_to_detectors[measurement_index]:
                                syndrome[detector] ^= True
                            break
        else:
            remaining_layers = self._noiseless_layers[timeslice+1:]
            for layer in remaining_layers:
                for instruction in layer:
                    if isinstance(instruction, stim.CircuitRepeatBlock):
                        raise ValueError("There is a REPEAT block in the circuit.")
                    data = stim.gate_data(instruction.name)
                    if produces_measurements:=data.produces_measurements:
                        anticommuting_paulis = self._get_anticommuting_paulis(instruction.name)
                        for measurement_index, target in enumerate(
                            instruction.targets_copy(),
                            start=int(instruction.tag),
                        ):
                            if pauli_string[target.value] in anticommuting_paulis:
                                for detector in self._measurement_to_detectors[measurement_index]:
                                    syndrome[detector] ^= True
                    if is_reset:=data.is_reset:
                        for target in instruction.targets_copy():
                            pauli_string[target.value] = 'I'
                    if not (produces_measurements or is_reset):
                        pauli_string = pauli_string.after(instruction)
        
        return syndrome, forget_sign(pauli_string)


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


    def _error_event_to_pauli_string(self, name: str, targets: tuple[stim.GateTarget, ...]):
        """Convert an error event to a `stim.PauliString`.
        
        :param name: The name of the error event,
            which can be 'E' or a member of `ONE_QUBIT_ERROR_EVENTS`.
            The measurement faults do not affect the Pauli string.
        :param targets: A tuple of stim.GateTarget objects representing the qubits the error event acts on.

        :return: A stim.PauliString representing the error event.
        """
        pauli_string = stim.PauliString(self.noisy_circuit.num_qubits)
        if name == 'E':
            for target in targets:
                pauli_string[target.value] = target.pauli_type
        elif (basis := name[0]) != 'M':
            (target,) = targets
            pauli_string[target.value] = basis
        return pauli_string


    @staticmethod
    def _get_anticommuting_paulis(name: str):
        """Get the set of Paulis that anticommute with the measurement given by `name`."""
        if 'X' in name:
            return {2, 3}
        elif 'Y' in name:
            return {1, 3}
        elif 'Z' in name or name in {'M', 'MR'}:
            return {1, 2}
        else:
            raise NotImplementedError


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
    def _noiseless_layers(self) -> list[stim.Circuit]:
        """A list of stim.Circuit objects, one for each timeslice,
        representing the noiseless layers of the circuit.
        """
        return split_by_ticks(self.noisy_circuit.without_noise())
