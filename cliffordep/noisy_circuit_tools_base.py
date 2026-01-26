"""Module for enumerating faults in noisy stim circuits."""

from collections import defaultdict
from functools import cached_property
from typing import final

import stim

from cliffordep.noiseless_circuit_tools import split_by_ticks
from cliffordep.constants import DEPOLARIZE2_ERROR_EVENTS
from cliffordep.type_aliases import ErrorEvent, ErrorLocation


class BaseCultivationCircuit:
    """A class representing a noisy stim circuit with methods to analyze faults."""

    @final
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
    def _noiseless_layers(self) -> list[stim.Circuit]:
        """A list of stim.Circuit objects, one for each timeslice,
        representing the noiseless layers of the circuit.
        """
        return split_by_ticks(self.noisy_circuit.without_noise())


    def group_error_events_by_location(self):
        """Group all possible error events in a noisy circuit by their error location.

        Output:
        * A map from a `ErrorLocation` to a list of `ErrorEvent`s.
        Each list is sorted consistently e.g. the order of error events for DEPOLARIZE2 is
        II, IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ
        for targets A and B, where A < B.
        """
        _events: defaultdict[ErrorLocation, list[ErrorEvent]] = defaultdict(list)
        for timeslice, layer in enumerate(split_by_ticks(self.noisy_circuit)):
            for instruction in layer:
                if isinstance(instruction, stim.CircuitRepeatBlock):
                    raise ValueError("There is a REPEAT block in the circuit.")
                for basis in ('X', 'Y', 'Z'):
                    if instruction.name in {f'{basis}_ERROR', f'M{basis}'}:
                        for target in instruction.targets_copy():
                            _events[timeslice, instruction.name, (target,)].append(
                                (timeslice, instruction.name, (target,)))
                    elif instruction.name == 'DEPOLARIZE1':
                        for target in instruction.targets_copy():
                            _events[timeslice, instruction.name, (target,)].append(
                                (timeslice, f'{basis}_ERROR', (target,)))
                if instruction.name == 'DEPOLARIZE2':
                    for targets in instruction.target_groups():
                        target_1, target_2 = sorted(targets, key=lambda t: t.value)
                        group = _events[timeslice, instruction.name, (target_1, target_2)]
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