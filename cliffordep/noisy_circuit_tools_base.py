"""Module for enumerating faults in noisy stim circuits."""

from collections import defaultdict
from collections.abc import Sequence
from functools import cached_property
from typing import Literal

import stim

from cliffordep.noiseless_circuit_tools import split_by_ticks
from cliffordep.constants import DEPOLARIZE2_FAULTS
from cliffordep.type_aliases import Fault, FaultSource
from cliffordep.pauli_string_tools import push_through_transversal


class BaseCultivationCircuit:
    """A class representing a noisy stim circuit with methods to analyze faults.
    
    Instance attributes:
    * `noisy_circuit` a `stim.Circuit` annotated with noise.
    * `DATA_INDICES` a set of indices corresponding to the data qubits.
    * `STABILIZER_GENERATORS` the generators of the stabilizer group.
    * `LOGICAL_X` a Pauli string representing a logical X operator.
    * `LOGICAL_Z` ditto for Z.
    """


    def __init__(
            self,
            noisy_circuit: stim.Circuit,
            data_indices: Sequence[int],
            stabilizer_generators: Sequence[stim.PauliString],
            logical_x: stim.PauliString,
            logical_z: stim.PauliString,
    ) -> None:
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
        self.DATA_INDICES = data_indices
        self.STABILIZER_GENERATORS = stabilizer_generators
        self.LOGICAL_X = logical_x
        self.LOGICAL_Z = logical_z


    def __repr__(self):
        return f"""{self.__class__.__name__}(
        noisy_circuit={self.noisy_circuit},
        data_indices={self.DATA_INDICES},
        stabilizer_generators={self.STABILIZER_GENERATORS},
        logical_x={self.LOGICAL_X},
        logical_z={self.LOGICAL_Z}
    )"""


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


    def group_faults_by_source(self):
        """Group all possible faults in a noisy circuit by their source Stim gate.

        Output:
        * A map from a `FaultSource` to a set `Fault`s.
        """
        _faults: defaultdict[FaultSource, set[Fault]] = defaultdict(set)
        for timeslice, layer in enumerate(split_by_ticks(self.noisy_circuit)):
            for instruction in layer:
                if isinstance(instruction, stim.CircuitRepeatBlock):
                    raise ValueError("There is a REPEAT block in the circuit.")
                for basis in ('X', 'Y', 'Z'):
                    if instruction.name in {f'{basis}_ERROR', f'M{basis}'}:
                        for target in instruction.targets_copy():
                            _faults[timeslice, instruction.name, (target,)].add(
                                (timeslice, instruction.name, (target,)))
                    elif instruction.name == 'DEPOLARIZE1':
                        for target in instruction.targets_copy():
                            _faults[timeslice, instruction.name, (target,)].add(
                                (timeslice, f'{basis}_ERROR', (target,)))
                if instruction.name == 'DEPOLARIZE2':
                    for target_1, target_2 in instruction.target_groups():
                        group = _faults[timeslice, instruction.name, (target_1, target_2)]
                        for fault in DEPOLARIZE2_FAULTS:
                            match fault:
                                case ('I', basis):
                                    group.add((timeslice, f'{basis}_ERROR', (target_2,)))
                                case (basis, 'I'):
                                    group.add((timeslice, f'{basis}_ERROR', (target_1,)))
                                case (basis_1, basis_2):
                                    group.add((timeslice, 'E', (
                                        stim.target_pauli(target_1.value, basis_1),
                                        stim.target_pauli(target_2.value, basis_2),
                                    )))
        return dict(_faults)


    def string_to_logical_vector(self, cultivated_state: Literal['T', 'S', 'Z'], data_string: str):
        clifford = push_through_transversal(data_string, gate=cultivated_state)
        clifford.postselect_from_stabilizers(self.STABILIZER_GENERATORS)
        logical_vector = clifford.get_logical_amplitudes(self.LOGICAL_X, self.LOGICAL_Z)
        logical_vector.transfer_xy_to_iz(logical_state=cultivated_state)
        return logical_vector