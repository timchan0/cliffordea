"""Module for enumerating faults in noisy stim circuits."""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from functools import cached_property
from typing import Literal

import numpy as np
import numpy.typing as npt
import stim

from cliffordep.noiseless_circuit_tools import split_by_ticks
from cliffordep.constants import DEPOLARIZE2_FAULTS
from cliffordep.type_aliases import Fault, FaultSource
from cliffordep.pauli_string_tools import forget_sign, push_through_transversal


class CultivationCircuit:
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
            stabilizer_generators: Iterable[stim.PauliString],
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


    @cached_property
    def noiseless_circuit(self):
        return self.noisy_circuit.without_noise()


    def get_syndrome_and_effect(self, fault: Fault):
        """Get the syndrome and the resultant Pauli string after inserting a fault.
        
        Input:
        * `fault` the fault to analyze.

        Require:
        * No qubit is noisily measured more than once per tick in `self.noisy_circuit`.

        Output:
        * `syndrome` a tuple of booleans representing the syndrome, where each boolean
        indicates whether the corresponding detector has been flipped.
        * `effect` the effect of the fault when propagated to the end of the circuit,
        as an unsigned Pauli string.
        """
        timeslice, name, targets = fault
        pauli_string = self._fault_to_pauli_string(name=name, targets=targets)
        syndrome: npt.NDArray[np.bool_] = np.zeros(self.noisy_circuit.num_detectors, dtype=bool)
        if name.startswith('M'):
            # pauli_string is identity
            for instruction in self._noiseless_layers[timeslice]:
                if isinstance(instruction, stim.CircuitRepeatBlock):
                    raise ValueError("There is a REPEAT block in the circuit.")
                if instruction.num_measurements:
                    # TODO: store measurement index in `Fault` to avoid below search
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
                        # TODO: handle multiqubit measurements
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


    def _fault_to_pauli_string(self, name: str, targets: tuple[stim.GateTarget, ...]):
        """Convert a fault to a stim.PauliString.
        
        Input:
        * `name` the name of the fault,
        which can be 'E', 'X_ERROR', 'Y_ERROR', 'Z_ERROR', 'MX', 'MY', 'MZ'.
        The measurement faults do not affect the Pauli string.
        * `targets` a tuple of stim.GateTarget objects representing the qubits the fault acts on.

        Output:
        * A stim.PauliString representing the fault.
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
    def _measurement_to_detectors(self) -> dict[int, set[int]]:
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
        return dict(measurement_to_detectors)
    

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


    def restrict_to_data(self, effect: str):
        """Restrict `effect` to only the data qubits.

        Input:
        * `effect` an unsigned Pauli string.

        Output:
        * The unsigned Pauli string restricted to the data qubits.
        """
        return ''.join(effect[index] for index in self.DATA_INDICES)