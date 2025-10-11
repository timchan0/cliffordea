"""Module for enumerating faults in noisy stim circuits."""

import numpy as np
import numpy.typing as npt
import stim

from cliffordep.type_aliases import ErrorEvent
from cliffordep.pauli_string_tools import forget_sign
from cliffordep.noisy_circuit_tools_base import BaseCultivationCircuit


class CultivationCircuit(BaseCultivationCircuit):
    """A class representing a noisy stim circuit with methods to analyze faults.

    Extends `BaseCultivationCircuit`.
    """


    def get_syndrome_and_effect(self, error_event: ErrorEvent):
        """Get the syndrome and the resultant Pauli string after inserting an error event.
        
        Input:
        * `error_event` the error event to analyze.

        Require:
        * No qubit is noisily measured more than once per tick in `self.noisy_circuit`.

        Output:
        * `syndrome` a tuple of booleans representing the syndrome, where each boolean
        indicates whether the corresponding detector has been flipped.
        * `effect` the effect of the error event when propagated to the end of the circuit,
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


    def _error_event_to_pauli_string(self, name: str, targets: tuple[stim.GateTarget, ...]):
        """Convert an error event to a `stim.PauliString`.
        
        Input:
        * `name` the name of the error event,
        which can be 'E', 'X_ERROR', 'Y_ERROR', 'Z_ERROR', 'MX', 'MY', 'MZ'.
        The measurement faults do not affect the Pauli string.
        * `targets` a tuple of stim.GateTarget objects representing the qubits the error event acts on.

        Output:
        * A stim.PauliString representing the error event.
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