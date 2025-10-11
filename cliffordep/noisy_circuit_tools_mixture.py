"""Module for enumerating faults in noisy stim circuits."""

from typing import Literal

import numpy as np
import numpy.typing as npt
import stim

from cliffordep.type_aliases import ErrorEvent
from cliffordep.pauli_string_tools import forget_sign, CliffordString, Mixture
from cliffordep.noisy_circuit_tools_base import BaseCultivationCircuit


class CultivationCircuitMixture(BaseCultivationCircuit):
    """A class representing a noisy stim circuit with methods to analyze faults.

    Extends `BaseCultivationCircuit`.
    """


    def get_syndrome_and_effect(
            self,
            error_event: ErrorEvent,
            replace_s_with: Literal['T', 'S', 'Z'] = 'S',
    ):
        """Get the syndrome and the resultant Pauli string after inserting an error event.
        
        Input:
        * `error_event` the error event to analyze.
        * `replace_s_with` the unitary to push through if `unitary` is an S gate.
        Also affects S dagger gates.

        Require:
        * No qubit is noisily measured more than once per tick in `self.noisy_circuit`.

        Output:
        * The effect of the error event when propagated to the end of the circuit,
        as a `Mixture` object.
        """
        fault_timeslice, name, targets = error_event
        pauli_string = self._error_event_to_pauli_string(name=name, targets=targets)
        syndrome: npt.NDArray[np.bool_] = np.zeros(self.noisy_circuit.num_detectors, dtype=bool)
        if name.startswith('M'):
            # mixture is pure and contains identity Pauli string only
            for instruction in self._noiseless_layers[fault_timeslice]:
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
            mixture = Mixture({tuple(syndrome): [CliffordString({pauli_string: 1})]})
        else:
            mixture = Mixture({tuple(syndrome): [CliffordString({pauli_string: 1})]})
            remaining_layers = self._noiseless_layers[fault_timeslice+1:]
            for layer in remaining_layers:
                for instruction in layer:
                    if isinstance(instruction, stim.CircuitRepeatBlock):
                        raise ValueError("There is a REPEAT block in the circuit.")
                    mixture.push_through(
                        instruction=instruction,
                        measurement_to_detectors=self._measurement_to_detectors,
                        replace_s_with=replace_s_with,
                    )
        return mixture


    def _error_event_to_pauli_string(self, name: str, targets: tuple[stim.GateTarget, ...]):
        """Convert an error event to an unsigned Pauli string.
        
        Input:
        * `name` the name of the error event,
        which can be 'E', 'X_ERROR', 'Y_ERROR', 'Z_ERROR', 'MX', 'MY', 'MZ'.
        The measurement faults do not affect the Pauli string.
        * `targets` a tuple of stim.GateTarget objects representing the qubits the error event acts on.

        Output:
        * An unsigned Pauli string representing the error event.
        """
        # TODO: avoid using stim.PauliString altogether in this method
        pauli_string = stim.PauliString(self.noisy_circuit.num_qubits)
        if name == 'E':
            for target in targets:
                pauli_string[target.value] = target.pauli_type
        elif (basis := name[0]) != 'M':
            (target,) = targets
            pauli_string[target.value] = basis
        return forget_sign(pauli_string)