"""Module for brute-force-enumerating faults in noisy stim circuits."""

from collections import defaultdict, Counter
from collections.abc import Iterable
import itertools
from math import prod

import numpy as np
import numpy.typing as npt
import stim

from cliffordea.enum.combinators._base import BaseDisjointCombinator
from cliffordea.enum.types import ErrorLocation, ErrorEvent
from cliffordea.enum.cultivation_circuit import CultivationCircuit
from cliffordea.accept.pauli import forget_sign
from cliffordea.enum.combinators._base import error_event_count


class ErrorEventCombinator(BaseDisjointCombinator):
    """Group all error events in a noisy circuit by their error location.

    Extends `BaseDisjointCombinator`.
    Finds undetected fault combinations by brute force
    i.e. iterating through all fault combinations
    and recording which ones have zero detector signature.
    """

    def __init__(self, noisy_circuit, print_progress=False):
        _circuit = CultivationCircuit(noisy_circuit=noisy_circuit)
        _basis: defaultdict[
            ErrorLocation, dict[ErrorEvent, tuple[npt.NDArray[np.bool_], str]]] = defaultdict(dict)
        for source, group in _circuit.group_error_events_by_location().items():
            for error_event in group:
                signature, effect = _circuit.get_signature_and_effect(error_event)
                _basis[source][error_event] = (signature, effect)
        self.basis = dict(_basis)
        """A map from each error location to another map
        from each of the faults it can produce to a pair containing:

            * `signature` a tuple of booleans representing the detector
              signature of the fault.
            * `effect` a string representing the resultant Pauli effect.
        """
        if print_progress:
            print(f"Finished enumerating all faults.")
        self.circuit = _circuit
        """The noisy circuit to analyze."""

    def __repr__(self):
        return f"{self.__class__.__name__}({self.circuit})"


    def _get_undetected_configurations_for_fault_count(
            self,
            fault_count: int,
            print_progress: bool = False,
    ):
        result: defaultdict[str, Counter[int]] = defaultdict(Counter)
        if print_progress:
            print(f"Finding undetected {fault_count}-fault configurations...")
        if fault_count == 0:
            result['_'*self.circuit.noisy_circuit.num_qubits][1] += 1
        else:
            combos = itertools.combinations(self.basis.items(), fault_count)
            for combo in combos:
                pairs = itertools.product(*(fault_dict.values() for _, fault_dict in combo))
                for pair in pairs:
                    has_nonzero_signature = self._has_nonzero_signature(
                        signature for signature, _ in pair
                    )
                    if not has_nonzero_signature:
                        product_string = prod(
                            (stim.PauliString(effect) for _, effect in pair),
                            start=stim.PauliString()
                        )
                        probability_denominator = prod(
                            error_event_count(process_name)
                            for (_, process_name, _), _ in combo)
                        result[forget_sign(product_string)][probability_denominator] += 1
        if print_progress:
            print(f"Done. They have {len(result)} distinct resultant effects.")
        return dict(result)


    @staticmethod
    def _has_nonzero_signature(
            signatures: Iterable[npt.NDArray[np.bool_]],
    ) -> bool:
        """Return whether the XOR of detector signatures is nonzero.
        
        :param signatures: Nonempty, equal-length detector signatures.

        :return: Whether their elementwise XOR has any set bits.
        """
        sum_all: npt.NDArray[np.signedinteger] = sum(signatures) % 2 # type: ignore
        return any(sum_all)
