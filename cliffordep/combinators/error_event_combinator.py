"""Module for brute-force-enumerating faults in noisy stim circuits."""

from collections import defaultdict, Counter
from collections.abc import Iterable
import itertools
from math import prod

import numpy as np
import numpy.typing as npt
import stim

from cliffordep.combinators._base import BaseExclusiveCombinator
from cliffordep.type_aliases import ErrorLocation, ErrorEvent
from cliffordep.noisy_circuit_tools import CultivationCircuit
from cliffordep.pauli_string_tools import forget_sign
from cliffordep.combinators import error_event_count


class ErrorEventCombinator(BaseExclusiveCombinator):
    """Group all error events in a noisy circuit by their error location.

    Extends `BaseExclusiveCombinator`.
    Finds undetected fault combinations by brute force
    i.e. iterating through all fault combinations
    and recording which ones have trivial syndrome.
    """

    def __init__(self, noisy_circuit, print_progress=False):
        _circuit = CultivationCircuit(noisy_circuit=noisy_circuit)
        _basis: defaultdict[
            ErrorLocation, dict[ErrorEvent, tuple[npt.NDArray[np.bool_], str]]] = defaultdict(dict)
        for source, group in _circuit.group_error_events_by_location().items():
            for error_event in group:
                syndrome, effect = _circuit.get_syndrome_and_effect(error_event)
                _basis[source][error_event] = (syndrome, effect)
        self.basis = dict(_basis)
        """A map from each error location to another map
        from each of the faults it can produce to a pair containing:

            * `syndrome` a tuple of booleans representing the syndrome of the fault.
            * `effect` a string representing the resultant Pauli operator of the fault.
        """
        if print_progress:
            print(f"Finished enumerating all faults.")
        self.circuit = _circuit
        """The noisy circuit to analyze."""

    def __repr__(self):
        return f"{self.__class__.__name__}({self.circuit})"


    def _get_undetected_configurations_for_length(
            self,
            length: int,
            print_progress: bool = False,
    ):
        result: defaultdict[str, Counter[int]] = defaultdict(Counter)
        if print_progress:
            print(f"Finding undetected configurations of length {length}...")
        if length == 0:
            result['_'*self.circuit.noisy_circuit.num_qubits][1] += 1
        else:
            combos = itertools.combinations(self.basis.items(), length)
            for combo in combos:
                pairs = itertools.product(*(fault_dict.values() for _, fault_dict in combo))
                for pair in pairs:
                    any_defects = self._any_defects(syndrome for syndrome, _ in pair)
                    if not any_defects:
                        product_string = prod(
                            (stim.PauliString(effect) for _, effect in pair),
                            start=stim.PauliString()
                        )
                        fault_counts = prod(
                            error_event_count(process_name)
                            for (_, process_name, _), _ in combo)
                        result[forget_sign(product_string)][fault_counts] += 1
        if print_progress:
            print(f"Done. They lead to {len(result)} distinct errors.")
        return dict(result)


    @staticmethod
    def _any_defects(syndromes: Iterable[npt.NDArray[np.bool_]]) -> bool:
        """Return if any defects exist in the elementwise XOR of an iterable of syndromes.
        
        :param syndromes: A nonempty iterable of syndromes all of equal length.

        :return: Whether any defects exist in their elementwise XOR.
        """
        sum_all: npt.NDArray[np.signedinteger] = sum(syndromes) % 2 # type: ignore
        return any(sum_all)
