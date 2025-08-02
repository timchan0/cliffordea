"""Module for brute-force-enumerating faults in noisy stim circuits."""

from collections import defaultdict, Counter
from collections.abc import Iterable
import itertools
from math import prod

import numpy as np
import numpy.typing as npt
import stim

from cliffordep.type_aliases import FaultSource, Fault
from cliffordep.noisy_circuit_tools import fault_count, CultivationCircuit
from cliffordep.pauli_string_tools import unsigned_str
from cliffordep.combinators import Combinator


class FaultSourceBruteForceCombinator(Combinator):
    """Group all faults in a noisy circuit by their source Stim gate.

    Extends `Combinator`.
    
    Instance attributes:
    * `circuit` the noisy circuit to analyze.
    * `basis` a map from each fault source to another map
    from each of the faults it can produce to a pair containing:
        - `syndrome` a tuple of booleans representing the syndrome of the fault.
        - `effect` a string representing the resultant Pauli operator of the fault.
    """

    def __init__(
            self,
            circuit: CultivationCircuit,
            restrict_to_data: bool = True,
            print_progress: bool = False,
    ):
        _basis: defaultdict[
            FaultSource, dict[Fault, tuple[npt.NDArray[np.bool_], str]]] = defaultdict(dict)
        for source, group in circuit.group_faults_by_source().items():
            for fault in group:
                syndrome, effect = circuit.get_syndrome_and_effect(fault)
                if restrict_to_data:
                    effect = ''.join(effect[index] for index in circuit.DATA_INDICES)
                _basis[source][fault] = (syndrome, effect)
        self.basis = dict(_basis)
        if print_progress:
            print(f"Finished enumerating all faults.")
        super().__init__(circuit, restrict_to_data, print_progress)


    def _get_undetected_fault_combinations_for_length(
            self,
            length: int,
            print_progress: bool = False,
    ):
        result: defaultdict[str, Counter[int]] = defaultdict(Counter)
        if print_progress:
            print(f"Finding undetected combinations of length {length}...")
        if length == 0:
            first_fault_dict, *_ = self.basis.values()
            (_, first_effect), *_ = first_fault_dict.values()
            result['_'*len(first_effect)][1] += 1
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
                            fault_count(source_name)
                            for (_, source_name, _), _ in combo)
                        result[unsigned_str(product_string)][fault_counts] += 1
        if print_progress:
            print(f"Done. They lead to {len(result)} distinct effects.")
        return dict(result)


    @staticmethod
    def _any_defects(syndromes: Iterable[npt.NDArray[np.bool_]]) -> bool:
        """Return if any defects exist in the elementwise XOR of an iterable of syndromes.
        
        Input:
        * `syndromes` a nonempty iterable of syndromes all of equal length.

        Output:
        * whether any defects exist in their elementwise XOR.
        """
        sum_all: npt.NDArray[np.signedinteger] = sum(syndromes) % 2 # type: ignore
        return any(sum_all)