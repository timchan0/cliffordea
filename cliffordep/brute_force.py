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


class BruteCultivationCircuit(CultivationCircuit):
    """A class representing a noisy stim circuit with brute-force methods to analyze faults.

    Extends `CultivationCircuit`.
    """

    def group_faults_by_source(self, restrict_to_data: bool = True) -> dict[
        FaultSource,
        dict[Fault, tuple[npt.NDArray[np.bool_], str]]
    ]:
        """Group all possible faults in a noisy circuit by their source Stim gate.
        
        Input:
        * `restrict_to_data` whether to restrict all effects to only the data qubits.

        Output:
        * A map from a `FaultSource` to another map from a `Fault` to a tuple containing:
            - `syndrome` a tuple of booleans representing the syndrome of the fault.
            - `effect` a string representing the resultant Pauli operator of the fault.
        """
        _faults: defaultdict[
            FaultSource, dict[Fault, tuple[npt.NDArray[np.bool_], str]]] = defaultdict(dict)
        for source, group in self._group_faults_by_source().items():
            for fault in group:
                syndrome, effect = self._get_syndrome_and_effect(fault)
                if restrict_to_data:
                    effect = ''.join(effect[index] for index in self.DATA_INDICES)
                _faults[source][fault] = (syndrome, effect)
        return dict(_faults)


def undetected_combinations(
        effect_maps: dict[FaultSource, dict[Fault, tuple[npt.NDArray[np.bool_], str]]],
        length: int = 2,
):
    """Find all combinations of `length` faults from `effect_maps` that have trivial syndrome.
    
    Input:
    * `effect_maps` the output of `group_faults_by_source`.
    * `length` the length of combinations to find.

    Output:
    * a map from each effect to a counter of `length`-tuples of denominators.
    Each denominator represents a fault such that if all in the tuple occur,
    the syndrome will be trivial.
    """
    result: defaultdict[str, Counter[tuple[int, ...]]] = defaultdict(Counter)
    if length == 0:
        first_fault_dict, *_ = effect_maps.values()
        (_, first_effect), *_ = first_fault_dict.values()
        result['_'*len(first_effect)][()] += 1
    else:
        combos = itertools.combinations(effect_maps.items(), length)
        for combo in combos:
            pairs = itertools.product(*(fault_dict.values() for _, fault_dict in combo))
            for pair in pairs:
                any_defects = _any_defects(syndrome for syndrome, _ in pair)
                if not any_defects:
                    product_string = prod(
                        (stim.PauliString(effect) for _, effect in pair),
                        start=stim.PauliString()
                    )
                    fault_counts = tuple(sorted([
                        fault_count(source_name)
                        for (_, source_name, _), _ in combo]))
                    result[unsigned_str(product_string)][fault_counts] += 1
    return dict(result)


def _any_defects(syndromes: Iterable[npt.NDArray[np.bool_]]) -> bool:
    """Return if any defects exist in the elementwise XOR of an iterable of syndromes.
    
    Input:
    * `syndromes` a nonempty iterable of syndromes all of equal length.

    Output:
    * whether any defects exist in their elementwise XOR.
    """
    sum_all: npt.NDArray[np.signedinteger] = sum(syndromes) % 2 # type: ignore
    return any(sum_all)