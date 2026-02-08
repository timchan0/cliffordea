from collections import Counter, defaultdict
import itertools
import math
from typing import Literal

import stim

from cliffordep.combinators._base import BaseFaultCombinator
from cliffordep.noisy_circuit_tools_mixture import CultivationCircuitMixture
from cliffordep.combinators._base import get_trivial_syndrome_combinations
from cliffordep.pauli_string_tools import FrozenCliffordString
from cliffordep.type_aliases import FaultBagMixed


class FaultCombinatorMixed(BaseFaultCombinator):
    """Group faults by their syndrome then effect,
    where effects are mixtures of Clifford strings.
    
    Extends `BaseFaultCombinator`.

    Overriden attributes:
    * `basis` same as in `BaseFaultCombinator` but each effect is a normalized frozen Clifford string.
    """
    
    def __init__(
            self,
            noisy_circuit: stim.Circuit,
            print_progress: bool = False,
            replace_s_with: Literal['T', 'S', 'Z'] = 'S',
    ):
        """Input:
        * `noisy_circuit` the `CultivationCircuitMixture` to analyze.
        * `print_progress` whether to print progress.
        * `replace_s_with` the unitary to push through if `unitary` is an S gate.
        Also affects S dagger gates.
        """
        _circuit = CultivationCircuitMixture(noisy_circuit=noisy_circuit)
        _basis: defaultdict[
            tuple[bool, ...], dict[FrozenCliffordString, int]
        ] = defaultdict(dict)
        _index_to_bag: defaultdict[int, list[Counter[float]]] = defaultdict(lambda: [Counter(), Counter(), Counter()])
        for (_, process_name, _), group in _circuit.group_error_events_by_location().items():
            process_class = self._classify(process_name)
            for error_event in group:
                mixture = _circuit.get_syndrome_and_effect(error_event, replace_s_with=replace_s_with)
                for syndrome, submixture in mixture.submixtures.items():
                    for normalized_effect, probability in submixture.items():
                        assert normalized_effect.mutable_copy().norm_squared == 1
                        index = _basis[syndrome].setdefault(
                            normalized_effect, len(_index_to_bag))
                        _index_to_bag[index][process_class][probability] += 1
        self.basis: dict[tuple[bool, ...], dict[FrozenCliffordString, int]] = dict(_basis) # type: ignore
        self.index_to_bag: dict[int, FaultBagMixed] = { # type: ignore
            index: tuple(bag) for index, bag in _index_to_bag.items()}
        if print_progress:
            print(f"Finished enumerating all faults. {str(self)}")
        self.circuit = _circuit

    def __repr__(self):
        return f"{self.__class__.__name__}({self.circuit})"
    

    def _get_undetected_configurations_for_length(
            self,
            length: int,
            print_progress: bool = False
    ) -> dict[FrozenCliffordString, set[frozenset[int]]]:
        result: defaultdict[FrozenCliffordString, set[frozenset[int]]] = defaultdict(set)
        if length == 0:
            identity = FrozenCliffordString(frozenset({('_'*self.circuit.noisy_circuit.num_qubits, 1)}), 1)
            result[identity].add(frozenset())
        else:
            trivial_syndrome_combos = get_trivial_syndrome_combinations(self.basis.keys(), length)
            for syndrome_counter in trivial_syndrome_combos:
                options = self._syndrome_counter_to_options(syndrome_counter)
                for segment_product in itertools.product(*(option.items() for option in options)):
                    # note: this is probably just as fast as iterating through `segment_product` once
                    # segment_product never empty
                    product_effect: FrozenCliffordString = math.prod(
                        effect for effect, _ in segment_product) # type: ignore
                    candidates = itertools.product(*(
                        segments for _, segments in segment_product))
                    for candidate in candidates:
                        flattened = frozenset(itertools.chain.from_iterable(candidate))
                        result[product_effect].add(flattened)
        if print_progress:
            print(f"    {length}, leading to {len(result)} distinct errors.")
        return dict(result)
    

    def _syndrome_counter_to_options(
            self,
            syndrome_counter: Counter[tuple[bool, ...]],
    ) -> list[dict[FrozenCliffordString, set[frozenset[int]]]]:
        """Get configuration segments for each syndrome based on the counts in `syndrome_counter`.
        
        Input:
        * `syndrome_counter` dictates for each syndrome
        how many faults in `self.basis[syndrome]` should appear in the combination.
        E.g. `{syndrome1: 2, syndrome2: 1}`.

        Output:
        * `options` a list of options, one for each item in `syndrome_counter`
        e.g. `[option1, option2]`.
        Each option is a map from effect to a set of frozen sets
        of fault indices whose combination produces that effect.
        e.g. `{effect1: {frozenset1, frozenset2}, effect2: {frozenset3}}`
        where e.g. `frozenset1 = {1, 2}`.
        """
        options: list[dict[FrozenCliffordString, set[frozenset[int]]]] = []
        for syndrome, count in syndrome_counter.items():
            effects_to_segments = self._get_effect_to_segments(syndrome, count)
            options.append(effects_to_segments)
        return options
    

    def _get_effect_to_segments(
        self,
        syndrome: tuple[bool, ...],
        length: int,
    ):
        """Get a map from effect to the configurations with that resultant effect.
        
        Input:
        * `syndrome` defines the set of faults to take combinations from.
        * `length` the length of fault combinations to consider.

        Output:
        * a map from (nonzero) effect to a set of configurations.
        Each configuration is a frozen set of fault indices
        with that resultant effect.
        """
        result: defaultdict[FrozenCliffordString, set[frozenset[int]]] = defaultdict(set)
        fault_set = self.basis[syndrome].items()
        fault_combos = itertools.combinations(fault_set, length)
        for fault_combo in fault_combos:
            # e.g. fault_combo = (('effect1', 1), ('effect2', 2)) and is never empty
            # the effects in fault_combo are normalized
            product_effect: FrozenCliffordString = math.prod(
                effect for effect, _ in fault_combo) # type: ignore
            if product_effect.norm_squared == 0:
                # this is not a bug: two normalized effects can multiply to zero
                # e.g. (XX + YY)(XX - YY) = 0
                # this just means they are not unitary, which is possible
                # e.g. the first of the two above effects could have come from
                # (XX + XY + YX + YY)/2 and a measurement that collapses the state to (XX + YY)
                continue
            index_combo: frozenset[int] = frozenset(
                index for _, index in fault_combo)
            result[product_effect].add(index_combo)
        return dict(result)
    

    def get_index_to_odds(self, noise_level):
        raise NotImplementedError