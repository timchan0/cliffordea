from collections import Counter, defaultdict
import itertools
import math

import stim

from cliffordep.combinators._base import BaseErrorMechanismCombinator
from cliffordep.noisy_circuit_tools import CultivationCircuit
from cliffordep.combinators.fault_source_combinator import get_trivial_syndrome_combinations
from cliffordep.pauli_string_tools import forget_sign
from cliffordep.type_aliases import MechanismBag


class ErrorMechanismCombinator(BaseErrorMechanismCombinator):
    """Group error mechanisms by their syndrome then effect,
    where effects are pure Clifford strings.
    
    Extends `BaseErrorMechanismCombinator`.
    """
    
    def __init__(
            self,
            circuit: CultivationCircuit,
            print_progress: bool = False,
    ):
        _basis: defaultdict[
            tuple[bool, ...], dict[str, int]
        ] = defaultdict(dict)
        _index_to_bag: defaultdict[int, list[int]] = defaultdict(lambda: [0, 0, 0])
        for (_, source_name, _), group in circuit.group_faults_by_source().items():
            source_class = self._classify(source_name)
            for fault in group:
                syndrome, effect = circuit.get_syndrome_and_effect(fault)
                index = _basis[tuple(syndrome)].setdefault(effect, len(_index_to_bag))
                _index_to_bag[index][source_class] += 1
        self.basis: dict[tuple[bool, ...], dict[str, int]] = dict(_basis) # type: ignore
        self.index_to_bag: dict[int, MechanismBag] = { # type: ignore
            index: tuple(bag) for index, bag in _index_to_bag.items()}
        if print_progress:
            print(f"Finished enumerating all faults. {str(self)}")
        super().__init__(circuit, print_progress=print_progress)


    def get_undetected_mechanism_combinations( # type: ignore
            self,
            max_order: int,
            print_progress: bool = False,
    ) -> list[dict[str, set[frozenset[int]]]]:
        return super().get_undetected_mechanism_combinations(max_order, print_progress=print_progress) # type: ignore


    def _get_undetected_mechanism_combinations_for_length(
            self,
            length: int,
            print_progress: bool = False
    ) -> dict[str, set[frozenset[int]]]:
        result: defaultdict[str, set[frozenset[int]]] = defaultdict(set)
        if print_progress:
            print(f"Finding undetected combinations of length {length}...")
        if length == 0:
            result['_'*self.circuit.noisy_circuit.num_qubits].add(frozenset())
        else:
            trivial_syndrome_combos = get_trivial_syndrome_combinations(self.basis.keys(), length)
            for syndrome_counter in trivial_syndrome_combos:
                options = self._syndrome_counter_to_options(syndrome_counter)
                for segment_product in itertools.product(*(option.items() for option in options)):
                    # note: this is probably just as fast as iterating through `segment_product` once
                    # segment_product never empty
                    product_effect = forget_sign(math.prod(
                        stim.PauliString(effect) for effect, _ in segment_product)) # type: ignore
                    candidates = itertools.product(*(
                        segments for _, segments in segment_product))
                    for candidate in candidates:
                        flattened = frozenset(itertools.chain.from_iterable(candidate))
                        result[product_effect].add(flattened)
        if print_progress:
            print(f"Done. They lead to {len(result)} distinct effects.")
        return dict(result)
    

    def _syndrome_counter_to_options(
            self,
            syndrome_counter: Counter[tuple[bool, ...]],
    ) -> list[dict[str, set[frozenset[int]]]]:
        """Get mechanism combination segments for each syndrome based on the counts in `syndrome_counter`.
        
        Input:
        * `syndrome_counter` dictates for each syndrome
        how many mechanisms in `self.basis[syndrome]` should appear in the combination.
        E.g. `{syndrome1: 2, syndrome2: 1}`.

        Output:
        * `options` a list of options, one for each item in `syndrome_counter`
        e.g. `[option1, option2]`.
        Each option is a map from effect to a set of frozen sets
        of error mechanism indices whose combination produces that effect.
        e.g. `{effect1: {frozenset1, frozenset2}, effect2: {frozenset3}}`
        where e.g. `frozenset1 = {1, 2}`.
        """
        options: list[dict[str, set[frozenset[int]]]] = []
        for syndrome, count in syndrome_counter.items():
            effects_to_segments = self._get_effect_to_segments(syndrome, count)
            options.append(effects_to_segments)
        return options
    

    def _get_effect_to_segments(
        self,
        syndrome: tuple[bool, ...],
        length: int,
    ):
        """Get a map from effect to the error mechanism combinations with that resultant effect.
        
        Input:
        * `syndrome` defines the set of error mechanisms to take combinations from.
        * `length` the length of mechanism combinations to consider.

        Output:
        * a map from effect to a set of mechanism combinations.
        Each mechanism combination is a frozen set of error mechanism indices
        with that resultant effect.
        """
        result: defaultdict[str, set[frozenset[int]]] = defaultdict(set)
        mechanism_set = self.basis[syndrome].items()
        mechanism_combos = itertools.combinations(mechanism_set, length)
        for mechanism_combo in mechanism_combos:
            # e.g. mechanism_combo = (('effect1', 1), ('effect2', 2)) and is never empty
            product_effect = forget_sign(math.prod(
                stim.PauliString(effect) for effect, _ in mechanism_combo)) # type: ignore
            index_combo: frozenset[int] = frozenset(
                index for _, index in mechanism_combo)
            result[product_effect].add(index_combo)
        return dict(result)
    

    def get_index_to_odds(self, noise_level: float):
        index_to_odds: dict[int, float] = {}
        for index, bag in self.index_to_bag.items():
            decay_factor = math.prod(
                (1-2*self._decomposed_probability(process_class, noise_level))**count
                for process_class, count in enumerate(bag)
            )
            prob = (1 - decay_factor) / 2
            index_to_odds[index] = prob / (1 - prob)
        return index_to_odds