from collections import Counter, defaultdict
import itertools
import math
from typing import Literal

from cliffordep.combinators._base import BaseErrorMechanismCombinator
from cliffordep.noisy_circuit_tools_mixture import CultivationCircuitMixture
from cliffordep.combinators.fault_source_combinator import get_trivial_syndrome_combinations
from cliffordep.pauli_string_tools import CliffordString, FrozenCliffordString

MechanismBagMixed = tuple[Counter[float], Counter[float], Counter[float]]


class ErrorMechanismCombinatorMixed(BaseErrorMechanismCombinator):
    """Group error mechanisms by their syndrome then effect,
    where effects are mixtures of Clifford strings.
    
    Extends `BaseErrorMechanismCombinator`.

    Overriden attributes:
    * `basis` same as in `BaseErrorMechanismCombinator` but each effect is a normalized frozen Clifford string.
    """
    
    def __init__(
            self,
            circuit: CultivationCircuitMixture,
            print_progress: bool = False,
            replace_s_with: Literal['T', 'S', 'Z'] = 'S',
    ):
        """Input:
        * `circuit` the `CultivationCircuitMixture` to analyze.
        * `print_progress` whether to print progress.
        * `replace_s_with` the unitary to push through if `unitary` is an S gate.
        Also affects S dagger gates.
        """
        _basis: defaultdict[
            tuple[bool, ...], dict[FrozenCliffordString, int]
        ] = defaultdict(dict)
        _index_to_bag: defaultdict[int, list[Counter[float]]] = defaultdict(lambda: [Counter(), Counter(), Counter()])
        for (_, source_name, _), group in circuit.group_faults_by_source().items():
            source_class = self._classify(source_name)
            for fault in group:
                mixture = circuit.get_syndrome_and_effect(fault, replace_s_with=replace_s_with)
                for syndrome, submixture in mixture.submixtures.items():
                    for normalized_effect, probability in submixture.items():
                        assert normalized_effect.mutable_copy().norm_squared == 1
                        index = _basis[syndrome].setdefault(
                            normalized_effect, len(_index_to_bag))  # TODO: use an integer counter
                        _index_to_bag[index][source_class][probability] += 1
        self.basis = dict(_basis)
        self.index_to_bag: dict[int, MechanismBagMixed] = { # type: ignore
            index: tuple(bag) for index, bag in _index_to_bag.items()}
        if print_progress:
            print(f"Finished enumerating all faults. \
Found {self.mechanism_count} error mechanisms \
distributed among {self.syndrome_count} syndromes.")
        super().__init__(circuit, print_progress=print_progress)

    @property
    def syndrome_count(self) -> int:
        return len(self.basis)
    
    @property
    def mechanism_count(self) -> int:
        return sum(len(mechanisms) for mechanisms in self.basis.values())
    

    def get_undetected_mechanism_combinations(
            self,
            max_order: int,
            print_progress: bool = False,
    ):
        """Find all combinations of mechanisms up to `max_order` that have trivial syndrome.

        Input:
        * `max_order` the maximum order of probability to consider.
        * `print_progress` whether to print progress.

        Output:
        * A list whose kth entry is a map
        from each effect to a set of frozen sets of error mechanism indices.
        """
        return [self._get_undetected_mechanism_combinations_for_length(
            length, print_progress) for length in range(max_order + 1)]
    

    def _get_undetected_mechanism_combinations_for_length(
            self,
            length: int,
            print_progress: bool = False
    ) -> dict[FrozenCliffordString, set[frozenset[int]]]:
        """Find all combinations of `length` mechanisms that have trivial syndrome.
        
        Helper for `get_undetected_mechanism_combinations`.
        
        Input:
        * `length` the length of combinations to find.
        * `print_progress` whether to print progress.

        Output:
        * a map from each effect to a set of frozen sets of error mechanism indices.
        """
        result: defaultdict[FrozenCliffordString, set[frozenset[int]]] = defaultdict(set)
        if print_progress:
            print(f"Finding undetected combinations of length {length}...")
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
            print(f"Done. They lead to {len(result)} distinct effects.")
        return dict(result)
    

    def _syndrome_counter_to_options(
            self,
            syndrome_counter: Counter[tuple[bool, ...]],
    ) -> list[dict[FrozenCliffordString, set[frozenset[int]]]]:
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
        """Get a map from effect to the error mechanism combinations with that resultant effect.
        
        Input:
        * `syndrome` defines the set of error mechanisms to take combinations from.
        * `length` the length of mechanism combinations to consider.

        Output:
        * a map from (nonzero) effect to a set of mechanism combinations.
        Each mechanism combination is a frozen set of error mechanism indices
        with that resultant effect.
        """
        result: defaultdict[FrozenCliffordString, set[frozenset[int]]] = defaultdict(set)
        mechanism_set = self.basis[syndrome].items()
        mechanism_combos = itertools.combinations(mechanism_set, length)
        for mechanism_combo in mechanism_combos:
            # e.g. mechanism_combo = (('effect1', 1), ('effect2', 2)) and is never empty
            # the effects in mechanism_combo are normalized
            product_effect: FrozenCliffordString = math.prod(
                effect for effect, _ in mechanism_combo) # type: ignore
            if product_effect.norm_squared == 0:
                # this is not a bug: two normalized effects can multiply to zero
                # e.g. (XX + YY)(XX - YY) = 0
                # this just means they are not unitary, which is possible
                # e.g. the first of the two above effects could have come from
                # (XX + XY + YX + YY)/2 and a measurement that collapses the state to (XX + YY)
                continue
            index_combo: frozenset[int] = frozenset(
                index for _, index in mechanism_combo)
            result[product_effect].add(index_combo)
        return dict(result)