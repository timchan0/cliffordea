from collections import Counter, defaultdict
from collections.abc import Iterable
import itertools
import math
from typing import Literal

import stim

from cliffordep.combinators._base import Combinator
from cliffordep.noisy_circuit_tools import CultivationCircuit
from cliffordep.combinators.fault_source_combinator import get_trivial_syndrome_combinations
from cliffordep.pauli_string_tools import forget_sign
from cliffordep.type_aliases import MechanismBag


class ErrorMechanismCombinator(Combinator):
    """Group error mechanisms by their syndrome then effect.
    
    Extends `Combinator`.

    Instance attributes:
    * `circuit` the noisy circuit to analyze.
    * `basis` a map from each syndrome to another map from each effect
    to an error mechanism index.
    * `index_to_bag` a map from each error mechanism index to its mechanism bag.
    """
    
    def __init__(
            self,
            circuit: CultivationCircuit,
            restrict_to_data: bool = True,
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
                if restrict_to_data:
                    effect = circuit.restrict_to_data(effect)
                index = _basis[tuple(syndrome)].setdefault(effect, len(_index_to_bag))  # TODO: use a counter
                _index_to_bag[index][source_class] += 1
        self.basis = dict(_basis)
        self.index_to_bag: dict[int, MechanismBag] = { # type: ignore
            index: tuple(bag) for index, bag in _index_to_bag.items()}
        if print_progress:
            print(f"Finished enumerating all faults. \
Found {self.mechanism_count} error mechanisms \
distributed among {self.syndrome_count} syndromes.")
        super().__init__(circuit, restrict_to_data, print_progress)

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


    def get_kept_strings(
            self,
            mechanism_combinations: list[dict[str, set[frozenset[int]]]],
            cultivated_state: Literal['T', 'S', 'Z'] = 'T',
            print_progress: bool = False,
    ) -> list[tuple[
        dict[str, tuple[float, set[frozenset[int]]]],
        dict[str, tuple[float, set[frozenset[int]]]],
    ]]:
        """Construct the information needed to reconstruct the logical error probability for any noise level.

        Input:
        * `mechanism_combinations` the output of `get_undetected_mechanism_combinations`.
        * `cultivated_state` the target logical state cultivated.
        * `print_progress` whether to print progress.

        Output:
        * A list of pairs, one for each order. Each pair contains:
            - `identity_strings` a map from each effect that leads to logical identity to a pair containing:
                - the probability they are not rejected,
                - a set of frozen sets of error mechanism indices.
            - `error_strings` ditto for effects that lead to a logical error.
        """
        if print_progress:
            print(f"{cultivated_state} state cultivation:")
        result = [self._get_kept_strings(
                combinations_of_order,
                cultivated_state=cultivated_state,
                order=order if print_progress else None,
            ) for order, combinations_of_order in enumerate(mechanism_combinations)]
        return result
    

    def error_rate_per_kept_shot(
            self,
            all_string_leads: list[tuple[
                dict[str, tuple[float, set[frozenset[int]]]],
                dict[str, tuple[float, set[frozenset[int]]]],
            ]],
            noise_level: float,
            print_progress: bool = False,
    ) -> float:
        """Calculate the logical error rate per kept shot for a given noise level.

        Input:
        * `all_string_leads` the output of `get_kept_strings`.
        * `noise_level` the noise level to analyze.
        * `print_progress` whether to print progress.
        """
        index_to_odds = self.get_index_to_odds(noise_level)
        identity_odds, error_odds = 0, 0
        for length, (identity_strings, error_strings) in enumerate(all_string_leads):
            i_odds = self._sum_odds(identity_strings.values(), index_to_odds)
            e_odds = self._sum_odds(error_strings.values(), index_to_odds)
            identity_odds += i_odds
            error_odds += e_odds
            if print_progress:
                print(f'O(p^{length}) events:')
                print(f'Identity odds = {i_odds}')
                print(f'Error odds = {e_odds}')
        return error_odds / (identity_odds + error_odds)
    

    def _get_undetected_mechanism_combinations_for_length(
            self,
            length: int,
            print_progress: bool = False
    ) -> dict[str, set[frozenset[int]]]:
        """Find all combinations of `length` mechanisms that have trivial syndrome.
        
        Helper for `get_undetected_mechanism_combinations`.
        
        Input:
        * `length` the length of combinations to find.
        * `print_progress` whether to print progress.

        Output:
        * a map from each effect to a set of frozen sets of error mechanism indices.
        """
        result: defaultdict[str, set[frozenset[int]]] = defaultdict(set)
        if print_progress:
            print(f"Finding undetected combinations of length {length}...")
        if length == 0:
            first_effect_map, *_ = self.basis.values()
            first_effect, *_ = first_effect_map.keys()
            result['_'*len(first_effect)].add(frozenset())
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
    

    def _syndrome_counter_to_options(self, syndrome_counter: Counter[tuple[bool, ...]]) -> list[dict[str, set[frozenset[int]]]]:
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
    

    # TODO: define as overload method in `Combinator`
    def _get_kept_strings(
            self,
            combinations_of_order: dict[str, set[frozenset[int]]],
            cultivated_state: Literal['T', 'S', 'Z'] = 'T',
            order: None | int = None,
    ) -> tuple[
        dict[str, tuple[float, set[frozenset[int]]]],
        dict[str, tuple[float, set[frozenset[int]]]],
    ]:
        """Group postselected effects by whether they lead to identity or error.

        Helper for `get_kept_strings`.

        Input:
        * `combinations_of_order` a map from each effect to
        a set of frozen sets of error mechanism indices.
        * `cultivated_state` the target logical state cultivated.
        * `order` an optional parameter used only for printing progress.

        Output:
        * `identity_strings` a map from each effect that leads to logical identity,
        to the probability it is not rejected and a set of frozen set of mechanism indices.
        * `error_strings` ditto for effects that lead to a logical error.
        """
        identity_strings: dict[str, tuple[float, set[frozenset[int]]]] = {}
        error_strings: dict[str, tuple[float, set[frozenset[int]]]] = {}
        for data_string, combo_set in combinations_of_order.items():
            logical_vector = self.circuit.string_to_logical_vector(cultivated_state, data_string)
            # TODO: check if `LogicalVector.transfer_xy_to_iz` is unitary. If so, move from above method into below conditional block.
            if logical_vector.probability_mass:
                # TODO: use `logical_error_probability` instead of `is_logical_error`
                if logical_vector.is_logical_error:
                    error_strings[data_string] = (logical_vector.probability_mass, combo_set)
                else:
                    identity_strings[data_string] = (logical_vector.probability_mass, combo_set)
        if order is not None:
            print(f'For order {order}, {len(identity_strings)} ({len(error_strings)}) effects are stabilized and lead to identity (error).')
        return identity_strings, error_strings
    

    def get_index_to_odds(self, noise_level: float):
        """Get a map from error mechanism index to the odds of it flipping."""
        index_to_odds: dict[int, float] = {}
        for index, bag in self.index_to_bag.items():
            decay_factor = math.prod(
                (1-2*self._decomposed_probability(process_class, noise_level))**count
                for process_class, count in enumerate(bag)
            )
            prob = (1 - decay_factor) / 2
            index_to_odds[index] = prob / (1 - prob)
        return index_to_odds
    

    @staticmethod
    def _sum_odds(
            probability_combo_pairs: Iterable[tuple[float, set[frozenset[int]]]],
            index_to_odds: dict[int, float],
        ) -> float:
        """Sum the odds of all error mechanim combinations in `probability_combo_pairs`.
        
        Input:
        * `probability_combo_pairs` an iterable of pairs, each containing:
            - the probability the error mechanism combination is not rejected,
            - a set of frozen sets of error mechanism indices that defines the combination.
        * `index_to_odds` a map from each error mechanism index to the odds of it flipping.
        """
        return sum(probability_kept * sum(math.prod(
            index_to_odds[index]
            for index in combo)
            for combo in combos)
            for probability_kept, combos in probability_combo_pairs)


    @staticmethod
    def _classify(source_name: str) -> int:
        """Classify a fault source by how many faults it can make.

        Input:
        * `source_name` the name of the Stim gate that gives rise to faults.
        This can be 'DEPOLARIZE1', 'DEPOLARIZE2', 'X_ERROR', 'Y_ERROR', 'Z_ERROR', 'MX', 'MY', 'MZ'.

        Output:
        * An integer indicating the type of fault source:
            * 0 if it makes only 1 fault (X_ERROR, Y_ERROR, Z_ERROR, MX, MY, MZ).
            * 1 if it makes 3 faults (DEPOLARIZE1).
            * 2 if it makes 15 faults (DEPOLARIZE2).
        """
        if source_name in {'X_ERROR', 'Y_ERROR', 'Z_ERROR', 'MX', 'MY', 'MZ'}:
            return 0
        elif source_name == 'DEPOLARIZE1':
            return 1
        elif source_name == 'DEPOLARIZE2':
            return 2
        else:
            raise ValueError(f"Unknown fault source: {source_name}")


    @staticmethod
    def _decomposed_probability(
            class_: int,
            noise_level: float,
    ):
        """Return the probability of the independent processes of a fault source of class `class_`.

        Input:
        * `class_` the class of the fault source.
        It can be 0, 1, or 2.
        This depends on the number of independent processes
        that sequentially compose to equal the fault source.
        * `noise_level` a float in [0, 1].
        """
        p = noise_level
        if class_ == 0:
            return p
        elif class_ == 1:
            return 1/2 - math.sqrt(9 - 12*p)/6
        elif class_ == 2:
            return -15**(7/8)*(15 - 16*p)**(1/8)/30 + 1/2
        else:
            raise ValueError(f"Invalid `class_`: {class_}. Only 0, 1, and 2 are supported.")