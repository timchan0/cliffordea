import abc
from collections import Counter
from collections.abc import Iterable
import math
from typing import Literal

from cliffordep.noisy_circuit_tools import CultivationCircuit
from cliffordep.noisy_circuit_tools_mixture import CultivationCircuitMixture
from cliffordep.type_aliases import MechanismBag


class Combinator(abc.ABC):
    """Class to find all combinations of faults that lead to trivial syndrome.

    Instance attributes:
    * `circuit` the noisy circuit to analyze.
    """

    @abc.abstractmethod
    def __init__(
            self,
            circuit: 'CultivationCircuit | CultivationCircuitMixture',
            restrict_to_data: bool = True,
            print_progress: bool = False,
    ):
        """Input:
        * `circuit` the `CultivationCircuit` to analyze.
        * `restrict_to_data` whether to restrict all effects to only the data qubits.
        * `print_progress` whether to print progress.
        """
        self.circuit = circuit


class BaseFaultSourceCombinator(Combinator):
    """Group all faults in a noisy circuit.
    
    Extends `Combinator`.
    """

    def get_undetected_fault_combinations(
            self,
            max_order: int,
            print_progress: bool = False,
    ):
        """Find all combinations of faults up to `max_order` that have trivial syndrome.

        Input:
        * `max_order` the maximum order of probability to consider.
        * `print_progress` whether to print progress.

        Output:
        * A list whose kth entry is a map
        from each effect to a counter of denominators.
        Each denominator divides (noise level)^k to equal
        the probability an instance of that undetected combination of k faults occurs.
        """
        return [self._get_undetected_fault_combinations_for_length(
            length, print_progress) for length in range(max_order + 1)]


    def get_kept_strings(
            self,
            fault_combinations: list[dict[str, Counter[int]]],
            cultivated_state: Literal['T', 'S', 'Z'] = 'T',
            print_progress: bool = False,
    ) -> list[tuple[
        dict[str, tuple[float, Counter[int]]],
        dict[str, tuple[float, Counter[int]]],
    ]]:
        """Return all the information needed to reconstruct the logical error probability for any noise level.

        Input:
        * `fault_combinations` the output of `get_undetected_fault_combinations`.
        * `cultivated_state` the target logical state cultivated.
        * `print_progress` whether to print progress.

        Output:
        * A list of pairs, one for each order. Each pair contains:
            - `identity_strings` a map from each effect that leads to logical identity to a pair containing:
                - the probability they are not rejected,
                - a counter of denominators.
            - `error_strings` ditto for effects that lead to a logical error.
        """
        if print_progress:
            print(f"{cultivated_state} state cultivation:")
        result = [self._get_kept_strings(
                combinations_of_order,
                cultivated_state=cultivated_state,
                order=order if print_progress else None,
            ) for order, combinations_of_order in enumerate(fault_combinations)]
        return result


    @classmethod
    def error_rate_per_kept_shot(
            cls,
            all_string_leads: list[tuple[
                dict[str, tuple[float, Counter[int]]],
                dict[str, tuple[float, Counter[int]]],
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
        p_odds = noise_level / (1 - noise_level)
        identity_odds, error_odds = 0, 0
        for length, (identity_strings, error_strings) in enumerate(all_string_leads):
            single_odds = p_odds**length
            i_odds = single_odds * cls._get_normalized_counts(identity_strings.values())
            e_odds = single_odds * cls._get_normalized_counts(error_strings.values())
            identity_odds += i_odds
            error_odds += e_odds
            if print_progress:
                print(f'O(p^{length}) events:')
                print(f'Identity odds = {i_odds}')
                print(f'Error odds = {e_odds}')
        return error_odds / (identity_odds + error_odds)


    # TODO: define as overload method in `Combinator`
    def _get_kept_strings(
            self,
            combinations_of_order: dict[str, Counter[int]],
            cultivated_state: Literal['T', 'S', 'Z'] = 'T',
            order: None | int = None,
    ) -> tuple[
        dict[str, tuple[float, Counter[int]]],
        dict[str, tuple[float, Counter[int]]],
    ]:
        """Group postselected effects by whether they lead to identity or error.

        Helper for `get_kept_strings`.

        Input:
        * `combinations_of_order` a map from each effect to a counter of denominators.
        Each denominator divides (noise level)^order to equal
        the probability an instance of that undetected combination occurs.
        * `cultivated_state` the target logical state cultivated.
        * `order` an optional parameter used only for printing progress.

        Output:
        * `identity_strings` a map from each effect that leads to logical identity,
        to the probability it is not rejected and a counter of denominators.
        * `error_strings` ditto for effects that lead to a logical error.
        """
        identity_strings: dict[str, tuple[float, Counter[int]]] = {}
        error_strings: dict[str, tuple[float, Counter[int]]] = {}
        for data_string, denominators in combinations_of_order.items():
            logical_vector = self.circuit.string_to_logical_vector(cultivated_state, data_string)
            # TODO: check if `LogicalVector.transfer_xy_to_iz` is unitary. If so, move from above method into below conditional block.
            if logical_vector.probability_mass:
                # TODO: use `logical_error_probability` instead of `is_logical_error`
                if logical_vector.is_logical_error:
                    error_strings[data_string] = (logical_vector.probability_mass, denominators)
                else:
                    identity_strings[data_string] = (logical_vector.probability_mass, denominators)
        if order is not None:
            print(f'For order {order}, {len(identity_strings)} ({len(error_strings)}) effects are stabilized and lead to identity (error).')
        return identity_strings, error_strings


    @abc.abstractmethod
    def _get_undetected_fault_combinations_for_length(
            self,
            length: int,
            print_progress: bool = False,
    ) -> dict[str, Counter[int]]:
        """Find all combinations of `length` faults that have trivial syndrome.

        Helper for `self.get_undetected_fault_combinations()`.

        Input:
        * `length` the length of combinations to find.
        * `print_progress` whether to print progress.

        Output:
        * a map from each effect to a counter of denominators.
        Each denominator divides (noise level)^length to equal
        the probability an instance of that undetected combination occurs.
        """


    @classmethod
    def _get_normalized_counts(cls, probability_counter_pairs: Iterable[tuple[float, Counter[int]]]) -> float:
        return sum(
            probability_kept * cls._counter_to_normalized_total(counter)
            for probability_kept, counter in probability_counter_pairs
        )

    @staticmethod
    def _counter_to_normalized_total(counter: Counter[int]) -> float:
        return sum(count / denominator for denominator, count in counter.items())
    

class BaseErrorMechanismCombinator(Combinator):
    """Group error mechanisms by their syndrome then effect.
    
    Extends `Combinator`.

    Instance attributes:
    * `circuit` the noisy circuit to analyze.
    * `basis` a map from each syndrome to another map from each effect
    to an error mechanism index.
    * `index_to_bag` a map from each error mechanism index to its mechanism bag.
    """

    index_to_bag: dict[int, MechanismBag]


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
    ) -> float:
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
    

    @staticmethod
    def _sum_odds(
            probability_combo_pairs: Iterable[tuple[float, set[frozenset[int]]]],
            index_to_odds: dict[int, float],
        ) -> float:
        """Sum the odds of all error mechanism combinations in `probability_combo_pairs`.
        
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