import abc
from collections import Counter
from collections.abc import Iterable
from functools import cache
import math
from typing import Literal

from cliffordep.noisy_circuit_tools import CultivationCircuit
from cliffordep.noisy_circuit_tools_mixture import CultivationCircuitMixture
from cliffordep.type_aliases import FaultBag, FaultBagMixed
from cliffordep.pauli_string_tools import FrozenCliffordString, LogicalVector


class Combinator(abc.ABC):
    """Class to find all fault configurations that are undetectable i.e. lead to trivial syndrome.

    Instance attributes:
    * `circuit` the noisy circuit to analyze.
    """

    @abc.abstractmethod
    def __init__(
            self,
            circuit: 'CultivationCircuit | CultivationCircuitMixture',
            print_progress: bool = False,
    ):
        """Input:
        * `circuit` the `CultivationCircuit` to analyze.
        * `print_progress` whether to print progress.
        """
        self.circuit = circuit

    def __repr__(self):
        return f"{self.__class__.__name__}({self.circuit})"


class BaseExclusiveCombinator(Combinator):
    """Group all events in a noisy circuit where some events are exclusive.
    
    Extends `Combinator`.
    """

    def get_undetected_configurations(
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
        return [self._get_undetected_configurations_for_length(
            length, print_progress) for length in range(max_order + 1)]


    def get_kept_strings(
            self,
            configurations: list[dict[str, Counter[int]]],
            cultivated_state: Literal['T', 'S', 'Z'] = 'T',
            print_progress: bool = False,
    ) -> list[dict[str, tuple[LogicalVector, Counter[int]]]]:
        """Return all the information needed to reconstruct the logical error probability for any noise level.

        Input:
        * `configurations` the output of `self.get_undetected_configurations()`.
        * `cultivated_state` the target logical state cultivated.
        * `print_progress` whether to print progress.

        Output:
        * A list of maps, one for each order. Each one maps from each error
        (as an unsigned Pauli string) to a pair containing:
            - the resulting logical vector,
            - a counter of denominators.
        """
        if print_progress:
            print(f"{cultivated_state} state cultivation:")
        result = [self._get_kept_strings(
                combinations_of_order,
                cultivated_state=cultivated_state,
                order=order if print_progress else None,
            ) for order, combinations_of_order in enumerate(configurations)]
        return result


    @classmethod
    def error_rate_per_kept_shot(
            cls,
            all_string_leads: list[dict[str, tuple[LogicalVector, Counter[int]]]],
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
        for length, strings in enumerate(all_string_leads):
            single_odds = p_odds**length
            i_counts, e_counts = cls._get_normalized_counts(strings.values())
            i_odds = single_odds * i_counts
            e_odds = single_odds * e_counts
            if print_progress:
                print(f'O(p^{length}) events:')
                print(f'Identity odds = {i_odds}')
                print(f'Error odds = {e_odds}')
            identity_odds += i_odds
            error_odds += e_odds
        return error_odds / (identity_odds + error_odds)


    # TODO: define as overload method in `Combinator`
    def _get_kept_strings(
            self,
            combinations_of_order: dict[str, Counter[int]],
            cultivated_state: Literal['T', 'S', 'Z'] = 'T',
            order: None | int = None,
    ) -> dict[str, tuple[LogicalVector, Counter[int]]]:
        """Compute the logical vector for each postselected error.

        Helper for `get_kept_strings`.

        Input:
        * `combinations_of_order` a map from each effect to a counter of denominators.
        Each denominator divides (noise level)^order to equal
        the probability an instance of that undetected combination occurs.
        * `cultivated_state` the target logical state cultivated.
        * `order` an optional parameter used only for printing progress.

        Output:
        * `strings` a map from each error (as an unsigned Pauli string)
        to a pair containing:
            - the resulting logical vector,
            - a counter of denominators.
        """
        strings: dict[str, tuple[LogicalVector, Counter[int]]] = {}
        for pauli_string, denominators in combinations_of_order.items():
            logical_vector = self.circuit.string_to_logical_vector(cultivated_state, pauli_string)
            # TODO: check if `LogicalVector.transfer_xy_to_iz` is unitary. If so, move from above method into below conditional block.
            if logical_vector.probability_mass:
                strings[pauli_string] = (logical_vector, denominators)
        if order is not None:
            identity_weight = 0
            error_weight = 0
            for logical_vector, _ in strings.values():
                identity_weight += logical_vector.probability_of('I')
                error_weight += logical_vector.probability_of('Z')
            print(f'    {order}, {identity_weight} ({error_weight}) errors are kept and lead to identity (error).')
        return strings


    @abc.abstractmethod
    def _get_undetected_configurations_for_length(
            self,
            length: int,
            print_progress: bool = False,
    ) -> dict[str, Counter[int]]:
        """Find all combinations of `length` faults that have trivial syndrome.

        Helper for `self.get_undetected_configurations()`.

        Input:
        * `length` the length of combinations to find.
        * `print_progress` whether to print progress.

        Output:
        * a map from each effect to a counter of denominators.
        Each denominator divides (noise level)^length to equal
        the probability an instance of that undetected combination occurs.
        """


    @classmethod
    def _get_normalized_counts(cls, vector_counter_pairs: Iterable[tuple[LogicalVector, Counter[int]]]) -> tuple[float, float]:
        i_counts, e_counts = 0, 0
        for logical_vector, counter in vector_counter_pairs:
            normalized_total = cls._counter_to_normalized_total(counter)
            i_counts += logical_vector.probability_of('I') * normalized_total
            e_counts += logical_vector.probability_of('Z') * normalized_total
        return i_counts, e_counts

    @staticmethod
    def _counter_to_normalized_total(counter: Counter[int]) -> float:
        return sum(count / denominator for denominator, count in counter.items())
    

class BaseFaultCombinator(Combinator):
    """Group faults by their syndrome then effect, where all faults are independent.
    
    Extends `Combinator`.

    Instance attributes:
    * `circuit` the noisy circuit to analyze.
    * `basis` a map from each syndrome to another map from each effect to a fault index.
    * `index_to_bag` a map from each fault index to its fault bag.
    """

    index_to_bag: dict[int, FaultBag] | dict[int, FaultBagMixed]
    basis: dict[tuple[bool, ...], dict[str, int]] | dict[
            tuple[bool, ...], dict[FrozenCliffordString, int]]

    @property
    def syndrome_count(self) -> int:
        return len(self.basis)
    
    @property
    def fault_count(self) -> int:
        return sum(len(faults) for faults in self.basis.values())
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__} with {self.fault_count} faults distributed among {self.syndrome_count} syndromes."
    
    def print_basis(self):
        """Print the syndrome, effect, index, and fault bag of all faults."""
        lines = []
        for syndrome, effect_dict in self.basis.items():
            lines.append(''.join('1' if detector else '0' for detector in syndrome))
            for effect, index in effect_dict.items():
                bag = self.index_to_bag[index]
                bag_str = bag if type(bag[0]) is int else tuple(dict(b) for b in bag) # type: ignore
                lines.append(f"  {effect}: {index}, {bag_str}")
        print('\n'.join(lines))

    @cache
    def index_to_syndrome_and_effect(self, index: int) -> tuple[tuple[bool, ...], str | FrozenCliffordString]:
        """Get the syndrome and effect corresponding to a fault index.
        
        Input:
        * `index` the fault index.

        Output:
        * `syndrome` the syndrome of the fault.
        * `effect` the effect of the fault.
        """
        for syndrome, effect_dict in self.basis.items():
            for effect, effect_index in effect_dict.items():
                if effect_index == index:
                    return syndrome, effect
        raise ValueError(f"Index {index} not found in basis.")
    

    def get_undetected_configurations(
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
        from each effect to a set of frozen sets of fault indices.
        """
        if print_progress:
            print("Found undetected configurations of length...")
        return [self._get_undetected_configurations_for_length(
            length, print_progress) for length in range(max_order + 1)]


    @abc.abstractmethod
    def _get_undetected_configurations_for_length(
            self,
            length: int,
            print_progress: bool = False
    ) -> dict[str, set[frozenset[int]]] | dict[FrozenCliffordString, set[frozenset[int]]]:
        """Find all combinations of `length` faults that have trivial syndrome.
        
        Helper for `self.get_undetected_configurations()`.
        
        Input:
        * `length` the length of combinations to find.
        * `print_progress` whether to print progress.

        Output:
        * a map from each effect to a set of frozen sets of fault indices.
        """


    def get_kept_strings(
            self,
            configurations: list[dict[str, set[frozenset[int]]]],
            cultivated_state: Literal['T', 'S', 'Z'] = 'T',
            print_progress: bool = False,
    ) -> list[dict[str, tuple[LogicalVector, set[frozenset[int]]]]]:
        """Construct the information needed to reconstruct the logical error probability for any noise level.

        Input:
        * `configurations` the output of `self.get_undetected_configurations()`.
        * `cultivated_state` the target logical state cultivated.
        * `print_progress` whether to print progress.

        Output:
        * A list of maps, one for each order. Each one maps from each error
        (as an unsigned Pauli string) to a pair containing:
            - the resulting logical vector,
            - a set fault configurations (each represented by a frozen set of fault indices).
        """
        if print_progress:
            print(f"{cultivated_state} state cultivation: for order...")
        result = [self._get_kept_strings(
                combinations_of_order,
                cultivated_state=cultivated_state,
                order=order if print_progress else None,
            ) for order, combinations_of_order in enumerate(configurations)]
        return result
    

    # TODO: define as overload method in `Combinator`
    def _get_kept_strings(
            self,
            combinations_of_order: dict[str, set[frozenset[int]]],
            cultivated_state: Literal['T', 'S', 'Z'] = 'T',
            order: None | int = None,
    ) -> dict[str, tuple[LogicalVector, set[frozenset[int]]]]:
        """Compute the logical vector for each postselected error.

        Helper for `get_kept_strings`.

        Input:
        * `combinations_of_order` a map from each effect to
        a set of frozen sets of fault indices.
        * `cultivated_state` the target logical state cultivated.
        * `order` an optional parameter used only for printing progress.

        Output:
        * `strings` a map from each error (as an unsigned Pauli string)
        to a pair containing:
            - the resulting logical vector,
            - a set fault configurations (each represented by a frozen set of fault indices).
        """
        strings: dict[str, tuple[LogicalVector, set[frozenset[int]]]] = {}
        for pauli_string, combo_set in combinations_of_order.items():
            logical_vector = self.circuit.string_to_logical_vector(cultivated_state, pauli_string)
            # TODO: check if `LogicalVector.transfer_xy_to_iz` is unitary. If so, move from above method into below conditional block.
            if logical_vector.probability_mass:
                strings[pauli_string] = (logical_vector, combo_set)
        if order is not None:
            identity_weight = 0
            error_weight = 0
            for logical_vector, _ in strings.values():
                identity_weight += logical_vector.probability_of('I')
                error_weight += logical_vector.probability_of('Z')
            print(f'    {order}, {identity_weight} ({error_weight}) errors are kept and lead to identity (error).')
        return strings
    

    def error_rate_per_kept_shot(
            self,
            all_string_leads: list[dict[str, tuple[LogicalVector, set[frozenset[int]]]]],
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
        for length, strings in enumerate(all_string_leads):
            i_odds, e_odds = self._sum_odds(strings.values(), index_to_odds)
            if print_progress:
                print(f'O(p^{length}) events:')
                print(f'Identity odds = {i_odds}')
                print(f'Error odds = {e_odds}')
            identity_odds += i_odds
            error_odds += e_odds
        return error_odds / (identity_odds + error_odds)
    

    @abc.abstractmethod
    def get_index_to_odds(self, noise_level: float) -> dict[int, float]:
        """Get a map from fault index to the odds of it flipping."""


    @staticmethod
    def _classify(process_name: str) -> int:
        """Classify an error process by how many error events it can make.

        Input:
        * `process_name` the name of the Stim gate that gives rise to error events.
        This can be 'DEPOLARIZE1', 'DEPOLARIZE2', 'X_ERROR', 'Y_ERROR', 'Z_ERROR', 'MX', 'MY', 'MZ'.

        Output:
        * An integer indicating the type of error process:
            * 0 if it makes only 1 error event (X_ERROR, Y_ERROR, Z_ERROR, MX, MY, MZ).
            * 1 if it makes 3 error events (DEPOLARIZE1).
            * 2 if it makes 15 error events (DEPOLARIZE2).
        """
        if process_name in {'X_ERROR', 'Y_ERROR', 'Z_ERROR', 'MX', 'MY', 'MZ'}:
            return 0
        elif process_name == 'DEPOLARIZE1':
            return 1
        elif process_name == 'DEPOLARIZE2':
            return 2
        else:
            raise ValueError(f"Unknown error process: {process_name}")


    @staticmethod
    def _decomposed_probability(
            class_: int,
            noise_level: float,
    ) -> float:
        """Return the probability of the independent events of an error location of class `class_`.

        Input:
        * `class_` the class of the error location.
        It can be 0, 1, or 2.
        This depends on the number of independent events
        that sequentially compose to equal the error location.
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
            vector_combo_pairs: Iterable[tuple[LogicalVector, set[frozenset[int]]]],
            index_to_odds: dict[int, float],
        ) -> tuple[float, float]:
        """Sum the odds of all configurations in `vector_combo_pairs`.
        
        Input:
        * `vector_combo_pairs` an iterable of pairs, each containing:
            - a logical vector,
            - a set of frozen sets of fault indices that defines the combination.
        * `index_to_odds` a map from each fault index to the odds of it flipping.
        """
        i_odds, e_odds = 0, 0
        for logical_vector, set_of_configurations in vector_combo_pairs:
            prob = sum(math.prod(index_to_odds[index] for index in combo) for combo in set_of_configurations)
            i_odds += logical_vector.probability_of('I') * prob
            e_odds += logical_vector.probability_of('Z') * prob
        return i_odds, e_odds