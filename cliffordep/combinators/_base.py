import abc
from collections import Counter
from collections.abc import Iterable
from functools import cache
import itertools
import math

import numpy as np
import numpy.typing as npt
import stim

from cliffordep.type_aliases import FaultBag, FaultBagMixed, LogicalTriple
from cliffordep.pauli_string_tools import FrozenCliffordString


class Combinator(abc.ABC):
    """Class to find all fault configurations that are undetectable i.e. lead to trivial syndrome.

    Instance attributes:
    * `circuit` the noisy circuit to analyze.
    """

    @abc.abstractmethod
    def __init__(
            self,
            noisy_circuit: stim.Circuit,
            print_progress: bool = False,
    ):
        """Input:
        * `noisy_circuit` the stim circuit to analyze.
        * `print_progress` whether to print progress.
        """


class BaseExclusiveCombinator(Combinator):
    """Group all events in a noisy circuit where some events are exclusive.
    
    Extends `Combinator`.
    """

    def get_undetected_configurations(
            self,
            max_degree: int,
            print_progress: bool = False,
    ):
        """Find all combinations of faults up to `max_degree` that have trivial syndrome.

        Input:
        * `max_degree` the maximum power of probability to consider.
        * `print_progress` whether to print progress.

        Output:
        * A list whose kth entry is a map
        from each effect to a counter of denominators.
        Each denominator divides (noise level)^k to equal
        the probability an instance of that undetected combination of k faults occurs.
        """
        return [self._get_undetected_configurations_for_length(
            length, print_progress) for length in range(max_degree + 1)]


    @classmethod
    def error_rate_per_kept_shot(
            cls,
            all_string_leads: list[dict[str, tuple[float, float, Counter[int]]]],
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
        for degree, strings in enumerate(all_string_leads):
            single_odds = p_odds**degree
            i_counts, e_counts = cls._get_normalized_counts(strings.values())
            i_odds = single_odds * i_counts
            e_odds = single_odds * e_counts
            if print_progress:
                print(f'O(p^{degree}) events:')
                print(f'Identity odds = {i_odds}')
                print(f'Error odds = {e_odds}')
            identity_odds += i_odds
            error_odds += e_odds
        return error_odds / (identity_odds + error_odds)


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
    def _get_normalized_counts(cls, logical_triples: Iterable[tuple[float, float, Counter[int]]]) -> tuple[float, float]:
        i_counts, e_counts = 0, 0
        for accept_probability, logical_fidelity, counter in logical_triples:
            normalized_total = cls._counter_to_normalized_total(counter)
            i_counts += accept_probability * logical_fidelity * normalized_total
            e_counts += accept_probability * (1-logical_fidelity) * normalized_total
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
    

    def error_rate_per_kept_shot(
            self,
            all_string_leads: list[dict[str, LogicalTriple]],
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
        for degree, strings in enumerate(all_string_leads):
            i_odds, e_odds = _sum_odds(strings.values(), index_to_odds)
            if print_progress:
                print(f'O(p^{degree}) events:')
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
    

def _sum_odds(
        logical_triples: Iterable[LogicalTriple],
        index_to_odds: dict[int, float],
    ) -> tuple[float, float]:
    """Sum the odds of all configurations in `vector_combo_pairs`.
    
    Input:
    * `logical_triples` an iterable of triples, each containing:
        - an acceptance probability,
        - a logical fidelity,
        - a set of frozen sets of fault indices that defines the combination.
    * `index_to_odds` a map from each fault index to the odds of it flipping.
    """
    i_odds, e_odds = 0, 0
    for accept_probability, logical_fidelity, set_of_configurations in logical_triples:
        prob = sum(math.prod(index_to_odds[index] for index in combo) for combo in set_of_configurations)
        i_odds += accept_probability * logical_fidelity * prob
        e_odds += accept_probability * (1-logical_fidelity) * prob
    return i_odds, e_odds


def get_trivial_syndrome_combinations(
        syndromes: Iterable[tuple[bool, ...]],
        length: int,
) -> list[Counter[tuple[bool, ...]]]:
    """Find all combinations of g syndromes that result in a trivial syndrome.

    :param syndromes: An iterable of syndromes.
    :param length: The length g of combinations to find.
    :return trivial_combos: A list of counters, each one mapping a syndrome to
    the number of times it appears in the combination that sums to the trivial syndrome.
    """
    # TODO: find a basis for the kernel of the parity check matrix then take combinations of basis vectors
    if length == 0:
        return [Counter()]
    trivial_combos: list[Counter[tuple[bool, ...]]] = []
    combos = itertools.combinations_with_replacement(syndromes, length)
    for combo in combos:
        resultant_syndrome: npt.NDArray[np.int_] = np.array(combo).sum(axis=0) % 2
        if not any(resultant_syndrome):
            trivial_combos.append(Counter(combo))
    return trivial_combos