import abc
from collections import Counter
from collections.abc import Iterable, Iterator
import itertools

import numpy as np
import numpy.typing as npt
import stim

from cliffordea.constants import ONE_QUBIT_ERROR_EVENTS


class Combinator(abc.ABC):
    """Class to find all fault configurations that are undetectable i.e. lead to trivial syndrome."""

    @abc.abstractmethod
    def __init__(
            self,
            noisy_circuit: stim.Circuit,
            print_progress: bool = False,
    ):
        """
        :param noisy_circuit: The stim circuit to analyze.
        :param print_progress: Whether to print progress.
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

        :param max_degree: The maximum power of probability to consider.
        :param print_progress: Whether to print progress.

        :return configurations:
            A list whose kth entry is a map
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

        :param all_string_leads: Postselected logical-analysis results grouped
            by fault degree and final Pauli effect.
        :param noise_level: The noise level to analyze.
        :param print_progress: Whether to print progress.
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

        :param length: The length of combinations to find.
        :param print_progress: Whether to print progress.

        :return configurations: a map from each effect to a counter of denominators.
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


def get_trivial_syndrome_combinations(
        syndromes: Iterable[tuple[bool, ...]],
        length: int,
) -> Iterator[Counter[tuple[bool, ...]]]:
    """Find all combinations of g syndromes that result in a trivial syndrome.

    :param syndromes: An iterable of syndromes.
    :param length: The length g of combinations to find.
    :return trivial_combos: An iterator of counters, each one mapping
        a syndrome to the number of times it appears in the combination.
    """
    # TODO: find a basis for the kernel of the parity check matrix then take combinations of basis vectors
    if length == 0:
        yield Counter()
        return
    combos = itertools.combinations_with_replacement(syndromes, length)
    for combo in combos:
        resultant_syndrome: npt.NDArray[np.int_] = np.array(combo).sum(axis=0) % 2
        if not any(resultant_syndrome):
            yield Counter(combo)


def classify(process_name: str) -> int:
    """Classify an error process by how many error events it can make.

    :param process_name: The name of the Stim gate that gives rise to error events.
        This can be 'DEPOLARIZE1', 'DEPOLARIZE2',
        or a member of `ONE_QUBIT_ERROR_EVENTS`.

    :return process_class:
        An integer indicating the class of error process:

            * 0 if `process_name` is in `ONE_QUBIT_ERROR_EVENTS`.
            * 1 if it makes 3 error events (DEPOLARIZE1).
            * 2 if it makes 15 error events (DEPOLARIZE2).
    """
    if process_name in ONE_QUBIT_ERROR_EVENTS:
        return 0
    elif process_name == 'DEPOLARIZE1':
        return 1
    elif process_name == 'DEPOLARIZE2':
        return 2
    else:
        raise ValueError(f"Unknown error process: {process_name}")


def error_event_count(process_name: str) -> int:
    """Return the number of error events an error process can make.

    :param process_name: The name of the Stim gate that gives rise to error events.
        This can be 'DEPOLARIZE1', 'DEPOLARIZE2',
        or a member of `ONE_QUBIT_ERROR_EVENTS`.

    :return: The number of error events the error process can make.
        This indicates the noise strength divided by the probability of each error event.
    """
    return {0: 1, 1: 3, 2: 15}[classify(process_name)]
