import abc
from collections import Counter
from collections.abc import Iterable, Iterator
import itertools

import numpy as np
import numpy.typing as npt
import stim

from cliffordea.enum.constants import ONE_QUBIT_ERROR_EVENTS


class Combinator(abc.ABC):
    """Find fault configurations with zero detector signature."""

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


class BaseDisjointCombinator(Combinator):
    """Group events in a noisy circuit where some events are disjoint."""

    def get_undetected_configurations(
            self,
            max_fault_count: int,
            print_progress: bool = False,
    ):
        """Find undetected configurations up to `max_fault_count` faults.

        :param max_fault_count: The largest fault count to consider.
        :param print_progress: Whether to print progress.

        :return configurations:
            A list whose kth entry is a map
            from each effect to a counter of denominators.
            Each denominator divides (noise level)^k to equal
            the probability an instance of that undetected combination of k faults occurs.
        """
        return [self._get_undetected_configurations_for_fault_count(
            fault_count, print_progress)
            for fault_count in range(max_fault_count + 1)]


    @classmethod
    def error_rate_per_kept_shot(
            cls,
            kept_effects_by_fault_count: list[
                dict[str, tuple[float, float, Counter[int]]]
            ],
            noise_level: float,
            print_progress: bool = False,
    ) -> float:
        """Calculate the logical error rate per kept shot for a given noise level.

        :param kept_effects_by_fault_count: Postselected logical-analysis
            results grouped by fault count and resultant Pauli effect.
        :param noise_level: The noise level to analyze.
        :param print_progress: Whether to print progress.
        """
        p_odds = noise_level / (1 - noise_level)
        benign_odds, malignant_odds = 0, 0
        for fault_count, effects in enumerate(kept_effects_by_fault_count):
            single_odds = p_odds**fault_count
            benign_weight, malignant_weight = cls._get_normalized_weights(
                effects.values(),
            )
            fault_count_benign_odds = single_odds * benign_weight
            fault_count_malignant_odds = single_odds * malignant_weight
            if print_progress:
                print(f'{fault_count}-fault configurations:')
                print(f'Benign odds = {fault_count_benign_odds}')
                print(f'Malignant odds = {fault_count_malignant_odds}')
            benign_odds += fault_count_benign_odds
            malignant_odds += fault_count_malignant_odds
        return malignant_odds / (benign_odds + malignant_odds)


    @abc.abstractmethod
    def _get_undetected_configurations_for_fault_count(
            self,
            fault_count: int,
            print_progress: bool = False,
    ) -> dict[str, Counter[int]]:
        """Find undetected configurations containing `fault_count` faults.

        Helper for `self.get_undetected_configurations()`.

        :param fault_count: The number of faults in each configuration.
        :param print_progress: Whether to print progress.

        :return configurations: a map from each effect to a counter of denominators.
            Each denominator divides (noise level)^fault_count to equal
            the probability an instance of that undetected combination occurs.
        """


    @classmethod
    def _get_normalized_weights(cls, logical_triples: Iterable[tuple[float, float, Counter[int]]]) -> tuple[float, float]:
        benign_weight, malignant_weight = 0, 0
        for accept_probability, logical_fidelity, counter in logical_triples:
            normalized_total = cls._counter_to_normalized_total(counter)
            benign_weight += accept_probability * logical_fidelity * normalized_total
            malignant_weight += accept_probability * (1-logical_fidelity) * normalized_total
        return benign_weight, malignant_weight

    @staticmethod
    def _counter_to_normalized_total(counter: Counter[int]) -> float:
        return sum(count / denominator for denominator, count in counter.items())


def get_zero_signature_combinations(
        signatures: Iterable[tuple[bool, ...]],
        fault_count: int,
) -> Iterator[Counter[tuple[bool, ...]]]:
    """Find fault-signature combinations whose XOR is zero.

    :param signatures: An iterable of detector signatures.
    :param fault_count: The number of signatures in each combination.
    :return zero_signature_combinations: Counters mapping each signature to
        its multiplicity in a zero-signature combination.
    """
    # TODO: find a basis for the kernel of the parity check matrix then take combinations of basis vectors
    if fault_count == 0:
        yield Counter()
        return
    combos = itertools.combinations_with_replacement(signatures, fault_count)
    for combo in combos:
        resultant_signature: npt.NDArray[np.int_] = np.array(combo).sum(axis=0) % 2
        if not any(resultant_signature):
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
