from collections import Counter
from collections.abc import Iterable
from typing import Literal

def error_rate_per_kept_shot(
        all_string_leads: list[tuple[
            dict[str, tuple[float, Counter[int]]],
            dict[str, tuple[float, Counter[int]]],
        ]],
        noise_level: float,
        print_progress: bool = False,
) -> float:
    p_odds = noise_level / (1 - noise_level)
    identity_odds, error_odds = 0, 0
    for length, (identity_strings, error_strings) in enumerate(all_string_leads):
        single_odds = p_odds**length
        i_odds = single_odds * _get_normalized_counts(identity_strings.values())
        e_odds = single_odds * _get_normalized_counts(error_strings.values())
        identity_odds += i_odds
        error_odds += e_odds
        if print_progress:
            print(f'O(p^{length}) events:')
            print(f'Identity odds = {i_odds}')
            print(f'Error odds = {e_odds}')
    return error_odds / (identity_odds + error_odds)

def _get_normalized_counts(probability_counter_pairs: Iterable[tuple[float, Counter[int]]]) -> float:
    return sum(
        probability_kept * _counter_to_normalized_total(counter)
        for probability_kept, counter in probability_counter_pairs
    )

def _counter_to_normalized_total(counter: Counter[int]) -> float:
    return sum(count / denominator for denominator, count in counter.items())

def parity_probability(parity: Literal[0, 1], probabilities: Iterable[float]):
    """Calculate the probability of an odd or even number of events occurring.
    
    Input:
    * `parity` whether to calculate the probability of an even or odd number of events.
    * `probabilities` a sequence of probabilities of independent events.

    Output:
    * The probability the number of events occurring matches parity.
    """
    prob = 1 - parity
    for p in probabilities:
        prob = prob * (1 - p) + (1 - prob) * p
    return prob