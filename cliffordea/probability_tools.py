from collections.abc import Iterable
from typing import Literal

def parity_probability(parity: Literal[0, 1], probabilities: Iterable[float]):
    """Calculate the probability of an odd or even number of events occurring.
    
    :param parity: Whether to calculate the probability of an even or odd number of events.
    :param probabilities: A sequence of probabilities of independent events.

    :return: The probability the number of events occurring matches parity.
    """
    # TODO: test against decay factor method
    prob = 1 - parity
    for p in probabilities:
        prob = prob * (1 - p) + (1 - prob) * p
    return prob
