from collections.abc import Iterable
from typing import Literal

def parity_probability(parity: Literal[0, 1], probabilities: Iterable[float]):
    """Calculate the probability of an odd or even number of events occurring.
    
    Input:
    * `parity` whether to calculate the probability of an even or odd number of events.
    * `probabilities` a sequence of probabilities of independent events.

    Output:
    * The probability the number of events occurring matches parity.
    """
    # TODO: test against decay factor method
    prob = 1 - parity
    for p in probabilities:
        prob = prob * (1 - p) + (1 - prob) * p
    return prob