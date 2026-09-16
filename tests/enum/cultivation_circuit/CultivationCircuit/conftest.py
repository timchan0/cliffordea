from collections import Counter, defaultdict
from collections.abc import Iterable
import math

import pytest

from cliffordea.enum.combinators._base import error_event_count
from cliffordea.enum.types import ErrorLocation


@pytest.fixture
def update_undetected_configurations():
    """Update `undetected_configurations` with a candidate undetected combination of faults.

    Input:
    * `undetected_configurations` maps each effect to a counter of denominators.
    * `effect` the resultant (unsigned) Pauli string of the candidate combination.
    * `candidate` a sequence containing the candidate segment's faults
    of pairs each containing:
        - an error location
        - the number of disjoint error events realizing that fault.

    Side effect:
    * Update `undetected_configurations` with the undetected combination of faults if it is valid
    i.e. comprises distinct error locations.
    """
    def f(
        undetected_configurations: defaultdict[str, Counter[int]],
        effect: str,
        candidate: Iterable[tuple[ErrorLocation, int]],
):
        error_locations: set[ErrorLocation] = set()
        denominator_factors: list[int] = []
        realization_count = 1
        for error_location, count in candidate:
            if error_location in error_locations:
                return
            _, name, _ = error_location
            denominator_factors.append(error_event_count(name))
            realization_count *= count
            error_locations.add(error_location)
        probability_denominator = math.prod(denominator_factors)
        undetected_configurations[effect][probability_denominator] += (
            realization_count
        )
    return f
