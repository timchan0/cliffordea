from collections import Counter, defaultdict
from collections.abc import Iterable
import math

import pytest

from cliffordep.combinators._base import error_event_count
from cliffordep.type_aliases import ErrorLocation


@pytest.fixture
def update_undetected_configurations():
    """Update `undetected_configurations` with a candidate undetected combination of faults.

    Input:
    * `undetected_configurations` maps each effect to a counter of denominators.
    * `effect` the resultant (unsigned) Pauli string of the candidate combination.
    * `candidate` a sequence (whose length equals that of the candidate combination)
    of pairs each containing:
        - an error location
        - count the number of its faults that can be used as part of the undetected combination.

    Side effect:
    * Update `undetected_configurations` with the undetected combination of faults if it is valid
    i.e. comprises distinct error locations.
    """
    def f(
        undetected_configurations: defaultdict[str, Counter[int]],
        effect: str,
        candidate: Iterable[tuple[ErrorLocation, int]],
):
        # check for duplicate sources
        sources: set[ErrorLocation] = set()
        unsorted_denominators: list[int] = []
        tot = 1
        for source, count in candidate:
            if source in sources:
                return
            _, name, _ = source
            unsorted_denominators.append(error_event_count(name))
            tot *= count
            sources.add(source)
        denominators = math.prod(unsorted_denominators)
        undetected_configurations[effect][denominators] += tot
    return f