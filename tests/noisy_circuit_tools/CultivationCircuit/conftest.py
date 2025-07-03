from collections import Counter, defaultdict
from collections.abc import Iterable

import pytest

from cliffordep.noisy_circuit_tools import fault_count
from cliffordep.type_aliases import FaultSource


@pytest.fixture
def update_undetected_combinations():
    """Update `undetected_combinations` with a candidate undetected combination of faults.

    Input:
    * `undetected_combinations` maps each effect to a counter of denominators for each combination.
    * `effect` the resultant (unsigned) Pauli string of the candidate combination.
    * `candidate` a sequence (whose length equals that of the candidate combination)
    of pairs each containing:
        - a fault source
        - count the number of its faults that can be used as part of the undetected combination.

    Side effect:
    * Update `undetected_combinations` with the undetected combination of faults if it is valid
    i.e. comprises distinct fault sources.
    """
    def f(
        undetected_combinations: defaultdict[str, Counter[tuple[int, ...]]],
        effect: str,
        candidate: Iterable[tuple[FaultSource, int]],
):
        # check for duplicate sources
        sources: set[FaultSource] = set()
        unsorted_denominators: list[int] = []
        tot = 1
        for source, count in candidate:
            if source in sources:
                return
            _, name, _ = source
            unsorted_denominators.append(fault_count(name))
            tot *= count
            sources.add(source)
        denominators = tuple(sorted(unsorted_denominators))
        undetected_combinations[effect][denominators] += tot
    return f