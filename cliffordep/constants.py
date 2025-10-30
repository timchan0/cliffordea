import itertools
from typing import Literal


DEPOLARIZE2_ERROR_EVENTS: tuple[tuple[
    Literal['I', 'Z', 'X', 'Y'],
    Literal['I', 'Z', 'X', 'Y'],
], ...] = tuple(fault for fault in itertools.product(
    'IXYZ', repeat=2) if fault != ('I', 'I')) # type: ignore
"""A tuple of all possible 2-qubit error events, sorted lexicographically."""
# TODO: replace with stim.PauliString.iter_all(2, min_weight=1)?