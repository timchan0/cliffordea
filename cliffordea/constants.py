import itertools
from typing import Literal


ONE_QUBIT_ERROR_EVENTS = frozenset({
    'X_ERROR',
    'Y_ERROR',
    'Z_ERROR',
    'M',
    'MZ',
    'MX',
    'MY',
    'MR',
    'MRZ',
    'MRX',
    'MRY',
})
"""Stim process names that produce one elementary error event per target.

This includes one-qubit Pauli errors and measurement outcome flips, each
occurring with the process probability. Stim canonicalizes the aliases ``MZ``
and ``MRZ`` to ``M`` and ``MR``, respectively. They remain accepted here for
callers that construct error locations or events directly.
"""


DEPOLARIZE2_ERROR_EVENTS: tuple[tuple[
    Literal['I', 'Z', 'X', 'Y'],
    Literal['I', 'Z', 'X', 'Y'],
], ...] = tuple(fault for fault in itertools.product(
    'IXYZ', repeat=2) if fault != ('I', 'I')) # type: ignore
"""A tuple of all possible 2-qubit error events, sorted lexicographically."""
# TODO: replace with stim.PauliString.iter_all(2, min_weight=1)?
