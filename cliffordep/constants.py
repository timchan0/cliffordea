import itertools
from typing import Literal


DEPOLARIZE2_FAULTS: tuple[tuple[
    Literal['I', 'Z', 'X', 'Y'],
    Literal['I', 'Z', 'X', 'Y'],
], ...] = tuple(fault for fault in itertools.product(
    'IXYZ', repeat=2) if fault != ('I', 'I')) # type: ignore
# TODO: replace with stim.PauliString.iter_all(2, min_weight=1)?