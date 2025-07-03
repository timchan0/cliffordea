"""Module for manipulating Pauli strings."""

from collections.abc import Iterable
import itertools
from typing import Literal

from stim import PauliString
import numpy as np
from numpy import typing as npt


PUSH_THROUGH_Z: dict[int, tuple[str, ...]] = {
    0: ('I',),
    1: ('-X',),
    2: ('-Y',),
    3: ('Z',),
}
"""A map from each Pauli to the unnormalized superposition of Paulis
after pushing through a Z gate.
"""

PUSH_THROUGH_S: dict[int, tuple[str, ...]] = {
    0: ('I',),
    1: ('Y',),
    2: ('-X',),
    3: ('Z',),
}
"""A map from each Pauli to the unnormalized superposition of Paulis
after pushing through a S gate.
"""

PUSH_THROUGH_T: dict[int, tuple[str, ...]] = {
    0: ('I',),
    1: ('X', 'Y'),
    2: ('-X', 'Y'),
    3: ('Z',),
}
"""A map from each Pauli to the unnormalized superposition of Paulis
after pushing through a T gate.
"""

PUSH_THROUGH_MAP: dict[Literal['T', 'S', 'Z'], dict[int, tuple[str, ...]]] = {
    'T': PUSH_THROUGH_T,
    'S': PUSH_THROUGH_S,
    'Z': PUSH_THROUGH_Z,
}  # TODO: test against PauliString.after


def unsigned_str(pauli_string: str | PauliString):
    """Convert a stim.PauliString to a string without the global phase."""
    return str(pauli_string).replace('+', '').replace('-', '').replace('i', '')


def push_through_transversal(pauli_string: PauliString, gate: Literal['T', 'S', 'Z']):
    """Push a Pauli string through the same gate on each qubit.
    
    Input:
    * `pauli_string` the initial `PauliString`.
    * `gate` the gate to push through, either 'T', 'S', or 'Z'.

    Output:
    * A unitary Clifford string that results from
    pushing the input through a `gate` on each qubit.
    """
    map = PUSH_THROUGH_MAP[gate]
    sign = pauli_string.sign
    options: list[tuple[str, ...]] = [map[pauli] for pauli in pauli_string]
    return CliffordString([sign * _pauli_tuple_to_string(pauli)
        for pauli in itertools.product(*options)])


def _pauli_tuple_to_string(pauli_tuple: tuple[str, ...]):
    string = PauliString()
    for pauli in pauli_tuple:
        string += PauliString(pauli)
    return string


SIGNATURE_TO_INDEX = {
    (False, False): 0,
    (False, True): 1,
    (True, True): 2,
    (True, False): 3,
}
"""A map from each logical signature
i.e. 'anticommute with (logical X, logical Z)?',
to the Pauli with that signature.
"""


class CliffordString:
    """An error in the form of a superposition of Pauli strings.
    
    Instance attributes:
    * `terms` a list of signed Pauli strings that make up the superposition.
    * `denominator_squared` the divisor of each term, squared.
    """

    def __add__(self, other: 'CliffordString'):
        """Return the superposition of two Clifford strings."""
        if not isinstance(other, CliffordString):
            raise TypeError(f"Cannot add {type(other)} to CliffordString.")
        return CliffordString(self.terms + other.terms)

    def __init__(self, terms: list[PauliString], denominator_squared: None | float = None):
        self.terms = terms
        self.denominator_squared: float = len(terms) if denominator_squared is None else denominator_squared
        # self.terms_bag = defaultdict(complex)
        # for term in terms:
        #     self.terms_bag[unsigned_str(term)] += term.sign

    def __str__(self):
        return f'{self.denominator_squared}^(-1/2) ({" ".join(str(term) for term in self.terms)})'

    @property
    def probability_mass(self):
        """The probability of the surviving part of the Clifford string after any postselection.
        
        Require:
        * `self.terms` are all distinct.

        Output:
        * The probability mass of the Clifford string whose value is in [0, 1].
        """
        numerator = sum(abs(a.sign) for a in self.terms)
        return numerator / self.denominator_squared

    def postselect_from_stabilizers(self, stabilizer_generators: Iterable[PauliString]):
        """Kill all terms that do not commute with the stabilizers.
        
        Input:
        * `stabilizer_generators` the generators of the stabilizer group.

        Side effect:
        * `self.terms` is modified to only include terms that commute with all stabilizers.
        """
        stabilized: list[PauliString] = [term for term in self.terms if
        all(term.commutes(stabilizer) for stabilizer in stabilizer_generators)]
        self.terms = stabilized

    def commutes_or_unknown(self, other: PauliString):
        """Partial predicate for if the Clifford string commutes with a Pauli string.

        When this method returns `True`, the Clifford string definitely commutes with the Pauli string.
        When it returns `None`, the commutation relation is unknown.
        """
        return True if all(term.commutes(other) for term in self.terms) else None
    
    def get_logical_amplitudes(
            self,
            logical_x: PauliString,
            logical_z: PauliString,
    ):
        """Return the normalized amplitude of each logical class in the Clifford string.

        Require:
        * `self.terms` contains only terms that commute with all stabilizers
        i.e. `self.postselect_from_stabilizers` has been called.
        
        Input:
        * `logical_x` a Pauli string representing a logical X operator.
        * `logical_z` ditto for Z.

        Output:
        * A 4-vector of normalized amplitudes for the I, X, Y, Z logical classes in the Clifford string.
        """
        amplitudes = np.zeros(4, dtype=np.complex128)
        for term in self.terms:
            signature: tuple[bool, bool] = tuple(
                not term.commutes(logical) for logical in (logical_x, logical_z)) # type: ignore
            amplitudes[SIGNATURE_TO_INDEX[signature]] += term.sign
        return LogicalVector(amplitudes / self.denominator_squared**0.5)
    

SQRT2 = 2**0.5
ANY_ERROR = np.array([0, 1, 1, 1])
IDENTITY = np.array([1, 0, 0, 0])
"""Logical vector representing the logical I."""
PAULI_Z = np.array([0, 0, 0, 1])
"""Logical vector representing the logical Z."""
IH_XY = np.array([0, 1, 1, 0], dtype=np.complex128) / SQRT2
"""Logical vector representing the logical I * H_XY := (X + Y) / sqrt(2).
This stabilizes the logical T state."""
ZH_XY = 1j * np.array([0, -1, 1, 0], dtype=np.complex128) / SQRT2
"""Logical vector representing the logical Z * H_XY := i (Y - X) / sqrt(2)."""
IY = np.array([0, 0, 1, 0], dtype=np.complex128)
"""Logical vector representing the logical I * Y.
This stabilizes the logical S state."""
ZY = np.array([0, -1j, 0, 0], dtype=np.complex128)
"""Logical vector representing the logical Z * Y := -iX."""
InX = np.array([0, -1, 0, 0], dtype=np.complex128)
"""Logical vector representing the logical I * (-X).
This stabilizes the logical Z state := (1, -1) / sqrt(2)."""
ZnX = np.array([0, 0, -1j, 0], dtype=np.complex128)
"""Logical vector representing the logical Z * (-X) := -iY."""

GATE_TO_IS_AND_ZS: dict[Literal['T', 'S', 'Z'], tuple[
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128],
]] = {
    'T': (IH_XY, ZH_XY),
    'S': (IY, ZY),
    'Z': (InX, ZnX),
}
"""A map from gate G to the operators (IS, ZS) such that S stabilizes G|+>."""


class LogicalVector:
    """A vector of amplitudes for each logical class.
    
    Instance attributes:
    * `amplitudes` a 4-vector of normalized amplitudes
    for the I, X, Y, Z logical classes respectively.
    """

    def __init__(self, amplitudes: npt.NDArray[np.complex128]):
        self.amplitudes = amplitudes

    @property
    def probability_mass(self):
        """The probability of the surviving part of the logical vector after any postselection,
        whose value is in [0, 1].
        """
        return float(np.vdot(self.amplitudes, self.amplitudes).real)

    @property
    def is_logical_error(self):
        """Return whether `self` leads to a logical fidelity < 1."""
        return bool(np.vdot(ANY_ERROR, self.amplitudes))
    
    @property
    def logical_error_probability(self):
        """Return the probability of Z logical error.

        Require:
        * The amplitudes of the X and Y logical classes are zero.

        Output:
        * the probability the logical vector leads to Z logical error.
        This is a real number in the range [0, 1].
        """
        z_amplitude = self.amplitudes[3]
        return float(z_amplitude.real**2 + z_amplitude.imag**2)

    def transfer_xy_to_iz(
            self,
            logical_state: Literal['T', 'S', 'Z'],
            round_decimals: int = 12,
    ):
        """Transfer all X and Y amplitude to I and Z using knowledge of the logical state.
        
        E.g. if the logical state is T, use the fact that H_XY stabilizes the state.
        
        Input:
        * `logical_state` the state this logical vector acts on.
        * `round_decimals` the number of decimals to round all amplitudes to after transfer.

        Side effect:
        * Transfer all X and Y amplitude in `self.amplitudes` to I and Z amplitude.
        """
        # TODO: speed up by casting as a matrix multiplication
        I_stabilizer, Z_stabilizer = GATE_TO_IS_AND_ZS[logical_state]
        i_component = np.vdot(I_stabilizer, self.amplitudes)
        z_component = np.vdot(Z_stabilizer, self.amplitudes)
        # TODO: assume these are zero
        self.amplitudes -= i_component * I_stabilizer
        self.amplitudes -= z_component * Z_stabilizer
        self.amplitudes += i_component * IDENTITY
        self.amplitudes += z_component * PAULI_Z
        self.amplitudes = self.amplitudes.round(round_decimals)