"""Module for manipulating Pauli strings."""

import cmath
from collections import defaultdict
from dataclasses import dataclass
import itertools
import math
from typing import Literal

import stim
from stim import PauliString
import numpy as np
from numpy import typing as npt


PUSH_THROUGH_Z: dict[str, tuple[str, ...]] = {
    '_': ('I',),
    'X': ('-X',),
    'Y': ('-Y',),
    'Z': ('Z',),
}
"""A map from each Pauli to the unnormalized superposition of Paulis
after pushing through a Z gate.
"""

PUSH_THROUGH_S: dict[str, tuple[str, ...]] = {
    '_': ('I',),
    'X': ('Y',),
    'Y': ('-X',),
    'Z': ('Z',),
}
"""A map from each Pauli to the unnormalized superposition of Paulis
after pushing through an S gate.
"""

PUSH_THROUGH_S_DAG: dict[str, tuple[str, ...]] = {
    '_': ('I',),
    'X': ('-Y',),
    'Y': ('X',),
    'Z': ('Z',),
}
"""A map from each Pauli to the unnormalized superposition of Paulis
after pushing through an S dagger gate.
"""

PUSH_THROUGH_T: dict[str, tuple[str, ...]] = {
    '_': ('I',),
    'X': ('X', 'Y'),
    'Y': ('-X', 'Y'),
    'Z': ('Z',),
}
"""A map from each Pauli to the unnormalized superposition of Paulis
after pushing through a T gate.
"""

PUSH_THROUGH_T_DAG: dict[str, tuple[str, ...]] = {
    '_': ('I',),
    'X': ('X', '-Y'),
    'Y': ('X', 'Y'),
    'Z': ('Z',),
}
"""A map from each Pauli to the unnormalized superposition of Paulis
after pushing through a T dagger gate.
"""

PUSH_THROUGH_MAP: dict[str, dict[str, tuple[str, ...]]] = {
    'Z': PUSH_THROUGH_Z,
    'S': PUSH_THROUGH_S,
    'T': PUSH_THROUGH_T,
    'Z_DAG': PUSH_THROUGH_Z,
    'S_DAG': PUSH_THROUGH_S_DAG,
    'T_DAG': PUSH_THROUGH_T_DAG,
}

def forget_sign(pauli_string: PauliString):
    """Convert a `stim.PauliString` to a string without the global phase."""
    return str(pauli_string).replace('+', '').replace('-', '').replace('i', '')


def split_sign(pauli_string: PauliString):
    """Split a Pauli string into its sign and the unsigned part."""
    return pauli_string.sign, forget_sign(pauli_string)


def _tensor_paulis(*paulis: str):
    """Tensor product one or more signed Paulis.
    
    Input:
    * `paulis` a (possibly empty) tuple of signed Paulis e.g. ('X', '-iY').

    Output:
    * Their tensor product as a `stim.PauliString`.
    """
    return sum((PauliString(pauli) for pauli in paulis), start=PauliString())


def _reset_qubits(effect: str, indices: set[int]):
    """Reset specified qubits in an effect."""
    return ''.join('_' if i in indices else c for i, c in enumerate(effect))


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

_PRECISION = 12
"""Rounding precision for `FrozenCliffordString`."""


def _canonicalize(terms: dict[str, complex], denominator_squared: float):
    """Cast a Clifford string into canonical form.
    
    Canonical form means:
    * no terms have zero amplitude.
    * the phase of the lexicographically smallest term is 0.
    * the magnitude of the smallest amplitude is 1.
    * all values are rounded to 12 decimal digits.

    Input:
    * `terms, denominator_squared` defines the Clifford string to canonicalize.

    Output:
    * The canonicalized `(terms, denominator_squared)`.

    Side effects:
    * None.
    """
    new_terms = {term: amplitude for term, amplitude in terms.items() if amplitude}
    if new_terms:
        first_term = min(new_terms.keys())
        first_phase = cmath.phase(new_terms[first_term])
        phase_factor = cmath.exp(1j * first_phase)
        smallest_magnitude = min(abs(amplitude) for amplitude in new_terms.values())
        for term in new_terms.keys():
            unrounded = new_terms[term] / (phase_factor*smallest_magnitude)
            real = round(unrounded.real, _PRECISION)
            imag = round(unrounded.imag, _PRECISION)
            new_terms[term] = complex(real, imag)
        new_denominator_squared = round(denominator_squared / smallest_magnitude**2, _PRECISION)
    else:
        new_denominator_squared = 1
    return new_terms, new_denominator_squared


def _boolean_array_to_int(array: np.ndarray) -> int:
    """Convert a 1D boolean numpy array to an integer."""
    return int(''.join(array.astype(int).astype(str)), 2)


class CliffordString:
    """An error in the form of a superposition of Pauli strings.
    
    Instance attributes:
    * `terms` the Pauli strings that make up the superposition,
    in the form of a map from each unsigned Pauli string to its unnormalized amplitude.
    Allowed characters in each Pauli string are `'_', 'X', 'Y', 'Z'`.
    * `denominator_squared` the divisor of each amplitude, squared.
    """

    def __init__(
            self,
            terms: None | dict[str, complex] = None,
            denominator_squared: None | float = None,
    ):
        """Input:
        * `terms` a map from each unsigned Pauli string to its unnormalized amplitude.
        Allowed characters in each Pauli string are `'_', 'X', 'Y', 'Z'`.
        If not specified,
        the Clifford string is assumed to be zero.
        * `denominator_squared` the divisor of each amplitude, squared.
        If not specified,
        the Clifford string is automatically normalized to 1.
        """
        self.terms: defaultdict[str, complex] = defaultdict(
            complex, {} if terms is None else terms)
        if denominator_squared is None:
            self.denominator_squared: float = sum(
                abs(amplitude)**2 for amplitude in self.terms.values()
            ) if self.terms else 1
        else:
            self.denominator_squared = denominator_squared

    def __str__(self):
        return f'{self.denominator_squared}^(-1/2) [{" + ".join(
            f"{sign}{term}" for term, sign in self.terms.items()
        )}]'
    
    def __repr__(self):
        return f'CliffordString({dict(self.terms)}, {self.denominator_squared})'

    def __mul__(self, rhs: 'int | float | complex | CliffordString'):
        if isinstance(rhs, (int, float, complex)):
            return self._scaled(rhs)
        return self._multiply(self, rhs)
    
    def __rmul__(self, lhs: 'int | float | complex | CliffordString'):
        if isinstance(lhs, (int, float, complex)):
            return self._scaled(lhs)
        return self._multiply(lhs, self)

    def _scaled(self, scalar: int | float | complex):
        """Return a scaled copy of the Clifford string."""
        return CliffordString(
            {term: amplitude * scalar for term, amplitude in self.terms.items()},
            self.denominator_squared,
        )

    @staticmethod
    def _multiply(lhs: 'CliffordString', rhs: 'CliffordString'):
        """Return the product of two Clifford strings."""
        terms: defaultdict[str, complex] = defaultdict(complex)
        for l_term, l_amplitude in lhs.terms.items():
            l_ps = PauliString(l_term)
            for r_term, r_amplitude in rhs.terms.items():
                prod_sign, prod_string = split_sign(l_ps * PauliString(r_term))
                terms[prod_string] += prod_sign * l_amplitude * r_amplitude
        return CliffordString(terms, lhs.denominator_squared * rhs.denominator_squared)

    @property
    def norm_squared(self):
        numerator = sum(abs(amplitude)**2 for amplitude in self.terms.values())
        return numerator / self.denominator_squared

    def normalize(self):
        """Scale the original Clifford string so that its norm is 1."""
        self.denominator_squared = sum(abs(amplitude)**2 for amplitude in self.terms.values())


    def postselect_from_stabilizers(self, stabilizer_generators: dict[str, tuple[stim.PauliString, ...]]):
        """Kill all terms that do not commute with the stabilizers.
        
        Input:
        * `stabilizer_generators` the generators of the stabilizer group.

        Side effect:
        * `self.terms` is modified to only include terms that commute with all stabilizers.
        """
        killed: set[str] = set()
        for term in self.terms.keys():
            pauli_string = PauliString(term)
            if not all(
                pauli_string.commutes(generator)
                for generator_list in stabilizer_generators.values()
                for generator in generator_list
            ):
                killed.add(term)
        for term in killed:
            del self.terms[term]

    def commutes_or_unknown(self, other: PauliString):
        """Partial predicate for if the Clifford string commutes with a Pauli string.

        When this method returns `True`, the Clifford string definitely commutes with the Pauli string.
        When it returns `None`, the commutation relation is unknown.
        """
        return True if all(PauliString(term).commutes(other) for term in self.terms.keys()) else None

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
        for term, amplitude in self.terms.items():
            pauli_string = PauliString(term)
            signature: tuple[bool, bool] = tuple(
                not pauli_string.commutes(logical) for logical in (logical_x, logical_z)) # type: ignore
            if signature == (False, False):
                representative = PauliString(len(term))
            elif signature == (False, True):
                representative = logical_x
            elif signature == (True, True):
                representative = 1j * logical_x * logical_z
            else:  # (True, False)
                representative = logical_z
            quotient = pauli_string * representative
            # TODO: generalize this to color code of distance greater than 3
            sign = quotient.sign * 1j**len(quotient.pauli_indices("Y"))
            amplitudes[SIGNATURE_TO_INDEX[signature]] += sign * amplitude
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
"""A map from gate G to the operators (IA, ZA) such that A stabilizes G|+>.
The operators are are 4-vectors in the basis of logical Paulis I, X, Y, Z,
and form a basis in the subspace spanned by (X, Y).
"""


class LogicalVector:
    """A vector of amplitudes for each logical class.
    
    Instance attributes:
    * `amplitudes` a 4-vector of normalized amplitudes
    for the I, X, Y, Z logical classes respectively.
    """

    def __init__(self, amplitudes: npt.NDArray[np.complex128]):
        self.amplitudes = amplitudes

    def __str__(self):
        return str(self.amplitudes)
    
    def __repr__(self):
        return f"LogicalVector({self.amplitudes})"

    @property
    def probability_mass(self):
        """The probability of the surviving part of the logical vector after any postselection,
        whose value is in [0, 1].
        """
        return float(np.vdot(self.amplitudes, self.amplitudes).real)
    
    def probability_of(self, sector: Literal['I', 'X', 'Y', 'Z']):
        """Return the probability of a given logical sector.

        Input:
        * `sector` the logical sector to get the probability of.

        Output:
        * The probability the logical vector leads to that sector.
        This is not normalized by the probability mass,
        and is a real number in the range [0, 1].
        """
        index = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}[sector]
        amplitude: complex = self.amplitudes[index]
        return amplitude.real**2 + amplitude.imag**2

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
        i_component: complex = np.vdot(I_stabilizer, self.amplitudes) # type: ignore
        z_component: complex = np.vdot(Z_stabilizer, self.amplitudes) # type: ignore
        # TODO: assume these are zero
        self.amplitudes -= i_component * I_stabilizer
        self.amplitudes -= z_component * Z_stabilizer
        self.amplitudes += i_component * IDENTITY
        self.amplitudes += z_component * PAULI_Z
        self.amplitudes = self.amplitudes.round(round_decimals)