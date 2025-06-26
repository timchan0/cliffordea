"""Module for manipulating Pauli strings."""

import cmath
from collections import defaultdict
from collections.abc import Iterable, Sequence
import itertools
from typing import Literal

from stim import PauliString


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


def unsigned_str(pauli_string: str | PauliString):
    """Convert a stim.PauliString to a string without the global phase."""
    return str(pauli_string).replace('+', '').replace('-', '').replace('i', '')


def push_through_transversal(pauli_string: PauliString, gate: Literal['T', 'S', 'Z'] = 'T'):
    """Push a Pauli string through the same gate on each qubit.
    
    Input:
    * `pauli_string` the initial `PauliString`.
    * `gate` the gate to push through, either 'T', 'S', or 'Z'.

    Output:
    * A unitary Clifford string that results from
    pushing the input through a `gate` on each qubit.
    """
    map = PUSH_THROUGH_T if gate == 'T' else PUSH_THROUGH_S if gate == 'S' else PUSH_THROUGH_Z
    sign = pauli_string.sign
    options: list[tuple[str, ...]] = [map[pauli] for pauli in pauli_string]
    return CliffordString([sign * _pauli_tuple_to_string(pauli)
        for pauli in itertools.product(*options)])


class CliffordString:
    """An error in the form of a superposition of Pauli strings.
    
    Instance attributes:
    * `terms` a list of signed Pauli strings that make up the superposition.
    * `denominator_squared` the divisor of each term, squared.
    * `logical_amplitudes` maps each logical signature (which logicals do not commute)
    to the amplitude of the Clifford string with that signature.
    These amplitudes are not normalized by the denominator.
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
        * `self.logical_amplitudes` has been set with `self.set_logical_amplitudes()`.

        Output:
        * The probability mass of the Clifford string whose value is in [0, 1].
        """
        numerator = sum(a.real**2 + a.imag**2 for a in self.logical_amplitudes.values())
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
    
    def set_logical_amplitudes(
            self,
            logical_x: PauliString,
            logical_z: PauliString,
    ):
        """Return the unnormalized amplitude of each logical class in the Clifford string.

        Require:
        * `self.terms` contains only terms that commute with all stabilizers
        i.e. `self.postselect_from_stabilizers` has been called.
        
        Input:
        * `logical_x` a Pauli string representing a logical X operator.
        * `logical_z` ditto for Z.

        Side effect:
        * `self.logical_amplitudes` is updated according to `self.terms`.
        """
        # TODO: return 4-vector instead of a dict
        amplitudes: defaultdict[tuple[bool, ...], complex] = defaultdict(complex)
        for term in self.terms:
            signature = tuple(not term.commutes(logical) for logical in (logical_x, logical_z))
            amplitudes[signature] += term.sign
        self.logical_amplitudes = dict(amplitudes)
    
    def convert_hxy_to_identity(self):
        """Convert as much logical H_XY := (X + Y) / sqrt(2) as possible into logical identity.
        
        Require:
        * The state this Clifford string acts on is the logical T state.
        * logical signatures are 'anticommute with (logical X, logical Z)'.
        * `self.logical_amplitudes` has been set with `self.set_logical_amplitudes()`.

        Side effect:
        * In `self.logical_amplitudes`, converts as much logical H_XY amplitude
        into logical identity amplitude as possible.
        """
        # TODO: account for converting X - Y to Z
        new_amplitudes = defaultdict(complex, self.logical_amplitudes)
        logical_i = (False, False)
        logical_x = (False, True)
        logical_y = (True, True)
        r_x, phi_x = cmath.polar(new_amplitudes[logical_x])
        r_y, phi_y = cmath.polar(new_amplitudes[logical_y])
        if all((r_x, r_y, phi_x==phi_y)):
            subtractand = cmath.rect(min(r_x, r_y), phi_x)
            new_amplitudes[logical_x] -= subtractand
            new_amplitudes[logical_y] -= subtractand
            new_amplitudes[logical_i] += subtractand * 2**0.5
        self.logical_amplitudes = dict(new_amplitudes)
    
    @property
    def is_logical_error(self):
        """Return whether `logical_amplitudes` represents a logical error."""
        return any((abs(self.logical_amplitudes.get(signature, 0)) for signature in (
            (False, True),
            (True, True),
            (True, False),
        )))
    
    # BELOW ARE DEPRECATED METHODS
    
    @property
    def logical_error_probability(self) -> float:
        """Return the probability of any logical error due to the Clifford string.

        Require:
        * `self.terms` contains only terms that commute with all stabilizers.

        Output:
        * the probability the Clifford string leads to any logical error
        i.e. the fraction that anticommutes with at least one logical operator.
        This is a real number in the range [0, 1].
        """
        numerator = 0
        for signature, amplitude in self.logical_amplitudes.items():
            if any(signature):
                numerator += amplitude.real**2 + amplitude.imag**2
        return numerator / self.denominator_squared

    def factor_out_logical_h_xy(self):
        """Try factor out a logical H_XY := (X + Y) / sqrt(2) from the right.

        Require:
        * `self.terms` an even sequence of Pauli strings for the distance-3 color code,
        ordered such that the first plus the last term is divisible by the logical H_XY,
        the second plus the second last term is divisible by the logical H_XY, etc.
        
        Output:
        * `self` right-divided by the appropriate logical H_XY.
        * `pauli_indices` the indices of the qubits acted on by the appropriate logical H_XY.
        """
        first, *_ = self.terms
        if len(first) != 7:
            raise ValueError(f"Expected 7 qubits, got {len(first)}.")
        pauli_indices = first.pauli_indices(included_paulis='XY')
        proposed_x = PauliString('X' if index in pauli_indices else 'I' for index in range(7))
        proposed_y = PauliString('Y' if index in pauli_indices else 'I' for index in range(7))
        factors: list[PauliString] = []
        for k in range(len(self.terms)//2):
            term_1 = self.terms[k]
            term_2 = self.terms[-1-k]
            factor_1 = term_1 * proposed_x
            factor_2 = term_2 * proposed_y
            factor_3 = term_1 * proposed_y
            factor_4 = term_2 * proposed_x
            if factor_1 == factor_2:
                factor = factor_1
            elif factor_3 == factor_4:
                factor = factor_3
            else:
                raise ValueError(f"Cannot factor out H_XY from {term_1} {term_2}.")
            factors.append(factor)
        if len(pauli_indices) not in {3, 7}:
            raise ValueError(f"Factored out {
                '*'.join(f'X{i}' for i in pauli_indices)
            } + {
                '*'.join(f'Y{i}' for i in pauli_indices)
            } but these terms are not logical X and Y respectively.")
        return CliffordString(factors, self.denominator_squared/2), pauli_indices


def _pauli_tuple_to_string(pauli_tuple: tuple[str, ...]):
    string = PauliString()
    for pauli in pauli_tuple:
        string += PauliString(pauli)
    return string


def factor_out_logical_h_xy(undetected_terms: Sequence[PauliString]):
    """Factor out the correct logical H_XY from a sequence of undetected terms.

    Input:
    * `undetected_terms` an even sequence of pauli strings that commute with all stabilizers.
    
    Output:
    * `factors` the undetected terms divided by the appropriate logical H_XY.
    * `pauli_indices` the indices of the qubits acted on by the appropriate logical H_XY.
    """
    first_term, *_ = undetected_terms
    pauli_indices = first_term.pauli_indices(included_paulis='XY')
    proposed_x = PauliString('X' if index in pauli_indices else 'I' for index in range(7))
    proposed_y = PauliString('Y' if index in pauli_indices else 'I' for index in range(7))
    factors: list[PauliString] = []
    for k in range(len(undetected_terms)//2):
        term_1 = undetected_terms[k]
        term_2 = undetected_terms[-1-k]
        factor_1 = term_1 * proposed_x
        factor_2 = term_2 * proposed_y
        factor_3 = term_1 * proposed_y
        factor_4 = term_2 * proposed_x
        if factor_1 == factor_2:
            factor = factor_1
        elif factor_3 == factor_4:
            factor = factor_3
        else:
            raise ValueError(f"Cannot factor out H_XY from {term_1} {term_2}.")
        factors.append(factor)
    return factors, pauli_indices