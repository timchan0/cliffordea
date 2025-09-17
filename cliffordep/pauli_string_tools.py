"""Module for manipulating Pauli strings."""

import cmath
from collections import defaultdict
from collections.abc import Sequence
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
after pushing through a S gate.
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
    'T': PUSH_THROUGH_T,
    'S': PUSH_THROUGH_S,
    'Z': PUSH_THROUGH_Z,
    'T_DAG': PUSH_THROUGH_T_DAG,
}  # TODO: test against PauliString.after


def forget_sign(pauli_string: PauliString):
    """Convert a `stim.PauliString` to a string without the global phase."""
    return str(pauli_string).replace('+', '').replace('-', '').replace('i', '')


def split_sign(pauli_string: PauliString):
    """Split a Pauli string into its sign and the unsigned part."""
    return pauli_string.sign, forget_sign(pauli_string)


def push_through_transversal(pauli_string: str, gate: Literal['T', 'S', 'Z']):
    """Push an _unsigned_ Pauli string through the same gate on each qubit.
    
    Input:
    * `pauli_string` the unsigned Pauli string.
    * `gate` the gate to push through, either 'T', 'S', or 'Z'.

    Output:
    * A unitary Clifford string that results from
    pushing the input through a `gate` on each qubit.
    """
    map = PUSH_THROUGH_MAP[gate]
    options: list[tuple[str, ...]] = [map[pauli] for pauli in pauli_string]
    terms: dict[str, complex] = {}
    for pauli_tuple in itertools.product(*options):
        sign, child = split_sign(_tensor_paulis(*pauli_tuple))
        terms[child] = sign
    return CliffordString(terms)


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

    def frozen_copy(self):
        """Return a frozen and hashable copy of the Clifford string.
        
        Side effects:
        * None.
        """
        canonicalized_terms, canonicalized_denominator_squared = _canonicalize(self.terms, self.denominator_squared)
        return FrozenCliffordString(
            frozenset(canonicalized_terms.items()),
            canonicalized_denominator_squared,
        )


    def postselect_from_stabilizers(self, stabilizer_generators: Sequence[PauliString]):
        """Kill all terms that do not commute with the stabilizers.
        
        Input:
        * `stabilizer_generators` the generators of the stabilizer group.

        Side effect:
        * `self.terms` is modified to only include terms that commute with all stabilizers.
        """
        killed: set[str] = set()
        for term in self.terms.keys():
            pauli_string = PauliString(term)
            if not all(pauli_string.commutes(stabilizer) for stabilizer in stabilizer_generators):
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
            amplitudes[SIGNATURE_TO_INDEX[signature]] += amplitude
        return LogicalVector(amplitudes / self.denominator_squared**0.5)


@dataclass(frozen=True)
class FrozenCliffordString:
    terms: frozenset[tuple[str, complex]]
    denominator_squared: float

    def __str__(self):
        return f'{self.denominator_squared}^(-1/2) [{" + ".join(
            f"{sign}{term}" for term, sign in self.terms
        )}]'
    
    def __repr__(self):
        return f'FrozenCliffordString({self.terms}, {self.denominator_squared})'

    def __mul__(self, rhs: 'int | float | FrozenCliffordString'):
        if isinstance(rhs, (int, float)):
            return self._scaled(rhs)
        return self._multiply(self, rhs)
    
    def __rmul__(self, lhs: 'int | float | FrozenCliffordString'):
        if isinstance(lhs, (int, float)):
            return self._scaled(lhs)
        return self._multiply(lhs, self)

    # TODO: implement product so no need to canonicalize repeatedly
    @staticmethod
    def _multiply(lhs: 'FrozenCliffordString', rhs: 'FrozenCliffordString'):
        """Return the product of two frozen Clifford strings, canonicalized."""
        terms: defaultdict[str, complex] = defaultdict(complex)
        for l_term, l_amplitude in lhs.terms:
            l_ps = PauliString(l_term)
            for r_term, r_amplitude in rhs.terms:
                prod_sign, prod_string = split_sign(l_ps * PauliString(r_term))
                terms[prod_string] += prod_sign * l_amplitude * r_amplitude
        canonicalized_terms, canonicalized_denominator_squared = _canonicalize(
            terms,
            lhs.denominator_squared * rhs.denominator_squared,
        )
        return FrozenCliffordString(
            frozenset(canonicalized_terms.items()),
            canonicalized_denominator_squared,
        )

    def mutable_copy(self):
        """Return a mutable, unhashable copy of the Clifford string."""
        return CliffordString(
            terms=dict(self.terms),
            denominator_squared=self.denominator_squared,
        )
    
    @property
    def norm_squared(self):
        numerator = sum(abs(amplitude)**2 for _, amplitude in self.terms)
        return numerator / self.denominator_squared
    
    def _scaled(self, scalar: int | float):
        """Return a scaled frozen canonical copy of the Clifford string."""
        return FrozenCliffordString(
            terms=self.terms,
            denominator_squared=round(self.denominator_squared / scalar**2, _PRECISION),
        )
    

    def push_through_unitary(
            self,
            unitary: stim.CircuitInstruction,
            replace_s_with: Literal['T', 'S', 'Z'] = 'S',
    ):
        """Return the result of pushing the Clifford string through a unitary instruction.

        Input:
        * `unitary` the instruction.
        It MUST NOT produce measurements or have a reset.
        * `replace_s_with` the unitary to push through if `unitary` is an S gate.
        Also affects S dagger gates.

        Output:
        * The Clifford string after being pushed through the unitary
        represented as a new frozen Clifford string.
        """
        target_indices: set[int] = {target.value for target in unitary.targets_copy()}
        new_terms: defaultdict[str, complex] = defaultdict(complex)
        
        if replace_s_with == 'T' and (name:=unitary.name) in {'S', 'S_DAG'}:
            map = PUSH_THROUGH_MAP['T' if name=='S' else 'T_DAG']
            for term, amplitude in self.terms:
                options: list[tuple[str, ...]] = [
                    map[pauli] if index in target_indices else (pauli,)
                    for index, pauli in enumerate(term)]
                denominator: float = math.prod(len(option) for option in options)**0.5
                for pauli_tuple in itertools.product(*options):
                    sign, child = split_sign(_tensor_paulis(*pauli_tuple))
                    new_terms[child] += sign * amplitude / denominator
        
        else:
            if replace_s_with == 'Z' and unitary.name in {'S', 'S_DAG'}:
                unitary = stim.CircuitInstruction(
                    name='Z',
                    targets=unitary.targets_copy(),
                )
            for term, amplitude in self.terms:
                sign, unsigned = split_sign(PauliString(term).after(unitary))
                new_terms[unsigned] += sign * amplitude

        canonicalized_terms, canonicalized_denominator_squared = _canonicalize(new_terms, self.denominator_squared)
        return FrozenCliffordString(frozenset(canonicalized_terms.items()), canonicalized_denominator_squared)


    def push_through_reset(self, reset: stim.CircuitInstruction):
        """Return the result of pushing the Clifford string through a reset instruction.
        
        Input:
        * `reset` the reset instruction to push through.
        It must satisfy `stim.gate_data(reset.name).is_reset`.

        Output:
        * `mixture` a mixture of Clifford strings,
        in the form of a map from each normalized frozen Clifford string to its probability.
        The sum of these probabilities equals `self.norm_squared`.

        # Examples
        Pushing `(XX + YY)/sqrt(2)` through a reset on qubit 0 yields...
        * `_X` with probability 1/2,
        * `_Y` with probability 1/2.
        
        Pushing `(XX + XY + YX + YY)/2` through a reset on qubit 1 yields...
        * `(X_ + Y_)/sqrt(2)` with probability 1.
        """
        qubits_reset = {target.value for target in reset.targets_copy()}
        # `clifford_strings` maps the string of Paulis that were reset to a post-reset Clifford string
        clifford_strings: defaultdict[
            tuple[str, ...], defaultdict[str, complex]
        ] = defaultdict(lambda: defaultdict(complex))
        for term, amplitude in self.terms:
            paulis_reset = tuple(term[index] for index in qubits_reset)
            term_after = _reset_qubits(term, qubits_reset)
            clifford_strings[paulis_reset][term_after] += amplitude
        mixture: defaultdict[FrozenCliffordString, float] = defaultdict(float)
        for terms in clifford_strings.values():
            clifford_string = CliffordString(terms, self.denominator_squared)
            probability = clifford_string.norm_squared
            clifford_string.normalize()
            mixture[clifford_string.frozen_copy()] += probability
        return mixture
    

    def push_through_measurement(
            self,
            measurement: stim.CircuitInstruction,
            measurement_to_detectors: dict[int, set[int]],
            syndrome_before_push: tuple[bool, ...],
    ):
        """Return the result of pushing the Clifford string through a measurement instruction.
        
        Input:
        * `measurement` the measurement instruction.
        * `measurement_to_detectors` a map from each measurement index in the circuit
        to a set of indices of the detectors it flips.
        * `syndrome_before_push` the syndrome of the Clifford string
        before pushing it through the measurement.

        Output:
        * `mixture` a mixture of Clifford strings,
        represented as a map from each syndrome to a normalized Clifford string
        (that gives that syndrome when pushed through `measurement`) and its probability.
        """
        # TODO: give examples in docstring
        if measurement.name == 'MPP':
            clifford_strings = self._push_through_multiqubit_measurement(measurement, measurement_to_detectors, syndrome_before_push)
        else:
            clifford_strings = self._push_through_1_qubit_measurement(measurement, measurement_to_detectors, syndrome_before_push)
        mixture = self._push_through_measurement_helper(clifford_strings)
        return mixture

    def _push_through_1_qubit_measurement(
            self,
            measurement: stim.CircuitInstruction,
            measurement_to_detectors: dict[int, set[int]],
            syndrome_before_push: tuple[bool, ...],
    ):
        anticommuting_paulis = self._get_anticommuting_paulis(measurement.name)
        # `clifford_strings` maps each syndrome to a post-measurement Clifford string
        clifford_strings: defaultdict[
            tuple[bool, ...], defaultdict[str, complex]
        ] = defaultdict(lambda: defaultdict(complex))
        for term, amplitude in self.terms:
            syndrome = np.array(syndrome_before_push, dtype=bool)
            for measurement_index, target in enumerate(
                measurement.targets_copy(),
                start=int(measurement.tag),
            ):
                if term[target.value] in anticommuting_paulis:
                    for detector in measurement_to_detectors[measurement_index]:
                        syndrome[detector] ^= True
            clifford_strings[tuple(syndrome)][term] += amplitude
        return clifford_strings
    
    def _push_through_multiqubit_measurement(
            self,
            measurement: stim.CircuitInstruction,
            measurement_to_detectors: dict[int, set[int]],
            syndrome_before_push: tuple[bool, ...],
    ):
        # `clifford_strings` maps each syndrome to a post-measurement Clifford string
        clifford_strings: defaultdict[
            tuple[bool, ...], defaultdict[str, complex]
        ] = defaultdict(lambda: defaultdict(complex))
        for term, amplitude in self.terms:
            syndrome = np.array(syndrome_before_push, dtype=bool)
            for measurement_index, target_group in enumerate(
                measurement.target_groups(),
                start=int(measurement.tag),
            ):
                measurement_flips = False
                for target in target_group:
                    if term[target.value] in self._get_anticommuting_paulis(target.pauli_type):
                        measurement_flips ^= True
                if measurement_flips:
                    for detector in measurement_to_detectors[measurement_index]:
                        syndrome[detector] ^= True
            clifford_strings[tuple(syndrome)][term] += amplitude
        return clifford_strings

    def _push_through_measurement_helper(self, clifford_strings: dict[tuple[bool, ...], defaultdict[str, complex]]):
        mixture: dict[tuple[bool, ...], tuple[FrozenCliffordString, float]] = {}
        for syndrome, terms in clifford_strings.items():
            s = CliffordString(terms, self.denominator_squared)
            probability = s.norm_squared
            s.normalize()
            mixture[syndrome] = (s.frozen_copy(), probability)
        return mixture

    @staticmethod
    def _get_anticommuting_paulis(name: str):
        """Get the set of Paulis that anticommute with the measurement given by `name`."""
        if 'X' in name:
            return {'Y', 'Z'}
        elif 'Y' in name:
            return {'X', 'Z'}
        elif 'Z' in name or name in {'M', 'MR'}:
            return {'X', 'Y'}
        else:
            raise NotImplementedError


class Mixture:
    """A mixture of Clifford strings.
    
    Instance attributes:
    * `submixtures` a map from each syndrome to a map from each
    normalized (frozen) Clifford string (that gives that syndrome) to its probability.
    Each syndrome is a tuple of booleans, where each boolean
    indicates whether the corresponding detector has been flipped.
    """

    def __init__(self, submixtures: dict[tuple[bool, ...], list[CliffordString]]):
        _submixtures: defaultdict[
            tuple[bool, ...],
            defaultdict[FrozenCliffordString, float],
        ] = defaultdict(lambda: defaultdict(float))
        for syndrome, list_ in submixtures.items():
            for clifford_string in list_:
                probability = clifford_string.norm_squared
                clifford_string.normalize()
                _submixtures[syndrome][clifford_string.frozen_copy()] += probability
        self.submixtures = dict(_submixtures)


    def __str__(self) -> str:
        lines = []
        for syndrome, submixture in self.submixtures.items():
            lines.append(''.join('1' if detector else '0' for detector in syndrome))
            for clifford_string, probability in submixture.items():
                lines.append(f'  probability {probability}: {str(clifford_string)}')
        return '\n'.join(lines)


    def push_through(
            self,
            instruction: stim.CircuitInstruction,
            measurement_to_detectors: dict[int, set[int]],
            replace_s_with: Literal['T', 'S', 'Z'] = 'S',
    ):
        """Push the mixture through a circuit instruction.

        Input:
        * `instruction` the instruction.
        * `measurement_to_detectors` a map from each measurement index in the circuit
        to a set of indices of the detectors it flips.

        Side effect:
        * All the mixture's terms are pushed through the circuit instruction.
        """
        data = stim.gate_data(instruction.name)
        if produces_measurements := data.produces_measurements:
            self._push_through_measurement(
                instruction,
                measurement_to_detectors,
            )
        if is_reset := data.is_reset:
            self._push_through_reset(instruction)
        if not (produces_measurements or is_reset):
            self._push_through_unitary(instruction, replace_s_with=replace_s_with)


    def _push_through_unitary(
            self,
            unitary: stim.CircuitInstruction,
            replace_s_with: Literal['T', 'S', 'Z'] = 'S',    
    ):
        """Push the mixture through a unitary instruction.

        Input:
        * `unitary` the instruction.
        It MUST NOT produce measurements or have a reset.

        Side effect:
        * All the mixture's terms are pushed through the unitary instruction.
        """
        new_mixture: defaultdict[
            tuple[bool, ...],
            defaultdict[FrozenCliffordString, float],
        ] = defaultdict(lambda: defaultdict(float))
        for syndrome, submixture in self.submixtures.items():
            for frozen_string, probability in submixture.items():
                frozen_string_after = frozen_string.push_through_unitary(unitary, replace_s_with=replace_s_with)
                new_mixture[syndrome][frozen_string_after] += probability
        self.submixtures = dict(new_mixture)


    def _push_through_measurement(
            self,
            measurement: stim.CircuitInstruction,
            measurement_to_detectors: dict[int, set[int]],
    ):
        """Push the mixture through a measurement.
        
        Input:
        * `measurement` the measurement instruction.
        It must satisfy `stim.gate_data(measurement.name).produces_measurements`.
        * `measurement_to_detectors` a map from each measurement index in the circuit
        to a set of indices of the detectors it flips.

        Side effect:
        * All the mixture's terms are pushed through the measurement instruction.
        """
        new_mixture: defaultdict[
            tuple[bool, ...],
            defaultdict[FrozenCliffordString, float],
        ] = defaultdict(lambda: defaultdict(float))
        for syndrome_before_push, submixture in self.submixtures.items():
            for outer_string, outer_prob in submixture.items():
                inner_submixture = outer_string.push_through_measurement(
                    measurement,
                    measurement_to_detectors,
                    syndrome_before_push,
                )
                for new_syndrome, (inner_string, inner_prob) in inner_submixture.items():
                    new_mixture[new_syndrome][inner_string] += outer_prob * inner_prob
        self.submixtures = dict(new_mixture)


    def _push_through_reset(self, reset: stim.CircuitInstruction):
        """Push the mixture through a reset instruction.
        
        Input:
        * `reset` the reset instruction to push through.
        It must satisfy `stim.gate_data(reset.name).is_reset`.

        Side effect:
        * All the mixture's terms are pushed through the reset instruction.
        """
        new_mixture: defaultdict[
            tuple[bool, ...],
            defaultdict[FrozenCliffordString, float],
        ] = defaultdict(lambda: defaultdict(float))
        for syndrome, submixture in self.submixtures.items():
            for outer_string, outer_prob in submixture.items():
                inner_submixture = outer_string.push_through_reset(reset)
                for inner_string, inner_prob in inner_submixture.items():
                    new_mixture[syndrome][inner_string] += outer_prob * inner_prob
        self.submixtures = dict(new_mixture)


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

    @property
    def is_error(self):
        """Return whether `self` leads to a logical fidelity < 1."""
        return bool(np.vdot(ANY_ERROR, self.amplitudes))
    
    @property
    def error_probability(self):
        """Return the probability of Z logical error.

        Require:
        * The amplitudes of the X and Y logical classes are zero.

        Output:
        * the probability the logical vector leads to Z logical error
        given the logical vector has survived.
        This is a real number in the range [0, 1].
        """
        z_amplitude: complex = self.amplitudes[3]
        return (z_amplitude.real**2 + z_amplitude.imag**2) / self.probability_mass

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