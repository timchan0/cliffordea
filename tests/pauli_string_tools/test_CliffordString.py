import cmath
from collections import defaultdict
from unittest import mock

import numpy as np
import pytest
from stim import PauliString

from cliffordep.pauli_string_tools import CliffordString, SQRT2

_EPSILON = 1e-13


class TestInit:

    def test_empty(self):
        cs = CliffordString()
        assert cs.terms == defaultdict(complex)
        assert cs.denominator_squared == 1

    def test_terms(self, terms):
        cs = CliffordString(terms)
        assert cs.terms == defaultdict(complex, terms)
        assert cs.denominator_squared == 5

    @pytest.mark.parametrize("denominator_squared", [1, 2, 3])
    def test_denominator_squared(self, denominator_squared: float):
        cs = CliffordString(denominator_squared=denominator_squared)
        assert cs.terms == defaultdict(complex)
        assert cs.denominator_squared == denominator_squared
    
    @pytest.mark.parametrize("denominator_squared", [1, 2, 3])
    def test_both(self, terms: dict[str, complex], denominator_squared: float):
        cs = CliffordString(terms, denominator_squared=denominator_squared)
        assert cs.terms == defaultdict(complex, terms)
        assert cs.denominator_squared == denominator_squared


class TestScaled:

    def test_int(self, string_1: CliffordString):
        cs = string_1._scaled(2)
        assert cs.terms == defaultdict(complex, {"X": 4, "Y": -2j})
        assert cs.denominator_squared == string_1.denominator_squared

    def test_float(self, string_1: CliffordString):
        cs = string_1._scaled(1.5)
        assert cs.terms == defaultdict(complex, {"X": 3, "Y": -1.5j})
        assert cs.denominator_squared == string_1.denominator_squared

    def test_complex(self, string_1: CliffordString):
        cs = string_1._scaled(1 + 2j)
        assert cs.terms == defaultdict(complex, {"X": 2 + 4j, "Y": 2 - 1j})
        assert cs.denominator_squared == string_1.denominator_squared


class TestMultiply:

    def test_both_empty(self):
        cs = CliffordString._multiply(CliffordString(), CliffordString())
        assert cs.terms == defaultdict(complex)
        assert cs.denominator_squared == 1

    def test_left_empty(self, string_1):
        cs = CliffordString._multiply(CliffordString(), string_1)
        assert cs.terms == defaultdict(complex)
        assert cs.denominator_squared == string_1.denominator_squared

    def test_right_empty(self, string_1):
        cs = CliffordString._multiply(string_1, CliffordString())
        assert cs.terms == defaultdict(complex)
        assert cs.denominator_squared == string_1.denominator_squared

    def test_square(self, string_1: CliffordString):
        cs = CliffordString._multiply(string_1, string_1)
        assert cs.terms == defaultdict(complex, {"_": 3, "Z": 0})
        assert cs.denominator_squared == 25

    def test_different(self, string_1: CliffordString, string_2: CliffordString):
        cs = CliffordString._multiply(string_1, string_2)
        assert cs.terms == defaultdict(complex, {"X": 1j, "Y": -1})
        assert cs.denominator_squared == string_1.denominator_squared * string_2.denominator_squared

    def test_annihilate(self):
        s1 = CliffordString({'XX': 1, 'YY': 1})
        s2 = CliffordString({'XX': 1, 'YY': -1})
        cs = CliffordString._multiply(s1, s2)
        assert dict(cs.terms) == {"__": 0, "ZZ": 0}
        assert cs.denominator_squared == 4


class TestMul:

    _scaled = 'cliffordep.pauli_string_tools.CliffordString._scaled'
    _multiply = 'cliffordep.pauli_string_tools.CliffordString._multiply'

    def test_int(self, string_1: CliffordString):
        with mock.patch(self._scaled) as mock_scaled:
            _ = string_1 * 2
            mock_scaled.assert_called_once_with(2)

    def test_float(self, string_1: CliffordString):
        with mock.patch(self._scaled) as mock_scaled:
            _ = string_1 * 1.5
            mock_scaled.assert_called_once_with(1.5)

    def test_complex(self, string_1: CliffordString):
        with mock.patch(self._scaled) as mock_scaled:
            _ = string_1 * (1 + 2j)
            mock_scaled.assert_called_once_with(1 + 2j)

    def test_clifford_string(self, string_1: CliffordString, string_2: CliffordString):
        with mock.patch(self._multiply) as mock_multiply:
            _ = string_1 * string_2
            mock_multiply.assert_called_once_with(string_1, string_2)


def test_norm_squared(string_2: CliffordString):
    assert string_2.norm_squared == 2/1.5


def test_normalize(string_2: CliffordString):
    string_2.normalize()
    assert string_2.norm_squared == 1


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


class TestCanonicalize:

    def test_already_canonical(self, terms: dict[str, complex]):
        new_terms, new_denominator_squared = _canonicalize(terms, 5)
        assert new_terms == terms
        assert new_denominator_squared == 5

    @pytest.mark.parametrize("zero", (0, 0j, -0, -0+0j))
    def test_kill_empty_terms(self, terms: dict[str, complex], string_1: CliffordString, zero: complex):
        string_1.terms['Z'] = zero
        assert string_1.terms == defaultdict(complex, terms | {'Z': zero})
        new_terms, new_denominator_squared = _canonicalize(string_1.terms, string_1.denominator_squared)
        assert new_terms == defaultdict(complex, terms)
        assert new_denominator_squared == 5

    @pytest.mark.parametrize("phase", (0, cmath.pi/2, cmath.pi, 3*cmath.pi/2, cmath.pi/4))
    def test_remove_global_phase(self, string_1: CliffordString, phase: float):
        string_1j = CliffordString({
            term: cmath.exp(1j * phase) * amplitude
            for term, amplitude in string_1.terms.items()
        })
        new_terms, new_denominator_squared = _canonicalize(string_1j.terms, string_1j.denominator_squared)
        assert new_terms == string_1.terms
        assert new_denominator_squared == string_1.denominator_squared

    @pytest.mark.parametrize("scalar", (1.5, 2))
    def test_scale(self, string_1: CliffordString, scalar: float):
        string_2 = string_1._scaled(scalar)
        string_2.denominator_squared *= scalar**2
        new_terms, new_denominator_squared = _canonicalize(string_2.terms, string_2.denominator_squared)
        assert new_terms == string_1.terms
        assert new_denominator_squared == string_1.denominator_squared

    def test_zero(self):
        cs = CliffordString({"X": 0}, denominator_squared=2)
        new_terms, new_denominator_squared = _canonicalize(cs.terms, cs.denominator_squared)
        assert new_terms == defaultdict(complex)
        assert new_denominator_squared == 1

    def test_round(self, string_1: CliffordString):
        string_2 = CliffordString(
            {term: amplitude+_EPSILON for term, amplitude in string_1.terms.items()},
            string_1.denominator_squared+_EPSILON
        )
        assert string_1.terms != string_2.terms
        assert string_1.denominator_squared != string_2.denominator_squared
        new_terms, new_denominator_squared = _canonicalize(string_2.terms, string_2.denominator_squared)
        assert new_terms == string_1.terms
        assert new_denominator_squared == string_1.denominator_squared


class TestGetLogicalAmplitudes:

    WEIGHTS = (3, 7)

    @pytest.fixture(params=WEIGHTS, ids=lambda w: f'X{w}')
    def logical_x(self, request):
        return PauliString('X' * request.param)
    
    @pytest.fixture(params=WEIGHTS, ids=lambda w: f'Z{w}')
    def logical_z(self, request):
        return PauliString('Z' * request.param)

    @pytest.mark.parametrize("weight", WEIGHTS)
    def test_logical_x(self, weight, logical_x, logical_z):
        cs = CliffordString({'X' * weight: 1})
        logical_vector = cs.get_logical_amplitudes(logical_x, logical_z)
        assert np.array_equal(logical_vector.amplitudes, np.array([0, 1, 0, 0]))
    
    @pytest.mark.parametrize("weight", WEIGHTS)
    def test_logical_z(self, weight, logical_x, logical_z):
        cs = CliffordString({'Z' * weight: 1})
        logical_vector = cs.get_logical_amplitudes(logical_x, logical_z)
        assert np.array_equal(logical_vector.amplitudes, np.array([0, 0, 0, 1]))

    @pytest.mark.parametrize("weight", WEIGHTS)
    def test_y_tensor_w(self, weight, logical_x, logical_z):
        cs = CliffordString({'Y' * weight: 1})
        logical_vector = cs.get_logical_amplitudes(logical_x, logical_z)
        assert np.array_equal(logical_vector.amplitudes, np.array([0, 0, -1, 0]))

    @pytest.mark.parametrize("weight", WEIGHTS)
    def test_logical_y(self, weight, logical_x, logical_z):
        cs = CliffordString({'Y' * weight: -1})
        logical_vector = cs.get_logical_amplitudes(logical_x, logical_z)
        assert np.array_equal(logical_vector.amplitudes, np.array([0, 0, 1, 0]))

    @pytest.mark.parametrize("weight", WEIGHTS)
    def test_logical_H_XY(self, weight, logical_x, logical_z):
        cs = CliffordString({'X' * weight: 1, 'Y' * weight: -1})
        logical_vector = cs.get_logical_amplitudes(logical_x, logical_z)
        assert np.array_equal(logical_vector.amplitudes, np.array([0, 1, 1, 0])/SQRT2)

    @pytest.mark.parametrize("weight", WEIGHTS)
    def test_logical_iZH_XY(self, weight, logical_x, logical_z):
        cs = CliffordString({'X' * weight: 1, 'Y' * weight: 1})
        logical_vector = cs.get_logical_amplitudes(logical_x, logical_z)
        assert np.array_equal(logical_vector.amplitudes, np.array([0, 1, -1, 0])/SQRT2)