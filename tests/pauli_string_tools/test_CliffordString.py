import cmath
from collections import defaultdict
from unittest import mock

import pytest
import stim

from cliffordep.pauli_string_tools import CliffordString, FrozenCliffordString, SQRT2

_EPSILON = 1e-13


@pytest.fixture
def string_4() -> CliffordString:
    return CliffordString({"XX": 4, "XY": -2j, "YX": -2j, "YY": -1}, denominator_squared=1.5)


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


def test_frozen_copy(string_1: CliffordString):
    with mock.patch('cliffordep.pauli_string_tools.CliffordString._canonicalize') as mock_canonicalize:
        frozen_copy = string_1.frozen_copy()
        mock_canonicalize.assert_called_once_with()
    assert isinstance(frozen_copy, FrozenCliffordString)
    assert frozen_copy.terms == frozenset({("X", 2), ("Y", -1j)})
    assert frozen_copy.denominator_squared == 5


def test_norm_squared(string_2: CliffordString):
    assert string_2.norm_squared == 2/1.5


def test_normalize(string_2: CliffordString):
    string_2.normalize()
    assert string_2.norm_squared == 1


class TestCanonicalize:

    @pytest.mark.parametrize("zero", (0, 0j, -0, -0+0j))
    def test_kill_empty_terms(self, terms: dict[str, complex], string_1: CliffordString, zero: complex):
        string_1.terms['Z'] = zero
        assert string_1.terms == defaultdict(complex, terms | {'Z': zero})
        string_1._canonicalize()
        assert string_1.terms == defaultdict(complex, terms)
        assert string_1.denominator_squared == 5

    @pytest.mark.parametrize("phase", (0, cmath.pi/2, cmath.pi, 3*cmath.pi/2, cmath.pi/4))
    def test_remove_global_phase(self, string_1: CliffordString, phase: float):
        string_1j = CliffordString({
            term: cmath.exp(1j * phase) * amplitude
            for term, amplitude in string_1.terms.items()
        })
        string_1j._canonicalize()
        assert string_1j.terms == string_1.terms
        assert string_1j.denominator_squared == string_1.denominator_squared

    @pytest.mark.parametrize("scalar", (1.5, 2))
    def test_scale(self, string_1: CliffordString, scalar: float):
        string_2 = string_1._scaled(scalar)
        string_2.denominator_squared *= scalar**2
        string_2._canonicalize()
        assert string_2.terms == string_1.terms
        assert string_2.denominator_squared == string_1.denominator_squared

    def test_zero(self):
        cs = CliffordString({"X": 0}, denominator_squared=2)
        cs._canonicalize()
        assert cs.terms == defaultdict(complex)
        assert cs.denominator_squared == 1

    def test_round(self, string_1: CliffordString):
        string_2 = CliffordString(
            {term: amplitude+_EPSILON for term, amplitude in string_1.terms.items()},
            string_1.denominator_squared+_EPSILON
        )
        assert string_1.terms != string_2.terms
        assert string_1.denominator_squared != string_2.denominator_squared
        string_2._canonicalize()
        assert string_1.terms == string_2.terms
        assert string_1.denominator_squared == string_2.denominator_squared


class TestPushThroughUnitary:

    def test_x(self, string_1: CliffordString):
        string_1.push_through_unitary(stim.CircuitInstruction('X', [0]))
        assert string_1.terms == defaultdict(complex, {"X": 2, "Y": 1j})
        assert string_1.denominator_squared == 5

    def test_y(self, string_1: CliffordString):
        string_1.push_through_unitary(stim.CircuitInstruction('Y', [0]))
        assert string_1.terms == defaultdict(complex, {"X": -2, "Y": -1j})
        assert string_1.denominator_squared == 5

    def test_z(self, string_1: CliffordString):
        string_1.push_through_unitary(stim.CircuitInstruction('Z', [0]))
        assert string_1.terms == defaultdict(complex, {"X": -2, "Y": 1j})
        assert string_1.denominator_squared == 5

    def test_h(self, string_1: CliffordString):
        string_1.push_through_unitary(stim.CircuitInstruction('H', [0]))
        assert string_1.terms == defaultdict(complex, {"Z": 2, "Y": 1j})
        assert string_1.denominator_squared == 5

    def test_s(self, string_1: CliffordString):
        string_1.push_through_unitary(stim.CircuitInstruction('S', [0]))
        assert string_1.terms == defaultdict(complex, {"X": 1j, "Y": 2})
        assert string_1.denominator_squared == 5

    def test_x_through_t(self):
        string = CliffordString({'X': 1})
        string.push_through_unitary(stim.CircuitInstruction('S', [0]), replace_s_with='T')
        assert dict(string.terms) == {"X": 1/SQRT2, "Y": 1/SQRT2}
        assert string.denominator_squared == 1

    def test_y_through_t(self):
        string = CliffordString({'Y': 1})
        string.push_through_unitary(stim.CircuitInstruction('S', [0]), replace_s_with='T')
        assert dict(string.terms) == {"X": -1/SQRT2, "Y": 1/SQRT2}
        assert string.denominator_squared == 1

    def test_z_through_t(self):
        string = CliffordString({'Z': 1})
        string.push_through_unitary(stim.CircuitInstruction('S', [0]), replace_s_with='T')
        assert dict(string.terms) == {"Z": 1}
        assert string.denominator_squared == 1

    def test_x_through_t_twice(self):
        string = CliffordString({'X': 1})
        for _ in range(2):
            string.push_through_unitary(stim.CircuitInstruction('S', [0]), replace_s_with='T')
        assert len(string.terms) == 2
        assert string.terms['X'] == 0
        assert abs(string.terms['Y'] - 1) < _EPSILON
        assert string.denominator_squared == 1

    def test_y_through_t_twice(self):
        string = CliffordString({'Y': 1})
        for _ in range(2):
            string.push_through_unitary(stim.CircuitInstruction('S', [0]), replace_s_with='T')
        assert len(string.terms) == 2
        assert abs(string.terms['X'] + 1) < _EPSILON
        assert string.terms['Y'] == 0
        assert string.denominator_squared == 1

    def test_z_through_t_twice(self):
        string = CliffordString({'Z': 1})
        for _ in range(2):
            string.push_through_unitary(stim.CircuitInstruction('S', [0]), replace_s_with='T')
        assert dict(string.terms) == {"Z": 1}
        assert string.denominator_squared == 1

    def test_s_replaced_with_t_1(self, string_1: CliffordString):
        string_1.push_through_unitary(stim.CircuitInstruction('S', [0]), replace_s_with='T')
        assert dict(string_1.terms) == {"X": (2+1j)/SQRT2, "Y": (2-1j)/SQRT2}
        assert string_1.denominator_squared == 5

    @pytest.mark.parametrize("replacement", ('T', 'Z'))
    def test_s_replaced_with_t_or_Z_unaffected(self, string_2: CliffordString, replacement):
        string_2.push_through_unitary(stim.CircuitInstruction('S', [0]), replace_s_with=replacement)
        assert dict(string_2.terms) == {"_": 1j, "Z": -1j}
        assert string_2.denominator_squared == 1.5

    def test_s_replaced_with_z(self, string_1: CliffordString):
        string_1.push_through_unitary(stim.CircuitInstruction('S', [0]), replace_s_with='Z')
        assert dict(string_1.terms) == {"X": -2, "Y": 1j}
        assert string_1.denominator_squared == 5

    def test_cnot(self, string_3: CliffordString):
        string_3.push_through_unitary(stim.CircuitInstruction('CX', [0, 1]))
        assert string_3.terms == defaultdict(complex, {"X_": 2, "XZ": 1j})
        assert string_3.denominator_squared == 5

    # @pytest.mark.parametrize("basis", ('X', 'Y', 'Z'))
    # def test_measure(self, string_3: CliffordString, basis: str):
    #     string_3.push_through_unitary(stim.CircuitInstruction(f'M{basis}', [0]))
    #     assert string_3.terms == defaultdict(complex, {"XX": 2, "YY": -1j})
    #     assert string_3.denominator_squared == 5


class TestPushThroughReset:

    BASES = ('X', 'Y', 'Z')

    @pytest.mark.parametrize("basis", BASES)
    def test_probability_preserved(self, string_2: CliffordString, basis: str):
        mixture = string_2.push_through_reset(stim.CircuitInstruction(f'R{basis}', [0]))
        assert mixture == defaultdict(float, {
            FrozenCliffordString(frozenset({("_", 1)}), 1): string_2.norm_squared,
        })
        assert string_2.norm_squared != 1

    @pytest.mark.parametrize("basis", BASES)
    @pytest.mark.parametrize("qubit,term_1,term_2", [(0, "_X", "_Y"), (1, "X_", "Y_")])
    def test_reset_entangled(
        self,
        string_3: CliffordString,
        basis: str,
        qubit: int,
        term_1: str,
        term_2: str,
    ):
        string_3.denominator_squared = 1.5
        mixture = string_3.push_through_reset(stim.CircuitInstruction(f'R{basis}', [qubit]))
        normed_result = {
            FrozenCliffordString(frozenset({(term_1, 1)}), 1): 4/5,
            FrozenCliffordString(frozenset({(term_2, 1)}), 1): 1/5,
        }
        assert mixture.keys() == normed_result.keys()
        for key, val in normed_result.items():
            assert (mixture[key] - val * string_3.norm_squared) < _EPSILON
        assert string_3.norm_squared != 1

    @pytest.mark.parametrize("basis", BASES)
    def test_all_reset(self, string_3: CliffordString, basis: str):
        mixture = string_3.push_through_reset(stim.CircuitInstruction(f'R{basis}', [0, 1]))
        assert mixture == defaultdict(float, {
            FrozenCliffordString(frozenset({("__", 1)}), 1): 1,
        })

    @pytest.mark.parametrize("basis", BASES)
    def test_reset_separable(self, string_4: CliffordString, basis: str):
        mixture = string_4.push_through_reset(stim.CircuitInstruction(f'R{basis}', [0]))
        assert mixture == defaultdict(float, {
            FrozenCliffordString(frozenset({("_X", 2), ("_Y", -1j)}), 5): string_4.norm_squared,
        })
        mixture = string_4.push_through_reset(stim.CircuitInstruction(f'R{basis}', [1]))
        assert mixture == defaultdict(float, {
            FrozenCliffordString(frozenset({("X_", 2), ("Y_", -1j)}), 5): string_4.norm_squared,
        })
        assert string_4.norm_squared != 1


class TestPushThroughMeasurement:

    BASES = ('X', 'Y', 'Z')

    # @pytest.mark.parametrize("basis", BASES)
    # def test_probability_preserved(self, string_2: CliffordString, basis: str):
    #     mixture = string_2.push_through_reset(stim.CircuitInstruction(f'R{basis}', [0]))
    #     assert mixture == defaultdict(float, {
    #         FrozenCliffordString(frozenset({("_", 1)}), 1): string_2.norm_squared,
    #     })
    #     assert string_2.norm_squared != 1

    # @pytest.mark.parametrize("basis", BASES)
    # @pytest.mark.parametrize("qubit,term_1,term_2", [(0, "_X", "_Y"), (1, "X_", "Y_")])
    # def test_reset_entangled(
    #     self,
    #     string_3: CliffordString,
    #     basis: str,
    #     qubit: int,
    #     term_1: str,
    #     term_2: str,
    # ):
    #     string_3.denominator_squared = 1.5
    #     mixture = string_3.push_through_reset(stim.CircuitInstruction(f'R{basis}', [qubit]))
    #     normed_result = {
    #         FrozenCliffordString(frozenset({(term_1, 1)}), 1): 4/5,
    #         FrozenCliffordString(frozenset({(term_2, 1)}), 1): 1/5,
    #     }
    #     assert mixture.keys() == normed_result.keys()
    #     for key, val in normed_result.items():
    #         assert (mixture[key] - val * string_3.norm_squared) < _EPSILON
    #     assert string_3.norm_squared != 1

    def test_all_measure_z(self, string_3: CliffordString):
        mixture = string_3.push_through_measurement(
            stim.CircuitInstruction('MZ', [0, 1], tag='0'),
            measurement_to_detectors={0: {0}, 1: {1}},
            syndrome_before_push=(False, False),
        )
        assert mixture == {(1, 1): (FrozenCliffordString(frozenset({("XX", 2), ("YY", -1j)}), 5), 1)}
        
    @pytest.mark.parametrize("basis", ('X', 'Y'))
    def test_all_measure_x_or_y(self, string_3: CliffordString, basis: str):
        mixture = string_3.push_through_measurement(
            stim.CircuitInstruction(f'M{basis}', [0, 1], tag='0'),
            measurement_to_detectors={0: {0}, 1: {1}},
            syndrome_before_push=(False, False),
        )
        assert mixture == {
            (int(basis=='Y'), int(basis=='Y')): (FrozenCliffordString(frozenset({("XX", 1)}), 1), 4/5),
            (int(basis=='X'), int(basis=='X')): (FrozenCliffordString(frozenset({("YY", 1)}), 1), 1/5),
        }

    # @pytest.mark.parametrize("basis", BASES)
    @pytest.mark.parametrize("index", (0, 1))
    def test_measure_separable_z(self, string_4: CliffordString, index: int):
        mixture = string_4.push_through_measurement(
            stim.CircuitInstruction(f'MZ', [index], tag='0'),
            measurement_to_detectors={0: {0}},
            syndrome_before_push=(False,),
        )
        expected_result = {(True,): ((FrozenCliffordString(frozenset({
                ("XX", 4),
                ("XY", -2j),
                ("YX", -2j),
                ("YY", -1),
            }), 25), string_4.norm_squared))}
        assert mixture.keys() == expected_result.keys()
        for key, (expected_frozen_string, expected_prob) in expected_result.items():
            frozen_string, prob = mixture[key]
            assert frozen_string == expected_frozen_string
            assert (prob - expected_prob) < _EPSILON
        assert string_4.norm_squared != 1