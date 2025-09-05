from unittest import mock

import pytest
import stim

from cliffordep.pauli_string_tools import CliffordString, FrozenCliffordString, SQRT2, _canonicalize, _PRECISION

_EPSILON = 1e-13


@pytest.fixture
def empty_frozen():
    return FrozenCliffordString(frozenset(), 1)


@pytest.fixture
def frozen_X():
    return FrozenCliffordString(frozenset({('X', 1)}), 1)

@pytest.fixture
def frozen_Y():
    return FrozenCliffordString(frozenset({('Y', 1)}), 1)

@pytest.fixture
def frozen_Z():
    return FrozenCliffordString(frozenset({('Z', 1)}), 1)

@pytest.fixture
def frozen_string_1(string_1: CliffordString):
    return string_1.frozen_copy()

@pytest.fixture
def frozen_string_2(string_2: CliffordString):
    return string_2.frozen_copy()

@pytest.fixture
def frozen_string_3(string_3: CliffordString):
    return string_3.frozen_copy()

@pytest.fixture
def frozen_string_4(string_4: CliffordString):
    return string_4.frozen_copy()


def test_not_equals(frozen_string_1: FrozenCliffordString, frozen_string_2: FrozenCliffordString):
    assert frozen_string_1 != frozen_string_2


class TestMul:

    _scaled = 'cliffordep.pauli_string_tools.FrozenCliffordString._scaled'
    _multiply = 'cliffordep.pauli_string_tools.FrozenCliffordString._multiply'

    def test_int(self, frozen_string_1: FrozenCliffordString):
        with mock.patch(self._scaled) as mock_scaled:
            _ = frozen_string_1 * 2
            mock_scaled.assert_called_once_with(2)

    def test_float(self, frozen_string_1: FrozenCliffordString):
        with mock.patch(self._scaled) as mock_scaled:
            _ = frozen_string_1 * 1.5
            mock_scaled.assert_called_once_with(1.5)

    def test_clifford_string(self, frozen_string_1: FrozenCliffordString, frozen_string_2: FrozenCliffordString):
        with mock.patch(self._multiply) as mock_multiply:
            _ = frozen_string_1 * frozen_string_2
            mock_multiply.assert_called_once_with(frozen_string_1, frozen_string_2)


class TestScaled:

    def test_int(self, frozen_string_1: FrozenCliffordString):
        cs = frozen_string_1._scaled(2)
        assert set(cs.terms) == {("X", 2), ("Y", -1j)}
        assert cs.denominator_squared == round(frozen_string_1.denominator_squared / 2**2, _PRECISION)

    def test_float(self, frozen_string_1: FrozenCliffordString):
        cs = frozen_string_1._scaled(1.5)
        assert set(cs.terms) == {("X", 2), ("Y", -1j)}
        assert cs.denominator_squared == round(frozen_string_1.denominator_squared / 1.5**2, _PRECISION)


class TestMultiply:

    def test_both_empty(self, empty_frozen: FrozenCliffordString):
        squared = FrozenCliffordString._multiply(empty_frozen, empty_frozen)
        assert squared.terms == frozenset()
        assert squared.denominator_squared == 1

    def test_left_empty(self, empty_frozen: FrozenCliffordString, frozen_string_1: FrozenCliffordString):
        cs = FrozenCliffordString._multiply(empty_frozen, frozen_string_1)
        assert cs.terms == frozenset()
        assert cs.denominator_squared == 1

    def test_right_empty(self, empty_frozen: FrozenCliffordString, frozen_string_1: FrozenCliffordString):
        cs = FrozenCliffordString._multiply(frozen_string_1, empty_frozen)
        assert cs.terms == frozenset()
        assert cs.denominator_squared == 1

    def test_square(self, frozen_string_1: FrozenCliffordString):
        cs = FrozenCliffordString._multiply(frozen_string_1, frozen_string_1)
        assert cs.terms == frozenset({("_", 1)})
        assert cs.denominator_squared == round(25/9, _PRECISION)

    def test_different(self, frozen_string_1: FrozenCliffordString, frozen_string_2: FrozenCliffordString):
        cs = FrozenCliffordString._multiply(frozen_string_1, frozen_string_2)
        assert set(cs.terms) == {("X", 1), ("Y", 1j)}
        assert cs.denominator_squared == frozen_string_1.denominator_squared * frozen_string_2.denominator_squared

    def test_annihilate(self):
        s1 = FrozenCliffordString(frozenset({('XX', 1), ('YY', 1)}), 2)
        s2 = FrozenCliffordString(frozenset({('XX', 1), ('YY', -1)}), 2)
        cs = FrozenCliffordString._multiply(s1, s2)
        assert set(cs.terms) == frozenset()
        assert cs.denominator_squared == 1


class TestPushThroughUnitary:

    def test_x(self, frozen_string_1: FrozenCliffordString):
        pushed = frozen_string_1.push_through_unitary(stim.CircuitInstruction('X', [0]))
        assert set(pushed.terms) == {("X", 2), ("Y", 1j)}
        assert pushed.denominator_squared == 5

    def test_y(self, frozen_string_1: FrozenCliffordString):
        pushed = frozen_string_1.push_through_unitary(stim.CircuitInstruction('Y', [0]))
        assert set(pushed.terms) == {("X", 2), ("Y", 1j)}
        assert pushed.denominator_squared == 5

    def test_z(self, frozen_string_1: FrozenCliffordString):
        pushed = frozen_string_1.push_through_unitary(stim.CircuitInstruction('Z', [0]))
        assert set(pushed.terms) == {("X", 2), ("Y", -1j)}
        assert pushed.denominator_squared == 5

    def test_h(self, frozen_string_1: FrozenCliffordString):
        pushed = frozen_string_1.push_through_unitary(stim.CircuitInstruction('H', [0]))
        assert set(pushed.terms) == {("Z", -2j), ("Y", 1)}
        assert pushed.denominator_squared == 5

    def test_s(self, frozen_string_1: FrozenCliffordString):
        pushed = frozen_string_1.push_through_unitary(stim.CircuitInstruction('S', [0]))
        assert set(pushed.terms) == {("X", 1), ("Y", -2j)}
        assert pushed.denominator_squared == 5

    def test_x_through_t(self, frozen_X: FrozenCliffordString):
        pushed = frozen_X.push_through_unitary(stim.CircuitInstruction('S', [0]), replace_s_with='T')
        assert set(pushed.terms) == {("X", 1), ("Y", 1)}
        assert pushed.denominator_squared == 2

    def test_y_through_t(self, frozen_Y: FrozenCliffordString):
        pushed = frozen_Y.push_through_unitary(stim.CircuitInstruction('S', [0]), replace_s_with='T')
        assert set(pushed.terms) == {("X", 1), ("Y", -1)}
        assert pushed.denominator_squared == 2

    def test_z_through_t(self, frozen_Z: FrozenCliffordString):
        pushed = frozen_Z.push_through_unitary(stim.CircuitInstruction('S', [0]), replace_s_with='T')
        assert set(pushed.terms) == {("Z", 1)}
        assert pushed.denominator_squared == 1

    def test_x_through_t_twice(self, frozen_X: FrozenCliffordString):
        pushed = frozen_X.push_through_unitary(stim.CircuitInstruction('S', [0]), replace_s_with='T')
        pushed = pushed.push_through_unitary(stim.CircuitInstruction('S', [0]), replace_s_with='T')
        assert set(pushed.terms) == {("Y", 1)}
        assert pushed.denominator_squared == 1

    def test_y_through_t_twice(self, frozen_Y: FrozenCliffordString):
        pushed = frozen_Y.push_through_unitary(stim.CircuitInstruction('S', [0]), replace_s_with='T')
        pushed = pushed.push_through_unitary(stim.CircuitInstruction('S', [0]), replace_s_with='T')
        assert set(pushed.terms) == {("X", 1)}
        assert pushed.denominator_squared == 1

    def test_z_through_t_twice(self, frozen_Z: FrozenCliffordString):
        pushed = frozen_Z.push_through_unitary(stim.CircuitInstruction('S', [0]), replace_s_with='T')
        pushed = pushed.push_through_unitary(stim.CircuitInstruction('S', [0]), replace_s_with='T')
        assert dict(pushed.terms) == {"Z": 1}
        assert pushed.denominator_squared == 1

    def test_s_replaced_with_t_1(self, frozen_string_1: FrozenCliffordString):
        pushed = frozen_string_1.push_through_unitary(stim.CircuitInstruction('S', [0]), replace_s_with='T')
        canonicalized_terms, canonicalized_denominator_squared = _canonicalize(
            {"X": (2+1j)/SQRT2, "Y": (2-1j)/SQRT2}, 5)
        assert dict(pushed.terms) == canonicalized_terms
        assert pushed.denominator_squared == canonicalized_denominator_squared

    @pytest.mark.parametrize("replacement", ('T', 'Z'))
    def test_s_replaced_with_t_or_Z_unaffected(self, frozen_string_2: FrozenCliffordString, replacement):
        pushed = frozen_string_2.push_through_unitary(stim.CircuitInstruction('S', [0]), replace_s_with=replacement)
        assert pushed == FrozenCliffordString(frozenset({("Z", 1), ("_", -1)}), 1.5)

    def test_s_replaced_with_z(self, frozen_string_1: FrozenCliffordString):
        pushed = frozen_string_1.push_through_unitary(stim.CircuitInstruction('S', [0]), replace_s_with='Z')
        assert pushed == FrozenCliffordString(frozenset({("X", 2), ("Y", -1j)}), 5)

    def test_cnot(self, frozen_string_3: FrozenCliffordString):
        pushed = frozen_string_3.push_through_unitary(stim.CircuitInstruction('CX', [0, 1]))
        assert pushed == FrozenCliffordString(frozenset({("X_", -2j), ("XZ", 1)}), 5)

    # @pytest.mark.parametrize("basis", ('X', 'Y', 'Z'))
    # def test_measure(self, string_3: CliffordString, basis: str):
    #     string_3.push_through_unitary(stim.CircuitInstruction(f'M{basis}', [0]))
    #     assert string_3.terms == defaultdict(complex, {"XX": 2, "YY": -1j})
    #     assert string_3.denominator_squared == 5


class TestPushThroughReset:

    BASES = ('X', 'Y', 'Z')

    @pytest.mark.parametrize("basis", BASES)
    def test_probability_preserved(self, frozen_string_2: FrozenCliffordString, basis: str):
        mixture = frozen_string_2.push_through_reset(stim.CircuitInstruction(f'R{basis}', [0]))
        assert dict(mixture) == {
            FrozenCliffordString(frozenset({("_", 1)}), 1): frozen_string_2.norm_squared}
        assert frozen_string_2.norm_squared != 1

    @pytest.mark.parametrize("basis", BASES)
    @pytest.mark.parametrize("qubit,term_1,term_2", [(0, "_X", "_Y"), (1, "X_", "Y_")])
    def test_reset_entangled(
        self,
        basis: str,
        qubit: int,
        term_1: str,
        term_2: str,
    ):
        fcs = FrozenCliffordString(frozenset({("XX", 2), ("YY", -1j)}), 1.5)
        mixture = fcs.push_through_reset(stim.CircuitInstruction(f'R{basis}', [qubit]))
        normed_result = {
            FrozenCliffordString(frozenset({(term_1, 1)}), 1): 4/5,
            FrozenCliffordString(frozenset({(term_2, 1)}), 1): 1/5,
        }
        assert mixture.keys() == normed_result.keys()
        for key, val in normed_result.items():
            assert (mixture[key] - val * fcs.norm_squared) < _EPSILON
        assert fcs.norm_squared != 1

    @pytest.mark.parametrize("basis", BASES)
    def test_all_reset(self, frozen_string_3: FrozenCliffordString, basis: str):
        mixture = frozen_string_3.push_through_reset(stim.CircuitInstruction(f'R{basis}', [0, 1]))
        assert dict(mixture) == {FrozenCliffordString(frozenset({("__", 1)}), 1): 1}

    @pytest.mark.parametrize("basis", BASES)
    def test_reset_separable(self, frozen_string_4: FrozenCliffordString, basis: str):
        mixture = frozen_string_4.push_through_reset(stim.CircuitInstruction(f'R{basis}', [0]))
        assert dict(mixture) == {
            FrozenCliffordString(frozenset({("_X", 2), ("_Y", -1j)}), 5): frozen_string_4.norm_squared}
        mixture = frozen_string_4.push_through_reset(stim.CircuitInstruction(f'R{basis}', [1]))
        assert dict(mixture) == {
            FrozenCliffordString(frozenset({("X_", 2), ("Y_", -1j)}), 5): frozen_string_4.norm_squared}
        assert frozen_string_4.norm_squared != 1


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

    def test_all_measure_z(self, frozen_string_3: FrozenCliffordString):
        mixture = frozen_string_3.push_through_measurement(
            stim.CircuitInstruction('MZ', [0, 1], tag='0'),
            measurement_to_detectors={0: {0}, 1: {1}},
            syndrome_before_push=(False, False),
        )
        assert mixture == {(1, 1): (FrozenCliffordString(frozenset({("XX", 2), ("YY", -1j)}), 5), 1)}
        
    @pytest.mark.parametrize("basis", ('X', 'Y'))
    def test_all_measure_x_or_y(self, frozen_string_3: FrozenCliffordString, basis: str):
        mixture = frozen_string_3.push_through_measurement(
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
    def test_measure_separable_z(self, frozen_string_4: FrozenCliffordString, index: int):
        mixture = frozen_string_4.push_through_measurement(
            stim.CircuitInstruction(f'MZ', [index], tag='0'),
            measurement_to_detectors={0: {0}},
            syndrome_before_push=(False,),
        )
        expected_result = {(True,): ((FrozenCliffordString(frozenset({
                ("XX", 4),
                ("XY", -2j),
                ("YX", -2j),
                ("YY", -1),
            }), 25), frozen_string_4.norm_squared))}
        assert mixture.keys() == expected_result.keys()
        for key, (expected_frozen_string, expected_prob) in expected_result.items():
            frozen_string, prob = mixture[key]
            assert frozen_string == expected_frozen_string
            assert (prob - expected_prob) < _EPSILON
        assert frozen_string_4.norm_squared != 1