import numpy as np

import pytest

from cliffordep.pauli_trace import _character_sum, _nonnegative_character_sum, brute_force_project_product_trace, projector_product_trace


@pytest.mark.parametrize("assume_nonnegative", [False, True])
def test_small_random(assume_nonnegative: bool):
    import random
    random.seed(42)
    for qubit_count in (1, 2, 3, 4):
        for _ in range(50):
            pauli_count = random.randint(1, min(7, 2 * qubit_count + 3))
            px = []
            pz = []
            signs = []
            for _ in range(pauli_count):
                xv = [random.randint(0, 1) for _ in range(qubit_count)]
                zv = [random.randint(0, 1) for _ in range(qubit_count)]
                
                # ensure an even number of Y Pauli tensor factors
                y_indices = [i for i in range(qubit_count) if xv[i] == 1 and zv[i] == 1]
                if len(y_indices) % 2 == 1:
                    flip_index = random.choice(y_indices)
                    xv[flip_index] = 0
                
                px.append(xv)
                pz.append(zv)
                signs.append(random.randint(0, 1))
            tab, _, _ = brute_force_project_product_trace(px, pz, signs)
            tdet = projector_product_trace(px, pz, signs, mode='deterministic', assume_nonnegative=assume_nonnegative)
            if assume_nonnegative:
                assert abs(tab) == tdet
            else:
                assert tab == tdet
            tbrute = projector_product_trace(px, pz, signs, mode='brute', assume_nonnegative=assume_nonnegative)
            assert tab == tbrute


@pytest.mark.parametrize("assume_nonnegative", [False, True])
def test_four_yys(assume_nonnegative: bool):
    qubit_count = 2
    tdet = projector_product_trace(
        px=[np.array([1, 1]) for _ in range(qubit_count)],
        pz=[np.array([1, 1]) for _ in range(qubit_count)],
        signs=[0 for _ in range(qubit_count)],
        mode='deterministic',
        assume_nonnegative=assume_nonnegative,
    )
    assert tdet == 2


@pytest.mark.parametrize("assume_nonnegative", [False, True])
def test_commuting_case(assume_nonnegative: bool):
    qubit_count = 5
    rank = 3
    px = []
    pz = []
    signs = []
    for i in range(rank):
        x = [0] * qubit_count
        z = [0] * qubit_count
        z[i] = 1
        px.append(x)
        pz.append(z)
        signs.append(0)
    px.extend(px[:2])
    pz.extend(pz[:2])
    signs.extend([0, 0])
    t = projector_product_trace(px, pz, signs, mode='deterministic', assume_nonnegative=assume_nonnegative)
    assert t == 2 ** qubit_count // (2 ** rank)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        _ = projector_product_trace([np.array([0, 1])], [np.array([0])], [0])


@pytest.mark.parametrize("assume_nonnegative", [False, True])
def test_x_z(assume_nonnegative: bool):
    px = [np.array([1]), np.array([0])]
    pz = [np.array([0]), np.array([1])]
    signs = [0, 0]
    t = projector_product_trace(px, pz, signs, assume_nonnegative=assume_nonnegative)
    assert t == 1/2


@pytest.mark.parametrize("assume_nonnegative", [False, True])
def test_repeated_paulis(assume_nonnegative: bool):
    px = [np.array([1, 0]), np.array([1, 0])]
    pz = [np.array([0, 0]), np.array([0, 0])]
    signs = [0, 0]
    t = projector_product_trace(px, pz, signs, assume_nonnegative=assume_nonnegative)
    assert t == 2


def test_assume_nonnegative_flag_path():
    # Choose px=0 so pairing_matrix is zero => cross_C = 0.
    # Choose signs=0 so linear = 0.
    # Then the quadratic character sum is S = 2^k >= 0, so the opt-in fast path is valid.
    px = [np.array([0]), np.array([0])]
    pz = [np.array([1]), np.array([1])]
    signs = [0, 0]

    t_default = projector_product_trace(
        px, pz, signs, mode='deterministic')
    t_fast = projector_product_trace(
        px, pz, signs, mode='deterministic', assume_nonnegative=True)
    assert t_default == t_fast
    assert t_fast == 1


@pytest.mark.parametrize("qubit_count", range(1, 5))
@pytest.mark.parametrize("assume_nonnegative", [False, True])
def test_minus_identity(qubit_count, assume_nonnegative: bool):
    px = [np.zeros(qubit_count, dtype=int)]
    pz = [np.zeros(qubit_count, dtype=int)]
    signs = [1]
    t = projector_product_trace(px, pz, signs, assume_nonnegative=assume_nonnegative)
    assert t == 0

@pytest.mark.parametrize("assume_nonnegative", [False, True])
def test_xzxz(assume_nonnegative: bool):
    px = [np.array([k]) for k in (1, 0, 1, 0)]
    pz = [np.array([k]) for k in (0, 1, 0, 1)]
    signs = [0, 0, 0, 0]
    t = projector_product_trace(px, pz, signs, assume_nonnegative=assume_nonnegative)
    assert t == 1/4


class TestCharacterSumQuadraticGF2Alternating:

    def test_elimination_matches_abs_congruence_reduction(self):
        rng = np.random.default_rng(123)
        for k in [0, 1, 2, 3, 4, 5, 8, 12]:
            for _ in range(200):
                # Random hollow symmetric matrix A.
                upper = rng.integers(0, 2, size=(k, k), dtype=np.uint8)
                upper = np.triu(upper, k=1)
                a = upper ^ upper.T
                linear = rng.integers(0, 2, size=(k,), dtype=np.uint8)

                s_full = _character_sum(a, linear)
                s_nonneg = _nonnegative_character_sum(a, linear)
                assert s_nonneg == abs(s_full)