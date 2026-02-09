import numpy as np

import pytest

from cliffordep.pauli_trace import brute_force_trace_symplectic, trace_of_projector_product_symplectic


def test_small_random():
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
            tab, _, _ = brute_force_trace_symplectic(px, pz, signs, qubit_count)
            tdet = trace_of_projector_product_symplectic(px, pz, signs, qubit_count, mode='deterministic')
            assert tab == tdet
            tauto = trace_of_projector_product_symplectic(px, pz, signs, qubit_count, enum_threshold=10, mode='auto')
            assert tab == tauto


def test_four_yys():
    y_count = 2
    tdet = trace_of_projector_product_symplectic(
        px=[np.array([1, 1]) for _ in range(y_count)],
        pz=[np.array([1, 1]) for _ in range(y_count)],
        signs=[0 for _ in range(y_count)],
        qubit_count=2,
        mode='deterministic',
    )
    assert tdet == 2


def test_commuting_case():
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
    t = trace_of_projector_product_symplectic(px, pz, signs, qubit_count, mode='deterministic')
    assert t == 2 ** qubit_count // (2 ** rank)


def test_invalid_inputs():
    try:
        _ = trace_of_projector_product_symplectic([np.array([0, 1])], [np.array([0])], [0], qubit_count=2)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass


def test_x_z():
    qubit_count = 1
    px = [np.array([1]), np.array([0])]
    pz = [np.array([0]), np.array([1])]
    signs = [0, 0]
    t = trace_of_projector_product_symplectic(px, pz, signs, qubit_count)
    assert t == 1/2


def test_repeated_paulis():
    qubit_count = 2
    px = [np.array([1, 0]), np.array([1, 0])]
    pz = [np.array([0, 0]), np.array([0, 0])]
    signs = [0, 0]
    t = trace_of_projector_product_symplectic(px, pz, signs, qubit_count)
    assert t == 2


@pytest.mark.parametrize("qubit_count", range(1, 5))
def test_minus_identity(qubit_count):
    px = [np.zeros(qubit_count, dtype=int)]
    pz = [np.zeros(qubit_count, dtype=int)]
    signs = [1]
    t = trace_of_projector_product_symplectic(px, pz, signs, qubit_count)
    assert t == 0

def test_xzxz():
    px = [np.array([k]) for k in (1, 0, 1, 0)]
    pz = [np.array([k]) for k in (0, 1, 0, 1)]
    signs = [0, 0, 0, 0]
    t = trace_of_projector_product_symplectic(px, pz, signs, qubit_count=1)
    assert t == 1/4