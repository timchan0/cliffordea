from typing import Literal

import numpy as np
import pytest

from cliffordep.pauli_trace._assumers import NoAssumptions, Nonnegative
from cliffordep.pauli_trace.main import brute_force_projector_product_trace, ProjectorProductTracer


@pytest.fixture(params=[False, True])
def deterministic_tracer(request):
    return ProjectorProductTracer(assume_nonnegative=request.param)

@pytest.fixture(params=[False, True])
def brute_tracer(request):
    return ProjectorProductTracer(mode='brute', assume_nonnegative=request.param)


def test_small_random(deterministic_tracer: ProjectorProductTracer, brute_tracer: ProjectorProductTracer):
    import random
    random.seed(42)
    for qubit_count in (1, 2, 3, 4):
        for _ in range(50):
            pauli_count = random.randint(1, min(7, 2 * qubit_count + 3))
            px = []
            pz = []
            j_powers: list[Literal[0, 1, 2, 3]] = []
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
                j_powers.append(random.choice((0, 2)))
            tab, _, _ = brute_force_projector_product_trace(px, pz, j_powers)
            tdet = deterministic_tracer.trace(px, pz, j_powers)
            if deterministic_tracer.ASSUME_NONNEGATIVE:
                assert abs(tab) == tdet
            else:
                assert tab == tdet
            tbrute = brute_tracer.trace(px, pz, j_powers)
            assert tab == tbrute


def test_four_yys(deterministic_tracer: ProjectorProductTracer):
    qubit_count = 2
    tdet = deterministic_tracer.trace(
        px=[np.array([1, 1]) for _ in range(qubit_count)],
        pz=[np.array([1, 1]) for _ in range(qubit_count)],
        j_powers=[0 for _ in range(qubit_count)],
    )
    assert tdet == 2


def test_commuting_case(deterministic_tracer: ProjectorProductTracer):
    qubit_count = 5
    rank = 3
    px = []
    pz = []
    j_powers: list[Literal[0, 1, 2, 3]] = []
    for i in range(rank):
        x = [0] * qubit_count
        z = [0] * qubit_count
        z[i] = 1
        px.append(x)
        pz.append(z)
        j_powers.append(0)
    px.extend(px[:2])
    pz.extend(pz[:2])
    j_powers.extend([0, 0])
    t = deterministic_tracer.trace(px, pz, j_powers)
    assert t == 2 ** qubit_count // (2 ** rank)


def test_invalid_inputs(deterministic_tracer: ProjectorProductTracer):
    with pytest.raises(ValueError):
        _ = deterministic_tracer.trace([np.array([0, 1])], [np.array([0])], [0])


def test_x_z(deterministic_tracer: ProjectorProductTracer):
    px = [np.array([1]), np.array([0])]
    pz = [np.array([0]), np.array([1])]
    j_powers = (0, 0)
    t = deterministic_tracer.trace(px, pz, j_powers)
    assert t == 1/2


def test_repeated_paulis(deterministic_tracer: ProjectorProductTracer):
    px = [np.array([1, 0]), np.array([1, 0])]
    pz = [np.array([0, 0]), np.array([0, 0])]
    j_powers = (0, 0)
    t = deterministic_tracer.trace(px, pz, j_powers)
    assert t == 2


def test_assume_nonnegative_flag_path(deterministic_tracer: ProjectorProductTracer):
    # Choose px=0 so pairing_matrix is zero => cross_C = 0.
    # Choose j_power=0 so linear = 0.
    # Then the quadratic character sum is S = 2^k >= 0, so the opt-in fast path is valid.
    px = [np.array([0]), np.array([0])]
    pz = [np.array([1]), np.array([1])]
    j_power = (0, 0)
    trace = deterministic_tracer.trace(px, pz, j_power)
    assert trace == 1


@pytest.mark.parametrize("qubit_count", range(1, 5))
def test_minus_identity(qubit_count, deterministic_tracer: ProjectorProductTracer):
    px = [np.zeros(qubit_count, dtype=int)]
    pz = [np.zeros(qubit_count, dtype=int)]
    j_powers = (2,)
    t = deterministic_tracer.trace(px, pz, j_powers)
    assert t == 0

def test_xzxz(deterministic_tracer: ProjectorProductTracer):
    px = [np.array([k]) for k in (1, 0, 1, 0)]
    pz = [np.array([k]) for k in (0, 1, 0, 1)]
    j_powers = (0, 0, 0, 0)
    t = deterministic_tracer.trace(px, pz, j_powers)
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

                s_full = NoAssumptions.sum(linear, a)
                s_nonneg = Nonnegative.sum(linear, a)
                assert s_nonneg == abs(s_full)