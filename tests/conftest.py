import pytest
import stim

import cliffordep
from cliffordep.noisy_circuit_tools import CultivationCircuit
from cliffordep.combinators import ErrorEventCombinator


@pytest.fixture
def noisy_d3_double_cat_check_circuit():
    """Noisy version of the distance-3 double cat check circuit."""
    circuit = cliffordep.circuits.D3DoubleCatCheckA6.INNER_CIRCUIT
    noisy_circuit = cliffordep.noise.uniformly_depolarize(circuit, noise_level=1e-3)
    # remove last layer of depolarizing noise
    noisy_circuit = noisy_circuit[:-1]
    return noisy_circuit


@pytest.fixture
def noisy_d3_double_cat_check(noisy_d3_double_cat_check_circuit: stim.Circuit):
    """Noisy version of the distance-3 double cat check circuit."""
    d3_double_cat_check = cliffordep.circuits.D3DoubleCatCheckA6()
    return cliffordep.CultivationCircuit(
        noisy_d3_double_cat_check_circuit,
        stabilizer_generators=d3_double_cat_check.STABILIZER_GENERATORS,
        logical_x=d3_double_cat_check.LOGICAL_X,
        logical_z=d3_double_cat_check.LOGICAL_Z,
    )


@pytest.fixture
def d3_double_cat_check_brute(noisy_d3_double_cat_check: CultivationCircuit) -> ErrorEventCombinator:
    """Exclusive fault brute-force combinator for the distance-3 double cat check circuit."""
    return ErrorEventCombinator(noisy_d3_double_cat_check)