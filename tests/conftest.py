import pytest
import stim

import cliffordep
from cliffordep.combinators import ErrorEventCombinator


@pytest.fixture
def noisy_d3_double_cat_check_circuit():
    """Noisy version of the distance-3 double cat check circuit."""
    circuit = cliffordep.circuits.D3DoubleCatCheckA6.INNER_CIRCUIT
    noisy_circuit = cliffordep.noise.uniformly_depolarize(circuit, noise_level=1e-3)
    return noisy_circuit


@pytest.fixture
def noisy_d3_double_cat_check(noisy_d3_double_cat_check_circuit: stim.Circuit):
    """Noisy version of the distance-3 double cat check circuit."""
    return cliffordep.CultivationCircuit(noisy_d3_double_cat_check_circuit)


@pytest.fixture
def d3_double_cat_check_brute(noisy_d3_double_cat_check_circuit: stim.Circuit) -> ErrorEventCombinator:
    """Exclusive fault brute-force combinator for the distance-3 double cat check circuit."""
    return ErrorEventCombinator(noisy_d3_double_cat_check_circuit)