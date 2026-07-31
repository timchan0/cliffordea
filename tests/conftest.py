import pytest
import stim

import cliffordep
from cliffordep.combinators import ErrorEventCombinator, FaultCombinator
from cliffordep.logical_analyzers import CliffordLogicalAnalyzer, SuperpositionLogicalAnalyzer


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


@pytest.fixture(scope='session')
def both_analyzers():
    """Build both logical analyzers and enumerate the shared distance-3 configurations once."""
    noise_level = 1e-3
    max_order = 4
    circuit = cliffordep.circuits.D3DoubleCatCheckA6()
    noisy_circuit = cliffordep.noise.uniformly_depolarize(
        circuit.INNER_CIRCUIT,
        noise_level=noise_level,
    )
    combinator = FaultCombinator(noisy_circuit=noisy_circuit)
    superposition_analyzer = SuperpositionLogicalAnalyzer(
        data_indices=circuit.DATA_INDICES,
        stabilizer_generators=circuit.STABILIZER_GENERATORS_RESTRICTED,
        logical_s=circuit.LOGICAL_S,
    )
    clifford_analyzer = CliffordLogicalAnalyzer(
        data_indices=circuit.DATA_INDICES,
        stabilizer_generators=circuit.STABILIZER_GENERATORS_RESTRICTED,
        logical_s=circuit.LOGICAL_S,
    )
    configurations = combinator.get_undetected_configurations(max_order=max_order)
    return superposition_analyzer, clifford_analyzer, configurations
