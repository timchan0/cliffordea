import pytest
import stim

import cliffordea
from cliffordea.enum.combinators import ErrorEventCombinator, FaultCombinator
from cliffordea.accept.logical_analyzers import CliffordLogicalAnalyzer, SuperpositionLogicalAnalyzer


@pytest.fixture
def noisy_d3_double_cat_check_circuit():
    """Noisy version of the distance-3 double-check circuit."""
    circuit = cliffordea.enum.circuits.DoubleCheck(ancilla_count=6).INNER_CIRCUIT
    noisy_circuit = cliffordea.enum.noise.uniformly_depolarize(circuit, noise_level=1e-3)
    return noisy_circuit


@pytest.fixture
def noisy_d3_double_cat_check(noisy_d3_double_cat_check_circuit: stim.Circuit):
    """Noisy version of the distance-3 double-check circuit."""
    return cliffordea.enum.CultivationCircuit(noisy_d3_double_cat_check_circuit)


@pytest.fixture
def d3_double_cat_check_brute(noisy_d3_double_cat_check_circuit: stim.Circuit) -> ErrorEventCombinator:
    """Disjoint-event brute-force combinator for the distance-3 double check."""
    return ErrorEventCombinator(noisy_d3_double_cat_check_circuit)


@pytest.fixture(scope='session')
def d3_combinator_and_kept_effects_by_analyzer():
    """Build shared distance-3 kept-effect regression results.

    The superposition analyzer is evaluated through three faults for comparison
    with the Clifford analyzer. The Clifford analyzer is evaluated through
    four faults so the same session-scoped result can support the frozen
    higher-fault-count regression.

    :return: The fault combinator, the superposition-analyzer results, and the
        Clifford-analyzer results.
    """
    noise_level = 1e-3
    circuit = cliffordea.enum.circuits.DoubleCheck(ancilla_count=6)
    noisy_circuit = cliffordea.enum.noise.uniformly_depolarize(
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
    superposition_kept_effects = combinator.get_kept_effects(
        logical_analyzer=superposition_analyzer,
        max_fault_count=3,
        cultivated_states=('S', 'T'),
    )
    clifford_kept_effects = combinator.get_kept_effects(
        logical_analyzer=clifford_analyzer,
        max_fault_count=4,
        cultivated_states=('S', 'T'),
    )
    return combinator, superposition_kept_effects, clifford_kept_effects
