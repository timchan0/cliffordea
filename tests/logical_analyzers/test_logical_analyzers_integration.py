import pytest

from cliffordep.combinators import FaultCombinator
from cliffordep.logical_analyzers import SuperpositionLogicalAnalyzer, TableauLogicalAnalyzer
from cliffordep import noise
from cliffordep import circuits


@pytest.fixture
def both_analyzers():
    NOISE_LEVEL = 1e-3
    max_order = 4
    circuit = circuits.D3DoubleCatCheckA6()
    noisy_circuit = noise.uniformly_depolarize(
        circuit.INNER_CIRCUIT,
        noise_level=NOISE_LEVEL,
    )
    combinator = FaultCombinator(noisy_circuit=noisy_circuit)
    superposition_analyzer = SuperpositionLogicalAnalyzer(
        data_indices=circuit.DATA_INDICES,
        stabilizer_generators=circuit.STABILIZER_GENERATORS_RESTRICTED,
        logical_s=circuit.LOGICAL_S,
    )
    tableau_analyzer = TableauLogicalAnalyzer(
        data_indices=circuit.DATA_INDICES,
        stabilizer_generators=circuit.STABILIZER_GENERATORS_RESTRICTED,
        logical_s=circuit.LOGICAL_S,
        # mode='brute',
    )
    configurations = combinator.get_undetected_configurations(max_order=max_order)
    return superposition_analyzer, tableau_analyzer, configurations


@pytest.mark.parametrize("state", ['S', 'T'])
def test_superposition_vs_tableau(state, both_analyzers: tuple[
    SuperpositionLogicalAnalyzer,
    TableauLogicalAnalyzer,
    list[dict[str, set[frozenset[int]]]],
]):
    superposition_analyzer, tableau_analyzer, configurations = both_analyzers
    kept_strings_1 = superposition_analyzer.get_kept_strings(
        configurations=configurations,
        cultivated_state=state,
    )
    kept_strings_2 = tableau_analyzer.get_kept_strings(
        configurations=configurations,
        cultivated_state=state,
    )
    for degree, _ in enumerate(configurations):
        for kept_string, logical_triple_1 in kept_strings_1[degree].items():
            assert kept_string in kept_strings_2[degree], f"{kept_string} missing in kept_strings_2[{degree}]"
            logical_triple_2 = kept_strings_2[degree][kept_string]
            assert logical_triple_1 == logical_triple_2, f"Mismatch for {kept_string}: {logical_triple_1} vs {logical_triple_2}"
        for kept_string in kept_strings_2[degree]:
            assert kept_string in kept_strings_1[degree], f"{kept_string} missing in kept_strings_1[{degree}]"

    # Correct result:
    # S state cultivation: for order...
    #     0, 1.0 (0.0) errors are kept and lead to identity (error).
    #     1, 2.0 (0.0) errors are kept and lead to identity (error).
    #     2, 6.0 (0.0) errors are kept and lead to identity (error).
    #     3, 16.0 (24.0) errors are kept and lead to identity (error).
    #     4, 46.0 (64.0) errors are kept and lead to identity (error).
    # T state cultivation: for order...
    #     0, 1.0 (0.0) errors are kept and lead to identity (error).
    #     1, 2.0 (0.0) errors are kept and lead to identity (error).
    #     2, 2.75 (1.0) errors are kept and lead to identity (error).
    #     3, 18.5 (23.75) errors are kept and lead to identity (error).
    #     4, 55.75 (55.5) errors are kept and lead to identity (error).