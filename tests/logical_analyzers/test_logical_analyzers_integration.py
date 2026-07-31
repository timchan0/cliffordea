import pytest

from cliffordep.logical_analyzers import CliffordLogicalAnalyzer, SuperpositionLogicalAnalyzer


@pytest.mark.parametrize("state", ['S', 'T'])
def test_superposition_vs_clifford(state, both_analyzers: tuple[
    SuperpositionLogicalAnalyzer,
    CliffordLogicalAnalyzer,
    list[dict[str, set[frozenset[int]]]],
]):
    """Check that both logical analyzer implementations keep the same strings."""
    superposition_analyzer, clifford_analyzer, configurations = both_analyzers
    superposition_kept_strings = superposition_analyzer.get_kept_strings(
        configurations=configurations,
        cultivated_state=state,
    )
    clifford_kept_strings = clifford_analyzer.get_kept_strings(
        configurations=configurations,
        cultivated_state=state,
    )
    for degree, _ in enumerate(configurations):
        for kept_string, superposition_triple in superposition_kept_strings[degree].items():
            assert kept_string in clifford_kept_strings[degree], f"{kept_string} missing in clifford_kept_strings[{degree}]"
            clifford_triple = clifford_kept_strings[degree][kept_string]
            assert superposition_triple == clifford_triple, f"Mismatch for {kept_string}: {superposition_triple} vs {clifford_triple}"
        for kept_string in clifford_kept_strings[degree]:
            assert kept_string in superposition_kept_strings[degree], f"{kept_string} missing in superposition_kept_strings[{degree}]"

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
