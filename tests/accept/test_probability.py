import stim

from cliffordea.accept import trivial_syndrome_probability
from cliffordea.accept.logical_analyzers import (
    CliffordLogicalAnalyzer,
    LOGICAL_COEFFICIENTS,
)
from cliffordea.accept.pauli import pauli_mask
from cliffordea.enum import circuits


def test_identity_error_is_always_accepted():
    """An identity error preserves every stabilizer and is accepted with certainty."""
    encoder = stim.Tableau(3)

    assert trivial_syndrome_probability(
        encoder,
        stim.Tableau(3),
        logical_qubit_count=2,
        logical_coefficients={0: 1.0},
    ) == 1.0


def test_two_logical_qubits_use_the_declared_stabilizer_split():
    """The public algorithm supports a code with more than one logical qubit."""
    encoder = stim.Tableau(3)
    error = stim.Tableau.from_named_gate("H") + stim.Tableau(2)

    assert trivial_syndrome_probability(
        encoder,
        error,
        logical_qubit_count=2,
        logical_coefficients={0: 1.0},
    ) == 0.5


def test_distance_three_example_has_quarter_acceptance():
    """The standalone paper algorithm reproduces the known D3 probability."""
    circuit = circuits.DoubleCheck(ancilla_count=6)
    analyzer = CliffordLogicalAnalyzer(
        data_indices=circuit.DATA_INDICES,
        stabilizer_generators=circuit.STABILIZER_GENERATORS_RESTRICTED,
        logical_s=circuit.LOGICAL_S,
    )
    error = analyzer.LOGICAL["T"](pauli_mask("YX_X___"))

    assert trivial_syndrome_probability(
        analyzer.encoder,
        error,
        logical_qubit_count=1,
        logical_coefficients=LOGICAL_COEFFICIENTS["T"],
    ) == 0.25
