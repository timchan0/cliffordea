import inspect

import cliffordea


def test_root_api_exposes_only_the_three_topic_modules():
    """The clean-break root API names only accept, enum, and sim."""
    assert cliffordea.__all__ == ["accept", "enum", "sim"]
    assert not hasattr(cliffordea, "FaultCombinator")
    assert not hasattr(cliffordea, "CliffordLogicalAnalyzer")


def test_accept_api_exposes_logical_pauli_coefficients():
    """The acceptance API publishes its readable logical-state input type."""
    assert "LogicalPauliCoefficients" in cliffordea.accept.__all__
    assert hasattr(cliffordea.accept, "LogicalPauliCoefficients")


def test_enum_api_uses_current_paper_terminology():
    """The enumeration API exposes only the clean-break terminology."""
    assert "DisjointFaultCombinator" in cliffordea.enum.__all__
    assert hasattr(cliffordea.enum, "DisjointFaultCombinator")
    assert not hasattr(cliffordea.enum, "FaultCombinatorExclusive")
    assert hasattr(
        cliffordea.enum.CultivationCircuit,
        "get_signature_and_effect",
    )
    assert not hasattr(
        cliffordea.enum.CultivationCircuit,
        "get_syndrome_and_effect",
    )
    assert hasattr(cliffordea.enum.FaultCombinator, "signature_count")
    assert not hasattr(cliffordea.enum.FaultCombinator, "syndrome_count")
    kept_parameters = inspect.signature(
        cliffordea.enum.FaultCombinator.get_kept_effects,
    ).parameters
    assert "max_fault_count" in kept_parameters
    assert "max_order" not in kept_parameters
    disjoint_parameters = inspect.signature(
        cliffordea.enum.DisjointFaultCombinator.get_undetected_configurations,
    ).parameters
    assert "max_fault_count" in disjoint_parameters
    assert "max_degree" not in disjoint_parameters
