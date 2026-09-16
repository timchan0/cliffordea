import cliffordea


def test_root_api_exposes_only_the_three_topic_modules():
    """The clean-break root API names only accept, enum, and sim."""
    assert cliffordea.__all__ == ["accept", "enum", "sim"]
    assert not hasattr(cliffordea, "FaultCombinator")
    assert not hasattr(cliffordea, "CliffordLogicalAnalyzer")
