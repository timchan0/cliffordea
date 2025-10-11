import pytest

from cliffordep.combinators import FaultCombinatorExclusive, ErrorEventCombinator


class TestGroupFaultsByEffect():

    def test_sum_len_faults(self, d3_double_cat_check_fault_source_combinator: FaultCombinatorExclusive):
        assert sum(len(faults) for faults in d3_double_cat_check_fault_source_combinator.values()) == 158


class TestUndetectedCombinationsOnD3DoubleCatCheck():
    """Tests on the `d3_double_cat_check` circuit."""

    @pytest.mark.parametrize("length", [0, 1, 2])
    def test_against_brute_force(
        self,
        d3_double_cat_check_fault_source_combinator: FaultCombinatorExclusive,
        d3_double_cat_check_brute: ErrorEventCombinator,
        length: int,
    ):
        result_1 = d3_double_cat_check_fault_source_combinator._get_undetected_configurations_for_length(length)
        result_2 = d3_double_cat_check_brute._get_undetected_configurations_for_length(length)
        assert result_1 == result_2