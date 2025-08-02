import pytest

from cliffordep.combinators import FaultSourceCombinator, FaultSourceBruteForceCombinator


class TestGroupFaultsByEffect():

    def test_sum_len_faults(self, d3_double_cat_check_fault_source_combinator: FaultSourceCombinator):
        assert sum(len(faults) for faults in d3_double_cat_check_fault_source_combinator.values()) == 158


class TestUndetectedCombinationsOnD3DoubleCatCheck():
    """Tests on the `d3_double_cat_check` circuit."""

    @pytest.mark.parametrize("length", [0, 1, 2])
    def test_against_brute_force(
        self,
        d3_double_cat_check_fault_source_combinator: FaultSourceCombinator,
        d3_double_cat_check_brute: FaultSourceBruteForceCombinator,
        length: int,
    ):
        result_1 = d3_double_cat_check_fault_source_combinator._get_undetected_fault_combinations_for_length(length)
        result_2 = d3_double_cat_check_brute._get_undetected_fault_combinations_for_length(length)
        assert result_1 == result_2