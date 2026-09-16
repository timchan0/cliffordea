import pytest

from cliffordea.enum.combinators import DisjointFaultCombinator, ErrorEventCombinator


class TestGroupFaultsByEffect():

    def test_fault_count(self, d3_double_cat_check_disjoint_fault_combinator: DisjointFaultCombinator):
        assert sum(len(faults) for faults in d3_double_cat_check_disjoint_fault_combinator.values()) == 158


class TestUndetectedCombinationsOnD3DoubleCatCheck():
    """Tests on the `d3_double_cat_check` circuit."""

    @pytest.mark.parametrize("fault_count", [0, 1, 2])
    def test_against_brute_force(
        self,
        d3_double_cat_check_disjoint_fault_combinator: DisjointFaultCombinator,
        d3_double_cat_check_brute: ErrorEventCombinator,
        fault_count: int,
    ):
        result_1 = d3_double_cat_check_disjoint_fault_combinator._get_undetected_configurations_for_fault_count(
            fault_count,
        )
        result_2 = d3_double_cat_check_brute._get_undetected_configurations_for_fault_count(
            fault_count,
        )
        assert result_1 == result_2
