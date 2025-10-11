import itertools

import stim

import cliffordep
from cliffordep.combinators import FaultCombinatorExclusive
from cliffordep.pauli_string_tools import FrozenCliffordString


class TestD3DoubleCatCheck():
    """Tests on the `d3_double_cat_check` circuit."""
    

    def test_correct_length(self, d3_double_cat_check_fault_source_combinator: FaultCombinatorExclusive):
        """Test returns a dictionary with the correct key count."""
        assert type(d3_double_cat_check_fault_source_combinator) is FaultCombinatorExclusive
        assert type(d3_double_cat_check_fault_source_combinator.basis) is dict
        assert d3_double_cat_check_fault_source_combinator.syndrome_count == 19


    def test_undetected_pairs(self, d3_double_cat_check_fault_source_combinator: FaultCombinatorExclusive):
        """Test that undetected pairs commute with the flag measurements at the end of `d3_double_cat_check`."""
        for _, resultant_paulis in d3_double_cat_check_fault_source_combinator.items():
            pairs = itertools.combinations(resultant_paulis.items(), 2)
            for (resultant_pauli_1, faults_1), (resultant_pauli_2, faults_2) in pairs:
                string_1 = stim.PauliString(resultant_pauli_1)
                string_2 = stim.PauliString(resultant_pauli_2)
                prod = string_1 * string_2
                if not any(name=='MX' for _, name, _ in (faults_1|faults_2).keys()):
                    for index in cliffordep.circuits.D3DoubleCatCheckA6.ANCILLA_INDICES:
                        assert prod[index] not in FrozenCliffordString._get_anticommuting_paulis('MX')