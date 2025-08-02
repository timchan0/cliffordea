import itertools

import stim
import pytest

import cliffordep
from cliffordep.noisy_circuit_tools import CultivationCircuit, FaultSourceCombinator


class TestD3DoubleCatCheck():
    """Tests on the `d3_double_cat_check` circuit."""


    @pytest.fixture
    def unrestricted_fault_source_combinator(self, noisy_d3_double_cat_check: CultivationCircuit) -> FaultSourceCombinator:
        """Fault source combinator for the distance-3 double cat check circuit."""
        combinator = FaultSourceCombinator(noisy_d3_double_cat_check, restrict_to_data=False)
        return combinator
    

    def test_correct_length(self, d3_double_cat_check_fault_source_combinator: FaultSourceCombinator):
        """Test returns a dictionary with the correct key count."""
        assert type(d3_double_cat_check_fault_source_combinator) is FaultSourceCombinator
        assert type(d3_double_cat_check_fault_source_combinator.basis) is dict
        assert d3_double_cat_check_fault_source_combinator.syndrome_count == 19


    def test_undetected_pairs(self, unrestricted_fault_source_combinator: FaultSourceCombinator):
        """Test that undetected pairs commute with the flag measurements at the end of `d3_double_cat_check`."""
        for _, resultant_paulis in unrestricted_fault_source_combinator.items():
            pairs = itertools.combinations(resultant_paulis.items(), 2)
            for (resultant_pauli_1, faults_1), (resultant_pauli_2, faults_2) in pairs:
                string_1 = stim.PauliString(resultant_pauli_1)
                string_2 = stim.PauliString(resultant_pauli_2)
                prod = string_1 * string_2
                if not any(name=='MX' for _, name, _ in (faults_1|faults_2).keys()):
                    for index in cliffordep.circuits.D3DoubleCatCheckA6.ANCILLA_INDICES:
                        assert prod[index] not in CultivationCircuit._get_anticommuting_paulis('MX')