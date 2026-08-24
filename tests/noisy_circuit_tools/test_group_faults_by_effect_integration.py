import itertools

import stim

import cliffordep
from cliffordep.combinators import FaultCombinatorExclusive


def _get_anticommuting_paulis(name: str):
    """Get the set of Paulis that anticommute with the measurement given by `name`."""
    if 'X' in name:
        return {'Y', 'Z'}
    elif 'Y' in name:
        return {'X', 'Z'}
    elif 'Z' in name or name in {'M', 'MR'}:
        return {'X', 'Y'}
    else:
        raise NotImplementedError


class TestD3DoubleCatCheck():
    """Tests on the `d3_double_cat_check` circuit."""
    

    def test_correct_length(self, d3_double_cat_check_fault_source_combinator: FaultCombinatorExclusive):
        """Test returns a dictionary with the correct key count."""
        assert type(d3_double_cat_check_fault_source_combinator) is FaultCombinatorExclusive
        assert type(d3_double_cat_check_fault_source_combinator.basis) is dict
        assert d3_double_cat_check_fault_source_combinator.syndrome_count == 19


    def test_undetected_pairs(self, d3_double_cat_check_fault_source_combinator: FaultCombinatorExclusive):
        """Test that undetected pairs commute with the flag measurements at the end of `d3_double_cat_check`."""
        d3a6 = cliffordep.circuits.Distance3DoubleCheck(ancilla_count=6)
        d3a6_ancilla_indices = {q for q in range(d3a6.INNER_CIRCUIT.num_qubits) if q not in d3a6.DATA_INDICES}
        for _, resultant_paulis in d3_double_cat_check_fault_source_combinator.items():
            pairs = itertools.combinations(resultant_paulis.items(), 2)
            for (resultant_pauli_1, faults_1), (resultant_pauli_2, faults_2) in pairs:
                string_1 = stim.PauliString(resultant_pauli_1)
                string_2 = stim.PauliString(resultant_pauli_2)
                prod = string_1 * string_2
                if not any(name=='MX' for _, name, _ in (faults_1|faults_2).keys()):
                    for index in d3a6_ancilla_indices:
                        assert prod[index] not in _get_anticommuting_paulis('MX')