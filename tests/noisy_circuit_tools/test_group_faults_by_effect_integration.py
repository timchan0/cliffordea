import itertools

import stim
import pytest

import cliffordep
from cliffordep.noisy_circuit_tools import _get_anticommuting_paulis, CultivationCircuit
from cliffordep.type_aliases import EffectMap


class TestD3DoubleCatCheck():
    """Tests on the `d3_double_cat_check` circuit."""


    @pytest.fixture
    def unrestricted_effect_maps(self, noisy_d3_double_cat_check: CultivationCircuit) -> dict[tuple[bool, ...], EffectMap]:
        """Output of `group_faults_by_effect` for the distance-3 double cat check circuit."""
        effect_maps = noisy_d3_double_cat_check.group_faults_by_effect(False)
        return effect_maps
    

    def test_correct_length(self, d3_double_cat_check_effect_maps: dict[tuple[bool, ...], EffectMap]):
        """Test returns a dictionary with the correct key count."""
        assert type(d3_double_cat_check_effect_maps) is dict
        assert len(d3_double_cat_check_effect_maps) == 19


    def test_undetected_pairs(self, unrestricted_effect_maps: dict[tuple[bool, ...], EffectMap]):
        """Test that undetected pairs commute with the flag measurements at the end of `d3_double_cat_check`."""
        for _, resultant_paulis in unrestricted_effect_maps.items():
            pairs = itertools.combinations(resultant_paulis.items(), 2)
            for (resultant_pauli_1, faults_1), (resultant_pauli_2, faults_2) in pairs:
                string_1 = stim.PauliString(resultant_pauli_1)
                string_2 = stim.PauliString(resultant_pauli_2)
                prod = string_1 * string_2
                if not any(name=='MX' for _, name, _ in (faults_1|faults_2).keys()):
                    for index in cliffordep.circuits.D3DoubleCatCheckA6.ANCILLA_INDICES:
                        assert prod[index] not in _get_anticommuting_paulis('MX')