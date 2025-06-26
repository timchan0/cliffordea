import numpy as np
import numpy.typing as npt
import pytest

from cliffordep import brute_force
from cliffordep.noisy_circuit_tools import CultivationCircuit
from cliffordep.type_aliases import EffectMap, FaultSource, Fault


class TestGroupFaultsByEffect():

    def test_sum_len_effect_maps(self, d3_double_cat_check_effect_maps: dict[tuple[bool, ...], EffectMap]):
        assert sum(len(effect_map) for effect_map in d3_double_cat_check_effect_maps.values()) == 158


class TestUndetectedCombinationsOnD3DoubleCatCheck():
    """Tests on the `d3_double_cat_check` circuit."""

    @pytest.mark.parametrize("length", [0, 1, 2])
    def test_against_brute_force(
        self,
        noisy_d3_double_cat_check: CultivationCircuit,
        d3_double_cat_check_effect_maps: dict[tuple[bool, ...], EffectMap],
        d3_double_cat_check_group_brute: dict[FaultSource, dict[Fault, tuple[npt.NDArray[np.bool_], str]]],
        length: int,
    ):
        result_1 = noisy_d3_double_cat_check._get_undetected_fault_combinations_for_length(d3_double_cat_check_effect_maps, length=length)
        result_2 = brute_force.undetected_combinations(d3_double_cat_check_group_brute, length=length)
        assert result_1 == result_2