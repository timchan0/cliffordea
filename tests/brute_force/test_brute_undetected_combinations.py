from collections import Counter, defaultdict
import itertools

import numpy as np
import numpy.typing as npt
import stim

from cliffordep.noisy_circuit_tools import fault_count
from cliffordep.brute_force import undetected_combinations
from cliffordep.pauli_string_tools import unsigned_str
from cliffordep.type_aliases import Fault, FaultSource

class TestD3DoubleCatCheck():
    """Tests on the `d3_double_cat_check` circuit."""

    def test_length_0_exact(self, d3_double_cat_check_group_brute):
        result = undetected_combinations(d3_double_cat_check_group_brute, length=0)
        trivial_effect = '_'*7
        assert len(result) == 1
        assert trivial_effect in result
        assert result[trivial_effect] == Counter({(): 1})
    
    def test_length_0(
            self,
            d3_double_cat_check_group_brute: dict[FaultSource, dict[Fault, tuple[npt.NDArray[np.bool_], str]]]
    ):
        goal: defaultdict[str, Counter[tuple[int, ...]]] = defaultdict(Counter)
        first_fault_dict, *_ = d3_double_cat_check_group_brute.values()
        (_, first_effect), *_ = first_fault_dict.values()
        goal['_'*len(first_effect)][()] += 1
        
        result = undetected_combinations(d3_double_cat_check_group_brute, length=0)
        assert result == dict(goal)

    def test_length_1(
            self,
            d3_double_cat_check_group_brute: dict[FaultSource, dict[Fault, tuple[npt.NDArray[np.bool_], str]]]
    ):
        goal: defaultdict[str, Counter[tuple[int, ...]]] = defaultdict(Counter)
        for (_, source_name, _), fault_dict_1 in d3_double_cat_check_group_brute.items():
            for (syndrome_1, effect_1) in fault_dict_1.values():
                if not any(syndrome_1):
                    goal[effect_1][(fault_count(source_name),)] += 1

        result = undetected_combinations(d3_double_cat_check_group_brute, length=1)
        assert result == dict(goal)
    
    def test_length_2(
            self,
            d3_double_cat_check_group_brute: dict[FaultSource, dict[Fault, tuple[npt.NDArray[np.bool_], str]]]
    ):
        length = 2
        goal: defaultdict[str, Counter[tuple[int, ...]]] = defaultdict(Counter)
        combos = itertools.combinations(d3_double_cat_check_group_brute.items(), length)
        for ((_, source_name_1, _), fault_dict_1), ((_, source_name_2, _), fault_dict_2) in combos:
            pairs = itertools.product(fault_dict_1.values(), fault_dict_2.values())
            for (syndrome_1, effect_1), (syndrome_2, effect_2) in pairs:
                if (syndrome_1 == syndrome_2).all():
                    product_string = stim.PauliString(effect_1) * stim.PauliString(effect_2)
                    fault_counts = tuple(sorted([
                        fault_count(source_name)
                        for source_name in (source_name_1, source_name_2)]))
                    goal[unsigned_str(product_string)][fault_counts] += 1

        result = undetected_combinations(d3_double_cat_check_group_brute, length=length)
        assert result == dict(goal)
    
    # def test_length_3(self, d3_double_cat_check_group):
    #     length = 3
    #     products: defaultdict[str, Counter[tuple[int, ...]]] = defaultdict(Counter)
    #     combos = itertools.combinations(d3_double_cat_check_group.items(), length)
    #     for ((*_, denominator_1), fault_dict_1), ((*_, denominator_2), fault_dict_2), ((*_, denominator_3), fault_dict_3) in combos:
    #         triples = itertools.product(
    #             fault_dict_1.values(),
    #             fault_dict_2.values(),
    #             fault_dict_3.values(),
    #         )
    #         for (syndrome_1, effect_1), (syndrome_2, effect_2), (syndrome_3, effect_3) in triples:
    #             any_defects = _any_defects((syndrome_1, syndrome_2, syndrome_3))
    #             if not any_defects:
    #                 product_string = stim.PauliString(effect_1) * stim.PauliString(effect_2) * stim.PauliString(effect_3)
    #                 products[unsigned_str(product_string)][denominator_1, denominator_2, denominator_3] += 1

    #     result = undetected_products(d3_double_cat_check_group, length=length)
    #     assert result == dict(products)