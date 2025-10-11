import itertools
from collections import Counter, defaultdict
from math import prod

import numpy as np
import pytest
import stim

from cliffordep.combinators import ErrorEventCombinator, error_event_count
from cliffordep.pauli_string_tools import forget_sign


class TestGetUndetectedConfigurationsForLength():
    """Tests on the `d3_double_cat_check` circuit."""

    def test_length_0_exact(self, d3_double_cat_check_brute: ErrorEventCombinator):
        result = d3_double_cat_check_brute._get_undetected_configurations_for_length(0)
        trivial_effect = '_'*d3_double_cat_check_brute.circuit.noisy_circuit.num_qubits
        assert len(result) == 1
        assert trivial_effect in result
        assert result[trivial_effect] == Counter({1: 1})

    def test_length_0(
            self,
            d3_double_cat_check_brute: ErrorEventCombinator
    ):
        goal: defaultdict[str, Counter[int]] = defaultdict(Counter)
        goal['_'*d3_double_cat_check_brute.circuit.noisy_circuit.num_qubits][1] += 1

        result = d3_double_cat_check_brute._get_undetected_configurations_for_length(0)
        assert result == dict(goal)

    def test_length_1(
            self,
            d3_double_cat_check_brute: ErrorEventCombinator
    ):
        goal: defaultdict[str, Counter[int]] = defaultdict(Counter)
        for (_, process_name, _), fault_dict_1 in d3_double_cat_check_brute.basis.items():
            for (syndrome_1, effect_1) in fault_dict_1.values():
                if not any(syndrome_1):
                    goal[effect_1][error_event_count(process_name)] += 1

        result = d3_double_cat_check_brute._get_undetected_configurations_for_length(1)
        assert result == dict(goal)

    def test_length_2(
            self,
            d3_double_cat_check_brute: ErrorEventCombinator
    ):
        length = 2
        goal: defaultdict[str, Counter[int]] = defaultdict(Counter)
        combos = itertools.combinations(d3_double_cat_check_brute.basis.items(), length)
        for ((_, process_name_1, _), fault_dict_1), ((_, process_name_2, _), fault_dict_2) in combos:
            pairs = itertools.product(fault_dict_1.values(), fault_dict_2.values())
            for (syndrome_1, effect_1), (syndrome_2, effect_2) in pairs:
                if (syndrome_1 == syndrome_2).all():
                    product_string = stim.PauliString(effect_1) * stim.PauliString(effect_2)
                    fault_counts = prod(
                        error_event_count(process_name)
                        for process_name in (process_name_1, process_name_2))
                    goal[forget_sign(product_string)][fault_counts] += 1

        result = d3_double_cat_check_brute._get_undetected_configurations_for_length(length)
        assert result == dict(goal)


class TestAnyDefects:

    def test_1(self):
        assert ErrorEventCombinator._any_defects([np.array([True, False, True])])

    def test_2(self):
        syndromes = (
            np.array([True, False, False]),
            np.array([True, True, False]),
        )
        assert ErrorEventCombinator._any_defects(syndromes)

    def test_length_1(self):
        syndromes = (
            np.array([True]),
            np.array([False]),
            np.array([False]),
        )
        assert ErrorEventCombinator._any_defects(syndromes)

    def test_all_false(self):
        syndromes = (
            np.array([False, False, False]),
            np.array([False, False, False]),
            np.array([False, False, False]),
        )
        assert not ErrorEventCombinator._any_defects(syndromes)

    def test_all_true(self):
        syndromes = (
            np.array([True, True, True]),
            np.array([True, True, True]),
            np.array([True, True, True]),
        )
        assert ErrorEventCombinator._any_defects(syndromes)

    def test_mixed(self):
        syndromes = (
            np.array([True, False, True]),
            np.array([False, True, True]),
            np.array([True, True, False]),
        )
        assert not ErrorEventCombinator._any_defects(syndromes)

    def test_different_lengths(self):
        syndromes = (
            np.array([True, False]),
            np.array([False, True, True]),
        )
        with pytest.raises(ValueError):
            ErrorEventCombinator._any_defects(syndromes)

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
    #                 products[forget_sign(product_string)][denominator_1, denominator_2, denominator_3] += 1

    #     result = undetected_products(d3_double_cat_check_group, length=length)
    #     assert result == dict(products)