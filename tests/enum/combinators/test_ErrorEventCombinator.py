import itertools
from collections import Counter, defaultdict
from math import prod

import numpy as np
import pytest
import stim

from cliffordea.enum.combinators import ErrorEventCombinator
from cliffordea.enum.combinators._base import error_event_count
from cliffordea.accept.pauli import forget_sign


class TestGetUndetectedConfigurationsForFaultCount():
    """Tests on the `d3_double_cat_check` circuit."""

    def test_fault_count_0_exact(self, d3_double_cat_check_brute: ErrorEventCombinator):
        result = d3_double_cat_check_brute._get_undetected_configurations_for_fault_count(0)
        trivial_effect = '_'*d3_double_cat_check_brute.circuit.noisy_circuit.num_qubits
        assert len(result) == 1
        assert trivial_effect in result
        assert result[trivial_effect] == Counter({1: 1})

    def test_fault_count_0(
            self,
            d3_double_cat_check_brute: ErrorEventCombinator
    ):
        goal: defaultdict[str, Counter[int]] = defaultdict(Counter)
        goal['_'*d3_double_cat_check_brute.circuit.noisy_circuit.num_qubits][1] += 1

        result = d3_double_cat_check_brute._get_undetected_configurations_for_fault_count(0)
        assert result == dict(goal)

    def test_fault_count_1(
            self,
            d3_double_cat_check_brute: ErrorEventCombinator
    ):
        goal: defaultdict[str, Counter[int]] = defaultdict(Counter)
        for (_, process_name, _), fault_dict_1 in d3_double_cat_check_brute.basis.items():
            for (signature_1, effect_1) in fault_dict_1.values():
                if not any(signature_1):
                    goal[effect_1][error_event_count(process_name)] += 1

        result = d3_double_cat_check_brute._get_undetected_configurations_for_fault_count(1)
        assert result == dict(goal)

    def test_fault_count_2(
            self,
            d3_double_cat_check_brute: ErrorEventCombinator
    ):
        fault_count = 2
        goal: defaultdict[str, Counter[int]] = defaultdict(Counter)
        combos = itertools.combinations(
            d3_double_cat_check_brute.basis.items(),
            fault_count,
        )
        for ((_, process_name_1, _), fault_dict_1), ((_, process_name_2, _), fault_dict_2) in combos:
            pairs = itertools.product(fault_dict_1.values(), fault_dict_2.values())
            for (signature_1, effect_1), (signature_2, effect_2) in pairs:
                if (signature_1 == signature_2).all():
                    product_string = stim.PauliString(effect_1) * stim.PauliString(effect_2)
                    probability_denominator = prod(
                        error_event_count(process_name)
                        for process_name in (process_name_1, process_name_2))
                    goal[forget_sign(product_string)][probability_denominator] += 1

        result = d3_double_cat_check_brute._get_undetected_configurations_for_fault_count(
            fault_count,
        )
        assert result == dict(goal)


class TestHasNonzeroSignature:

    def test_1(self):
        assert ErrorEventCombinator._has_nonzero_signature([np.array([True, False, True])])

    def test_2(self):
        signatures = (
            np.array([True, False, False]),
            np.array([True, True, False]),
        )
        assert ErrorEventCombinator._has_nonzero_signature(signatures)

    def test_single_bit(self):
        signatures = (
            np.array([True]),
            np.array([False]),
            np.array([False]),
        )
        assert ErrorEventCombinator._has_nonzero_signature(signatures)

    def test_all_false(self):
        signatures = (
            np.array([False, False, False]),
            np.array([False, False, False]),
            np.array([False, False, False]),
        )
        assert not ErrorEventCombinator._has_nonzero_signature(signatures)

    def test_all_true(self):
        signatures = (
            np.array([True, True, True]),
            np.array([True, True, True]),
            np.array([True, True, True]),
        )
        assert ErrorEventCombinator._has_nonzero_signature(signatures)

    def test_mixed(self):
        signatures = (
            np.array([True, False, True]),
            np.array([False, True, True]),
            np.array([True, True, False]),
        )
        assert not ErrorEventCombinator._has_nonzero_signature(signatures)

    def test_different_lengths(self):
        signatures = (
            np.array([True, False]),
            np.array([False, True, True]),
        )
        with pytest.raises(ValueError):
            ErrorEventCombinator._has_nonzero_signature(signatures)
