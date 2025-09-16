import itertools

from collections import Counter, defaultdict
import stim

from cliffordep.combinators import FaultSourceCombinator, fault_count
from cliffordep.noisy_circuit_tools import CultivationCircuit
from cliffordep.pauli_string_tools import forget_sign


class TestMeasurementToDetectors:

    def test_multiple_qubits_and_detectors(self, multi_qubit_detector_circuit: CultivationCircuit):
        assert multi_qubit_detector_circuit._measurement_to_detectors == {0: {0}, 1: {1}}


class TestNoiselessLayers:

    def test_empty_layer(self):
        circuit = CultivationCircuit(
            noisy_circuit=stim.Circuit("""TICK
            R 0"""),
            data_indices= (0,),
            stabilizer_generators=(stim.PauliString(),),
            logical_x=stim.PauliString(),
            logical_z=stim.PauliString(),
        )
        assert circuit._noiseless_layers == [stim.Circuit(), stim.Circuit("R 0")]

    def test_multiple_qubits_and_detectors(self, multi_qubit_detector_circuit: CultivationCircuit):
        assert multi_qubit_detector_circuit._noiseless_layers == [
            stim.Circuit("CX 0 1"),
            stim.Circuit("""MZ[0] 0 1
            DETECTOR rec[-2]
            DETECTOR rec[-1]"""),
        ]


class TestUndetectedCombinationsOnD3DoubleCatCheck():
    """Tests on the `d3_double_cat_check` circuit."""

    def test_length_0_exact(self, d3_double_cat_check_fault_source_combinator: FaultSourceCombinator):
        result = d3_double_cat_check_fault_source_combinator._get_undetected_fault_combinations_for_length(0)
        trivial_effect = '_'*d3_double_cat_check_fault_source_combinator.circuit.noisy_circuit.num_qubits
        assert len(result) == 1
        assert trivial_effect in result
        assert result[trivial_effect] == Counter({1: 1})

    def test_length_0(self, d3_double_cat_check_fault_source_combinator: FaultSourceCombinator):
        goal: defaultdict[str, Counter[int]] = defaultdict(Counter)
        goal['_'*d3_double_cat_check_fault_source_combinator.circuit.noisy_circuit.num_qubits][1] += 1

        result = d3_double_cat_check_fault_source_combinator._get_undetected_fault_combinations_for_length(0)
        assert result == dict(goal)

    def test_length_1(self, d3_double_cat_check_fault_source_combinator: FaultSourceCombinator):
        goal: defaultdict[str, Counter[int]] = defaultdict(Counter)
        for syndrome, effect_map in d3_double_cat_check_fault_source_combinator.items():
            if not any(syndrome):
                for effect, counter in effect_map.items():
                    for (_, source_name, _), count in counter.items():
                        goal[effect][fault_count(source_name)] += count

        result = d3_double_cat_check_fault_source_combinator._get_undetected_fault_combinations_for_length(1)
        assert result == dict(goal)

    def test_length_2(
            self,
            d3_double_cat_check_fault_source_combinator: FaultSourceCombinator,
            update_undetected_combinations,
    ):
        goal: defaultdict[str, Counter[int]] = defaultdict(Counter)
        length = 2
        for effect_map in d3_double_cat_check_fault_source_combinator.values():

            # consider pairs (i, j) where i != j
            pairs = itertools.combinations(effect_map.items(), length)
            for (effect_1, counter_1), (effect_2, counter_2) in pairs:
                product_effect = forget_sign(stim.PauliString(effect_1) * stim.PauliString(effect_2))
                candidates = itertools.product(counter_1.items(), counter_2.items())
                for candidate in candidates:
                    update_undetected_combinations(goal, product_effect, candidate)

            # consider pairs (i, i)
            for effect, counter in effect_map.items():
                product_effect = '_'*len(effect)
                candidates = itertools.combinations_with_replacement(counter.items(), length)
                for candidate in candidates:
                    update_undetected_combinations(goal, product_effect, candidate)

        result = d3_double_cat_check_fault_source_combinator._get_undetected_fault_combinations_for_length(length)
        assert result == dict(goal)