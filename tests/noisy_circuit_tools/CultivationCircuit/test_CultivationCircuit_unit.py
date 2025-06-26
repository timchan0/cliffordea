import itertools

from collections import Counter, defaultdict
import stim

from cliffordep.noisy_circuit_tools import CultivationCircuit, _update_undetected_combinations, fault_count
from cliffordep.pauli_string_tools import unsigned_str
from cliffordep.type_aliases import EffectMap


class TestMeasurementToDetectors:

    def test_multiple_qubits_and_detectors(self, multi_qubit_detector_circuit):
        assert multi_qubit_detector_circuit._measurement_to_detectors == {(1, 0): {0}, (1, 1): {1}}


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

    def test_multiple_qubits_and_detectors(self, multi_qubit_detector_circuit):
        assert multi_qubit_detector_circuit._noiseless_layers == [
            stim.Circuit("CX 0 1"),
            stim.Circuit("""MZ 0 1
            DETECTOR rec[-2]
            DETECTOR rec[-1]"""),
        ]


class TestUndetectedCombinationsOnD3DoubleCatCheck():
    """Tests on the `d3_double_cat_check` circuit."""

    def test_length_0_exact(self, noisy_d3_double_cat_check: CultivationCircuit, d3_double_cat_check_effect_maps: dict[tuple[bool, ...], EffectMap]):
        result = noisy_d3_double_cat_check._get_undetected_fault_combinations_for_length(d3_double_cat_check_effect_maps, length=0)
        trivial_effect = '_'*7
        assert len(result) == 1
        assert trivial_effect in result
        assert result[trivial_effect] == Counter({(): 1})

    def test_length_0(self, noisy_d3_double_cat_check: CultivationCircuit, d3_double_cat_check_effect_maps: dict[tuple[bool, ...], EffectMap]):
        goal: defaultdict[str, Counter[tuple[int, ...]]] = defaultdict(Counter)
        first_effect_map, *_ = d3_double_cat_check_effect_maps.values()
        first_effect, *_ = first_effect_map.keys()
        goal['_'*len(first_effect)][()] += 1

        result = noisy_d3_double_cat_check._get_undetected_fault_combinations_for_length(d3_double_cat_check_effect_maps, length=0)
        assert result == dict(goal)

    def test_length_1(self, noisy_d3_double_cat_check: CultivationCircuit, d3_double_cat_check_effect_maps: dict[tuple[bool, ...], EffectMap]):
        goal: defaultdict[str, Counter[tuple[int, ...]]] = defaultdict(Counter)
        for syndrome, effect_map in d3_double_cat_check_effect_maps.items():
            if not any(syndrome):
                for effect, counter in effect_map.items():
                    for (_, source_name, _), count in counter.items():
                        goal[effect][fault_count(source_name),] += count

        result = noisy_d3_double_cat_check._get_undetected_fault_combinations_for_length(d3_double_cat_check_effect_maps, length=1)
        assert result == dict(goal)

    def test_length_2(self, noisy_d3_double_cat_check: CultivationCircuit, d3_double_cat_check_effect_maps: dict[tuple[bool, ...], EffectMap]):
        goal: defaultdict[str, Counter[tuple[int, ...]]] = defaultdict(Counter)
        length = 2
        for effect_map in d3_double_cat_check_effect_maps.values():

            # consider pairs (i, j) where i != j
            pairs = itertools.combinations(effect_map.items(), length)
            for (effect_1, counter_1), (effect_2, counter_2) in pairs:
                product_effect = unsigned_str(stim.PauliString(effect_1) * stim.PauliString(effect_2))
                candidates = itertools.product(counter_1.items(), counter_2.items())
                for candidate in candidates:
                    _update_undetected_combinations(goal, product_effect, candidate)

            # consider pairs (i, i)
            for effect, counter in effect_map.items():
                product_effect = '_'*len(effect)
                candidates = itertools.combinations_with_replacement(counter.items(), length)
                for candidate in candidates:
                    _update_undetected_combinations(goal, product_effect, candidate)

        result = noisy_d3_double_cat_check._get_undetected_fault_combinations_for_length(d3_double_cat_check_effect_maps, length=length)
        assert result == dict(goal)