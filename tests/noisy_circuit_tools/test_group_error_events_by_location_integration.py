import numpy as np
import stim

from cliffordea.combinators import FaultCombinator
from cliffordea.combinators.fault_combinator import (
    _iter_zero_syndrome_configurations,
)
from cliffordea.constants import ONE_QUBIT_ERROR_EVENTS
from cliffordea.noisy_circuit_tools import CultivationCircuit
from cliffordea.type_aliases import ErrorEvent, ErrorLocation


class TestD3DoubleCatCheck:
    """Tests on the `d3_double_cat_check` circuit."""


    def test_correct_length(
            self,
            noisy_d3_double_cat_check_circuit: stim.Circuit,
            d3_double_cat_check_grouped_by_location: dict[ErrorLocation, list[ErrorEvent]],
    ):
        """Test returns a dictionary with the correct key count."""
        assert type(d3_double_cat_check_grouped_by_location) is dict

        noisy_gates = {
            'DEPOLARIZE1',
            'DEPOLARIZE2',
            *ONE_QUBIT_ERROR_EVENTS,
        }
        count = sum(len(instruction.target_groups())
                    for instruction in noisy_d3_double_cat_check_circuit
                    if isinstance(instruction, stim.CircuitInstruction)
                    and instruction.name in noisy_gates)
        assert len(d3_double_cat_check_grouped_by_location) == count


    def test_measure_reset_fault_regression(
            self,
            noisy_d3_double_cat_check_circuit: stim.Circuit,
            noisy_d3_double_cat_check: CultivationCircuit,
            d3_double_cat_check_grouped_by_location: dict[ErrorLocation, list[ErrorEvent]],
    ):
        """The central MRX readout flip joins the correct existing fault.

        Its detector pattern and final identity effect match an existing fault,
        so adding the missing event increases that fault's multiplicity instead
        of creating a new fault.
        """
        target = stim.GateTarget(5)
        event = (6, 'MRX', (target,))
        assert len(d3_double_cat_check_grouped_by_location) == 120
        assert sum(map(len, d3_double_cat_check_grouped_by_location.values())) == 620
        assert d3_double_cat_check_grouped_by_location[event] == [event]

        syndrome, effect = noisy_d3_double_cat_check.get_syndrome_and_effect(event)
        assert np.array_equal(
            syndrome,
            np.array([True, False, False, True, False, False, False]),
        )
        assert effect == '_' * noisy_d3_double_cat_check_circuit.num_qubits

        combinator = FaultCombinator(noisy_d3_double_cat_check_circuit)
        fault_index = combinator.basis[0b1001][0]
        assert combinator.fault_count == 158
        assert combinator.index_to_bag[fault_index] == (1, 0, 5)


    def test_undetected_configuration_counts(
            self,
            d3_combinator_and_kept_effects_by_analyzer,
    ):
        """Count the undetected distance-3 configurations and their final errors by order.

        The first reference list counts the fault sets that jointly flip no
        detectors. The second counts the distinct final Pauli errors produced
        by those sets. Together they catch unintended changes to enumeration.
        """
        combinator, *_ = d3_combinator_and_kept_effects_by_analyzer
        syndromes, fault_effects = zip(*combinator._indexed_faults, strict=True)
        configuration_counts = []
        effect_counts = []
        for order in range(5):
            configuration_count = 0
            resultant_effects = set()
            for effect, _ in _iter_zero_syndrome_configurations(
                    syndromes=syndromes,
                    effects=fault_effects,
                    order=order,
            ):
                configuration_count += 1
                resultant_effects.add(effect)
            configuration_counts.append(configuration_count)
            effect_counts.append(len(resultant_effects))

        assert configuration_counts == [
            1,
            32,
            1003,
            29656,
            769733,
        ]
        assert effect_counts == [
            1,
            32,
            413,
            2544,
            8048,
        ]
