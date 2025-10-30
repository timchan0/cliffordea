from collections import defaultdict
import numpy as np
import stim
import pytest

import cliffordep
from cliffordep.type_aliases import ErrorEvent, ErrorLocation
from cliffordep.noisy_circuit_tools import CultivationCircuit


class TestD3DoubleCatCheck:
    """Tests on the `d3_double_cat_check` circuit."""

    @pytest.fixture
    def ungrouped(self, d3_double_cat_check_grouped_by_location: dict[ErrorLocation, list[ErrorEvent]]):
        result: list[ErrorEvent] = sum(d3_double_cat_check_grouped_by_location.values(), [])
        return result

    def test_with_flip_simulator(
            self,
            ungrouped: list[ErrorEvent],
            noisy_d3_double_cat_check: CultivationCircuit,
    ):
        """Test the correctness of `group_error_events_by_location`
        by comparing results with a `stim.FlipSimulator`.
        """
        sim = stim.FlipSimulator(
            batch_size=1,
            disable_stabilizer_randomization=True,
            num_qubits=noisy_d3_double_cat_check.noisy_circuit.num_qubits,
        )
        for fault in ungrouped:
            self._test_with_flip_simulator(noisy_d3_double_cat_check, sim, fault)

    @staticmethod
    def _test_with_flip_simulator(
            circuit: CultivationCircuit,
            sim: stim.FlipSimulator,
            fault: ErrorEvent,
    ):

        syndrome, effect = circuit.get_syndrome_and_effect(fault)
        
        timeslice, name, targets = fault
        if name.startswith('M'):
            sim_syndrome = np.zeros(circuit.noisy_circuit.num_detectors, dtype=bool)
            for instruction in circuit._noiseless_layers[timeslice]:
                if isinstance(instruction, stim.CircuitRepeatBlock):
                    raise ValueError
                if instruction.num_measurements:
                    for measurement_index, target_group in enumerate(
                        instruction.target_groups(),
                        start=int(instruction.tag),
                    ):
                        if set(target_group) == set(targets):
                            for detector in circuit._measurement_to_detectors[measurement_index]:
                                sim_syndrome[detector] ^= True
                            break
            sim_syndrome = tuple(sim_syndrome)
            sim_effect = circuit.noisy_circuit.num_qubits*'_'
        else:
            faulty_circuit = cliffordep.insert_fault(
                    circuit.noiseless_circuit,
                    timeslice=timeslice,
                    name=name,
                    targets=targets,
                )
            sim.clear()
            for instruction in faulty_circuit:
                sim.do(instruction)
                if isinstance(instruction, stim.CircuitRepeatBlock):
                    raise ValueError("REPEAT blocks not handled in this simulation.")
                    # `FlipSimulator` does not erase the X error after the application of RX
                if stim.gate_data(instruction.name).is_reset:
                    for target in instruction.targets_copy():
                        sim.set_pauli_flip(
                                'I',
                                qubit_index=target.value,
                                instance_index=0,
                            )
            sim_syndrome = tuple(sim.get_detector_flips(instance_index=0))
            sim_effect = cliffordep.forget_sign(sim.peek_pauli_flips(instance_index=0))
        
        # assert the syndrome matches
        assert np.array_equal(sim_syndrome, syndrome)
        # assert the resultant Pauli string matches (up to a global phase)
        assert sim_effect == effect


def test_measurement_error_gives_syndrome():
    # Error event: measurement error at timeslice 0, qubit 0
    circuit = CultivationCircuit(
        noisy_circuit=stim.Circuit("""MZ 0
                                   DETECTOR rec[-1]"""),
        stabilizer_generators=(stim.PauliString(),),
        logical_x=stim.PauliString(),
        logical_z=stim.PauliString(),
    )
    circuit._measurement_to_detectors = defaultdict(set, {0: {0}})
    fault = (0, "MZ", (stim.GateTarget(0),))
    # For a 1-qubit, 1-detector circuit, measurement 0 flips detector 0
    syndrome, effect = circuit.get_syndrome_and_effect(fault)
    assert np.array_equal(syndrome, np.array([True]))
    assert effect == "_"

def test_pauli_error_gives_effect_and_syndrome():
    circuit = CultivationCircuit(
        noisy_circuit=stim.Circuit("""
            H 0
            TICK
            MZ 0
            DETECTOR rec[-1]
        """),
        stabilizer_generators=(stim.PauliString(),),
        logical_x=stim.PauliString(),
        logical_z=stim.PauliString(),
    )
    # Error event: X_ERROR at timeslice 0, qubit 0 (after H)
    fault = (0, "X_ERROR", (stim.GateTarget(0),))
    syndrome, effect = circuit.get_syndrome_and_effect(fault)
    # X anticommutes with MZ, so syndrome flips
    assert np.array_equal(syndrome, np.array([True]))
    assert effect == "X"

def test_no_syndrome_for_commuting_pauli():
    circuit = CultivationCircuit(
        noisy_circuit=stim.Circuit("""H 0
        MZ 0
        DETECTOR rec[-1]"""),
        stabilizer_generators=(stim.PauliString(),),
        logical_x=stim.PauliString(),
        logical_z=stim.PauliString(),
    )
    # Error event: Z_ERROR at timeslice 0, qubit 0 (commutes with MZ)
    fault = (0, "Z_ERROR", (stim.GateTarget(0),))
    syndrome, effect = circuit.get_syndrome_and_effect(fault)
    assert np.array_equal(syndrome, np.array([False]))
    assert effect == "Z"

def test_reset_removes_pauli():
    # Circuit: X_ERROR at t=0, qubit 0; R 0 at t=1
    circuit = CultivationCircuit(
        noisy_circuit=stim.Circuit("""TICK
                                   R 0"""),
        stabilizer_generators=(stim.PauliString(),),
        logical_x=stim.PauliString(),
        logical_z=stim.PauliString(),
    )
    fault = (0, "X_ERROR", (stim.GateTarget(0),))
    _, effect = circuit.get_syndrome_and_effect(fault)
    # After reset, effect should be identity
    assert effect == "_"

def test_multiple_qubits_and_detectors(multi_qubit_detector_circuit: CultivationCircuit):
    fault = (0, "X_ERROR", (stim.GateTarget(1),))
    syndrome, effect = multi_qubit_detector_circuit.get_syndrome_and_effect(fault)
    # X on qubit 1 after CX is X1
    assert effect == "_X"
    # Should only flip detector 1 if X on 1 anticommutes with MZ 1
    assert np.array_equal(syndrome, np.array([False, True]))
