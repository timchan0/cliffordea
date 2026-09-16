from collections import defaultdict
import numpy as np
import stim
import pytest

from cliffordea.accept.pauli import forget_sign
from cliffordea.enum.circuit_tools import insert_error_events
from cliffordea.enum.types import ErrorEvent, ErrorLocation
from cliffordea.enum.cultivation_circuit import CultivationCircuit


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
        for error_event in ungrouped:
            self._test_with_flip_simulator(noisy_d3_double_cat_check, sim, error_event)

    @staticmethod
    def _test_with_flip_simulator(
            circuit: CultivationCircuit,
            sim: stim.FlipSimulator,
            error_event: ErrorEvent,
    ):

        signature, effect = circuit.get_signature_and_effect(error_event)
        
        timeslice, name, targets = error_event
        if name.startswith('M'):
            sim_signature = np.zeros(circuit.noisy_circuit.num_detectors, dtype=bool)
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
                                sim_signature[detector] ^= True
                            break
            sim_signature = tuple(sim_signature)
            sim_effect = circuit.noisy_circuit.num_qubits*'_'
        else:
            faulty_circuit = insert_error_events(
                circuit.noiseless_circuit,
                weighted_error_events=((error_event, 1),),
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
            sim_signature = tuple(sim.get_detector_flips(instance_index=0))
            sim_effect = forget_sign(sim.peek_pauli_flips(instance_index=0))
        
        # assert the signature matches
        assert np.array_equal(sim_signature, signature)
        # assert the resultant Pauli string matches (up to a global phase)
        assert sim_effect == effect


def test_measurement_error_gives_signature():
    # Error event: measurement error at timeslice 0, qubit 0
    circuit = CultivationCircuit(noisy_circuit=stim.Circuit("""MZ 0
                                   DETECTOR rec[-1]"""))
    circuit._measurement_to_detectors = defaultdict(set, {0: {0}})
    fault = (0, "MZ", (stim.GateTarget(0),))
    # For a 1-qubit, 1-detector circuit, measurement 0 flips detector 0
    signature, effect = circuit.get_signature_and_effect(fault)
    assert np.array_equal(signature, np.array([True]))
    assert effect == "_"


def test_measure_reset_error_preserves_reset_gate():
    """An MRX outcome flip changes the detector result without adding a Pauli.

    Re-inserting the event for inspection must still use ``MRX`` rather than
    ``MX`` so that the visualized faulty circuit also resets the measured qubit.
    """
    circuit = CultivationCircuit(noisy_circuit=stim.Circuit("""
        MRX(0.001) 0
        DETECTOR rec[-1]
    """))
    event = (0, "MRX", (stim.GateTarget(0),))

    signature, effect = circuit.get_signature_and_effect(event)
    assert np.array_equal(signature, np.array([True]))
    assert effect == "_"

    probability = 0.25
    faulty_circuit = insert_error_events(
        circuit=circuit.noiseless_circuit,
        weighted_error_events=((event, probability),),
    )
    measurement = faulty_circuit[0]
    assert isinstance(measurement, stim.CircuitInstruction)
    assert measurement.name == "MRX"
    assert measurement.gate_args_copy() == [probability]
    assert stim.gate_data(measurement.name).is_reset


def test_pauli_error_gives_effect_and_signature():
    circuit = CultivationCircuit(noisy_circuit=stim.Circuit("""
            H 0
            TICK
            MZ 0
            DETECTOR rec[-1]
        """))
    # Error event: X_ERROR at timeslice 0, qubit 0 (after H)
    fault = (0, "X_ERROR", (stim.GateTarget(0),))
    signature, effect = circuit.get_signature_and_effect(fault)
    # X anticommutes with MZ, so signature flips
    assert np.array_equal(signature, np.array([True]))
    assert effect == "X"

def test_no_signature_for_commuting_pauli():
    circuit = CultivationCircuit(noisy_circuit=stim.Circuit("""H 0
        MZ 0
        DETECTOR rec[-1]"""))
    # Error event: Z_ERROR at timeslice 0, qubit 0 (commutes with MZ)
    fault = (0, "Z_ERROR", (stim.GateTarget(0),))
    signature, effect = circuit.get_signature_and_effect(fault)
    assert np.array_equal(signature, np.array([False]))
    assert effect == "Z"

def test_reset_removes_pauli():
    # Circuit: X_ERROR at t=0, qubit 0; R 0 at t=1
    circuit = CultivationCircuit(noisy_circuit=stim.Circuit("""TICK
                                   R 0"""))
    fault = (0, "X_ERROR", (stim.GateTarget(0),))
    _, effect = circuit.get_signature_and_effect(fault)
    # After reset, effect should be identity
    assert effect == "_"

def test_multiple_qubits_and_detectors(multi_qubit_detector_circuit: CultivationCircuit):
    fault = (0, "X_ERROR", (stim.GateTarget(1),))
    signature, effect = multi_qubit_detector_circuit.get_signature_and_effect(fault)
    # X on qubit 1 after CX is X1
    assert effect == "_X"
    # Should only flip detector 1 if X on 1 anticommutes with MZ 1
    assert np.array_equal(signature, np.array([False, True]))


@pytest.mark.parametrize(
    'instruction_text',
    [
        'H 0',
        'S 0',
        'S_DAG 0',
        'CX 0 1',
        'CZ 0 1',
        'SWAP 0 1',
        'ISWAP 0 1',
        'SQRT_XX 0 1',
        'SPP X0*Z1',
    ],
)
def test_reverse_responses_match_stim_clifford_conjugation(
        instruction_text: str,
):
    """Match every local X/Y/Z response with Stim conjugation."""
    instruction = stim.CircuitInstruction(instruction_text)
    circuit = CultivationCircuit(stim.Circuit(
        f'TICK\n{instruction_text}',
    ))

    for qubit in range(circuit.noisy_circuit.num_qubits):
        for basis in 'XYZ':
            event = (
                0,
                f'{basis}_ERROR',
                (stim.GateTarget(qubit),),
            )
            signature, effect = circuit.get_signature_and_effect(event)
            generator = stim.PauliString(circuit.noisy_circuit.num_qubits)
            generator[qubit] = basis
            expected = forget_sign(generator.after(instruction))

            assert not signature.any()
            assert effect == expected


@pytest.mark.parametrize(
    ('measurement_name', 'commuting_basis', 'anticommuting_basis'),
    [
        ('M', 'Z', 'X'),
        ('MX', 'X', 'Z'),
        ('MY', 'Y', 'X'),
    ],
)
def test_reverse_responses_respect_measurement_basis(
        measurement_name: str,
        commuting_basis: str,
        anticommuting_basis: str,
):
    """Flip a detector exactly when an incoming Pauli anticommutes."""
    circuit = CultivationCircuit(stim.Circuit(f"""
        TICK
        {measurement_name} 0
        DETECTOR rec[-1]
    """))

    commuting_signature, commuting_effect = circuit.get_signature_and_effect((
        0,
        f'{commuting_basis}_ERROR',
        (stim.GateTarget(0),),
    ))
    anticommuting_signature, anticommuting_effect = (
        circuit.get_signature_and_effect((
            0,
            f'{anticommuting_basis}_ERROR',
            (stim.GateTarget(0),),
        ))
    )

    assert np.array_equal(commuting_signature, np.array([False]))
    assert commuting_effect == commuting_basis
    assert np.array_equal(anticommuting_signature, np.array([True]))
    assert anticommuting_effect == anticommuting_basis


@pytest.mark.parametrize(
    ('measurement_name', 'commuting_basis', 'anticommuting_basis'),
    [
        ('MR', 'Z', 'X'),
        ('MRX', 'X', 'Z'),
        ('MRY', 'Y', 'X'),
    ],
)
def test_measurement_resets_erase_final_effects(
        measurement_name: str,
        commuting_basis: str,
        anticommuting_basis: str,
):
    """Retain measurement response while erasing the incoming Pauli."""
    circuit = CultivationCircuit(stim.Circuit(f"""
        TICK
        {measurement_name} 0
        DETECTOR rec[-1]
    """))

    commuting_analysis = circuit.get_signature_and_effect((
        0,
        f'{commuting_basis}_ERROR',
        (stim.GateTarget(0),),
    ))
    anticommuting_analysis = circuit.get_signature_and_effect((
        0,
        f'{anticommuting_basis}_ERROR',
        (stim.GateTarget(0),),
    ))

    assert np.array_equal(commuting_analysis[0], np.array([False]))
    assert commuting_analysis[1] == '_'
    assert np.array_equal(anticommuting_analysis[0], np.array([True]))
    assert anticommuting_analysis[1] == '_'


@pytest.mark.parametrize('reset_name', ['R', 'RX', 'RY'])
def test_pure_resets_erase_both_generator_responses(reset_name: str):
    """Erase every incoming Pauli across each pure-reset basis."""
    circuit = CultivationCircuit(stim.Circuit(f'TICK\n{reset_name} 0'))

    for basis in 'XYZ':
        signature, effect = circuit.get_signature_and_effect((
            0,
            f'{basis}_ERROR',
            (stim.GateTarget(0),),
        ))
        assert not signature.any()
        assert effect == '_'


def test_detector_parity_xors_generator_responses():
    """Cancel a detector flipped by both halves of a correlated Pauli."""
    circuit = CultivationCircuit(stim.Circuit("""
        TICK
        M 0 1
        DETECTOR rec[-1] rec[-2]
    """))
    event = (
        0,
        'E',
        (stim.target_x(0), stim.target_x(1)),
    )

    signature, effect = circuit.get_signature_and_effect(event)

    assert np.array_equal(signature, np.array([False]))
    assert effect == 'XX'


def test_mpp_response_uses_each_target_pauli_basis():
    """Propagate detector response through a mixed-Pauli measurement."""
    circuit = CultivationCircuit(stim.Circuit("""
        TICK
        MPP X0*Y1
        DETECTOR rec[-1]
    """))

    signature, effect = circuit.get_signature_and_effect((
        0,
        'Z_ERROR',
        (stim.GateTarget(0),),
    ))

    assert np.array_equal(signature, np.array([True]))
    assert effect == 'Z_'


def test_measurement_events_use_precomputed_global_indices():
    """Resolve several same-layer measurements without revisiting the layer."""
    circuit = CultivationCircuit(stim.Circuit("""
        M(0.001) 2 0
        MX(0.001) 1
        DETECTOR rec[-3] rec[-1]
        DETECTOR rec[-2] rec[-1]
    """))
    reverse_data = circuit._reverse_propagation_data
    expected_indices = {
        (0, (2,)): 0,
        (0, (0,)): 1,
        (0, (1,)): 2,
    }
    assert reverse_data.measurement_index_by_event == expected_indices
    circuit.__dict__['_noiseless_layers'] = None

    events_and_signature_masks = [
        ((0, 'M', (stim.GateTarget(2),)), 0b01),
        ((0, 'M', (stim.GateTarget(0),)), 0b10),
        ((0, 'MX', (stim.GateTarget(1),)), 0b11),
    ]
    for event, expected_signature_mask in events_and_signature_masks:
        signature_mask, effect_mask = circuit._get_signature_and_effect_masks(
            event,
        )
        assert signature_mask == expected_signature_mask
        assert effect_mask == 0
