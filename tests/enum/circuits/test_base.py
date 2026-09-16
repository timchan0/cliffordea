import pytest
import stim

from cliffordea.enum import circuits


@pytest.mark.parametrize("ancilla_count, flag_count", [
    (1, 0),
    (2, 0),
    (3, 0),
    (4, 0),
    (5, 0),
    (6, 0),
    (7, 0),
    (6, 3),
    (19, 0),
])
def test_find_data_indices(ancilla_count, flag_count):
    """Test DATA_INDICES agrees with hardcoded values."""
    if ancilla_count == 19:
        circuit = circuits.DoubleCheck(distance=5, ancilla_count=19, flag_count=flag_count)
    else:
        circuit = circuits.DoubleCheck(
            ancilla_count=ancilla_count,
            flag_count=flag_count,
        )
    _DATA_INDICES = {
        (1, 0): (0, 1, 3, 4, 5, 6, 7),
        (2, 0): (0, 1, 3, 4, 5, 7, 8),
        (3, 0): (0, 2, 4, 5, 6, 8, 9),
        (4, 0): (0, 1, 4, 5, 6, 9, 10),
        (5, 0): (0, 3, 5, 7, 8, 10, 11),
        (6, 0): (0, 3, 5, 7, 8, 10, 11),
        (7, 0): (0, 3, 5, 7, 8, 10, 11),
        (6, 3): (0, 3, 7, 9, 11, 13, 14),
        (19, 0): tuple(sorted((3, 5, 0, 9, 14, 22, 32, 29, 34, 31, 24, 26, 20, 18, 13, 7, 11, 16, 36)))
    }
    assert circuit.DATA_INDICES == _DATA_INDICES[ancilla_count, flag_count]


@pytest.mark.parametrize("distance, ancilla_count, flag_count, stabilizer_generators", [
    (3, 6, 0, {
        (0, 3, 5, 8),
        (3, 5, 7, 10),
        (5, 8, 10, 11),
    }),
    (5, 19, 0, {tuple(sorted(indices)) for indices in [
        (0, 9, 5, 3), (14, 32, 29, 22), (11, 16, 24, 18, 13, 7),
        (22, 29, 34, 31, 24, 16), (3, 5, 11, 7), (13, 18, 26, 20),
        (9, 14, 22, 16, 11, 5), (24, 31, 26, 18), (29, 32, 36, 34),
    ]}),
])
def test_find_stabilizer_generators(distance, ancilla_count, flag_count, stabilizer_generators):
    """Test STABILIZER_GENERATORS agrees with hardcoded values."""
    circuit = circuits.DoubleCheck(distance, ancilla_count, flag_count)
    assert set(circuit._STABILIZER_GENERATOR_INDICES) == stabilizer_generators


@pytest.mark.parametrize("ancilla_count, flag_count", [
    (19, 0),
    (19, 3),
    (19, 18),
])
def test_find_logical_s_gate_distance_5(ancilla_count, flag_count):
    """Test LOGICAL_S agrees with hardcoded values."""
    circuit = circuits.DoubleCheck(distance=5, ancilla_count=19, flag_count=flag_count)
    _LOGICAL_S = {
        (19, 0): stim.Circuit(
"""
S 0 5 7 14 16 18 20 29 31 36
S_DAG 9 11 13 22 24 26 34 32 3
"""
        ),
        (19, 3): stim.Circuit(
"""
S 16 18 20 22 32 34 6 8 0 39
S_DAG 3 10 12 14 26 24 28 35 37
"""
        ),
        (19, 18): stim.Circuit(
"""
S 21 23 25 27 44 46 7 9 0 54
S_DAG 4 15 35 17 19 39 37 51 49
"""
        ),
    }
    assert circuit.LOGICAL_S.to_tableau() == _LOGICAL_S[ancilla_count, flag_count].to_tableau()


@pytest.mark.parametrize("ancilla_count", (1, 2, 3, 5, 6, 7))
def test_find_logical_s_gate_distance_3(ancilla_count):
    """Test LOGICAL_S agrees with hardcoded values."""
    _MAJORITY_INDICES = (0, 2, 3, 6)
    # Indices of the qubits that have S applied to them for logical S gate
    # in the order given by `DATA_INDICES`.
    circuit = circuits.DoubleCheck(ancilla_count=ancilla_count)
    s_indices = {circuit.DATA_INDICES[index] for index in _MAJORITY_INDICES}
    for instruction in circuit.LOGICAL_S:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            raise NotImplementedError("LOGICAL_S should not contain repeat blocks.")
        if instruction.name == 'S':
            targets = {target.value for target in instruction.targets_copy()}
            assert targets == s_indices
        elif instruction.name != 'S_DAG':
            raise NotImplementedError(f"LOGICAL_S should only contain S and S_DAG instructions, got {instruction.name}.")
