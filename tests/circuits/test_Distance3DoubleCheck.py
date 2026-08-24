import pytest
import stim

from cliffordep import circuits
from cliffordep.circuits import Distance3DoubleCheck, D3A6


@pytest.fixture
def distance3_ancilla6():
    return D3A6()


identity_tensor_7 = stim.PauliString(7)
STABILIZER_GENERATORS_RESTRICTED = {
    'X': (
        stim.PauliString('X0*X1*X2*X4') * identity_tensor_7,
        stim.PauliString('X1*X2*X3*X5') * identity_tensor_7,
        stim.PauliString('X2*X4*X5*X6') * identity_tensor_7,
    ),
    'Z': (
        stim.PauliString('Z0*Z1*Z2*Z4') * identity_tensor_7,
        stim.PauliString('Z1*Z2*Z3*Z5') * identity_tensor_7,
        stim.PauliString('Z2*Z4*Z5*Z6') * identity_tensor_7,
    ),
}

def test_stabilizer_generators_restricted(distance3_ancilla6: Distance3DoubleCheck):
    assert distance3_ancilla6.STABILIZER_GENERATORS_RESTRICTED == STABILIZER_GENERATORS_RESTRICTED

def test_logicals_restricted(distance3_ancilla6: Distance3DoubleCheck):
    assert distance3_ancilla6.LOGICAL_X_RESTRICTED == stim.PauliString('X0*X1*X3') * identity_tensor_7
    assert distance3_ancilla6.LOGICAL_Z_RESTRICTED == stim.PauliString('Z0*Z1*Z3') * identity_tensor_7


def test_DATA_INDICES():
    """Test DATA_INDICES agrees with hardcoded values."""

    all_circuits: dict[
        tuple[int, int],
        circuits.Distance3DoubleCheck,
    ] = {
        (1, 0): circuits.D3A1(),
        (2, 0): circuits.D3A2(),
        (3, 0): circuits.D3A3(),
        (4, 0): circuits.D3A4(),
        (5, 0): circuits.D3A5(),
        (6, 0): circuits.D3A6(),
        (7, 0): circuits.D3A7(),
        (6, 2): circuits.D3A6F2(),
        (6, 3): circuits.D3A6F3(),
    }
    _DATA_INDICES = {
        (1, 0): (0, 1, 3, 4, 5, 6, 7),
        (2, 0): (0, 1, 3, 4, 5, 7, 8),
        (3, 0): (0, 2, 4, 5, 6, 8, 9),
        (4, 0): (0, 2, 4, 6, 7, 9, 10),
        (5, 0): (0, 3, 5, 7, 8, 10, 11),
        (6, 0): (0, 3, 5, 7, 8, 10, 11),
        (7, 0): (0, 3, 5, 7, 8, 10, 11),
        (6, 2): (0, 3, 7, 9, 10, 12, 13),
        (6, 3): (0, 3, 7, 9, 11, 13, 14),
    }
    for ancilla_flag_counts, circuit in all_circuits.items():
        assert circuit.DATA_INDICES == _DATA_INDICES[ancilla_flag_counts]