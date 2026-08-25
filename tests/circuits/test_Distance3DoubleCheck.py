import pytest
import stim

from cliffordep.circuits import Distance3DoubleCheck


@pytest.fixture
def distance3_ancilla6():
    return Distance3DoubleCheck(ancilla_count=6)


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