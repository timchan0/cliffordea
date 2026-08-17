import pytest
import stim

from cliffordep.circuits import OriginalD3ColorCodeLayout, D3DoubleCatCheckA6


@pytest.fixture
def original_d3_color_code():
    return D3DoubleCatCheckA6()


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

def test_stabilizer_generators_restricted(original_d3_color_code: OriginalD3ColorCodeLayout):
    assert original_d3_color_code.STABILIZER_GENERATORS_RESTRICTED == STABILIZER_GENERATORS_RESTRICTED

def test_logicals_restricted(original_d3_color_code: OriginalD3ColorCodeLayout):
    assert original_d3_color_code.LOGICAL_X_RESTRICTED == stim.PauliString('X0*X1*X3') * identity_tensor_7
    assert original_d3_color_code.LOGICAL_Z_RESTRICTED == stim.PauliString('Z0*Z1*Z3') * identity_tensor_7