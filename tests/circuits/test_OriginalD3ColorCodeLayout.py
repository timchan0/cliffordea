import pytest
import stim

from cliffordep.circuits import _OriginalD3ColorCodeLayout


@pytest.fixture
def original_d3_color_code():
    return _OriginalD3ColorCodeLayout()


STABILIZER_GENERATORS_RESTRICTED = (
    stim.PauliString('X0*X1*X2*X4'),
    stim.PauliString('Z0*Z1*Z2*Z4'),
    stim.PauliString('X1*X2*X3*X5'),
    stim.PauliString('Z1*Z2*Z3*Z5'),
    stim.PauliString('X2*X4*X5*X6'),
    stim.PauliString('Z2*Z4*Z5*Z6'),
)

def test_stabilizer_generators_restricted(original_d3_color_code: _OriginalD3ColorCodeLayout):
    assert original_d3_color_code.STABILIZER_GENERATORS_RESTRICTED == STABILIZER_GENERATORS_RESTRICTED

def test_logicals_restricted(original_d3_color_code: _OriginalD3ColorCodeLayout):
    assert original_d3_color_code.LOGICAL_X_RESTRICTED == stim.PauliString('X0*X1*X3')
    assert original_d3_color_code.LOGICAL_Z_RESTRICTED == stim.PauliString('Z0*Z1*Z3')