import pytest
import stim

from cliffordep.circuits import _D3DoubleCatCheckA6Or7


@pytest.fixture
def d3_double_cat_check_a6_or_7():
    return _D3DoubleCatCheckA6Or7()


STABILIZER_GENERATORS_RESTRICTED = (
    stim.PauliString('X0*X1*X2*X4'),
    stim.PauliString('Z0*Z1*Z2*Z4'),
    stim.PauliString('X1*X2*X3*X5'),
    stim.PauliString('Z1*Z2*Z3*Z5'),
    stim.PauliString('X2*X4*X5*X6'),
    stim.PauliString('Z2*Z4*Z5*Z6'),
)

def test_stabilizer_generators_restricted(d3_double_cat_check_a6_or_7: _D3DoubleCatCheckA6Or7):
    assert d3_double_cat_check_a6_or_7.STABILIZER_GENERATORS_RESTRICTED == STABILIZER_GENERATORS_RESTRICTED

def test_logicals_restricted(d3_double_cat_check_a6_or_7: _D3DoubleCatCheckA6Or7):
    assert d3_double_cat_check_a6_or_7.LOGICAL_X_RESTRICTED == stim.PauliString('X0*X1*X3')
    assert d3_double_cat_check_a6_or_7.LOGICAL_Z_RESTRICTED == stim.PauliString('Z0*Z1*Z3')