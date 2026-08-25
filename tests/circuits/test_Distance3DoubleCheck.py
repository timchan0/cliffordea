import pytest
import stim

from cliffordep.circuits import Distance3DoubleCheck
from cliffordep.pauli_string_tools import forget_sign

@pytest.fixture
def distance3_ancilla6():
    return Distance3DoubleCheck(ancilla_count=6)


identity_tensor_7 = stim.PauliString(7)
STABILIZER_GENERATORS_RESTRICTED = {b: {
    f'{b}{b}{b}_{b}__',
    f'_{b}{b}{b}_{b}_',
    f'__{b}_{b}{b}{b}',
} for b in ('X', 'Z')}
@pytest.mark.parametrize("basis", ('X', 'Z'))
def test_stabilizer_generators_restricted(basis, distance3_ancilla6: Distance3DoubleCheck):
    set_ = {
        forget_sign(pauli_string) for pauli_string
        in distance3_ancilla6.STABILIZER_GENERATORS_RESTRICTED[basis]
    }
    assert set_ == STABILIZER_GENERATORS_RESTRICTED[basis]