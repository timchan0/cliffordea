import pytest
import stim

from cliffordep.pauli_string_tools import split_sign

@pytest.mark.parametrize("pauli", ('_', 'X', 'Y', 'Z'))
@pytest.mark.parametrize("prefix, sign", [('', 1), ('-', -1), ('i', 1j), ('-i', -1j)])
@pytest.mark.parametrize("qubit_count", (1, 2))
def test_split_sign(pauli, prefix, sign, qubit_count):
    unsigned = qubit_count * pauli
    s = stim.PauliString(prefix + unsigned)
    assert split_sign(s) == (sign, unsigned)