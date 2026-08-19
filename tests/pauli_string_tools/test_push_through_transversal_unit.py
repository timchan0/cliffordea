import pytest
import stim
from stim import PauliString

from cliffordep.combinators.fault_combinator import (
    _unsigned_pauli_string_to_mask,
)
from cliffordep.logical_analyzers import _TransversalGate
from cliffordep.pauli_string_tools import PauliSum, tensor_paulis, split_sign


def _mask(unsigned_string: str) -> int:
    """Pack a test Pauli into the analyzer's native mask representation.

    :param unsigned_string: The unsigned Pauli string to pack.
    :return: The packed X/Z-support mask.
    """
    return _unsigned_pauli_string_to_mask(unsigned_string)


@pytest.mark.parametrize("gate", ["T", "S", "Z"])
def test_identity(gate):
    ps = "__"
    transversal_gate = _TransversalGate(2*gate)
    result = transversal_gate.conjugate(_mask(ps))
    assert isinstance(result, PauliSum)
    assert dict(result.terms) == {ps: 1}


def test_empty():
    ps = ''
    transversal_gate = _TransversalGate('')
    result = transversal_gate.conjugate(_mask(ps))
    assert dict(result.terms) == {ps: 1}


class TestTensorPaulis:

    def test_empty(self):
        assert tensor_paulis() == PauliString()
    
    def test_1(self):
        assert tensor_paulis("-iX") == PauliString("-iX")

    def test_2(self):
        assert tensor_paulis("X", "Y") == PauliString("XY")
        assert tensor_paulis("-X", "Y") == PauliString("-XY")
        assert tensor_paulis("X", "-Y") == PauliString("-XY")
        assert tensor_paulis("-X", "-Y") == PauliString("XY")
        assert tensor_paulis("Y", "-iZ") == PauliString("-iYZ")
        assert tensor_paulis("iY", "-iZ") == PauliString("YZ")

    def test_3(self):
        assert tensor_paulis("-X", "I", "-iZ") == PauliString("iX_Z")


class TestTGate:

    T_TENSOR_1 = _TransversalGate("T")
    T_TENSOR_2 = _TransversalGate("TT")
    
    def test_x(self):
        ps = "X"
        result = self.T_TENSOR_1.conjugate(_mask(ps))
        assert dict(result.terms) == {"X": 1, "Y": 1}

    def test_y(self):
        ps = "Y"
        result = self.T_TENSOR_1.conjugate(_mask(ps))
        assert dict(result.terms) == {"X": -1, "Y": 1}

    def test_z(self):
        ps = "Z"
        result = self.T_TENSOR_1.conjugate(_mask(ps))
        assert dict(result.terms) == {"Z": 1}

    def test_mixed(self):
        ps = "X_"
        result = self.T_TENSOR_2.conjugate(_mask(ps))
        assert dict(result.terms) == {"X_": 1, "Y_": 1}

    def test_two_qubits(self):
        ps = "XY"
        result = self.T_TENSOR_2.conjugate(_mask(ps))
        assert dict(result.terms) == {"XX": -1, "XY": 1, "YX": -1, "YY": 1}


class TestSGate:

    S_TENSOR_1 = _TransversalGate("S")
    S_TENSOR_2 = _TransversalGate("SS")

    def test_x(self):
        ps = "X"
        result = self.S_TENSOR_1.conjugate(_mask(ps))
        assert dict(result.terms) == {"Y": 1}

    def test_y(self):
        ps = "Y"
        result = self.S_TENSOR_1.conjugate(_mask(ps))
        assert dict(result.terms) == {"X": -1}

    def test_z(self):
        ps = "Z"
        result = self.S_TENSOR_1.conjugate(_mask(ps))
        assert dict(result.terms) == {"Z": 1}

    def test_mixed(self):
        ps = "X_"
        result = self.S_TENSOR_2.conjugate(_mask(ps))
        assert dict(result.terms) == {"Y_": 1}

    def test_two_qubits(self):
        ps = "XY"
        result = self.S_TENSOR_2.conjugate(_mask(ps))
        assert dict(result.terms) == {"YX": -1}


class TestZGate:

    Z_TENSOR_1 = _TransversalGate("Z")
    Z_TENSOR_2 = _TransversalGate("ZZ")

    def test_x(self):
        ps = "X"
        result = self.Z_TENSOR_1.conjugate(_mask(ps))
        assert dict(result.terms) == {"X": -1}

    def test_y(self):
        ps = "Y"
        result = self.Z_TENSOR_1.conjugate(_mask(ps))
        assert dict(result.terms) == {"Y": -1}

    def test_z(self):
        ps = "Z"
        result = self.Z_TENSOR_1.conjugate(_mask(ps))
        assert dict(result.terms) == {"Z": 1}

    def test_mixed(self):
        ps = "X_"
        result = self.Z_TENSOR_2.conjugate(_mask(ps))
        assert dict(result.terms) == {"X_": -1}

    def test_two_qubits(self):
        ps = "XY"
        result = self.Z_TENSOR_2.conjugate(_mask(ps))
        assert dict(result.terms) == {"XY": 1}


@pytest.mark.parametrize("gate", ["Z", "S", "S_DAG"])
@pytest.mark.parametrize("ps", ["_", "X", "Y", "Z"])
def test_against_pauli_string_after(gate, ps):
    instruction = stim.CircuitInstruction(gate, [0])
    after = PauliString(ps).after(instruction)
    sign, unsigned_ps = split_sign(after)
    transversal_gate = _TransversalGate([gate])
    assert transversal_gate.conjugate(_mask(ps)).terms == {unsigned_ps: sign}
