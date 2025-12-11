import pytest
import stim
from stim import PauliString

from cliffordep.pauli_string_tools import push_through_transversal, CliffordString, _tensor_paulis, split_sign


@pytest.mark.parametrize("gate", ["T", "S", "Z"])
def test_identity(gate):
    ps = "__"
    result = push_through_transversal(ps, gate=gate)
    assert isinstance(result, CliffordString)
    assert dict(result.terms) == {ps: 1}

@pytest.mark.parametrize("gate", ["T", "S", "Z"])
def test_empty(gate):
    ps = ''
    result = push_through_transversal(ps, gate=gate)
    assert dict(result.terms) == {ps: 1}


class TestTensorPaulis:

    def test_empty(self):
        assert _tensor_paulis() == PauliString()
    
    def test_1(self):
        assert _tensor_paulis("-iX") == PauliString("-iX")

    def test_2(self):
        assert _tensor_paulis("X", "Y") == PauliString("XY")
        assert _tensor_paulis("-X", "Y") == PauliString("-XY")
        assert _tensor_paulis("X", "-Y") == PauliString("-XY")
        assert _tensor_paulis("-X", "-Y") == PauliString("XY")
        assert _tensor_paulis("Y", "-iZ") == PauliString("-iYZ")
        assert _tensor_paulis("iY", "-iZ") == PauliString("YZ")

    def test_3(self):
        assert _tensor_paulis("-X", "I", "-iZ") == PauliString("iX_Z")


class TestTGate:

    GATE = "T"

    def test_x(self):
        ps = "X"
        result = push_through_transversal(ps, gate=self.GATE)
        assert dict(result.terms) == {"X": 1, "Y": 1}

    def test_y(self):
        ps = "Y"
        result = push_through_transversal(ps, gate=self.GATE)
        assert dict(result.terms) == {"X": -1, "Y": 1}

    def test_z(self):
        ps = "Z"
        result = push_through_transversal(ps, gate=self.GATE)
        assert dict(result.terms) == {"Z": 1}

    def test_mixed(self):
        ps = "X_"
        result = push_through_transversal(ps, gate=self.GATE)
        assert dict(result.terms) == {"X_": 1, "Y_": 1}

    def test_two_qubits(self):
        ps = "XY"
        result = push_through_transversal(ps, gate=self.GATE)
        assert dict(result.terms) == {"XX": -1, "XY": 1, "YX": -1, "YY": 1}

    def test_sign_rejected(self):
        ps = "-X"
        with pytest.raises(KeyError):
            push_through_transversal(ps, gate=self.GATE)


class TestSGate:

    GATE = "S"

    def test_x(self):
        ps = "X"
        result = push_through_transversal(ps, gate=self.GATE)
        assert dict(result.terms) == {"Y": 1}

    def test_y(self):
        ps = "Y"
        result = push_through_transversal(ps, gate=self.GATE)
        assert dict(result.terms) == {"X": -1}

    def test_z(self):
        ps = "Z"
        result = push_through_transversal(ps, gate=self.GATE)
        assert dict(result.terms) == {"Z": 1}

    def test_mixed(self):
        ps = "X_"
        result = push_through_transversal(ps, gate=self.GATE)
        assert dict(result.terms) == {"Y_": 1}

    def test_two_qubits(self):
        ps = "XY"
        result = push_through_transversal(ps, gate=self.GATE)
        assert dict(result.terms) == {"YX": -1}

    def test_sign_rejected(self):
        ps = "-X"
        with pytest.raises(KeyError):
            push_through_transversal(ps, gate=self.GATE)


class TestZGate:

    GATE = "Z"

    def test_x(self):
        ps = "X"
        result = push_through_transversal(ps, gate=self.GATE)
        assert dict(result.terms) == {"X": -1}

    def test_y(self):
        ps = "Y"
        result = push_through_transversal(ps, gate=self.GATE)
        assert dict(result.terms) == {"Y": -1}

    def test_z(self):
        ps = "Z"
        result = push_through_transversal(ps, gate=self.GATE)
        assert dict(result.terms) == {"Z": 1}

    def test_mixed(self):
        ps = "X_"
        result = push_through_transversal(ps, gate=self.GATE)
        assert dict(result.terms) == {"X_": -1}

    def test_two_qubits(self):
        ps = "XY"
        result = push_through_transversal(ps, gate=self.GATE)
        assert dict(result.terms) == {"XY": 1}

    def test_sign_rejected(self):
        ps = "-X"
        with pytest.raises(KeyError):
            push_through_transversal(ps, gate=self.GATE)


@pytest.mark.parametrize("gate", ["Z", "S", "S_DAG"])
@pytest.mark.parametrize("ps", ["_", "X", "Y", "Z"])
def test_against_pauli_string_after(gate, ps):
    instruction = stim.CircuitInstruction(gate, [0])
    after = PauliString(ps).after(instruction)
    sign, unsigned_ps = split_sign(after)
    assert push_through_transversal(ps, gate=gate).terms == {unsigned_ps: sign}