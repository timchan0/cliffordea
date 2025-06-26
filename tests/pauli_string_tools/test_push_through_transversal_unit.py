import pytest
from stim import PauliString

from cliffordep.pauli_string_tools import push_through_transversal, CliffordString


@pytest.mark.parametrize("gate", ["T", "S", "Z"])
def test_identity(gate):
    ps = PauliString("__")
    result = push_through_transversal(ps, gate=gate)
    assert isinstance(result, CliffordString)
    assert result.terms == [ps]

@pytest.mark.parametrize("gate", ["T", "S", "Z"])
def test_empty(gate):
    ps = PauliString()
    result = push_through_transversal(ps, gate=gate)
    assert result.terms == [ps]


class TestTGate:

    GATE = "T"

    def test_x(self):
        ps = PauliString("X")
        result = push_through_transversal(ps, gate=self.GATE)
        assert result.terms == [ps, PauliString("Y")]

    def test_y(self):
        ps = PauliString("Y")
        result = push_through_transversal(ps, gate=self.GATE)
        assert result.terms == [PauliString("-X"), ps]

    def test_z(self):
        ps = PauliString("Z")
        result = push_through_transversal(ps, gate=self.GATE)
        assert result.terms == [ps]

    def test_mixed(self):
        ps = PauliString("X_")
        result = push_through_transversal(ps, gate=self.GATE)
        assert result.terms == [ps, PauliString("Y_")]

    def test_two_qubits(self):
        ps = PauliString("XY")
        result = push_through_transversal(ps, gate=self.GATE)
        assert result.terms == [PauliString(s) for s in ["-XX", "XY", "-YX", "YY"]]

    def test_sign_preserved(self):
        ps = PauliString("-X")
        result = push_through_transversal(ps, gate=self.GATE)
        assert result.terms == [PauliString("-X"), PauliString("-Y")]


class TestSGate:

    GATE = "S"

    def test_x(self):
        ps = PauliString("X")
        result = push_through_transversal(ps, gate=self.GATE)
        assert result.terms == [PauliString("Y")]

    def test_y(self):
        ps = PauliString("Y")
        result = push_through_transversal(ps, gate=self.GATE)
        assert result.terms == [PauliString("-X")]

    def test_z(self):
        ps = PauliString("Z")
        result = push_through_transversal(ps, gate=self.GATE)
        assert result.terms == [ps]

    def test_mixed(self):
        ps = PauliString("X_")
        result = push_through_transversal(ps, gate=self.GATE)
        assert result.terms == [PauliString("Y_")]

    def test_two_qubits(self):
        ps = PauliString("XY")
        result = push_through_transversal(ps, gate=self.GATE)
        assert result.terms == [PauliString("-YX")]

    def test_sign_preserved(self):
        ps = PauliString("-X")
        result = push_through_transversal(ps, gate=self.GATE)
        assert result.terms == [PauliString("-Y")]


class TestZGate:

    GATE = "Z"

    def test_x(self):
        ps = PauliString("X")
        result = push_through_transversal(ps, gate=self.GATE)
        assert result.terms == [PauliString("-X")]

    def test_y(self):
        ps = PauliString("Y")
        result = push_through_transversal(ps, gate=self.GATE)
        assert result.terms == [PauliString("-Y")]

    def test_z(self):
        ps = PauliString("Z")
        result = push_through_transversal(ps, gate=self.GATE)
        assert result.terms == [ps]

    def test_mixed(self):
        ps = PauliString("X_")
        result = push_through_transversal(ps, gate=self.GATE)
        assert result.terms == [PauliString("-X_")]

    def test_two_qubits(self):
        ps = PauliString("XY")
        result = push_through_transversal(ps, gate=self.GATE)
        assert result.terms == [PauliString("XY")]

    def test_sign_preserved(self):
        ps = PauliString("-X")
        result = push_through_transversal(ps, gate=self.GATE)
        assert result.terms == [PauliString("X")]