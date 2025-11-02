import stim

class _OriginalD3ColorCodeLayout:
    """Common constants for the distance-3 double cat-check circuit.
    
    Class constants:
    * `DATA_INDICES` the indices of the data qubits.
    * `ANCILLA_INDICES` the indices of the ancilla qubits.
    * `FULL_CIRCUIT` the full double check circuit.
    * `INNER_CIRCUIT` the double check circuit _within_ the transversal T gates:
        * The first timeslice of this circuit is an MPP measurement of all data qubits,
        which the first logical H_XY measurement is checked against.
        * If the ancilla qubit is used again after measurement,
        the measurement should be MR instead of M (or MRX instead of MX).
        When M/MX/MY is applied to a qubit,
        this indicates the qubit will no longer be used
        so the noise model will no longer apply noise to it.
        * The last timeslice of this circuit is a reset of all non-data qubits.
        This is important as it lowers the number of errors that must be enumerated,
        thus improving the numeric performance considerably.
    """

    _STABILIZER_GENERATOR_INDICES = (
        (0, 1, 2, 4),
        (1, 2, 3, 5),
        (2, 4, 5, 6),
    )
    """Indices of the stabilizer generators restricted to the 7 data qubits
    in the order given by `DATA_INDICES`.
    """

    _LOGICAL_INDICES = (0, 1, 3)
    """Indices of the logicals restricted to the 7 data qubits
    in the order given by `DATA_INDICES`.
    """

    STABILIZER_GENERATORS_RESTRICTED = tuple(
        stim.PauliString('*'.join(f'{basis}{index}' for index in indices))
        for indices in _STABILIZER_GENERATOR_INDICES
        for basis in ('X', 'Z')
    )
    """Generators restricted to the 7 data qubits
    in the order given by `DATA_INDICES`.
    """

    INNER_CIRCUIT: stim.Circuit

    def __init__(self):
        self.DATA_INDICES: tuple[int, ...]
        self.STABILIZER_GENERATORS = tuple(
        stim.PauliString('*'.join(f'{basis}{self.DATA_INDICES[index]}' for index in indices))
        for indices in self._STABILIZER_GENERATOR_INDICES
        for basis in ('X', 'Z')
    )
        (self.LOGICAL_X_RESTRICTED, self.LOGICAL_Z_RESTRICTED) = tuple(
            stim.PauliString('*'.join(f'{basis}{index}' for index in self._LOGICAL_INDICES))
            for basis in ('X', 'Z')
        )
        (self.LOGICAL_X, self.LOGICAL_Z) = tuple(
            stim.PauliString('*'.join(f'{basis}{self.DATA_INDICES[index]}' for index in self._LOGICAL_INDICES))
            for basis in ('X', 'Z')
        )


class D3DoubleCatCheckA1(_OriginalD3ColorCodeLayout):
    """The distance-3 double cat-check circuit using 1 ancilla."""
    
    DATA_INDICES = (0, 1, 3, 4, 5, 6, 7)

    ANCILLA_INDICES = (2,)

    INNER_CIRCUIT = stim.Circuit(
"""
QUBIT_COORDS(0, 0) 0
QUBIT_COORDS(1, 1) 1
QUBIT_COORDS(2, 0) 2
QUBIT_COORDS(2, 1) 3
QUBIT_COORDS(2, 3) 4
QUBIT_COORDS(3, 0) 5
QUBIT_COORDS(3, 2) 6
QUBIT_COORDS(4, 0) 7
#!pragma POLYGON(0,0,1,0.25) 5 7 6 3
#!pragma POLYGON(0,1,0,0.25) 4 6 3 1
#!pragma POLYGON(1,0,0,0.25) 1 3 5 0
MPP X0*X1*X3*X4*X5*X6*X7
TICK
RX 2
TICK
CX 2 3
TICK
CX 3 0
TICK
CX 3 1
TICK
CX 3 4
TICK
CX 3 5
TICK
CX 3 6
TICK
CX 3 7
TICK
MRX 3
DETECTOR(0, 0, 0) rec[-1] rec[-2]
TICK
CX 3 7
TICK
CX 3 6
TICK
CX 3 5
TICK
CX 3 4
TICK
CX 3 1
TICK
CX 3 0
TICK
CX 2 3
TICK
MX 2
DETECTOR(2, 1, 1) rec[-1] rec[-2]
TICK
RX 2
"""
    )

    CIRCUIT = stim.Circuit(
"""
QUBIT_COORDS(0, 0) 0
QUBIT_COORDS(1, 1) 1
QUBIT_COORDS(2, 0) 2
QUBIT_COORDS(2, 1) 3
QUBIT_COORDS(2, 3) 4
QUBIT_COORDS(3, 0) 5
QUBIT_COORDS(3, 2) 6
QUBIT_COORDS(4, 0) 7
#!pragma POLYGON(0,0,1,0.25) 5 7 6 3
#!pragma POLYGON(0,1,0,0.25) 4 6 3 1
#!pragma POLYGON(1,0,0,0.25) 1 3 5 0
TICK
MPP Y0*Y5*Y7*Y6*Y4*Y3*Y1
TICK
MPP X5*X0*X1*X3
TICK
MPP X1*X3*X6*X4
TICK
MPP X7*X5*X3*X6
TICK
MPP Z5*Z0*Z1*Z3
TICK
MPP Z4*Z6*Z3*Z1
TICK
MPP Z7*Z5*Z3*Z6
TICK
RX 2
TICK
S_DAG 0 1 3 4 5 6 7
TICK
CX 2 3
TICK
CX 3 0
TICK
CX 3 1
TICK
CX 3 4
TICK
CX 3 5
TICK
CX 3 6
TICK
CX 3 7
TICK
MX 3
DETECTOR(0, 0, 0) rec[-1] rec[-8]
TICK
RX 3
TICK
CX 3 7
TICK
CX 3 6
TICK
CX 3 5
TICK
CX 3 4
TICK
CX 3 1
TICK
CX 3 0
TICK
CX 2 3
TICK
S 3 6 7 5 4 1 0
TICK
MX 2
DETECTOR(2, 1, 1) rec[-1] rec[-2]
TICK
MPP X5*X0*X1*X3
DETECTOR(3, 0, 2) rec[-1] rec[-3] rec[-9]
TICK
MPP X4*X6*X3*X1
DETECTOR(1, 1, 3) rec[-1] rec[-4] rec[-9]
TICK
MPP X7*X5*X3*X6
DETECTOR(4, 0, 4) rec[-1] rec[-5] rec[-9]
TICK
MPP Z5*Z0*Z1*Z3
DETECTOR(3, 0, 5) rec[-1] rec[-9]
TICK
MPP Z4*Z6*Z3*Z1
DETECTOR(2, 3, 6) rec[-1] rec[-9]
TICK
MPP Z7*Z5*Z3*Z6
DETECTOR(4, 0, 7) rec[-1] rec[-9]
TICK
MPP Y0*Y5*Y7*Y6*Y4*Y3*Y1
OBSERVABLE_INCLUDE(0) rec[-1]
"""
    )


class D3DoubleCatCheckA2(_OriginalD3ColorCodeLayout):
    """The distance-3 double cat-check circuit using 2 ancillas."""

    DATA_INDICES = (0, 1, 3, 4, 5, 7, 8)

    ANCILLA_INDICES = (2, 6)

    INNER_CIRCUIT = stim.Circuit(
"""
QUBIT_COORDS(0, 0) 0
QUBIT_COORDS(1, 1) 1
QUBIT_COORDS(2, 0) 2
QUBIT_COORDS(2, 1) 3
QUBIT_COORDS(2, 3) 4
QUBIT_COORDS(3, 0) 5
QUBIT_COORDS(3, 1) 6
QUBIT_COORDS(3, 2) 7
QUBIT_COORDS(4, 0) 8
#!pragma POLYGON(0,0,1,0.25) 5 8 7 3
#!pragma POLYGON(0,1,0,0.25) 4 7 3 1
#!pragma POLYGON(1,0,0,0.25) 1 3 5 0
MPP X0*X1*X3*X4*X5*X7*X8
TICK
RX 6 2
TICK
CX 6 5 2 3
TICK
CX 6 8 3 4
TICK
CX 3 1 6 7
TICK
CX 3 6
TICK
CX 3 0
TICK
MRX 3
DETECTOR(0, 0, 0) rec[-1] rec[-2]
TICK
CX 3 0
TICK
CX 3 6
TICK
CX 3 1 6 7
TICK
CX 6 8 3 4
TICK
CX 6 5 2 3
TICK
MX 6 2
DETECTOR(3, 1, 1) rec[-2]
DETECTOR(2, 1, 1) rec[-1] rec[-3]
TICK
RX 6 2
"""
    )

    CIRCUIT = stim.Circuit(
"""
QUBIT_COORDS(0, 0) 0
QUBIT_COORDS(1, 1) 1
QUBIT_COORDS(2, 0) 2
QUBIT_COORDS(2, 1) 3
QUBIT_COORDS(2, 3) 4
QUBIT_COORDS(3, 0) 5
QUBIT_COORDS(3, 1) 6
QUBIT_COORDS(3, 2) 7
QUBIT_COORDS(4, 0) 8
#!pragma POLYGON(0,0,1,0.25) 5 8 7 3
#!pragma POLYGON(0,1,0,0.25) 4 7 3 1
#!pragma POLYGON(1,0,0,0.25) 1 3 5 0
TICK
MPP Y0*Y5*Y8*Y7*Y4*Y3*Y1
TICK
MPP X5*X0*X1*X3
TICK
MPP X1*X3*X7*X4
TICK
MPP X8*X5*X3*X7
TICK
MPP Z5*Z0*Z1*Z3
TICK
MPP Z4*Z7*Z3*Z1
TICK
MPP Z8*Z5*Z3*Z7
TICK
RX 6 2
TICK
S_DAG 0 1 3 4 5 7 8
TICK
CX 6 5 2 3
TICK
CX 6 8 3 4
TICK
CX 3 1 6 7
TICK
CX 3 6
TICK
CX 3 0
TICK
MX 3
DETECTOR(0, 0, 0) rec[-1] rec[-8]
TICK
RX 3
TICK
CX 3 0
TICK
CX 3 6
TICK
CX 3 1 6 7
TICK
CX 6 8 3 4
TICK
CX 6 5 2 3
TICK
S 3 7 8 5 4 1 0
TICK
MX 6 2
DETECTOR(3, 1, 1) rec[-2]
DETECTOR(2, 1, 1) rec[-1] rec[-3]
TICK
MPP X5*X0*X1*X3
DETECTOR(3, 0, 2) rec[-1] rec[-4] rec[-10]
TICK
MPP X4*X7*X3*X1
DETECTOR(1, 1, 3) rec[-1] rec[-5] rec[-10]
TICK
MPP X8*X5*X3*X7
DETECTOR(4, 0, 4) rec[-1] rec[-6] rec[-10]
TICK
MPP Z5*Z0*Z1*Z3
DETECTOR(3, 0, 5) rec[-1] rec[-10]
TICK
MPP Z4*Z7*Z3*Z1
DETECTOR(2, 3, 6) rec[-1] rec[-10]
TICK
MPP Z8*Z5*Z3*Z7
DETECTOR(4, 0, 7) rec[-1] rec[-10]
TICK
MPP Y0*Y5*Y8*Y7*Y4*Y3*Y1
OBSERVABLE_INCLUDE(0) rec[-1] rec[-9]
"""
    )


class D3DoubleCatCheckA3(_OriginalD3ColorCodeLayout):
    """The distance-3 double cat-check circuit using 3 ancillas."""

    DATA_INDICES = (0, 2, 4, 5, 6, 8, 9)

    ANCILLA_INDICES = (1, 3, 7)

    INNER_CIRCUIT = stim.Circuit(
"""
QUBIT_COORDS(0, 0) 0
QUBIT_COORDS(1, 0) 1
QUBIT_COORDS(1, 1) 2
QUBIT_COORDS(2, 0) 3
QUBIT_COORDS(2, 1) 4
QUBIT_COORDS(2, 3) 5
QUBIT_COORDS(3, 0) 6
QUBIT_COORDS(3, 1) 7
QUBIT_COORDS(3, 2) 8
QUBIT_COORDS(4, 0) 9
#!pragma POLYGON(0,0,1,0.25) 6 9 8 4
#!pragma POLYGON(0,1,0,0.25) 5 8 4 2
#!pragma POLYGON(1,0,0,0.25) 2 4 6 0
MPP X0*X2*X4*X5*X6*X8*X9
TICK
RX 7 3 1
TICK
CX 7 6 3 4 1 2
TICK
CX 7 9 2 0 4 5
TICK
CX 4 2 7 8
TICK
CX 4 7
TICK
MRX 4
DETECTOR(0, 0, 0) rec[-1] rec[-2]
TICK
CX 4 7
TICK
CX 4 2 7 8
TICK
CX 7 9 2 0 4 5
TICK
CX 7 6 3 4 1 2
TICK
MX 7 3 1
DETECTOR(3, 1, 1) rec[-3]
DETECTOR(2, 1, 1) rec[-2] rec[-4]
DETECTOR(1, 0, 1) rec[-1]
TICK
RX 7 3 1
"""
    )

    CIRCUIT = stim.Circuit(
"""
QUBIT_COORDS(0, 0) 0
QUBIT_COORDS(1, 0) 1
QUBIT_COORDS(1, 1) 2
QUBIT_COORDS(2, 0) 3
QUBIT_COORDS(2, 1) 4
QUBIT_COORDS(2, 3) 5
QUBIT_COORDS(3, 0) 6
QUBIT_COORDS(3, 1) 7
QUBIT_COORDS(3, 2) 8
QUBIT_COORDS(4, 0) 9
#!pragma POLYGON(0,0,1,0.25) 6 9 8 4
#!pragma POLYGON(0,1,0,0.25) 5 8 4 2
#!pragma POLYGON(1,0,0,0.25) 2 4 6 0
TICK
MPP Y0*Y6*Y9*Y8*Y5*Y4*Y2
TICK
MPP X6*X0*X2*X4
TICK
MPP X2*X4*X8*X5
TICK
MPP X9*X6*X4*X8
TICK
MPP Z6*Z0*Z2*Z4
TICK
MPP Z5*Z8*Z4*Z2
TICK
MPP Z9*Z6*Z4*Z8
TICK
RX 7 3 1
TICK
S_DAG 0 2 4 5 6 8 9
TICK
CX 7 6 3 4 1 2
TICK
CX 7 9 2 0 4 5
TICK
CX 4 2 7 8
TICK
CX 4 7
TICK
MX 4
DETECTOR(0, 0, 0) rec[-1] rec[-8]
TICK
RX 4
TICK
CX 4 7
TICK
CX 4 2 7 8
TICK
CX 7 9 2 0 4 5
TICK
CX 7 6 3 4 1 2
TICK
S 4 8 9 6 5 2 0
TICK
MX 7 3 1
DETECTOR(3, 1, 1) rec[-3]
DETECTOR(2, 1, 1) rec[-2] rec[-4]
DETECTOR(1, 0, 1) rec[-1]
TICK
MPP X6*X0*X2*X4
DETECTOR(3, 0, 2) rec[-1] rec[-5] rec[-11]
TICK
MPP X5*X8*X4*X2
DETECTOR(1, 1, 3) rec[-1] rec[-6] rec[-11]
TICK
MPP X9*X6*X4*X8
DETECTOR(4, 0, 4) rec[-1] rec[-7] rec[-11]
TICK
MPP Z6*Z0*Z2*Z4
DETECTOR(3, 0, 5) rec[-1] rec[-11]
TICK
MPP Z5*Z8*Z4*Z2
DETECTOR(2, 3, 6) rec[-1] rec[-11]
TICK
MPP Z9*Z6*Z4*Z8
DETECTOR(4, 0, 7) rec[-1] rec[-11]
TICK
MPP Y0*Y6*Y9*Y8*Y5*Y4*Y2
OBSERVABLE_INCLUDE(0) rec[-1] rec[-10]
"""
    )


class D3DoubleCatCheckA4(_OriginalD3ColorCodeLayout):
    """The distance-3 double cat-check circuit using 4 ancillas."""

    DATA_INDICES = (0, 2, 4, 6, 7, 9, 10)

    ANCILLA_INDICES = (1, 3, 5, 8)

    INNER_CIRCUIT = stim.Circuit(
"""
QUBIT_COORDS(0, 0) 0
QUBIT_COORDS(1, 0) 1
QUBIT_COORDS(1, 1) 2
QUBIT_COORDS(2, 0) 3
QUBIT_COORDS(2, 1) 4
QUBIT_COORDS(2, 2) 5
QUBIT_COORDS(2, 3) 6
QUBIT_COORDS(3, 0) 7
QUBIT_COORDS(3, 1) 8
QUBIT_COORDS(3, 2) 9
QUBIT_COORDS(4, 0) 10
#!pragma POLYGON(0,0,1,0.25) 7 10 9 4
#!pragma POLYGON(0,1,0,0.25) 6 9 4 2
#!pragma POLYGON(1,0,0,0.25) 2 4 7 0
MPP X0*X2*X4*X6*X7*X9*X10
TICK
RX 8 3 1 5
TICK
CX 8 7 5 6 3 4 1 2
TICK
CX 4 5 8 10 2 0
TICK
CX 4 2 8 9
TICK
CX 4 8
TICK
MRX 4
DETECTOR(0, 0, 0) rec[-1] rec[-2]
TICK
CX 4 8
TICK
CX 4 2 8 9
TICK
CX 4 5 8 10 2 0
TICK
CX 8 7 5 6 3 4 1 2
TICK
MX 8 3 1 5
DETECTOR(3, 1, 1) rec[-4]
DETECTOR(2, 1, 1) rec[-3] rec[-5]
DETECTOR(1, 0, 1) rec[-2]
DETECTOR(2, 2, 1) rec[-1]
TICK
RX 8 3 1 5
"""
    )

    CIRCUIT = stim.Circuit(
"""
QUBIT_COORDS(0, 0) 0
QUBIT_COORDS(1, 0) 1
QUBIT_COORDS(1, 1) 2
QUBIT_COORDS(2, 0) 3
QUBIT_COORDS(2, 1) 4
QUBIT_COORDS(2, 2) 5
QUBIT_COORDS(2, 3) 6
QUBIT_COORDS(3, 0) 7
QUBIT_COORDS(3, 1) 8
QUBIT_COORDS(3, 2) 9
QUBIT_COORDS(4, 0) 10
#!pragma POLYGON(0,0,1,0.25) 7 10 9 4
#!pragma POLYGON(0,1,0,0.25) 6 9 4 2
#!pragma POLYGON(1,0,0,0.25) 2 4 7 0
TICK
MPP Y0*Y7*Y10*Y9*Y6*Y4*Y2
TICK
MPP X7*X0*X2*X4
TICK
MPP X2*X4*X9*X6
TICK
MPP X10*X7*X4*X9
TICK
MPP Z7*Z0*Z2*Z4
TICK
MPP Z6*Z9*Z4*Z2
TICK
MPP Z10*Z7*Z4*Z9
TICK
RX 8 3 1 5
TICK
S_DAG 0 2 4 6 7 9 10
TICK
CX 8 7 5 6 3 4 1 2
TICK
CX 4 5 8 10 2 0
TICK
CX 4 2 8 9
TICK
CX 4 8
TICK
MX 4
DETECTOR(0, 0, 0) rec[-1] rec[-8]
TICK
RX 4
TICK
CX 4 8
TICK
CX 4 2 8 9
TICK
CX 4 5 8 10 2 0
TICK
CX 8 7 5 6 3 4 1 2
TICK
S 4 9 10 7 6 2 0
TICK
MX 8 3 1 5
DETECTOR(3, 1, 1) rec[-4]
DETECTOR(2, 1, 1) rec[-3] rec[-5]
DETECTOR(1, 0, 1) rec[-2]
DETECTOR(2, 2, 1) rec[-1]
TICK
MPP X7*X0*X2*X4
DETECTOR(3, 0, 2) rec[-1] rec[-6] rec[-12]
TICK
MPP X6*X9*X4*X2
DETECTOR(1, 1, 3) rec[-1] rec[-7] rec[-12]
TICK
MPP X10*X7*X4*X9
DETECTOR(4, 0, 4) rec[-1] rec[-8] rec[-12]
TICK
MPP Z7*Z0*Z2*Z4
DETECTOR(3, 0, 5) rec[-1] rec[-12]
TICK
MPP Z6*Z9*Z4*Z2
DETECTOR(2, 3, 6) rec[-1] rec[-12]
TICK
MPP Z10*Z7*Z4*Z9
DETECTOR(4, 0, 7) rec[-1] rec[-12]
TICK
MPP Y0*Y7*Y10*Y9*Y6*Y4*Y2
OBSERVABLE_INCLUDE(0) rec[-1] rec[-8] rec[-11]
"""
    )


class D3DoubleCatCheckA5(_OriginalD3ColorCodeLayout):
    """The distance-3 double cat-check circuit using 5 ancillas."""

    DATA_INDICES = (0, 3, 5, 7, 8, 10, 11)

    ANCILLA_INDICES = (1, 2, 4, 6, 9)

    INNER_CIRCUIT = stim.Circuit(
"""
QUBIT_COORDS(0, 0) 0
QUBIT_COORDS(0, 1) 1
QUBIT_COORDS(1, 0) 2
QUBIT_COORDS(1, 1) 3
QUBIT_COORDS(2, 0) 4
QUBIT_COORDS(2, 1) 5
QUBIT_COORDS(2, 2) 6
QUBIT_COORDS(2, 3) 7
QUBIT_COORDS(3, 0) 8
QUBIT_COORDS(3, 1) 9
QUBIT_COORDS(3, 2) 10
QUBIT_COORDS(4, 0) 11
#!pragma POLYGON(0,0,1,0.25) 8 11 10 5
#!pragma POLYGON(0,1,0,0.25) 7 10 5 3
#!pragma POLYGON(1,0,0,0.25) 3 5 8 0
MPP X0*X3*X5*X7*X8*X10*X11
TICK
RX 9 4 2 6 1
TICK
CX 1 0 9 8 6 7 4 5 2 3
TICK
CX 3 1 5 6 9 11
TICK
CX 5 3 9 10
TICK
CX 5 9
TICK
MRX 5
DETECTOR(0, 0, 0) rec[-1] rec[-2]
TICK
CX 5 9
TICK
CX 5 3 9 10
TICK
CX 3 1 5 6 9 11
TICK
CX 1 0 9 8 6 7 4 5 2 3
TICK
MX 9 4 2 6 1
DETECTOR(3, 1, 1) rec[-5]
DETECTOR(2, 1, 1) rec[-4] rec[-6]
DETECTOR(1, 0, 1) rec[-3]
DETECTOR(2, 2, 1) rec[-2]
DETECTOR(0, 1, 1) rec[-1]
TICK
RX 9 4 2 6 1
"""
    )

    CIRCUIT = stim.Circuit(
"""
QUBIT_COORDS(0, 0) 0
QUBIT_COORDS(0, 1) 1
QUBIT_COORDS(1, 0) 2
QUBIT_COORDS(1, 1) 3
QUBIT_COORDS(2, 0) 4
QUBIT_COORDS(2, 1) 5
QUBIT_COORDS(2, 2) 6
QUBIT_COORDS(2, 3) 7
QUBIT_COORDS(3, 0) 8
QUBIT_COORDS(3, 1) 9
QUBIT_COORDS(3, 2) 10
QUBIT_COORDS(4, 0) 11
#!pragma POLYGON(0,0,1,0.25) 8 11 10 5
#!pragma POLYGON(0,1,0,0.25) 7 10 5 3
#!pragma POLYGON(1,0,0,0.25) 3 5 8 0
TICK
MPP Y0*Y8*Y11*Y10*Y7*Y5*Y3
TICK
MPP X8*X0*X3*X5
TICK
MPP X3*X5*X10*X7
TICK
MPP X11*X8*X5*X10
TICK
MPP Z8*Z0*Z3*Z5
TICK
MPP Z7*Z10*Z5*Z3
TICK
MPP Z11*Z8*Z5*Z10
TICK
RX 9 4 2 6 1
TICK
S_DAG 0 3 5 7 8 10 11
TICK
CX 1 0 9 8 6 7 4 5 2 3
TICK
CX 3 1 5 6 9 11
TICK
CX 5 3 9 10
TICK
CX 5 9
TICK
MX 5
DETECTOR(0, 0, 0) rec[-1] rec[-8]
TICK
RX 5
TICK
CX 5 9
TICK
CX 5 3 9 10
TICK
CX 3 1 5 6 9 11
TICK
CX 1 0 9 8 6 7 4 5 2 3
TICK
S 5 10 11 8 7 3 0
TICK
MX 9 4 2 6 1
DETECTOR(3, 1, 1) rec[-5]
DETECTOR(2, 1, 1) rec[-4] rec[-6]
DETECTOR(1, 0, 1) rec[-3]
DETECTOR(2, 2, 1) rec[-2]
DETECTOR(0, 1, 1) rec[-1]
TICK
MPP X8*X0*X3*X5
DETECTOR(3, 0, 2) rec[-1] rec[-7] rec[-13]
TICK
MPP X7*X10*X5*X3
DETECTOR(1, 1, 3) rec[-1] rec[-8] rec[-13]
TICK
MPP X11*X8*X5*X10
DETECTOR(4, 0, 4) rec[-1] rec[-9] rec[-13]
TICK
MPP Z8*Z0*Z3*Z5
DETECTOR(3, 0, 5) rec[-1] rec[-13]
TICK
MPP Z7*Z10*Z5*Z3
DETECTOR(2, 3, 6) rec[-1] rec[-13]
TICK
MPP Z11*Z8*Z5*Z10
DETECTOR(4, 0, 7) rec[-1] rec[-13]
TICK
MPP Y0*Y8*Y11*Y10*Y7*Y5*Y3
OBSERVABLE_INCLUDE(0) rec[-1] rec[-8] rec[-9] rec[-12]
"""
    )


class D3DoubleCatCheckA6(_OriginalD3ColorCodeLayout):
    """The distance-3 double cat-check circuit using 6 ancillas.
    
    This is the circuit used in the paper.
    """

    DATA_INDICES = (0, 3, 5, 7, 8, 10, 11)

    ANCILLA_INDICES = (12, 9, 4, 2, 6, 1)

    INNER_CIRCUIT = stim.Circuit(
"""
QUBIT_COORDS(0, 0) 0
QUBIT_COORDS(0, 1) 1
QUBIT_COORDS(1, 0) 2
QUBIT_COORDS(1, 1) 3
QUBIT_COORDS(2, 0) 4
QUBIT_COORDS(2, 1) 5
QUBIT_COORDS(2, 2) 6
QUBIT_COORDS(2, 3) 7
QUBIT_COORDS(3, 0) 8
QUBIT_COORDS(3, 1) 9
QUBIT_COORDS(3, 2) 10
QUBIT_COORDS(4, 0) 11
QUBIT_COORDS(4, 1) 12
#!pragma POLYGON(0,0,1,0.25) 8 11 10 5
#!pragma POLYGON(0,1,0,0.25) 7 10 5 3
#!pragma POLYGON(1,0,0,0.25) 3 5 8 0
MPP X0*X8*X11*X10*X7*X5*X3
TICK
RX 12 9 4 2 6 1
TICK
CX 1 0 9 8 6 7 4 5 2 3 12 11
TICK
CX 3 1 5 6 9 12
TICK
CX 5 3 9 10
TICK
CX 5 9
TICK
MRX 5
DETECTOR(0, 0, 0) rec[-1] rec[-2]
TICK
CX 5 9
TICK
CX 5 3 9 10
TICK
CX 3 1 5 6 9 12
TICK
CX 1 0 9 8 6 7 12 11 4 5 2 3
TICK
MX 12 9 4 2 6 1
DETECTOR(4, 1, 1) rec[-6]
DETECTOR(3, 1, 1) rec[-5]
DETECTOR(2, 1, 1) rec[-4] rec[-7]
DETECTOR(1, 0, 1) rec[-3]
DETECTOR(2, 2, 1) rec[-2]
DETECTOR(0, 1, 1) rec[-1]
TICK
RX 12 9 4 2 6 1
"""
    )

    CIRCUIT = stim.Circuit(
"""
QUBIT_COORDS(0, 0) 0
QUBIT_COORDS(0, 1) 1
QUBIT_COORDS(1, 0) 2
QUBIT_COORDS(1, 1) 3
QUBIT_COORDS(2, 0) 4
QUBIT_COORDS(2, 1) 5
QUBIT_COORDS(2, 2) 6
QUBIT_COORDS(2, 3) 7
QUBIT_COORDS(3, 0) 8
QUBIT_COORDS(3, 1) 9
QUBIT_COORDS(3, 2) 10
QUBIT_COORDS(4, 0) 11
QUBIT_COORDS(4, 1) 12
#!pragma POLYGON(0,0,1,0.25) 8 11 10 5
#!pragma POLYGON(0,1,0,0.25) 7 10 5 3
#!pragma POLYGON(1,0,0,0.25) 3 5 8 0
TICK
MPP Y0*Y8*Y11*Y10*Y7*Y5*Y3
TICK
MPP X8*X0*X3*X5
TICK
MPP X3*X5*X10*X7
TICK
MPP X11*X8*X5*X10
TICK
MPP Z8*Z0*Z3*Z5
TICK
MPP Z7*Z10*Z5*Z3
TICK
MPP Z11*Z8*Z5*Z10
TICK
RX 12 9 4 2 6 1
TICK
# CZ sweep[7] 0 sweep[7] 3 sweep[7] 5 sweep[7] 8
# CX sweep[8] 0 sweep[8] 3 sweep[8] 5 sweep[8] 8
# CZ sweep[9] 3 sweep[9] 5 sweep[9] 7 sweep[9] 10
# CX sweep[10] 3 sweep[10] 5 sweep[10] 7 sweep[10] 10
# CZ sweep[11] 5 sweep[11] 8 sweep[11] 10 sweep[11] 11
# CX sweep[12] 5 sweep[12] 8 sweep[12] 10 sweep[12] 11
S_DAG 0 3 5 7 8 10 11
# CZ sweep[0] 0
# CZ sweep[1] 3
# CZ sweep[2] 5
# CZ sweep[3] 7
# CZ sweep[4] 8
# CZ sweep[5] 10
# CZ sweep[6] 11
TICK
CX 1 0 9 8 6 7 4 5 2 3 12 11
TICK
CX 3 1 5 6 9 12
TICK
CX 5 3 9 10
TICK
CX 5 9
TICK
# CZ sweep[0] 5
# CZ sweep[1] 5
# CZ sweep[2] 5
# CZ sweep[3] 5
# CZ sweep[4] 5
# CZ sweep[5] 5
# CZ sweep[6] 5
MX 5
DETECTOR(0, 0, 0) rec[-1] rec[-8]
TICK
RX 5
# CZ sweep[0] 5
# CZ sweep[1] 5
# CZ sweep[2] 5
# CZ sweep[3] 5
# CZ sweep[4] 5
# CZ sweep[5] 5
# CZ sweep[6] 5
TICK
CX 5 9
TICK
CX 5 3 9 10
TICK
CX 3 1 5 6 9 12
TICK
CX 1 0 9 8 6 7 12 11 4 5 2 3
TICK
# CZ sweep[0] 0
# CZ sweep[1] 3
# CZ sweep[2] 5
# CZ sweep[3] 7
# CZ sweep[4] 8
# CZ sweep[5] 10
# CZ sweep[6] 11
# CZ sweep[13] 0 sweep[13] 3 sweep[13] 5 sweep[13] 8
# CX sweep[14] 0 sweep[14] 3 sweep[14] 5 sweep[14] 8
# CZ sweep[15] 3 sweep[15] 5 sweep[15] 7 sweep[15] 10
# CX sweep[16] 3 sweep[16] 5 sweep[16] 7 sweep[16] 10
# CZ sweep[17] 5 sweep[17] 8 sweep[17] 10 sweep[17] 11
# CX sweep[18] 5 sweep[18] 8 sweep[18] 10 sweep[18] 11
S 5 10 11 8 7 3 0
TICK
MX 12 9 4 2 6 1
DETECTOR(4, 1, 1) rec[-6]
DETECTOR(3, 1, 1) rec[-5]
DETECTOR(2, 1, 1) rec[-4] rec[-7]
DETECTOR(1, 0, 1) rec[-3]
DETECTOR(2, 2, 1) rec[-2]
DETECTOR(0, 1, 1) rec[-1]
TICK
MPP X8*X0*X3*X5
DETECTOR(3, 0, 2) rec[-1] rec[-8] rec[-14]
TICK
MPP X7*X10*X5*X3
DETECTOR(1, 1, 3) rec[-1] rec[-9] rec[-14]
TICK
MPP X11*X8*X5*X10
DETECTOR(4, 0, 4) rec[-1] rec[-10] rec[-14]
TICK
MPP Z8*Z0*Z3*Z5
DETECTOR(3, 0, 5) rec[-1] rec[-14]
TICK
MPP Z7*Z10*Z5*Z3
DETECTOR(2, 3, 6) rec[-1] rec[-14]
TICK
MPP Z11*Z8*Z5*Z10
DETECTOR(4, 0, 7) rec[-1] rec[-14]
TICK
MPP Y0*Y8*Y11*Y10*Y7*Y5*Y3
OBSERVABLE_INCLUDE(0) rec[-1] rec[-8] rec[-9] rec[-12] rec[-13]
"""
    )
    """Copied from the Stim file in `make_chunk_d3_double_cat_check()`
    in `code/src/cultiv/_construction/_cultivation_stage.py`.
    """


class D3DoubleCatCheckA7(_OriginalD3ColorCodeLayout):
    """The distance-3 double cat-check circuit using 7 ancillas."""

    DATA_INDICES = (0, 3, 5, 7, 8, 10, 11)

    ANCILLA_INDICES = (13, 12, 9, 4, 2, 6, 1)

    INNER_CIRCUIT = stim.Circuit(
"""
QUBIT_COORDS(0, 0) 0
QUBIT_COORDS(0, 1) 1
QUBIT_COORDS(1, 0) 2
QUBIT_COORDS(1, 1) 3
QUBIT_COORDS(2, 0) 4
QUBIT_COORDS(2, 1) 5
QUBIT_COORDS(2, 2) 6
QUBIT_COORDS(2, 3) 7
QUBIT_COORDS(3, 0) 8
QUBIT_COORDS(3, 1) 9
QUBIT_COORDS(3, 2) 10
QUBIT_COORDS(4, 0) 11
QUBIT_COORDS(4, 1) 12
QUBIT_COORDS(4, 2) 13
#!pragma POLYGON(0,0,1,0.25) 8 11 10 5
#!pragma POLYGON(0,1,0,0.25) 7 10 5 3
#!pragma POLYGON(1,0,0,0.25) 3 5 8 0
MPP X0*X3*X5*X7*X8*X10*X11
TICK
RX 13 12 9 4 2 6 1
TICK
CX 1 0 9 8 6 7 4 5 2 3 13 10 12 11
TICK
CX 3 1 5 6 9 10
TICK
CX 5 3 9 12
TICK
CX 5 9
TICK
MRX 5
DETECTOR(0, 0, 0) rec[-1] rec[-2]
TICK
CX 5 9
TICK
CX 5 3 9 12
TICK
CX 3 1 5 6 9 10
TICK
CX 1 0 9 8 6 7 4 5 2 3 13 10 12 11
TICK
MX 13 12 9 4 2 6 1
DETECTOR(4, 2, 1) rec[-7]
DETECTOR(4, 1, 1) rec[-6]
DETECTOR(3, 1, 1) rec[-5]
DETECTOR(2, 1, 1) rec[-4] rec[-8]
DETECTOR(1, 0, 1) rec[-3]
DETECTOR(2, 2, 1) rec[-2]
DETECTOR(0, 1, 1) rec[-1]
TICK
RX 13 12 9 4 2 6 1
"""
    )

    CIRCUIT = stim.Circuit(
"""
QUBIT_COORDS(0, 0) 0
QUBIT_COORDS(0, 1) 1
QUBIT_COORDS(1, 0) 2
QUBIT_COORDS(1, 1) 3
QUBIT_COORDS(2, 0) 4
QUBIT_COORDS(2, 1) 5
QUBIT_COORDS(2, 2) 6
QUBIT_COORDS(2, 3) 7
QUBIT_COORDS(3, 0) 8
QUBIT_COORDS(3, 1) 9
QUBIT_COORDS(3, 2) 10
QUBIT_COORDS(4, 0) 11
QUBIT_COORDS(4, 1) 12
QUBIT_COORDS(4, 2) 13
#!pragma POLYGON(0,0,1,0.25) 8 11 10 5
#!pragma POLYGON(0,1,0,0.25) 7 10 5 3
#!pragma POLYGON(1,0,0,0.25) 3 5 8 0
TICK
MPP Y0*Y3*Y5*Y7*Y8*Y10*Y11
TICK
MPP X0*X3*X5*X8
TICK
MPP Z0*Z3*Z5*Z8
TICK
MPP X3*X5*X7*X10
TICK
MPP Z3*Z5*Z7*Z10
TICK
MPP X5*X8*X10*X11
TICK
MPP Z5*Z8*Z10*Z11
TICK
RX 13 12 9 4 2 6 1
TICK
S_DAG 0 3 5 7 8 10 11
TICK
CX 1 0 9 8 6 7 4 5 2 3 13 10 12 11
TICK
CX 3 1 5 6 9 10
TICK
CX 5 3 9 12
TICK
CX 5 9
TICK
MX 5
DETECTOR(0, 0, 0) rec[-1] rec[-8]
TICK
RX 5
TICK
CX 5 9
TICK
CX 5 3 9 12
TICK
CX 3 1 5 6 9 10
TICK
CX 1 0 9 8 6 7 4 5 2 3 13 10 12 11
TICK
S 5 10 11 8 7 3 0
TICK
MX 13 12 9 4 2 6 1
DETECTOR(4, 2, 1) rec[-7]
DETECTOR(4, 1, 1) rec[-6]
DETECTOR(3, 1, 1) rec[-5]
DETECTOR(2, 1, 1) rec[-4] rec[-8]
DETECTOR(1, 0, 1) rec[-3]
DETECTOR(2, 2, 1) rec[-2]
DETECTOR(0, 1, 1) rec[-1]
TICK
MPP Z5*Z8*Z10*Z11
DETECTOR(2, 1, 2) rec[-1] rec[-10]
TICK
MPP X5*X8*X10*X11
DETECTOR(2, 1, 3) rec[-1] rec[-10] rec[-12]
TICK
MPP Z3*Z5*Z7*Z10
DETECTOR(1, 1, 4) rec[-1] rec[-14]
TICK
MPP X3*X5*X7*X10
DETECTOR(1, 1, 5) rec[-1] rec[-12] rec[-16]
TICK
MPP Z0*Z3*Z5*Z8
DETECTOR(0, 0, 6) rec[-1] rec[-18]
TICK
MPP X0*X3*X5*X8
DETECTOR(0, 0, 7) rec[-1] rec[-14] rec[-20]
TICK
MPP Y0*Y3*Y5*Y7*Y8*Y10*Y11
OBSERVABLE_INCLUDE(0) rec[-1] rec[-8] rec[-9] rec[-12] rec[-13]
"""
    )