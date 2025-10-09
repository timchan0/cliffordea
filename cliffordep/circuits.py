import stim

class _OriginalD3ColorCodeLayout:
    """Common constants for the distance-3 double cat-check circuit using 1, 6, or 7 ancillas."""

    DATA_INDICES = (0, 3, 5, 7, 8, 10, 11)
    """Indices of the data qubits."""

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


    def __init__(self):
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

    ANCILLA_INDICES = (4,)
    """Indices of the ancilla qubits."""

    INNER_CIRCUIT = stim.Circuit(
"""
QUBIT_COORDS(0, 0) 0
QUBIT_COORDS(1, 1) 3
QUBIT_COORDS(2, 0) 4
QUBIT_COORDS(2, 1) 5
QUBIT_COORDS(2, 3) 7
QUBIT_COORDS(3, 0) 8
QUBIT_COORDS(3, 2) 10
QUBIT_COORDS(4, 0) 11
#!pragma POLYGON(0,0,1,0.25) 8 11 10 5
#!pragma POLYGON(0,1,0,0.25) 7 10 5 3
#!pragma POLYGON(1,0,0,0.25) 3 5 8 0
H 0 3 5 7 8 10 11
RX 4
TICK
CX 4 5
TICK
CX 5 0
TICK
CX 5 3
TICK
CX 5 7
TICK
CX 5 8
TICK
CX 5 10
TICK
CX 5 11
TICK
MX 5
DETECTOR(0, 0, 0) rec[-1]
TICK
RX 5
TICK
CX 5 11
TICK
CX 5 10
TICK
CX 5 8
TICK
CX 5 7
TICK
CX 5 3
TICK
CX 5 0
TICK
CX 4 5
TICK
# added reset after measurement
MRX 4
DETECTOR(2, 1, 1) rec[-1] rec[-2]
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
RX 4
TICK
S_DAG 0 3 5 7 8 10 11
TICK
CX 4 5
TICK
CX 5 0
TICK
CX 5 3
TICK
CX 5 7
TICK
CX 5 8
TICK
CX 5 10
TICK
CX 5 11
TICK
MX 5
DETECTOR(0, 0, 0) rec[-1] rec[-8]
TICK
RX 5
TICK
CX 5 11
TICK
CX 5 10
TICK
CX 5 8
TICK
CX 5 7
TICK
CX 5 3
TICK
CX 5 0
TICK
CX 4 5
TICK
S 5 10 11 8 7 3 0
TICK
# added reset after measurement
MRX 4
DETECTOR(2, 1, 1) rec[-1] rec[-2]
TICK
MPP X8*X0*X3*X5
DETECTOR(3, 0, 2) rec[-1] rec[-3] rec[-9]
TICK
MPP X7*X10*X5*X3
DETECTOR(1, 1, 3) rec[-1] rec[-4] rec[-9]
TICK
MPP X11*X8*X5*X10
DETECTOR(4, 0, 4) rec[-1] rec[-5] rec[-9]
TICK
MPP Z8*Z0*Z3*Z5
DETECTOR(3, 0, 5) rec[-1] rec[-9]
TICK
MPP Z7*Z10*Z5*Z3
DETECTOR(2, 3, 6) rec[-1] rec[-9]
TICK
MPP Z11*Z8*Z5*Z10
DETECTOR(4, 0, 7) rec[-1] rec[-9]
TICK
MPP Y0*Y8*Y11*Y10*Y7*Y5*Y3
OBSERVABLE_INCLUDE(0) rec[-1]  # unsure about this
"""
    )


class D3DoubleCatCheckA6(_OriginalD3ColorCodeLayout):
    """The distance-3 double cat-check circuit using 6 ancillas.
    
    This is the circuit used in the paper.
    """

    ANCILLA_INDICES = (12, 9, 4, 2, 6, 1)
    """Indices of the ancilla qubits."""

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
H 0 3 5 7 8 10 11
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
MX 5
DETECTOR(0, 0, 0) rec[-1]
TICK
RX 5
TICK
CX 5 9
TICK
CX 5 3 9 10
TICK
CX 3 1 5 6 9 12
TICK
CX 1 0 9 8 6 7 12 11 4 5 2 3
TICK
# added reset after measurement
MRX 12 9 4 2 6 1
DETECTOR(4, 1, 1) rec[-6]
DETECTOR(3, 1, 1) rec[-5]
DETECTOR(2, 1, 1) rec[-4] rec[-7]
DETECTOR(1, 0, 1) rec[-3]
DETECTOR(2, 2, 1) rec[-2]
DETECTOR(0, 1, 1) rec[-1]
"""
    )
    """Adapted from the Stim file in `make_chunk_d3_double_cat_check()`
    in `code/src/cultiv/_construction/_cultivation_stage.py`.
    """

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
# added reset after measurement
MRX 12 9 4 2 6 1
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

    ANCILLA_INDICES = (13, 12, 9, 4, 2, 6, 1)
    """Indices of the ancilla qubits."""

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
H 0 3 5 7 8 10 11
RX 13 12 9 4 2 6 1
TICK
CX 1 0 9 8 6 7 4 5 2 3 13 10 12 11
TICK
CX 3 1 5 6 12 13
TICK
CX 5 3 9 12
TICK
CX 5 9
TICK
MX 5
DETECTOR(0, 0, 0) rec[-1]
TICK
RX 5
TICK
CX 5 9
TICK
CX 5 3 9 12
TICK
CX 3 1 5 6 12 13
TICK
CX 1 0 9 8 6 7 4 5 2 3 13 10 12 11
TICK
# added reset after measurement
MRX 13 12 9 4 2 6 1
DETECTOR(4, 2, 1) rec[-7]
DETECTOR(4, 1, 1) rec[-6]
DETECTOR(3, 1, 1) rec[-5]
DETECTOR(2, 1, 1) rec[-4] rec[-8]
DETECTOR(1, 0, 1) rec[-3]
DETECTOR(2, 2, 1) rec[-2]
DETECTOR(0, 1, 1) rec[-1]
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
CX 3 1 5 6 12 13
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
CX 3 1 5 6 12 13
TICK
CX 1 0 9 8 6 7 4 5 2 3 13 10 12 11
TICK
S 5 10 11 8 7 3 0
TICK
# added reset after measurement
MRX 13 12 9 4 2 6 1
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
OBSERVABLE_INCLUDE(0) rec[-1] rec[-8] rec[-9] rec[-12] rec[-13] rec[-14]
"""
    )