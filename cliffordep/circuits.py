import stim

class _D3DoubleCatCheckA6Or7:
    """Common constants for the distance-3 double cat-check circuit using 6 or 7 ancillas."""

    DATA_INDICES = (0, 3, 5, 7, 8, 10, 11)
    """Indices of the data qubits."""

    STABILIZER_GENERATORS = (
        stim.PauliString('X0*X1*X2*X4'),
        stim.PauliString('Z0*Z1*Z2*Z4'),
        stim.PauliString('X1*X2*X3*X5'),
        stim.PauliString('Z1*Z2*Z3*Z5'),
        stim.PauliString('X2*X4*X5*X6'),
        stim.PauliString('Z2*Z4*Z5*Z6'),
    )
    """Generators restricted to the 7 data qubits
    in the order given by `DATA_INDICES`.
    """

    LOGICAL_X = stim.PauliString('X0*X1*X3')
    LOGICAL_Z = stim.PauliString('Z0*Z1*Z3')


class D3DoubleCatCheckA7(_D3DoubleCatCheckA6Or7):
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
MX 13 12 9 4 2 6 1
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
OBSERVABLE_INCLUDE(0) rec[-1] rec[-8] rec[-9] rec[-12] rec[-13] rec[-14]
"""
    )


class D3DoubleCatCheckA6(_D3DoubleCatCheckA6Or7):
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
MX 12 9 4 2 6 1
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