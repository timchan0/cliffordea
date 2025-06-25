import stim

D3_DOUBLE_CAT_CHECK_PREOPT = stim.Circuit("""
RX 0 1 2 3 4 5 6
TICK
CX 0 7 1 8 2 9 3 10 4 11 5 12 6 13
TICK
CX 1 0
TICK
CX 2 1 5 6
TICK
CX 3 2 4 5
TICK
CX 3 4

TICK
MRX 3
DETECTOR rec[-1]

TICK
CX 3 4
TICK
CX 3 2 4 5
TICK
CX 2 1 5 6
TICK
CX 1 0
TICK
CX 0 7 1 8 2 9 3 10 4 11 5 12 6 13
TICK
MX 0 1 2 3 4 5 6
DETECTOR rec[-7]
DETECTOR rec[-6]
DETECTOR rec[-5]
DETECTOR rec[-4]
DETECTOR rec[-3]
DETECTOR rec[-2]
DETECTOR rec[-1]
""")

D3_DOUBLE_CAT_CHECK = stim.Circuit("""
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
    """)
"""Adapted from the Stim file in `make_chunk_d3_double_cat_check()`
in `code/src/cultiv/_construction/_cultivation_stage.py`.
"""

D3_DOUBLE_CAT_CHECK_DATA_INDICES = (0, 3, 5, 7, 8, 10, 11)
"""Indices of the data qubits in `D3_DOUBLE_CAT_CHECK`."""

D3_DOUBLE_CAT_CHECK_ANCILLA_INDICES = (12, 9, 4, 2, 6, 1)
"""Indices of the ancilla qubits in `D3_DOUBLE_CAT_CHECK`."""

D3_DOUBLE_CAT_CHECK_STABILIZER_GENERATORS = (
    stim.PauliString('X0*X1*X2*X4'),
    stim.PauliString('Z0*Z1*Z2*Z4'),
    stim.PauliString('X1*X2*X3*X5'),
    stim.PauliString('Z1*Z2*Z3*Z5'),
    stim.PauliString('X2*X4*X5*X6'),
    stim.PauliString('Z2*Z4*Z5*Z6'),
)
"""Generators restricted to the 7 data qubits
in the order given by `D3_DOUBLE_CAT_CHECK_DATA_INDICES`.
"""

D3_DOUBLE_CAT_CHECK_LOGICAL_X = stim.PauliString('X0*X1*X3')
D3_DOUBLE_CAT_CHECK_LOGICAL_Z = stim.PauliString('Z0*Z1*Z3')