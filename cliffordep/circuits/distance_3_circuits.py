from pathlib import Path

import stim


_DATA_QUBIT_COUNT = 7
_STIM_FILES_DIR = Path(__file__).with_name("stim_files")


class Distance3DoubleCheck:
    """Common constants for the distance-3 double-check circuit."""

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

    _MAJORITY_INDICES = (0, 2, 3, 6)
    """Indices of the qubits that have S applied to them for logical S gate
    in the order given by `DATA_INDICES`.
    """

    DATA_INDICES: tuple[int, ...]
    """The indices of the data qubits."""
    ANCILLA_INDICES: tuple[int, ...]
    """The indices of the ancilla qubits."""
    FLAG_INDICES: tuple[int, ...] = ()
    """The indices of the flag qubits."""

    def __init__(self):
        _flag_count = len(self.FLAG_INDICES)
        _flag_id = f'f{_flag_count}' if _flag_count else ''
        circuit_name = f'd3a{len(self.ANCILLA_INDICES)}{_flag_id}.stim'
        self.INNER_CIRCUIT = stim.Circuit.from_file(
            _STIM_FILES_DIR / 'inner_circuits' / circuit_name)
        """The double-check circuit _within_ the two layers of T gates:
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
        self.CIRCUIT = stim.Circuit.from_file(
            _STIM_FILES_DIR / 'full_circuits' / circuit_name)
        """The full double-check circuit."""
        logical_identity = stim.PauliString(self.INNER_CIRCUIT.num_qubits)
        self.STABILIZER_GENERATORS = {basis: tuple(stim.PauliString(
            '*'.join(f'{basis}{self.DATA_INDICES[index]}' for index in indices)
        ) * logical_identity for indices in self._STABILIZER_GENERATOR_INDICES) for basis in ('X', 'Z')}
        self.STABILIZER_GENERATORS_RESTRICTED = {basis: tuple(stim.PauliString(
            basis if data_index in indices
            else '_' for data_index in range(_DATA_QUBIT_COUNT) # type: ignore
        ) for indices in self._STABILIZER_GENERATOR_INDICES) for basis in ('X', 'Z')}
        """Generators restricted to the 7 data qubits
        in the order given by `DATA_INDICES`.
        """
        (self.LOGICAL_X_RESTRICTED, self.LOGICAL_Z_RESTRICTED) = tuple(
            stim.PauliString(
                basis if data_index in self._LOGICAL_INDICES
                else '_' for data_index in range(_DATA_QUBIT_COUNT) # type: ignore
            ) for basis in ('X', 'Z')
        )
        (self.LOGICAL_X, self.LOGICAL_Z) = tuple(
            stim.PauliString('*'.join(f'{basis}{self.DATA_INDICES[index]}' for index in self._LOGICAL_INDICES)) * logical_identity
            for basis in ('X', 'Z')
        )
        _majority_indices_unrestricted = {self.DATA_INDICES[index] for index in self._MAJORITY_INDICES} 
        self.LOGICAL_S: stim.Circuit = stim.Circuit()
        self.LOGICAL_S.append('S', _majority_indices_unrestricted)
        self.LOGICAL_S.append('S_DAG', set(self.DATA_INDICES) - _majority_indices_unrestricted)


class D3A1(Distance3DoubleCheck):
    """The distance-3 double-check circuit using 1 ancilla."""
    DATA_INDICES = (0, 1, 3, 4, 5, 6, 7)
    ANCILLA_INDICES = (2,)


class D3A2(Distance3DoubleCheck):
    """The distance-3 double-check circuit using 2 ancillas."""
    DATA_INDICES = (0, 1, 3, 4, 5, 7, 8)
    ANCILLA_INDICES = (2, 6)


class D3A3(Distance3DoubleCheck):
    """The distance-3 double-check circuit using 3 ancillas."""
    DATA_INDICES = (0, 2, 4, 5, 6, 8, 9)
    ANCILLA_INDICES = (1, 3, 7)


class D3A4(Distance3DoubleCheck):
    """The distance-3 double-check circuit using 4 ancillas."""
    DATA_INDICES = (0, 2, 4, 6, 7, 9, 10)
    ANCILLA_INDICES = (1, 3, 5, 8)


class D3A5(Distance3DoubleCheck):
    """The distance-3 double-check circuit using 5 ancillas."""
    DATA_INDICES = (0, 3, 5, 7, 8, 10, 11)
    ANCILLA_INDICES = (1, 2, 4, 6, 9)


class D3A6(Distance3DoubleCheck):
    """The distance-3 double-check circuit using 6 ancillas.
    
    This is the circuit used in the paper.
    """
    DATA_INDICES = (0, 3, 5, 7, 8, 10, 11)
    ANCILLA_INDICES = (12, 9, 4, 2, 6, 1)


class D3A7(Distance3DoubleCheck):
    """The distance-3 double-check circuit using 7 ancillas."""
    DATA_INDICES = (0, 3, 5, 7, 8, 10, 11)
    ANCILLA_INDICES = (13, 12, 9, 4, 2, 6, 1)


class D3A6F2(Distance3DoubleCheck):
    """The distance-3 double-check circuit using 6 ancillas and 2 flags."""
    DATA_INDICES = (0, 3, 7, 9, 10, 12, 13)
    ANCILLA_INDICES = (14, 11, 6, 2, 8, 1)
    FLAG_INDICES = (4, 5)


class D3A6F3(Distance3DoubleCheck):
    """The distance-3 double-check circuit using 6 ancillas and 3 flags."""
    DATA_INDICES = (0, 3, 7, 9, 11, 13, 14)
    ANCILLA_INDICES = (15, 12, 6, 2, 8, 1)
    FLAG_INDICES = (4, 5, 10)


class D3A6F5(Distance3DoubleCheck):
    """The distance-3 double-check circuit using 6 ancillas and 5 flags."""
    DATA_INDICES = (0, 4, 7, 9, 12, 14, 16)
    ANCILLA_INDICES = (17, 13, 6, 3, 8, 1)
    FLAG_INDICES = (2, 5, 10, 15, 11)