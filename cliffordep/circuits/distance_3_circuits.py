from pathlib import Path

import stim

from cliffordep.circuits._base import find_data_indices, find_logical_s_gate


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

    def __init__(self, ancilla_count: int = 6, flag_count: int = 0):
        """Possible values for (ancilla_count, flag_count):
        
        (1, 0)

        (2, 0)

        (3, 0)

        (4, 0)

        (5, 0)

        (6, 0): This is the original circuit.

        (7, 0)

        (6, 2)

        (6, 3)

        (6, 5)
        """
        self.ANCILLA_COUNT = ancilla_count
        """The number of ancilla qubits in the circuit."""
        self.FLAG_COUNT = flag_count
        """The number of flag qubits in the circuit."""
        _flag_id = f'f{flag_count}' if flag_count else ''
        circuit_name = f'd3a{self.ANCILLA_COUNT}{_flag_id}.stim'
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
        self.DATA_INDICES = find_data_indices(inner_circuit=self.INNER_CIRCUIT)
        """The indices of the data qubits in ascending order."""

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
        (self.LOGICAL_X, self.LOGICAL_Z) = tuple(
            stim.PauliString('*'.join(f'{basis}{self.DATA_INDICES[index]}' for index in self._LOGICAL_INDICES)) * logical_identity
            for basis in ('X', 'Z')
        )
        self.LOGICAL_S = self.LOGICAL_S = find_logical_s_gate(circuit=self.CIRCUIT)