from pathlib import Path

import stim

from cliffordea.enum.circuits._base import find_data_indices, find_logical_s_gate, find_stabilizer_generators


_STIM_FILES_DIR = Path(__file__).with_name("stim_files")


class DoubleCheck:
    """The double-check circuit in magic state cultivation."""

    def __init__(
            self,
            distance: int = 3,
            ancilla_count: int = 6,
            flag_count: int = 0,
    ):
        """Possible values for (distance, ancilla_count, flag_count):
        
        (3, 1, 0)

        (3, 2, 0)

        (3, 3, 0)

        (3, 4, 0)

        (3, 5, 0)

        (3, 6, 0):
            This is the original distance-3 circuit.
        (3, 7, 0)

        (3, 6, 2)

        (3, 6, 3)

        (3, 6, 5)

        (15, 19, 0):
            This is the original distance-5 circuit.
        (15, 19, 3):
            This circuit has fault distance 4.
            The way this was designed was as follows:
            the unflagged version has 3 malignant 3-fault configurations.
            Each flag in this circuit detects exactly one of these configurations.
        (15, 19, 13):
            This circuit has fault distance 5.
            The way this was designed was as follows:
            Start with the 18-flag circuit,
            prune the five flags corresponding to the leaves of the spanning tree,
            delay the X-parity folding of the three horizontal branches of the tree
            by 1 tick (this can be done in the original circuit too),
            then minimize the flag lifespans as much as possible.
        (15, 19, 18):
            This circuit has fault distance 5.
            The way this was designed was as follows:
            once the X parity to be measured is positioned into a spanning tree,
            we measure all pairwise Z parities between
            neighboring qubits in the spanning tree.
            The three delayed Z parity measurements are necessary for fault distance 5;
            without them the fault distance drops back down to 3.
        """
        self.DISTANCE = distance
        """The code distance of the circuit."""
        self.ANCILLA_COUNT = ancilla_count
        """The number of ancilla qubits in the circuit."""
        self.FLAG_COUNT = flag_count
        """The number of flag qubits in the circuit."""
        _flag_id = f'f{flag_count}' if flag_count else ''
        circuit_name = f'd{distance}a{ancilla_count}{_flag_id}'
        circuit_directory = _STIM_FILES_DIR / circuit_name
        self.INNER_CIRCUIT = stim.Circuit.from_file(
            circuit_directory / 'inner.stim')
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
            circuit_directory / 'full.stim')
        """The full double-check circuit."""
        self._STABILIZER_GENERATOR_INDICES = find_stabilizer_generators(
                    circuit=self.CIRCUIT)
        """Indices of the stabilizer generators."""
        self.STABILIZER_GENERATORS = {basis: tuple(stim.PauliString(
            basis if index in indices else '_' for index in range(self.CIRCUIT.num_qubits) # type: ignore
        ) for indices in self._STABILIZER_GENERATOR_INDICES
        ) for basis in ('X', 'Z')}
        self.STABILIZER_GENERATORS_RESTRICTED = {basis: tuple(stim.PauliString(
            basis if index in indices else '_' for index in self.DATA_INDICES # type: ignore
        ) for indices in self._STABILIZER_GENERATOR_INDICES
        ) for basis in ('X', 'Z')}
        """Generators restricted to the 7 data qubits
        in the order given by `DATA_INDICES`.
        """
        (self.LOGICAL_X, self.LOGICAL_Z) = tuple(
            stim.PauliString(
                basis if index in self.DATA_INDICES
                else '_' for index in range(self.INNER_CIRCUIT.num_qubits)) # type: ignore
            for basis in ('X', 'Z')
        )
        self.LOGICAL_S = find_logical_s_gate(circuit=self.CIRCUIT)
