from pathlib import Path

import stim


_STIM_FILES_DIR = Path(__file__).with_name("stim_files")


class Distance5DoubleCheck:
    """Common constants for the distance-5 double-check circuit."""

    _UNRESTRICTED_STABILIZER_GENERATOR_INDICES = (
        (0, 9, 5, 3), (14, 32, 29, 22), (11, 16, 24, 18, 13, 7),
        (22, 29, 34, 31, 24, 16), (3, 5, 11, 7), (13, 18, 26, 20),
        (9, 14, 22, 16, 11, 5), (24, 31, 26, 18), (29, 32, 36, 34),
    )
    """Indices of the stabilizer generators."""

    TRANSVERSAL_S: stim.Circuit
    ANCILLA_INDICES: tuple[int, ...]
    """The indices of the ancilla qubits."""

    def __init__(self):
        circuit_name = f'd5a{len(self.ANCILLA_INDICES)}.stim'
        self.INNER_CIRCUIT = stim.Circuit.from_file(
            _STIM_FILES_DIR / 'inner_circuits' / circuit_name)
        self.CIRCUIT = stim.Circuit.from_file(
            _STIM_FILES_DIR / 'full_circuits' / circuit_name)
        """The full double-check circuit."""
        self.DATA_INDICES: tuple[int, ...]
        self.STABILIZER_GENERATORS = {basis: tuple(stim.PauliString(
            basis if index in indices else '_' for index in range(self.INNER_CIRCUIT.num_qubits) # type: ignore
        ) for indices in self._UNRESTRICTED_STABILIZER_GENERATOR_INDICES
        ) for basis in ('X', 'Z')}
        self.STABILIZER_GENERATORS_RESTRICTED = {basis: tuple(stim.PauliString(
            basis if index in indices else '_' for index in self.DATA_INDICES # type: ignore
        ) for indices in self._UNRESTRICTED_STABILIZER_GENERATOR_INDICES
        ) for basis in ('X', 'Z')}
        (self.LOGICAL_X, self.LOGICAL_Z) = tuple(
            stim.PauliString(
                basis if index in self.DATA_INDICES
                else '_' for index in range(self.INNER_CIRCUIT.num_qubits)) # type: ignore
            for basis in ('X', 'Z')
        )


class D5A19(Distance5DoubleCheck):
    """The distance-5 double-check circuit using 19 ancillas.
    
    This is the circuit used in the paper.
    """

    DATA_INDICES = tuple(sorted((3, 5, 0, 9, 14, 22, 32, 29, 34, 31, 24, 26, 20, 18, 13, 7, 11, 16, 36)))
    ANCILLA_INDICES = tuple(sorted((21, 28, 2, 4, 6, 8, 17, 19, 35, 1, 10, 15, 33, 30, 12, 23, 25, 27, 37)))

    LOGICAL_S = stim.Circuit(
"""
S 0 5 7 14 16 18 20 29 31 36
S_DAG 9 11 13 22 24 26 34 32 3
"""
    )