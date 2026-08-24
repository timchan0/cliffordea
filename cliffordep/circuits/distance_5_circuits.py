from pathlib import Path
import json

import stim

from cliffordep.circuits._base import find_data_indices, find_stabilizer_generators


_STIM_FILES_DIR = Path(__file__).with_name("stim_files")


class Distance5DoubleCheck:
    """Common constants for the distance-5 double-check circuit."""

    TRANSVERSAL_S: stim.Circuit
    ANCILLA_INDICES: tuple[int, ...]
    """The indices of the ancilla qubits."""

    def __init__(self):
        circuit_name = f'd5a{len(self.ANCILLA_INDICES)}.stim'
        self.INNER_CIRCUIT = stim.Circuit.from_file(
            _STIM_FILES_DIR / 'inner_circuits' / circuit_name)
        self.DATA_INDICES = find_data_indices(inner_circuit=self.INNER_CIRCUIT)
        """The indices of the data qubits in ascending order."""
        self.CIRCUIT = stim.Circuit.from_file(
            _STIM_FILES_DIR / 'full_circuits' / circuit_name)
        """The full double-check circuit."""
        self._STABILIZER_GENERATOR_INDICES = find_stabilizer_generators(
            circuit=self.CIRCUIT)
        """Indices of the stabilizer generators."""
        self._initialize_operators()

    def _initialize_operators(self) -> None:
        """Build stabilizers and logicals at the loaded circuit width."""
        self.STABILIZER_GENERATORS = {basis: tuple(stim.PauliString(
            basis if index in indices else '_' for index in range(self.CIRCUIT.num_qubits) # type: ignore
        ) for indices in self._STABILIZER_GENERATOR_INDICES
        ) for basis in ('X', 'Z')}
        self.STABILIZER_GENERATORS_RESTRICTED = {basis: tuple(stim.PauliString(
            basis if index in indices else '_' for index in self.DATA_INDICES # type: ignore
        ) for indices in self._STABILIZER_GENERATOR_INDICES
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
    ANCILLA_INDICES = tuple(sorted((21, 28, 2, 4, 6, 8, 17, 19, 35, 1, 10, 15, 33, 30, 12, 23, 25, 27, 37)))

    LOGICAL_S = stim.Circuit(
"""
S 0 5 7 14 16 18 20 29 31 36
S_DAG 9 11 13 22 24 26 34 32 3
"""
    )


class D5A19Flagged(D5A19):
    """A synthesized flagged D5A19 circuit loaded from its manifest."""

    def __init__(self, solution_id: str = "lowest_flags"):
        generated_directory = _STIM_FILES_DIR / "generated"
        manifest_path = generated_directory / f"d5a19_{solution_id}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No generated D5A19 flag solution named {solution_id!r}. "
                f"Run tim_code/workflow_d5_flags.py first."
            )
        manifest = json.loads(manifest_path.read_text())
        files = manifest["files"]
        self.INNER_CIRCUIT = stim.Circuit.from_file(
            generated_directory / files["inner_circuit"]
        )
        self.DATA_INDICES = find_data_indices(inner_circuit=self.INNER_CIRCUIT)
        """The indices of the data qubits in ascending order."""
        self.CIRCUIT = stim.Circuit.from_file(
            generated_directory / files["full_circuit"]
        )
        self._STABILIZER_GENERATOR_INDICES = find_stabilizer_generators(
            circuit=self.CIRCUIT)
        """Indices of the stabilizer generators."""
        first_flag_index = max((*self.DATA_INDICES, *self.ANCILLA_INDICES)) + 1
        self.FLAG_INDICES = tuple(
            range(first_flag_index, self.INNER_CIRCUIT.num_qubits)
        )
        self.SOLUTION_ID = solution_id
        self.SYNTHESIS_MANIFEST = manifest
        self._initialize_operators()
