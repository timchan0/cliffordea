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
    FLAG_INDICES: tuple[int, ...] = ()
    """The indices of the flag qubits."""

    def __init__(self):
        _flag_count = len(self.FLAG_INDICES)
        _flag_id = f'f{_flag_count}' if _flag_count else ''
        circuit_name = f'd5a{len(self.ANCILLA_INDICES)}{_flag_id}.stim'
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


class D5A19F3(Distance5DoubleCheck):
    """The distance-5 double-check circuit using 19 ancillas and 3 flags.

    This circuit has fault distance 4.
    The way this was designed was as follows:
    the unflagged version has 3 malignant fault configurations of weight 3.
    Each flag in this circuit detects exactly one of these configurations.
    """
    ANCILLA_INDICES = tuple(sorted((23, 31, 2, 5, 7, 9, 19, 21, 38, 1, 11, 17, 36, 33, 13, 25, 27, 29, 40)))
    FLAG_INDICES = tuple(sorted((4, 30, 15)))
    LOGICAL_S = stim.Circuit(
"""
S 16 18 20 22 32 34 6 8 0 39
S_DAG 3 10 12 14 26 24 28 35 37
"""
    )


class D5A19F15(Distance5DoubleCheck):
    """The distance-5 double-check circuit using 19 ancillas and 15 flags.
    
    This circuit has fault distance 3.
    The way this was designed was as follows:
    once the X parity to be measured is positioned into a spanning tree,
    we measure all but three pairwise Z parities between
    neighboring qubits in the spanning tree.
    """
    ANCILLA_INDICES = tuple(sorted((31, 40, 3, 6, 8, 10, 23, 25, 49, 1, 15, 21, 47, 42, 17, 33, 35, 37, 52)))
    FLAG_INDICES = tuple(sorted((2, 5, 11, 19, 27, 38, 44, 50, 39, 45, 28, 29, 30, 12, 13)))
    LOGICAL_S = stim.Circuit(
"""
S 20 22 24 26 41 43 7 9 0 51
S_DAG 4 14 32 16 18 36 34 48 46
"""
    )


class D5A19F18(Distance5DoubleCheck):
    """The distance-5 double-check circuit using 19 ancillas and 18 flags.
    
    This circuit has fault distance 5.
    The way this was designed was as follows:
    once the X parity to be measured is positioned into a spanning tree,
    we measure all pairwise Z parities between
    neighboring qubits in the spanning tree.
    The three delayed Z parity measurements are necessary for fault distance 5;
    without them we have `D5A19F15` of fault distance 3.
    """
    ANCILLA_INDICES = tuple(sorted((34, 43, 3, 6, 8, 10, 24, 26, 52, 1, 16, 22, 50, 45, 18, 36, 38, 40, 55)))
    FLAG_INDICES = tuple(sorted((2, 5, 11, 20, 28, 41, 47, 53, 42, 48, 31, 32, 33, 13, 14, 12, 29, 30)))
    LOGICAL_S = stim.Circuit(
"""
S 21 23 25 27 44 46 7 9 0 54
S_DAG 4 15 35 17 19 39 37 51 49
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
