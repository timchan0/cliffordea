from pathlib import Path
import json

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
        self._initialize_operators()

    def _initialize_operators(self) -> None:
        """Build stabilizers and logicals at the loaded circuit width."""
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


class D5A19Flagged(D5A19):
    """A synthesized flagged D5A19 circuit loaded from its manifest."""

    def __init__(self, solution_id: str = "lowest_flags"):
        """Load a circuit only after exhaustive order-four verification.

        :param self: The flagged distance-five circuit being initialized.
        :param solution_id: The generated circuit and manifest identifier.
        :return: None.
        """
        generated_directory = _STIM_FILES_DIR / "generated"
        manifest_path = generated_directory / f"d5a19_{solution_id}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No generated D5A19 flag solution named {solution_id!r}. "
                f"Run tim_code/workflow_d5_flags.py first."
            )
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("verified_through_order") != 4:
            raise ValueError(
                f"Generated D5A19 flag solution {solution_id!r} is unverified; "
                "its manifest must declare verified_through_order=4."
            )
        files = manifest["files"]
        self.INNER_CIRCUIT = stim.Circuit.from_file(
            generated_directory / files["inner_circuit"]
        )
        self.CIRCUIT = stim.Circuit.from_file(
            generated_directory / files["full_circuit"]
        )
        first_flag_index = max((*self.DATA_INDICES, *self.ANCILLA_INDICES)) + 1
        self.FLAG_INDICES = tuple(
            range(first_flag_index, self.INNER_CIRCUIT.num_qubits)
        )
        self.SOLUTION_ID = solution_id
        self.SYNTHESIS_MANIFEST = manifest
        self._initialize_operators()
