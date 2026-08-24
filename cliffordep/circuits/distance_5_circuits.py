from pathlib import Path
import json
from typing import Literal

import stim

from cliffordep.circuits._base import find_data_indices, find_stabilizer_generators, find_logical_s_gate


_STIM_FILES_DIR = Path(__file__).with_name("stim_files")


FlagCount = Literal[0, 3, 15, 18]


class Distance5DoubleCheck:
    """Common constants for the distance-5 double-check circuit using 19 ancillas."""

    ANCILLA_COUNT: int = 19
    """The indices of the ancilla qubits."""

    def __init__(self, flag_count: Literal[0, 3, 15, 18] = 0):
        """Possible values for flag_count:
        
        0:
            This is the circuit used in the paper.
        3:
            This circuit has fault distance 4.
            The way this was designed was as follows:
            the unflagged version has 3 malignant fault configurations of weight 3.
            Each flag in this circuit detects exactly one of these configurations.
        15:
            This circuit has fault distance 3.
            The way this was designed was as follows:
            once the X parity to be measured is positioned into a spanning tree,
            we measure all but three pairwise Z parities between
            neighboring qubits in the spanning tree.
        18:
            This circuit has fault distance 5.
            The way this was designed was as follows:
            once the X parity to be measured is positioned into a spanning tree,
            we measure all pairwise Z parities between
            neighboring qubits in the spanning tree.
            The three delayed Z parity measurements are necessary for fault distance 5;
            without them we have the 15-flag circuit of fault distance 3.
        """
        self.FLAG_COUNT = flag_count
        """The number of flag qubits in the circuit."""
        _flag_id = f'f{flag_count}' if flag_count else ''
        circuit_name = f'd5a{self.ANCILLA_COUNT}{_flag_id}.stim'
        self.INNER_CIRCUIT = stim.Circuit.from_file(
            _STIM_FILES_DIR / 'inner_circuits' / circuit_name)
        self.DATA_INDICES = find_data_indices(inner_circuit=self.INNER_CIRCUIT)
        """The indices of the data qubits in ascending order."""
        self.CIRCUIT = stim.Circuit.from_file(
            _STIM_FILES_DIR / 'full_circuits' / circuit_name)
        """The full double-check circuit."""
        self._initialize_operators()

    def _initialize_operators(self) -> None:
        """Build stabilizers and logicals at the loaded circuit width."""
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
        (self.LOGICAL_X, self.LOGICAL_Z) = tuple(
            stim.PauliString(
                basis if index in self.DATA_INDICES
                else '_' for index in range(self.INNER_CIRCUIT.num_qubits)) # type: ignore
            for basis in ('X', 'Z')
        )
        self.LOGICAL_S = find_logical_s_gate(circuit=self.CIRCUIT)


class D5A19Flagged(Distance5DoubleCheck):
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
        first_flag_index = len(self.DATA_INDICES) + self.ANCILLA_COUNT
        self.FLAG_COUNT = self.INNER_CIRCUIT.num_qubits - first_flag_index
        self.SOLUTION_ID = solution_id
        self.SYNTHESIS_MANIFEST = manifest
        self._initialize_operators()
