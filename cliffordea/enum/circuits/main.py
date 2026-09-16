from pathlib import Path

import stim

from cliffordea.enum.circuits._base import find_data_indices, find_logical_s_gate, find_stabilizer_generators


_STIM_FILES_DIR = Path(__file__).with_name("stim_files")


class LogicalMeasurement:
    """A logical-measurement circuit in magic state cultivation."""

    _CIRCUIT_PARENT: Path
    """Directory containing the Stim circuits for this measurement method."""

    def __init__(
            self,
            distance: int,
            ancilla_count: int,
            flag_count: int = 0,
    ):
        """Load a logical-measurement circuit and derive its code properties.

        :param self: The logical-measurement circuit being initialized.
        :param distance: The code distance of the circuit.
        :param ancilla_count: The number of ancilla qubits in the circuit.
        :param flag_count: The number of flag qubits in the circuit.
        :return: None.
        """
        self.DISTANCE = distance
        """The code distance of the circuit."""
        self.ANCILLA_COUNT = ancilla_count
        """The number of ancilla qubits in the circuit."""
        self.FLAG_COUNT = flag_count
        """The number of flag qubits in the circuit."""
        _flag_id = f'f{flag_count}' if flag_count else ''
        circuit_name = f'd{distance}a{ancilla_count}{_flag_id}'
        circuit_directory = self._CIRCUIT_PARENT / circuit_name
        self.INNER_CIRCUIT = stim.Circuit.from_file(
            circuit_directory / 'inner.stim')
        """The logical-measurement circuit within the two layers of T gates:
        * The circuit should contain an MPP measurement on the data qubits
        at the timeslice they enter the logical measurement circuit;
        the logical H_XY measurement is checked against this MPP measurement.
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

        self.FULL_CIRCUIT = stim.Circuit.from_file(
            circuit_directory / 'full.stim')
        """The logical-measurement circuit with MPPs before and after,
        representing the logical observable and the stabilizer generators.
        """
        self._STABILIZER_GENERATOR_INDICES = find_stabilizer_generators(circuit=self.FULL_CIRCUIT)
        """Indices of the stabilizer generators."""
        self.STABILIZER_GENERATORS = {basis: tuple(stim.PauliString(
            basis if index in indices else '_' for index in self.DATA_INDICES # type: ignore
        ) for indices in self._STABILIZER_GENERATOR_INDICES
        ) for basis in ('X', 'Z')}
        """Generators restricted to the n data qubits
        in the order given by `DATA_INDICES`.
        """
        (self.LOGICAL_X, self.LOGICAL_Z) = tuple(
            stim.PauliString(
                basis if index in self.DATA_INDICES
                else '_' for index in range(self.INNER_CIRCUIT.num_qubits)) # type: ignore
            for basis in ('X', 'Z')
        )
        self.LOGICAL_S = find_logical_s_gate(circuit=self.FULL_CIRCUIT)


class DoubleCheck(LogicalMeasurement):
    """A double check comprising two logical H_XY measurements.

    Possible ``(distance, ancilla_count, flag_count)`` values are:

    * ``(3, 1, 0)``
    * ``(3, 2, 0)``
    * ``(3, 3, 0)``
    * ``(3, 4, 0)`` is the circuit used in experimental MSC [Rosenfeld2026].
    * ``(3, 5, 0)``
    * ``(3, 6, 0)`` is the original distance-three circuit [Gidney2024].
    * ``(3, 6, 2)``
    * ``(3, 6, 3)``
    * ``(3, 6, 5)``
    * ``(3, 7, 0)``
    * ``(5, 19, 0)`` is the original distance-five circuit [Gidney2024].
    * ``(5, 19, 3)`` has fault distance four. The unflagged version has three
      malignant 3-fault configurations, and each flag detects exactly one
      of these configurations.
    * ``(5, 19, 13)`` has fault distance five. It was designed by starting
      with ``(5, 19, 18)``, removing the five flags at the leaves
      of the spanning tree, delaying the X-parity folding of the three
      horizontal branches of the tree by one tick (which can also be done in
      the original circuit), and then minimizing the flag lifespans as much as
      possible.
    * ``(5, 19, 18)`` has fault distance five. Once the X parity to be measured
      is positioned into a spanning tree, all pairwise Z parities between
      neighboring qubits in the spanning tree are measured. The three shorter-lived
      flags are necessary for fault distance five; without
      them, the fault distance drops back to three.
    """

    _CIRCUIT_PARENT = _STIM_FILES_DIR / "double_checks"

    def __init__(
            self,
            distance: int = 3,
            ancilla_count: int = 6,
            flag_count: int = 0,
    ):
        super().__init__(distance, ancilla_count, flag_count)


class ShortSingleCheck(LogicalMeasurement):
    """A single check using a transversally measured GHZ state.

    Available ``(distance, ancilla_count, flag_count)`` values are
    ``(3, 7, 0)`` and ``(3, 17, 1)``.
    """

    _CIRCUIT_PARENT = _STIM_FILES_DIR / "short_single_checks"

    def __init__(
            self,
            distance: int = 3,
            ancilla_count: int = 17,
            flag_count: int = 1,
    ):
        super().__init__(distance, ancilla_count, flag_count)


class LongSingleCheck(LogicalMeasurement):
    """A single check using GHZ-state encoding and unencoding.

    Available ``(distance, ancilla_count, flag_count)`` values are
    ``(3, 7, 0)`` and ``(3, 12, 0)``.
    """

    _CIRCUIT_PARENT = _STIM_FILES_DIR / "long_single_checks"

    def __init__(
            self,
            distance: int = 3,
            ancilla_count: int = 12,
            flag_count: int = 0,
    ):
        super().__init__(distance, ancilla_count, flag_count)
