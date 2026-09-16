from pathlib import Path

import pytest
import stim

from cliffordea.enum import circuits
from cliffordea.enum.circuits.main import LongSingleCheck, ShortSingleCheck


_STIM_FILES_DIR = Path(circuits.__file__).with_name("stim_files")


@pytest.mark.parametrize(
    "circuit_type,circuit_parent,ancilla_count,flag_count",
    [
        (circuits.ShortSingleCheck, "short_single_checks", 7, 0),
        (circuits.ShortSingleCheck, "short_single_checks", 17, 1),
        (circuits.LongSingleCheck, "long_single_checks", 7, 0),
        (circuits.LongSingleCheck, "long_single_checks", 12, 0),
    ],
)
def test_loads_single_check_circuit_family(
        circuit_type: type[ShortSingleCheck] | type[LongSingleCheck],
        circuit_parent,
        ancilla_count,
        flag_count,
):
    """Each single-check type loads and analyzes its own supplied Stim files.

    :param circuit_type: Concrete logical-measurement class under test.
    :param circuit_parent: Directory containing that measurement family.
    :param ancilla_count: Ancilla count identifying the supplied circuit.
    :param flag_count: Flag count identifying the supplied circuit.
    :return: None.
    """
    circuit = circuit_type(
        distance=3,
        ancilla_count=ancilla_count,
        flag_count=flag_count,
    )
    flag_id = f"f{flag_count}" if flag_count else ""
    circuit_directory = (
        _STIM_FILES_DIR
        / circuit_parent
        / f"d3a{ancilla_count}{flag_id}"
    )

    assert circuit.INNER_CIRCUIT == stim.Circuit.from_file(
        circuit_directory / "inner.stim"
    )
    assert circuit.FULL_CIRCUIT == stim.Circuit.from_file(
        circuit_directory / "full.stim"
    )
    assert len(circuit.DATA_INDICES) == 7
    assert all(
        len(generator) == len(circuit.DATA_INDICES)
        for generators in circuit.STABILIZER_GENERATORS.values()
        for generator in generators
    )
    assert len(circuit.LOGICAL_X) == circuit.INNER_CIRCUIT.num_qubits
    assert len(circuit.LOGICAL_Z) == circuit.INNER_CIRCUIT.num_qubits
    assert len(circuit.LOGICAL_S) > 0


@pytest.mark.parametrize(
    "circuit_type,circuit_parent,ancilla_count,flag_count",
    [
        (circuits.DoubleCheck, "double_checks", 6, 0),
        (circuits.ShortSingleCheck, "short_single_checks", 17, 1),
        (circuits.LongSingleCheck, "long_single_checks", 12, 0),
    ],
)
def test_logical_measurement_defaults(
        circuit_type,
        circuit_parent,
        ancilla_count,
        flag_count,
):
    """Each method defaults to an available distance-three circuit.

    :param circuit_type: Concrete logical-measurement class under test.
    :param circuit_parent: Directory containing that measurement family.
    :param ancilla_count: Expected default ancilla count for the family.
    :param flag_count: Expected default flag count for the family.
    :return: None.
    """
    circuit = circuit_type()
    flag_id = f"f{flag_count}" if flag_count else ""
    expected_directory = (
        _STIM_FILES_DIR / circuit_parent / f"d3a{ancilla_count}{flag_id}"
    )

    assert isinstance(circuit, circuits.LogicalMeasurement)
    assert circuit.DISTANCE == 3
    assert circuit.ANCILLA_COUNT == ancilla_count
    assert circuit.FLAG_COUNT == flag_count
    assert circuit.INNER_CIRCUIT == stim.Circuit.from_file(
        expected_directory / "inner.stim"
    )
    assert circuit.FULL_CIRCUIT == stim.Circuit.from_file(
        expected_directory / "full.stim"
    )
