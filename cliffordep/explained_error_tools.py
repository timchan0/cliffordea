"""Module for analyzing `stim.ExplainedError`s."""

from collections.abc import Sequence
import itertools

import stim

from cliffordep.noiseless_circuit_tools import split_by_ticks, compose_slices

def _nest_circuit_error_locations(explained_errors: list[stim.ExplainedError]):
    """Extract the circuit error locations from a list of explained errors.
    
    :param explained_errors: A list of stim.ExplainedError objects.
        Each explained error represents a variable node in the Tanner graph
        i.e. represents a set of detectors that flip due to that error.
        It may have multiple circuit error locations.
        Each circuit error location represents a single noise gate in the circuit.
        For more detail see the docstrings of `stim.ExplainedError` and `stim.CircuitErrorLocation`.

    :return nest: A list of lists of stim.CircuitErrorLocation
        objects, where each list corresponds to an explained error
        and contains all circuit error locations for that explained error.
    """
    nest: list[list[stim.CircuitErrorLocation]] = []
    for explained_error in explained_errors:
        nest.append(explained_error.circuit_error_locations)
    return nest


def insert_circuit_error_locations(
        circuit: stim.Circuit,
        circuit_error_locations: Sequence[stim.CircuitErrorLocation],
        probability: float = 1,
):
    """Insert circuit error locations into the circuit.
    
    :param circuit: A stim.Circuit object.
    :param circuit_error_locations: A list of stim.CircuitErrorLocation objects.
    :param probability: The probability of the error.

    :return: The circuit with the errors inserted at the specified locations.
    """
    circuits = split_by_ticks(circuit)
    for error_location in circuit_error_locations:
        match error_location.flipped_pauli_product:
            case [a]:
                if a.gate_target.qubit_value is None:
                    raise ValueError("Qubit value is None.")
                circuits[error_location.tick_offset].append(
                    f'{a.gate_target.pauli_type}_ERROR',
                    a.gate_target.qubit_value,
                    probability,
                )
            case [a, b]:
                circuits[error_location.tick_offset].append(
                    "E",
                    [a.gate_target, b.gate_target],
                    probability,
                )
            case _:
                raise ValueError("More than two correlated Paulis.")
    return compose_slices(circuits)


def insert_explained_errors(
        circuit: stim.Circuit,
        explained_errors: list[stim.ExplainedError],
        probability: float = 1,
):
    """Make a circuit with the error instance inserted,
    for each instance from the explained errors.
    
    :param circuit: A stim.Circuit object.
    :param explained_errors: A list of stim.ExplainedError objects.
        Each explained error may have multiple circuit error locations.
    :param probability: The probability of the error.

    :return circuits: A list of stim.Circuit objects,
        each with a different error instance inserted.
        Each error instance corresponds to a different combination of
        circuit error locations from the explained errors.
    """
    circuits: list[stim.Circuit] = []
    basis = _nest_circuit_error_locations(explained_errors)
    for circuit_error_locations in itertools.product(*basis):
        circuits.append(insert_circuit_error_locations(
            circuit,
            circuit_error_locations,
            probability=probability,
        ))
    return circuits
