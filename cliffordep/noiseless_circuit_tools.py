"""Module for inserting faults into circuits."""

from collections.abc import Iterable, Mapping, Sequence
from typing import cast

import stim

from cliffordep.type_aliases import (
    ErrorEvent,
    MeasurementEventKey,
    MeasurementLocation,
)


def measurement_event_key(
        timeslice: int,
        targets: Iterable[stim.GateTarget],
) -> MeasurementEventKey:
    """Return a canonical, picklable key for one measurement result.

    :param timeslice: The timeslice containing the measurement.
    :param targets: The targets belonging to one measurement result.
    :return event_key: The timeslice and sorted measured-qubit indices.
    """
    return timeslice, tuple(sorted(target.value for target in targets))


def measurement_locations_by_event(
        circuits: Sequence[stim.Circuit],
) -> dict[MeasurementEventKey, MeasurementLocation]:
    """Index measurement-result locations in split circuit timeslices.

    :param circuits: A circuit split into timeslices by :func:`split_by_ticks`.
    :return locations: Instruction and target-group indices keyed by
        measurement event.
    """
    locations: dict[MeasurementEventKey, MeasurementLocation] = {}
    for timeslice, circuit in enumerate(circuits):
        for instruction_index in range(len(circuit)):
            instruction = cast(
                stim.CircuitInstruction,
                circuit[instruction_index],
            )
            if instruction.num_measurements:
                for target_index, target_group in enumerate(
                        instruction.target_groups()):
                    locations[measurement_event_key(
                        timeslice,
                        target_group,
                    )] = instruction_index, target_index
    return locations


def split_by_ticks(circuit: stim.Circuit):
    """Split a circuit into a list of circuits, one for each timeslice.

    :param circuit: A stim.Circuit object.
    
    :return circuits: A list of new stim.Circuit objects, one for each timeslice.
        Its length is `circuit.num_ticks + 1`.
        Does not modify the input circuit.
    """
    circuits: list[stim.Circuit] = []
    current_circuit = stim.Circuit()
    for op in circuit:
        if op.name == "TICK":
            circuits.append(current_circuit)
            current_circuit = stim.Circuit()
        else:
            current_circuit.append(op)
    circuits.append(current_circuit)
    return circuits

def compose_slices(circuits: list[stim.Circuit]):
    """Inverse of `split_by_ticks`.
    
    :param circuits: A list of circuits, one for each timeslice.

    :return composed: The sequential composition of the circuits in `circuits`.
        Does not modify the input circuits.
    """
    composed = stim.Circuit()
    for circuit in circuits:
        composed += circuit
        composed.append("TICK") # type: ignore
    composed.pop()  # remove last TICK
    return composed


def insert_error_events(
        circuit: stim.Circuit,
        weighted_error_events: Iterable[tuple[ErrorEvent, float]],
        measurement_locations: Mapping[
            MeasurementEventKey,
            MeasurementLocation,
        ] | None = None,
) -> stim.Circuit:
    """Insert error events with individual probabilities into a circuit.

    :param circuit: A stim.Circuit object.
    :param weighted_error_events: Error events paired with their probabilities.
    :param measurement_locations: Instruction and target-group indices keyed by
        measurement event. When omitted, construct them only if a measurement
        event is present.
    :return: A new circuit with the error events inserted.
        Does not modify the input circuit.
        The tags are not preserved in the output circuit.
    """
    events = tuple(weighted_error_events)
    circuits = split_by_ticks(circuit)
    if measurement_locations is None:
        measurement_locations = (
            measurement_locations_by_event(circuits)
            if any(
                error_event[1].startswith('M')
                for error_event, _ in events
            )
            else {}
        )
    measurement_events: list[
        tuple[int, int, int, str, tuple[stim.GateTarget, ...], float]
    ] = []
    pauli_events: list[tuple[ErrorEvent, float]] = []
    for error_event, probability in events:
        timeslice, name, targets = error_event
        if name.startswith('M'):
            instruction_index, target_index = measurement_locations[
                measurement_event_key(timeslice, targets)
            ]
            measurement_events.append((
                timeslice,
                instruction_index,
                target_index,
                name,
                targets,
                probability,
            ))
        else:
            pauli_events.append((error_event, probability))

    measurement_events.sort(
        key=lambda event: event[:3],
        reverse=True,
    )
    for (
            timeslice,
            instruction_index,
            target_index,
            name,
            targets,
            probability,
    ) in measurement_events:
        target, = targets
        instruction = cast(
            stim.CircuitInstruction,
            circuits[timeslice][instruction_index],
        )
        instruction_targets = instruction.targets_copy()
        insertand = stim.Circuit()
        if targets_before := instruction_targets[:target_index]:
            insertand.append(stim.CircuitInstruction(name, targets_before))
        insertand.append(stim.CircuitInstruction(
            name,
            [target],
            [probability],
        ))
        if targets_after := instruction_targets[target_index + 1:]:
            insertand.append(stim.CircuitInstruction(name, targets_after))
        circuits[timeslice].pop(instruction_index)
        circuits[timeslice].insert(instruction_index, insertand)

    for (timeslice, name, targets), probability in pauli_events:
        circuits[timeslice].append(name, targets, probability)

    return compose_slices(circuits)
