"""Correct distance-5 cultivation proxies for T-compatible sampling.

See ``sim/README.md`` under "Distance-five corrected reference"
for the transformation rules and their rationale.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

import stim


Coordinate = tuple[float, float]
MeasurementKey = tuple[str, Coordinate]
DetectorProfileEntry = tuple[
    Coordinate,
    int,
    tuple[MeasurementKey, ...],
]

_PRE_GROWTH_X_CHECK_COORDS = (
    (4.0, 1.0),
    (3.0, 1.0),
    (2.0, 0.0),
    (1.0, 0.0),
    (2.0, 2.0),
    (0.0, 1.0),
)
_BELL_GROWTH_MEASUREMENT_COORDS = (
    (5.0, 2.0),
    (6.0, 2.0),
    (5.0, 4.0),
    (4.0, 5.0),
    (3.0, 5.0),
)
_FIRST_D5_Z_CHECK_COORDS = (
    (8.0, 1.0),
    (6.0, 4.0),
    (6.0, 2.0),
    (6.0, 0.0),
    (4.0, 5.0),
    (4.0, 3.0),
    (4.0, 1.0),
    (2.0, 2.0),
    (2.0, 0.0),
)
_FIRST_D5_X_CHECK_COORDS = (
    (7.0, 1.0),
    (5.0, 4.0),
    (5.0, 2.0),
    (5.0, 0.0),
    (3.0, 5.0),
    (3.0, 3.0),
    (3.0, 1.0),
    (1.0, 2.0),
    (1.0, 0.0),
)

_FEEDFORWARD_RULES: dict[
    MeasurementKey,
    tuple[tuple[str, Coordinate], ...],
] = {
    ("MX", (5.0, 0.0)): (
        ("CZ", (7.0, 0.0)),
        ("CZ", (8.0, 0.0)),
    ),
    ("MX", (3.0, 3.0)): (
        ("CZ", (3.0, 4.0)),
        ("CZ", (4.0, 6.0)),
    ),
    ("M", (6.0, 0.0)): (
        ("CX", (7.0, 0.0)),
        ("CX", (8.0, 0.0)),
        ("CX", (7.0, 2.0)),
        ("CX", (8.0, 0.0)),
    ),
    ("M", (6.0, 2.0)): (
        ("CX", (7.0, 0.0)),
        ("CX", (8.0, 0.0)),
    ),
    ("M", (4.0, 1.0)): (
        ("CX", (3.0, 4.0)),
        ("CX", (4.0, 6.0)),
    ),
    ("M", (4.0, 3.0)): (
        ("CX", (0.0, 0.0)),
        ("CX", (3.0, 0.0)),
        ("CX", (4.0, 6.0)),
    ),
    ("M", (4.0, 5.0)): (
        ("CX", (3.0, 4.0)),
        ("CX", (4.0, 6.0)),
    ),
}
_ODD_PARITY_SOURCE = ("M", (4.0, 3.0))

_REQUIRED_COORDS = frozenset(
    _PRE_GROWTH_X_CHECK_COORDS
    + _BELL_GROWTH_MEASUREMENT_COORDS
    + _FIRST_D5_Z_CHECK_COORDS
    + _FIRST_D5_X_CHECK_COORDS
    + tuple(
        target_coordinate
        for corrections in _FEEDFORWARD_RULES.values()
        for _, target_coordinate in corrections
    )
)

_EXPECTED_GROWTH_SIGNATURE_SHA256 = (
    "ba20de8564f7f68dde63833cab175711a73e956a6c026005ee264017fb2d4bbe"
)

_EXPECTED_DETECTOR_SOURCE_PROFILE: tuple[DetectorProfileEntry, ...] = (
    ((6.0, 2.0), 1, (("M", (6.0, 2.0)),)),
    ((4.0, 5.0), 1, (("M", (4.0, 5.0)),)),
    ((4.0, 1.0), 1, (("M", (4.0, 1.0)),)),
    ((6.0, 2.0), 2, (("M", (6.0, 2.0)),)),
    ((6.0, 0.0), 3, (("M", (6.0, 0.0)),)),
    (
        (6.0, 0.5),
        3,
        (("M", (6.0, 0.0)), ("M", (6.0, 2.0))),
    ),
    ((4.0, 3.0), 2, (("M", (4.0, 3.0)),)),
    (
        (4.0, 1.0),
        3,
        (("M", (4.0, 1.0)), ("M", (4.0, 5.0))),
    ),
    ((4.0, 2.0), 2, (("M", (4.0, 3.0)),)),
    ((5.0, 0.0), 2, (("MX", (5.0, 0.0)),)),
    ((3.0, 3.0), 2, (("MX", (3.0, 3.0)),)),
    (
        (0.0, 1.0),
        26,
        (
            ("M", (6.0, 2.0)),
            ("M", (4.0, 1.0)),
            ("M", (4.0, 3.0)),
            ("M", (4.0, 5.0)),
        ),
    ),
)


def circuit_instructions(
    circuit: stim.Circuit,
) -> tuple[stim.CircuitInstruction, ...]:
    """Return circuit instructions after rejecting unsupported repeat blocks.

    :param circuit: Cultivation circuit that must not contain repeat blocks.
    :return: Circuit instructions in execution order.
    :raises NotImplementedError: If the circuit contains a repeat block.
    """
    instructions: list[stim.CircuitInstruction] = []
    for index in range(len(circuit)):
        operation = circuit[index]
        if isinstance(operation, stim.CircuitRepeatBlock):
            raise NotImplementedError(
                "distance-5 correction does not support REPEAT blocks"
            )
        instructions.append(operation)
    return tuple(instructions)


def coordinate_maps(
    instructions: Sequence[stim.CircuitInstruction],
) -> tuple[dict[int, Coordinate], dict[Coordinate, int]]:
    """Map circuit qubits to two-dimensional spatial coordinates.

    :param instructions: Cultivation circuit instructions containing coordinates.
    :return: Spatial coordinate for each declared qubit.
    :return: Qubit identifier for each declared spatial coordinate.
    :raises RuntimeError: If required coordinates are missing or ambiguous.
    """
    coordinate_by_qubit = {}
    qubit_by_coordinate = {}
    for instruction in instructions:
        if instruction.name != "QUBIT_COORDS":
            continue
        args = instruction.gate_args_copy()
        coordinate = (float(args[0]), float(args[1]))
        qubit = instruction.targets_copy()[0].value
        if coordinate in qubit_by_coordinate:
            raise RuntimeError(
                f"distance-5 correction found duplicate coordinate {coordinate}"
            )
        coordinate_by_qubit[qubit] = coordinate
        qubit_by_coordinate[coordinate] = qubit

    missing = sorted(_REQUIRED_COORDS - qubit_by_coordinate.keys())
    if missing:
        raise RuntimeError(
            "distance-5 correction is missing required coordinates: "
            f"{missing!r}"
        )
    return coordinate_by_qubit, qubit_by_coordinate


def _instruction_coordinates(
    instruction: stim.CircuitInstruction,
    coordinate_by_qubit: dict[int, Coordinate],
) -> tuple[Coordinate, ...]:
    """Resolve the ordered qubit targets of one instruction to coordinates.

    :param instruction: Circuit instruction whose targets should be resolved.
    :param coordinate_by_qubit: Spatial coordinate for each declared qubit.
    :return: Ordered spatial coordinates of the instruction's qubit targets.
    :raises RuntimeError: If a targeted qubit has no spatial coordinate.
    """
    try:
        return tuple(
            coordinate_by_qubit[target.value]
            for target in instruction.targets_copy()
            if target.is_qubit_target
        )
    except KeyError as error:
        raise RuntimeError(
            "distance-5 correction encountered a target without coordinates: "
            f"qubit {error.args[0]}"
        ) from error


def _find_landmarks(
    instructions: Sequence[stim.CircuitInstruction],
    coordinate_by_qubit: dict[int, Coordinate],
) -> tuple[int, int, int, int]:
    """Locate the coordinate-defined growth and syndrome landmarks.

    :param instructions: Flattened circuit instructions in execution order.
    :param coordinate_by_qubit: Spatial coordinate for each declared qubit.
    :return: Index of the pre-growth X-check measurement.
    :return: Index of the Bell-growth measurement.
    :return: Index of the first distance-5 Z-check measurement.
    :return: Index of the closing tick after that syndrome round.
    :raises RuntimeError: If the expected landmark sequence is absent.
    """
    pre_growth_indices = [
        index
        for index, instruction in enumerate(instructions)
        if instruction.name == "MX"
        and _instruction_coordinates(instruction, coordinate_by_qubit)
        == _PRE_GROWTH_X_CHECK_COORDS
    ]
    if len(pre_growth_indices) != 1:
        raise RuntimeError(
            "distance-5 correction could not uniquely locate the "
            "pre-growth X-check"
        )
    pre_growth_index = pre_growth_indices[0]

    bell_growth_indices = [
        index
        for index, instruction in enumerate(instructions)
        if index > pre_growth_index
        and instruction.name == "M"
        and _instruction_coordinates(instruction, coordinate_by_qubit)
        == _BELL_GROWTH_MEASUREMENT_COORDS
    ]
    if len(bell_growth_indices) != 1:
        raise RuntimeError(
            "distance-5 correction could not uniquely locate the "
            "Bell-growth measurement"
        )
    bell_growth_index = bell_growth_indices[0]

    first_d5_index = next(
        (
            index
            for index, instruction in enumerate(instructions)
            if index > bell_growth_index
            and instruction.name == "M"
            and _instruction_coordinates(instruction, coordinate_by_qubit)
            == _FIRST_D5_Z_CHECK_COORDS
        ),
        None,
    )
    if first_d5_index is None:
        raise RuntimeError(
            "distance-5 correction could not locate the first "
            "distance-5 Z-check measurement"
        )
    first_d5_x_index = first_d5_index + 1
    if (
        first_d5_x_index >= len(instructions)
        or instructions[first_d5_x_index].name != "MX"
        or _instruction_coordinates(
            instructions[first_d5_x_index],
            coordinate_by_qubit,
        )
        != _FIRST_D5_X_CHECK_COORDS
    ):
        raise RuntimeError(
            "distance-5 correction found an incompatible first "
            "distance-5 X-check measurement"
        )

    closing_tick_index = next(
        (
            index
            for index in range(first_d5_x_index + 1, len(instructions))
            if instructions[index].name == "TICK"
        ),
        None,
    )
    if closing_tick_index is None:
        raise RuntimeError(
            "distance-5 correction found no closing tick after the first "
            "distance-5 syndrome round"
        )
    return (
        pre_growth_index,
        bell_growth_index,
        first_d5_index,
        closing_tick_index,
    )


def _target_signature(
    target: stim.GateTarget,
    coordinate_by_qubit: dict[int, Coordinate],
) -> tuple[str, object]:
    """Normalize one Stim target independently of integer qubit identifiers.

    :param target: Stim instruction target to normalize.
    :param coordinate_by_qubit: Spatial coordinate for each declared qubit.
    :return: Target kind and its coordinate or classical index.
    """
    if target.is_combiner:
        return ("combiner", "*")
    if target.is_measurement_record_target:
        return ("record", target.value)
    if target.is_sweep_bit_target:
        return ("sweep", target.value)
    if target.is_x_target:
        return ("X", coordinate_by_qubit[target.value])
    if target.is_y_target:
        return ("Y", coordinate_by_qubit[target.value])
    if target.is_z_target:
        return ("Z", coordinate_by_qubit[target.value])
    kind = "inverted" if target.is_inverted_result_target else "qubit"
    return (kind, coordinate_by_qubit[target.value])


def _growth_signature(
    instructions: Sequence[stim.CircuitInstruction],
    coordinate_by_qubit: dict[int, Coordinate],
    start: int,
    stop: int,
) -> str:
    """Hash the noiseless coordinate structure of the Bell-growth window.

    Detector definitions are validated separately through their measurement
    dependencies.

    :param instructions: Flattened circuit instructions in execution order.
    :param coordinate_by_qubit: Spatial coordinate for each declared qubit.
    :param start: Inclusive index of the pre-growth landmark.
    :param stop: Inclusive index of the first distance-5 closing tick.
    :return: SHA-256 digest of the coordinate-normalized growth structure.
    """
    window = stim.Circuit()
    for instruction in instructions[start : stop + 1]:
        window.append(instruction)
    signature = [
        (
            instruction.name,
            tuple(float(arg) for arg in instruction.gate_args_copy()),
            tuple(
                _target_signature(target, coordinate_by_qubit)
                for target in instruction.targets_copy()
            ),
        )
        for instruction in circuit_instructions(window.without_noise())
        if instruction.name != "DETECTOR"
    ]
    serialized = json.dumps(signature, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _source_measurement_indices(
    instructions: Sequence[stim.CircuitInstruction],
    coordinate_by_qubit: dict[int, Coordinate],
    first_d5_index: int,
) -> dict[MeasurementKey, int]:
    """Find absolute record indices of the seven feedforward measurements.

    :param instructions: Flattened circuit instructions in execution order.
    :param coordinate_by_qubit: Spatial coordinate for each declared qubit.
    :param first_d5_index: Index of the first distance-5 M instruction.
    :return: Absolute measurement index for each feedforward source.
    """
    measurement_count = sum(
        instruction.num_measurements
        for instruction in instructions[:first_d5_index]
    )
    result = {}
    for instruction in instructions[first_d5_index : first_d5_index + 2]:
        for offset, target in enumerate(instruction.targets_copy()):
            key = (instruction.name, coordinate_by_qubit[target.value])
            if key in _FEEDFORWARD_RULES:
                result[key] = measurement_count + offset
        measurement_count += instruction.num_measurements
    return result


def _detector_source_profile(
    instructions: Sequence[stim.CircuitInstruction],
    source_measurement_indices: dict[MeasurementKey, int],
) -> tuple[DetectorProfileEntry, ...]:
    """Describe detectors that reference feedforward source measurements.

    :param instructions: Flattened circuit instructions in execution order.
    :param source_measurement_indices: Absolute source measurement indices.
    :return: Detector spatial coordinates, sizes, and source keys.
    """
    key_by_measurement = {
        measurement: key
        for key, measurement in source_measurement_indices.items()
    }
    source_order = {
        key: index for index, key in enumerate(_FEEDFORWARD_RULES)
    }
    profile = []
    measurement_count = 0
    for instruction in instructions:
        if instruction.name == "DETECTOR":
            record_targets = [
                target
                for target in instruction.targets_copy()
                if target.is_measurement_record_target
            ]
            sources = sorted(
                (
                    key_by_measurement[measurement_count + target.value]
                    for target in record_targets
                    if measurement_count + target.value in key_by_measurement
                ),
                key=source_order.__getitem__,
            )
            if sources:
                detector_args = instruction.gate_args_copy()
                profile.append(
                    (
                        (
                            float(detector_args[0]),
                            float(detector_args[1]),
                        ),
                        len(record_targets),
                        tuple(sources),
                    )
                )
        measurement_count += instruction.num_measurements
    return tuple(profile)


def _validate_detector_profile(
    instructions: Sequence[stim.CircuitInstruction],
    source_measurement_indices: dict[MeasurementKey, int],
) -> None:
    """Require the detector dependencies expected by the healing rules.

    :param instructions: Flattened circuit instructions in execution order.
    :param source_measurement_indices: Absolute source measurement indices.
    :return: None.
    :raises RuntimeError: If the detector structure is incompatible or healed.
    """
    profile = _detector_source_profile(
        instructions,
        source_measurement_indices,
    )
    if profile != _EXPECTED_DETECTOR_SOURCE_PROFILE:
        raise RuntimeError(
            "distance-5 correction found an incompatible detector-source "
            "profile; the circuit may already be corrected"
        )


def _healed_detector_targets(
    instruction: stim.CircuitInstruction,
    measurement_count: int,
    source_indices: frozenset[int],
    odd_parity_index: int,
) -> list[stim.GateTarget]:
    """Remove invalid source records from one affected detector.

    :param instruction: Detector instruction to heal.
    :param measurement_count: Measurements produced before the detector.
    :param source_indices: Feedforward source measurement indices.
    :param odd_parity_index: Source removed from the long logical detector.
    :return: Original or healed detector targets.
    """
    targets = instruction.targets_copy()
    record_targets = [
        target for target in targets if target.is_measurement_record_target
    ]
    if len(record_targets) > 10:
        healed = [
            target
            for target in targets
            if not (
                target.is_measurement_record_target
                and measurement_count + target.value == odd_parity_index
            )
        ]
    else:
        healed = [
            target
            for target in targets
            if not (
                target.is_measurement_record_target
                and measurement_count + target.value in source_indices
            )
        ]
    return targets if not healed else healed


def _append_feedforward_corrections(
    output: stim.Circuit,
    qubit_by_coordinate: dict[Coordinate, int],
    source_measurement_indices: dict[MeasurementKey, int],
    measurement_count: int,
) -> None:
    """Append the coordinate-defined classical feedforward operations.

    :param output: Circuit receiving the corrections.
    :param qubit_by_coordinate: Qubit identifier for each spatial coordinate.
    :param source_measurement_indices: Absolute source measurement indices.
    :param measurement_count: Measurements produced before insertion.
    :return: None.
    """
    for source, corrections in _FEEDFORWARD_RULES.items():
        record_offset = measurement_count - source_measurement_indices[source]
        for gate, target_coordinate in corrections:
            output.append(
                gate,
                [
                    stim.target_rec(-record_offset),
                    qubit_by_coordinate[target_coordinate],
                ],
            )


def correct_d5_cultivation_circuit(
    circuit: stim.Circuit,
) -> stim.Circuit:
    """Add T-compatible feedforward and detector healing to a D5 S proxy.

    The input may already contain noise. The returned circuit remains a
    Stim-compatible S/S_DAG circuit; T/T_DAG substitution happens later in the
    SymFT framework.

    :param circuit: Distance-five inject-and-cultivate S-proxy circuit.
    :return: Circuit with feedforward and healed detectors.
    :raises RuntimeError: If correction-sensitive circuit structure has drifted.
    :raises NotImplementedError: If the circuit contains a repeat block.
    """
    instructions = circuit_instructions(circuit)
    coordinate_by_qubit, qubit_by_coordinate = coordinate_maps(instructions)
    (
        pre_growth_index,
        _,
        first_d5_index,
        closing_tick_index,
    ) = _find_landmarks(instructions, coordinate_by_qubit)
    signature = _growth_signature(
        instructions,
        coordinate_by_qubit,
        pre_growth_index,
        closing_tick_index,
    )
    if signature != _EXPECTED_GROWTH_SIGNATURE_SHA256:
        raise RuntimeError(
            "distance-5 correction found an incompatible coordinate-based "
            "Bell-growth layout"
        )

    source_measurement_indices = _source_measurement_indices(
        instructions,
        coordinate_by_qubit,
        first_d5_index,
    )
    _validate_detector_profile(instructions, source_measurement_indices)

    source_indices = frozenset(source_measurement_indices.values())
    odd_parity_index = source_measurement_indices[_ODD_PARITY_SOURCE]
    output = stim.Circuit()
    measurement_count = 0
    for index, instruction in enumerate(instructions):
        if instruction.name == "DETECTOR":
            output.append(
                "DETECTOR",
                _healed_detector_targets(
                    instruction,
                    measurement_count,
                    source_indices,
                    odd_parity_index,
                ),
                instruction.gate_args_copy(),
                tag=instruction.tag,
            )
        else:
            output.append(instruction)
        measurement_count += instruction.num_measurements
        if index == closing_tick_index:
            _append_feedforward_corrections(
                output,
                qubit_by_coordinate,
                source_measurement_indices,
                measurement_count,
            )
    return output
