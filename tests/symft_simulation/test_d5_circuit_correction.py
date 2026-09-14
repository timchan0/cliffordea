"""Tests for coordinate-based distance-five cultivation correction."""

from __future__ import annotations
from collections.abc import Sequence

import pytest
import stim

from cliffordea.symft_simulation.d5_circuit_correction import (
    circuit_instructions,
    coordinate_maps,
    correct_d5_cultivation_circuit,
)
from cliffordea.symft_simulation.msc_framework import (
    REFERENCE_PATH,
    TASK_SCHEMA_VERSION,
    build_tasks,
)


FULL_CIRCUITS_DIR = REFERENCE_PATH.parent
REFERENCE_PAIRS = (
    (
        "d5a19_inject+cultivate_uncorrected.stim",
        "d5a19_inject+cultivate.stim",
    ),
    (
        "d5a19_inject+cultivate_uncorrected_p1e-3.stim",
        "d5a19_inject+cultivate_p1e-3.stim",
    ),
)


def _read_circuit(filename: str) -> stim.Circuit:
    """Read one full cultivation circuit fixture.

    :param filename: Basename within the full-circuit fixture directory.
    :return: Parsed Stim circuit.
    """
    return stim.Circuit(
        (FULL_CIRCUITS_DIR / filename).read_text(encoding="utf-8")
    )


def _feedforward_coordinate_profile(
    circuit: stim.Circuit,
) -> tuple[tuple[str, int, tuple[float, float]], ...]:
    """Describe classical CX/CZ targets by record offset and coordinate.

    :param circuit: Circuit containing coordinate declarations and feedback.
    :return: Ordered gate, record offset, and target-coordinate entries.
    """
    instructions = circuit_instructions(circuit)
    coordinate_by_qubit, _ = coordinate_maps(instructions)

    profile = []
    for instruction in instructions:
        if instruction.name not in ("CX", "CZ"):
            continue
        targets = instruction.targets_copy()
        for index in range(0, len(targets), 2):
            source = targets[index]
            target = targets[index + 1]
            if source.is_measurement_record_target:
                profile.append(
                    (
                        instruction.name,
                        source.value,
                        coordinate_by_qubit[target.value],
                    )
                )
    return tuple(profile)


def _detector_record_profile(
    circuit: stim.Circuit,
) -> tuple[tuple[tuple[float, ...], tuple[int, ...]], ...]:
    """Describe all detector record targets independently of qubit IDs.

    :param circuit: Circuit whose detectors should be described.
    :return: Detector coordinates and relative measurement-record offsets.
    """
    return tuple(
        (
            tuple(float(arg) for arg in instruction.gate_args_copy()),
            tuple(
                target.value
                for target in instruction.targets_copy()
                if target.is_measurement_record_target
            ),
        )
        for instruction in circuit_instructions(circuit)
        if instruction.name == "DETECTOR"
    )


def _remapped_target(
    target: stim.GateTarget,
    qubit_map: dict[int, int],
) -> int | stim.GateTarget:
    """Remap a quantum target while preserving its Stim target kind.

    :param target: Original Stim target.
    :param qubit_map: Replacement identifier for every original qubit.
    :return: Remapped quantum target or unchanged classical target.
    """
    if (
        target.is_combiner
        or target.is_measurement_record_target
        or target.is_sweep_bit_target
    ):
        return target
    if target.is_x_target:
        return stim.target_x(qubit_map[target.value])
    if target.is_y_target:
        return stim.target_y(qubit_map[target.value])
    if target.is_z_target:
        return stim.target_z(qubit_map[target.value])
    if target.is_inverted_result_target:
        return stim.target_inv(qubit_map[target.value])
    return qubit_map[target.value]


def _reindex_with_spectators(circuit: stim.Circuit) -> stim.Circuit:
    """Shift all original IDs and add unused qubits at new coordinates.

    :param circuit: Circuit whose original qubits should be reindexed.
    :return: Equivalent circuit with five unused leading qubits.
    """
    qubit_map = {qubit: qubit + 5 for qubit in range(circuit.num_qubits)}
    result = stim.Circuit()
    for qubit in range(5):
        result.append("QUBIT_COORDS", [qubit], [100.0 + qubit, 100.0])
    for instruction in circuit_instructions(circuit):
        result.append(
            instruction.name,
            [
                _remapped_target(target, qubit_map)
                for target in instruction.targets_copy()
            ],
            instruction.gate_args_copy(),
            tag=instruction.tag,
        )
    return result


def _replace_instruction(
    circuit: stim.Circuit,
    index: int,
    *,
    targets: Sequence[int | stim.GateTarget] | None = None,
    args: list[float] | None = None,
) -> stim.Circuit:
    """Replace one instruction's targets or gate arguments in a test circuit.

    :param circuit: Source circuit to copy.
    :param index: Instruction index to replace.
    :param targets: Optional replacement targets.
    :param args: Optional replacement gate arguments.
    :return: Circuit containing the requested structural mutation.
    """
    result = stim.Circuit()
    for current_index, instruction in enumerate(
        circuit_instructions(circuit)
    ):
        if current_index == index:
            result.append(
                instruction.name,
                instruction.targets_copy() if targets is None else targets,
                instruction.gate_args_copy() if args is None else args,
                tag=instruction.tag,
            )
        else:
            result.append(instruction)
    return result


@pytest.mark.parametrize(("source_name", "corrected_name"), REFERENCE_PAIRS)
def test_committed_references_equal_fresh_correction(
    source_name: str,
    corrected_name: str,
) -> None:
    """Each committed corrected reference is reproducible from its source.

    :param source_name: Uncorrected source-circuit basename.
    :param corrected_name: Corrected output-circuit basename.
    :return: None.
    """
    corrected_text = (FULL_CIRCUITS_DIR / corrected_name).read_text(
        encoding="utf-8"
    )

    regenerated = correct_d5_cultivation_circuit(_read_circuit(source_name))

    assert corrected_text == str(regenerated) + "\n"


@pytest.mark.parametrize(("source_name", "corrected_name"), REFERENCE_PAIRS)
def test_correction_preserves_shape_phase_layers_and_noise(
    source_name: str,
    corrected_name: str,
) -> None:
    """Correction changes feedback and detectors without changing circuit shape.

    :param source_name: Uncorrected source-circuit basename.
    :param corrected_name: Corrected output-circuit basename.
    :return: None.
    """
    source = _read_circuit(source_name)
    corrected = _read_circuit(corrected_name)
    source_instructions = circuit_instructions(source)
    corrected_instructions = circuit_instructions(corrected)

    assert (
        corrected.num_qubits,
        corrected.num_measurements,
        corrected.num_detectors,
        corrected.num_observables,
    ) == (42, 112, 107, 1)
    assert len(_feedforward_coordinate_profile(corrected)) == 17
    assert [
        str(instruction)
        for instruction in corrected_instructions
        if instruction.name in ("S", "S_DAG")
    ] == [
        str(instruction)
        for instruction in source_instructions
        if instruction.name in ("S", "S_DAG")
    ]
    assert [
        str(instruction)
        for instruction in corrected_instructions
        if instruction.name
        in ("DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR")
    ] == [
        str(instruction)
        for instruction in source_instructions
        if instruction.name
        in ("DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR")
    ]
    assert [
        str(instruction)
        for instruction in corrected_instructions
        if instruction.num_measurements
    ] == [
        str(instruction)
        for instruction in source_instructions
        if instruction.num_measurements
    ]
    assert all(
        instruction.name not in ("T", "T_DAG")
        for instruction in corrected_instructions
    )


def test_noisy_and_noiseless_corrections_have_identical_ideal_structure() -> None:
    """Removing noise from the corrected noisy reference gives the ideal one.

    :return: None.
    """
    noiseless = _read_circuit("d5a19_inject+cultivate.stim")
    noisy = _read_circuit("d5a19_inject+cultivate_p1e-3.stim")

    assert noisy != noiseless
    assert noisy.without_noise() == noiseless


def test_correction_uses_coordinates_after_reindexing_with_spectators() -> None:
    """Shifted IDs and unused-coordinate qubits preserve correction semantics.

    :return: None.
    """
    source = _read_circuit("d5a19_inject+cultivate_uncorrected.stim")
    baseline = correct_d5_cultivation_circuit(source)
    remapped = _reindex_with_spectators(source)

    corrected_remapped = correct_d5_cultivation_circuit(remapped)

    assert corrected_remapped.num_qubits == 47
    assert corrected_remapped.num_measurements == baseline.num_measurements
    assert corrected_remapped.num_detectors == baseline.num_detectors
    assert _feedforward_coordinate_profile(
        corrected_remapped
    ) == _feedforward_coordinate_profile(baseline)
    assert _detector_record_profile(
        corrected_remapped
    ) == _detector_record_profile(baseline)


def test_correction_ignores_flags_and_detector_time_shifts() -> None:
    """Flag-only records and detector time shifts preserve D5 correction.

    :return: None.
    """
    source = _read_circuit("d5a19_inject+cultivate_uncorrected.stim")
    source_instructions = circuit_instructions(source)
    first_gate_index = next(
        index
        for index, instruction in enumerate(source_instructions)
        if instruction.name != "QUBIT_COORDS"
    )
    flag_qubit = source.num_qubits
    flag_coordinate = (100.0, 100.0)
    flagged = stim.Circuit()
    for index, instruction in enumerate(source_instructions):
        if index == first_gate_index:
            flagged.append("QUBIT_COORDS", [flag_qubit], flag_coordinate)
            flagged.append("R", [flag_qubit])
            flagged.append("M", [flag_qubit])
            flagged.append(
                "DETECTOR",
                [stim.target_rec(-1)],
                (*flag_coordinate, 0.0),
            )
            flagged.append("TICK") # type: ignore
        if instruction.name == "DETECTOR":
            args = instruction.gate_args_copy()
            args[2] += 1.0
            flagged.append(
                "DETECTOR",
                instruction.targets_copy(),
                args,
                tag=instruction.tag,
            )
        else:
            flagged.append(instruction)

    baseline = correct_d5_cultivation_circuit(source)
    corrected_flagged = correct_d5_cultivation_circuit(flagged)

    assert corrected_flagged.num_qubits == baseline.num_qubits + 1
    assert corrected_flagged.num_measurements == baseline.num_measurements + 1
    assert corrected_flagged.num_detectors == baseline.num_detectors + 1
    assert _feedforward_coordinate_profile(
        corrected_flagged
    ) == _feedforward_coordinate_profile(baseline)
    assert (
        (*flag_coordinate, 0.0),
        (-1,),
    ) in _detector_record_profile(corrected_flagged)


def test_correction_rejects_sensitive_structural_drift() -> None:
    """Growth, syndrome, and coordinate mutations fail before correction.

    :return: None.
    """
    source = _read_circuit("d5a19_inject+cultivate_uncorrected.stim")
    instructions = circuit_instructions(source)
    bell_measurement_index = next(
        index
        for index, instruction in enumerate(instructions)
        if instruction.name == "M"
        and [target.value for target in instruction.targets_copy()]
        == [26, 33, 28, 22, 16]
    )
    growth_gate_index = next(
        index
        for index in range(bell_measurement_index + 1, len(instructions))
        if instructions[index].name == "CX"
    )
    growth_targets = instructions[growth_gate_index].targets_copy()
    growth_targets[0] = stim.GateTarget(1)
    changed_growth = _replace_instruction(
        source,
        growth_gate_index,
        targets=growth_targets,
    )

    first_d5_index = next(
        index
        for index in range(bell_measurement_index + 1, len(instructions))
        if instructions[index].name == "M"
        and len(instructions[index].targets_copy()) == 9
    )
    x_check_index = first_d5_index + 1
    x_check_targets = instructions[x_check_index].targets_copy()
    x_check_targets[0] = stim.GateTarget(36)
    changed_x_check = _replace_instruction(
        source,
        x_check_index,
        targets=x_check_targets,
    )

    coordinate_index = next(
        index
        for index, instruction in enumerate(instructions)
        if instruction.name == "QUBIT_COORDS"
        and instruction.gate_args_copy() == [4.0, 6.0]
    )
    changed_coordinate = _replace_instruction(
        source,
        coordinate_index,
        args=[40.0, 60.0],
    )

    with pytest.raises(RuntimeError, match="Bell-growth layout"):
        correct_d5_cultivation_circuit(changed_growth)
    with pytest.raises(RuntimeError, match="incompatible first distance-five"):
        correct_d5_cultivation_circuit(changed_x_check)
    with pytest.raises(RuntimeError, match="missing required coordinates"):
        correct_d5_cultivation_circuit(changed_coordinate)


def test_correction_rejects_repeated_application() -> None:
    """An already-healed circuit is rejected instead of corrected twice.

    :return: None.
    """
    source = _read_circuit("d5a19_inject+cultivate_uncorrected.stim")
    corrected = correct_d5_cultivation_circuit(source)

    with pytest.raises(RuntimeError, match="already be corrected"):
        correct_d5_cultivation_circuit(corrected)


def test_correction_rejects_repeat_blocks() -> None:
    """Repeat blocks fail explicitly instead of being silently flattened.

    :return: None.
    """
    circuit = stim.Circuit(
        """
        REPEAT 2 {
            H 0
        }
        """
    )

    with pytest.raises(NotImplementedError, match="does not support REPEAT"):
        correct_d5_cultivation_circuit(circuit)


def test_corrected_t_only_task_has_distinct_schema_four_identity() -> None:
    """The corrected noisy reference creates one distinct named T task.

    :return: None.
    """
    uncorrected_path = FULL_CIRCUITS_DIR / (
        "d5a19_inject+cultivate_uncorrected_p1e-3.stim"
    )
    corrected_path = FULL_CIRCUITS_DIR / (
        "d5a19_inject+cultivate_p1e-3.stim"
    )
    uncorrected_task = build_tasks(
        uncorrected_path.read_text(encoding="utf-8"),
        uncorrected_path.stem,
        noise_levels=(0.001,),
        variants=("T",),
    )[0]
    corrected_tasks = build_tasks(
        corrected_path.read_text(encoding="utf-8"),
        corrected_path.stem,
        noise_levels=(0.001,),
        variants=("T",),
    )

    assert len(corrected_tasks) == 1
    corrected_task = corrected_tasks[0]
    assert corrected_task.json_metadata["schema_version"] == TASK_SCHEMA_VERSION
    assert corrected_task.json_metadata["schema_version"] == 4
    assert corrected_task.json_metadata["variant"] == "T"
    assert corrected_task.json_metadata["circuit_name"] == (
        "d5a19_inject+cultivate_p1e-3"
    )
    assert corrected_task.strong_id() != uncorrected_task.strong_id()
