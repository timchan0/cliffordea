"""Tests for the Sinter-backed SymFT cultivation framework."""

from __future__ import annotations

import collections
import json
import math
import re
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import sinter
import stim

from cliffordep.symft_simulation.msc_framework import (
    ACTIVE_THREADS_PREFIX,
    DEFAULT_CIRCUIT_NAME,
    DEFAULT_NOISE_LEVELS,
    PLOT_ONLY_AGGREGATE_KEY,
    PLOT_PROVENANCE_KEY,
    REFERENCE_PATH,
    STATS_FILENAME,
    STREAM_COUNT_PREFIX,
    CompiledSymftSinterSampler,
    _StreamSequence,
    _resume_noise_levels,
    aggregate_stats_for_plot,
    build_tasks,
    inspect_circuit,
    make_variant_text,
    next_stream_ids,
    read_plot_stats,
    sampler_settings,
    sha256_text,
    smoke_sample,
    validate_all_variants,
    validate_resume_stats,
)
from cliffordep.symft_simulation.run_simulation import (
    build_parser,
    resolve_circuit_name,
)


class FakeSymftCountsSampler:
    """Return deterministic SymFT-shaped count dictionaries."""

    def __init__(self, active_threads: int | None = None) -> None:
        """Initialize the fake count sampler.

        :param self: Fake sampler being initialized.
        :param active_threads: Fixed reported threads, or None to derive them.
        :return: None.
        """
        self.active_threads = active_threads
        self.calls: list[tuple[int, int]] = []

    def sample(self, *, shots: int, stream_id: int) -> dict[str, Any]:
        """Return internally consistent aggregate counts.

        :param self: Fake sampler receiving the call.
        :param shots: Attempted shots requested by the adapter.
        :param stream_id: Explicit SymFT random stream identifier.
        :return: SymFT-shaped aggregate count dictionary.
        """
        self.calls.append((shots, stream_id))
        return {
            "shots": shots,
            "accepted": shots - 2,
            "discarded": 2,
            "logical_errors": 1,
            "active_threads": (
                self.active_threads
                if self.active_threads is not None
                else min(8, math.ceil(shots / 2))
            ),
            "timing": {"sample_s": 0.25},
        }


class FixedCompiledSampler(sinter.CompiledSampler):
    """Produce fixed-size batches while exposing sequential stream IDs."""

    def __init__(self, next_stream_id: int) -> None:
        """Initialize a deterministic compiled sampler.

        :param self: Compiled sampler being initialized.
        :param next_stream_id: First stream identifier to report.
        :return: None.
        """
        self.next_stream_id = next_stream_id

    def handles_throttling(self) -> bool:
        """Disable Sinter's adaptive throttling for deterministic tests.

        :param self: Fixed compiled sampler.
        :return: True because the fake controls its own batch size.
        """
        return True

    def sample(self, suggested_shots: int) -> sinter.AnonTaskStats:
        """Return a fixed batch carrying the next fake stream identifier.

        :param self: Fixed compiled sampler.
        :param suggested_shots: Maximum shots requested by Sinter.
        :return: Deterministic anonymous task statistics.
        """
        shots = min(suggested_shots, 10)
        stream_id = self.next_stream_id
        self.next_stream_id += 1
        return sinter.AnonTaskStats(
            shots=shots,
            errors=1,
            custom_counts=collections.Counter(
                {f"{STREAM_COUNT_PREFIX}{stream_id}": 1}
            ),
        )


class FixedSampler(sinter.Sampler):
    """Compile deterministic samplers for Sinter resume tests."""

    def __init__(self, next_stream_id: int) -> None:
        """Initialize the deterministic sampler factory.

        :param self: Sampler factory being initialized.
        :param next_stream_id: First stream identifier to report.
        :return: None.
        """
        self.next_stream_id = next_stream_id

    def compiled_sampler_for_task(
        self,
        task: sinter.Task,
    ) -> sinter.CompiledSampler:
        """Create a deterministic sampler for a resume test task.

        :param self: Fixed sampler factory.
        :param task: Sinter task being compiled.
        :return: Deterministic compiled sampler.
        """
        return FixedCompiledSampler(self.next_stream_id)


def _fixed_task(case: str = "resume") -> sinter.Task:
    """Create one stable task for Sinter resume tests.

    :param case: Metadata label separating tasks in a shared CSV.
    :return: Task with an explicitly supplied empty detector error model.
    """
    return sinter.Task(
        circuit=stim.Circuit(),
        detector_error_model=stim.DetectorErrorModel(),
        decoder="fixed",
        json_metadata={"case": case},
    )


def _task_stat(
    task: sinter.Task,
    *,
    shots: int,
    errors: int,
    discards: int,
    stream_ids: tuple[int, ...],
) -> sinter.TaskStats:
    """Create one persisted statistic matching a task.

    :param task: Task supplying identity and metadata.
    :param shots: Attempted shot count.
    :param errors: Logical-error count.
    :param discards: Detector-rejected shot count.
    :param stream_ids: Random streams represented by the statistic.
    :return: Sinter statistic compatible with resume validation.
    """
    custom_counts = collections.Counter(
        {f"{STREAM_COUNT_PREFIX}{stream_id}": 1 for stream_id in stream_ids}
    )
    return sinter.TaskStats(
        strong_id=task.strong_id(),
        decoder=str(task.decoder),
        json_metadata=task.json_metadata,
        shots=shots,
        errors=errors,
        discards=discards,
        seconds=0.001,
        custom_counts=custom_counts,
    )


def _plot_source_stat(
    strong_id: str,
    metadata: dict[str, Any],
    *,
    decoder: str = "symft_counts",
    shots: int = 100,
    errors: int = 3,
    discards: int = 20,
) -> sinter.TaskStats:
    """Create one raw backend-specific statistic for plot aggregation tests.

    :param strong_id: Version- and backend-specific raw task identity.
    :param metadata: Raw SymFT metadata attached to the statistic.
    :param decoder: Decoder identity used in the aggregation key.
    :param shots: Attempted shots represented by the statistic.
    :param errors: Logical errors represented by the statistic.
    :param discards: Detector-rejected shots represented by the statistic.
    :return: Raw Sinter statistic with one resume-only stream counter.
    """
    return sinter.TaskStats(
        strong_id=strong_id,
        decoder=decoder,
        json_metadata=metadata,
        shots=shots,
        errors=errors,
        discards=discards,
        seconds=0.125,
        custom_counts=collections.Counter({f"{STREAM_COUNT_PREFIX}0": 1}),
    )


def test_s_reference_generates_exact_existing_t_and_s_circuits() -> None:
    """The S source swaps back to the exact previously sampled T circuits."""
    source_text = REFERENCE_PATH.read_text(encoding="utf-8")

    rows = validate_all_variants(source_text)

    assert len(rows) == 12
    assert sha256_text(source_text) == (
        "5003995c48830329a84e50560c49c1433b9dd8d23c06266a74272b6fce2d5afa"
    )
    assert not re.search(r"(?m)^T(?:_DAG)? ", source_text)
    assert re.search(r"(?m)^S(?:_DAG)? ", source_text)
    t_text = make_variant_text(source_text, 0.001, "T")
    assert sha256_text(t_text) == (
        "47ba7bb81fb84a043d2ecf9de3598dca41faf67f2d54a078399dc8dff6416f0d"
    )
    for noise_level in DEFAULT_NOISE_LEVELS:
        assert re.search(
            r"(?m)^T(?:_DAG)? ",
            make_variant_text(source_text, noise_level, "T"),
        )
        assert not re.search(
            r"(?m)^T(?:_DAG)? ",
            make_variant_text(source_text, noise_level, "S"),
        )


def test_tasks_use_s_proxies_and_identify_actual_variants() -> None:
    """Every task is Stim-compatible while T and S retain distinct IDs."""
    source_text = REFERENCE_PATH.read_text(encoding="utf-8")

    tasks = build_tasks(source_text)

    assert len(tasks) == 12
    assert len({task.strong_id() for task in tasks}) == 12
    for task in tasks:
        proxy_text = str(task.circuit)
        assert "T_DAG" not in proxy_text
        assert not re.search(r"(?m)^T ", proxy_text)
        actual_text = make_variant_text(
            source_text,
            float(task.json_metadata["noise_level"]),
            str(task.json_metadata["variant"]),
        )
        assert task.json_metadata["circuit_name"] == DEFAULT_CIRCUIT_NAME
        assert task.json_metadata["circuit_sha256"] == sha256_text(actual_text)
        assert type(task.detector_error_model) is stim.DetectorErrorModel


def test_custom_noise_levels_control_validation_and_task_order() -> None:
    """Custom levels produce one T/S pair per unique value in stable order."""
    source_text = REFERENCE_PATH.read_text(encoding="utf-8")

    rows = validate_all_variants(source_text, (0.004, 0.001, 0.004))
    tasks = build_tasks(
        source_text,
        noise_levels=(0.004, 0.001, 0.004),
    )

    assert [
        (row["noise_level"], row["variant"])
        for row in rows
    ] == [
        (0.001, "T"),
        (0.001, "S"),
        (0.004, "T"),
        (0.004, "S"),
    ]
    assert [
        (task.json_metadata["noise_level"], task.json_metadata["variant"])
        for task in tasks
    ] == [
        (0.004, "T"),
        (0.004, "S"),
        (0.001, "T"),
        (0.001, "S"),
    ]


def test_inspection_accepts_variable_circuit_shapes() -> None:
    """Reference inspection reports, but does not fix, all circuit counts."""
    compact = """\
QUBIT_COORDS(0, 0) 0
M 0
OBSERVABLE_INCLUDE(0) rec[-1]
"""
    expanded = """\
QUBIT_COORDS(0, 0) 0
QUBIT_COORDS(1, 0) 1
QUBIT_COORDS(2, 0) 2
M 0 1 2
DETECTOR rec[-1]
DETECTOR rec[-2]
OBSERVABLE_INCLUDE(0) rec[-1]
OBSERVABLE_INCLUDE(1) rec[-2]
"""

    assert inspect_circuit(compact) == {
        "num_qubits": 1,
        "num_measurements": 1,
        "num_detectors": 0,
        "num_observables": 1,
    }
    assert inspect_circuit(expanded) == {
        "num_qubits": 3,
        "num_measurements": 3,
        "num_detectors": 2,
        "num_observables": 2,
    }
    compact_tasks = build_tasks(compact, "compact")
    expanded_tasks = build_tasks(expanded, "expanded")
    assert len(compact_tasks) == 12
    assert len(expanded_tasks) == 12
    compact_task, *_ = compact_tasks
    expanded_task, *_ = expanded_tasks
    assert type(compact_task.circuit) is stim.Circuit
    assert type(expanded_task.circuit) is stim.Circuit
    assert compact_task.circuit.num_qubits == 1
    assert compact_task.circuit.num_detectors == 0
    assert compact_task.circuit.num_observables == 1
    assert expanded_task.circuit.num_qubits == 3
    assert expanded_task.circuit.num_detectors == 2
    assert expanded_task.circuit.num_observables == 2


def test_inspection_requires_observable_zero() -> None:
    """A reference without observable zero is rejected at inspection."""
    circuit_text = """\
M 0
OBSERVABLE_INCLUDE(1) rec[-1]
"""

    with pytest.raises(RuntimeError, match="does not define observable 0"):
        inspect_circuit(circuit_text)


def test_cuda_tasks_have_distinct_sampler_metadata() -> None:
    """CUDA is opt-in and creates task IDs distinct from CPU sampling."""
    source_text = REFERENCE_PATH.read_text(encoding="utf-8")

    cpu_task = build_tasks(source_text)[0]
    cuda_task = build_tasks(source_text, cuda=True)[0]

    assert cpu_task.strong_id() != cuda_task.strong_id()
    assert sampler_settings() == cpu_task.json_metadata["sampler"]
    assert cuda_task.json_metadata["sampler"] == {
        "batch": True,
        "observable": 0,
        "postselect_detectors": True,
        "threads": 1,
        "batch_size": 0,
        "sample_chunk_shots": 0,
        "cuda": True,
        "cuda_mode": "gpu",
        "shots_per_launch": 0,
        "threads_per_block": 0,
    }


def test_plot_aggregation_pools_cpu_and_cuda_preserving_provenance() -> None:
    """Equivalent CPU and CUDA rows become one plotting point, not resume data."""
    source_text = REFERENCE_PATH.read_text(encoding="utf-8")
    cpu_metadata = json.loads(json.dumps(build_tasks(source_text)[0].json_metadata))
    cuda_metadata = json.loads(
        json.dumps(build_tasks(source_text, cuda=True)[0].json_metadata)
    )
    cpu_metadata["symft_version"] = "0.1.1"
    cuda_metadata["symft_version"] = "0.1.0"
    cpu_stat = _plot_source_stat("cpu", cpu_metadata, shots=100, errors=3)
    cuda_stat = _plot_source_stat("cuda", cuda_metadata, shots=250, errors=9)

    pooled = aggregate_stats_for_plot([cpu_stat, cuda_stat])

    assert len(pooled) == 1
    stat = pooled[0]
    assert stat.strong_id.startswith("plot-")
    assert stat.shots == 350
    assert stat.errors == 12
    assert stat.discards == 40
    assert stat.seconds == 0.25
    assert stat.custom_counts == collections.Counter()
    assert stat.json_metadata[PLOT_ONLY_AGGREGATE_KEY] is True
    provenance = stat.json_metadata[PLOT_PROVENANCE_KEY]
    assert provenance["symft_versions"] == ["0.1.0", "0.1.1"]
    assert {
        json.dumps(configuration, sort_keys=True)
        for configuration in provenance["sampler_configurations"]
    } == {
        json.dumps(cpu_metadata["sampler"], sort_keys=True),
        json.dumps(cuda_metadata["sampler"], sort_keys=True),
    }


def test_plot_aggregation_keeps_physical_or_decoder_differences_separate() -> None:
    """A changed physical field or decoder always produces a separate point."""
    source_text = REFERENCE_PATH.read_text(encoding="utf-8")
    baseline = json.loads(json.dumps(build_tasks(source_text)[0].json_metadata))
    cases = [
        ("circuit_sha256", "different-circuit", None),
        ("noise_level", 0.123, None),
        ("variant", "different-variant", None),
        ("sampler.observable", 1, None),
        ("sampler.postselect_detectors", False, None),
        (None, None, "another-decoder"),
    ]
    for index, (field, value, decoder) in enumerate(cases):
        changed = json.loads(json.dumps(baseline))
        if field is not None:
            container = changed
            *parents, leaf = field.split(".")
            for parent in parents:
                container = container[parent]
            container[leaf] = value
        pooled = aggregate_stats_for_plot(
            [
                _plot_source_stat("baseline", baseline),
                _plot_source_stat(
                    f"changed-{index}",
                    changed,
                    decoder="symft_counts" if decoder is None else decoder,
                ),
            ]
        )
        assert len(pooled) == 2


def test_read_plot_stats_leaves_raw_csv_unchanged_and_cannot_resume(
    tmp_path: Path,
) -> None:
    """Derived plotting rows do not alter raw storage and resume is rejected."""
    source_text = REFERENCE_PATH.read_text(encoding="utf-8")
    raw_stat = _plot_source_stat(
        "raw",
        json.loads(json.dumps(build_tasks(source_text)[0].json_metadata)),
    )
    stats_path = tmp_path / STATS_FILENAME
    stats_path.write_text(
        sinter.CSV_HEADER + "\n" + raw_stat.to_csv_line(),
        encoding="utf-8",
    )
    before = stats_path.read_text(encoding="utf-8")

    pooled = read_plot_stats(stats_path)

    assert stats_path.read_text(encoding="utf-8") == before
    with pytest.raises(RuntimeError, match="plot-only aggregated"):
        validate_resume_stats(pooled, build_tasks(source_text))


def test_circuit_names_separate_otherwise_identical_sinter_tasks() -> None:
    """Human-readable circuit names participate in Sinter strong IDs."""
    source_text = REFERENCE_PATH.read_text(encoding="utf-8")

    alpha_tasks = build_tasks(source_text, "alpha")
    beta_tasks = build_tasks(source_text, "beta")

    assert {task.strong_id() for task in alpha_tasks}.isdisjoint(
        {task.strong_id() for task in beta_tasks}
    )
    assert {
        task.json_metadata["circuit_name"] for task in alpha_tasks
    } == {"alpha"}
    assert {
        task.json_metadata["circuit_name"] for task in beta_tasks
    } == {"beta"}


def test_compiled_adapter_maps_counts_caps_calls_and_advances_streams() -> None:
    """The adapter returns Sinter semantics and one new stream per capped call."""
    fake = FakeSymftCountsSampler()
    adapter = CompiledSymftSinterSampler(
        sampler=fake,
        sampler_info={
            "backend": "batch",
            "threads": 8,
            "sample_chunk_shots": 2,
        },
        call_shots=10,
        stream_sequence=_StreamSequence(7),
    )

    first = adapter.sample(25)
    second = adapter.sample(4)

    assert fake.calls == [(10, 7), (4, 8)]
    assert first.shots == 10
    assert first.errors == 1
    assert first.discards == 2
    assert first.seconds == 0.25
    assert first.custom_counts[f"{STREAM_COUNT_PREFIX}7"] == 1
    assert first.custom_counts[f"{ACTIVE_THREADS_PREFIX}5"] == 1
    assert second.shots == 4


def test_compiled_adapter_rejects_a_thread_count_mismatch() -> None:
    """A SymFT result using fewer threads than expected stops collection."""
    adapter = CompiledSymftSinterSampler(
        sampler=FakeSymftCountsSampler(active_threads=4),
        sampler_info={
            "backend": "batch",
            "threads": 8,
            "sample_chunk_shots": 2,
        },
        call_shots=10,
        stream_sequence=_StreamSequence(0),
    )

    with pytest.raises(RuntimeError, match="used 4 threads; expected 5"):
        adapter.sample(10)


def test_compiled_adapter_accepts_cuda_single_host_worker() -> None:
    """CUDA results report one host worker regardless of requested shot count."""
    adapter = CompiledSymftSinterSampler(
        sampler=FakeSymftCountsSampler(active_threads=1),
        sampler_info={
            "backend": "cuda",
            "threads": 1,
            "sample_chunk_shots": 1_048_576,
        },
        call_shots=10,
        stream_sequence=_StreamSequence(0),
    )

    stat = adapter.sample(10)

    assert stat.custom_counts[f"{ACTIVE_THREADS_PREFIX}1"] == 1


def test_compiled_adapter_rejects_cuda_host_worker_mismatch() -> None:
    """CUDA collection stops if SymFT does not report its one host worker."""
    adapter = CompiledSymftSinterSampler(
        sampler=FakeSymftCountsSampler(active_threads=2),
        sampler_info={
            "backend": "cuda",
            "threads": 1,
            "sample_chunk_shots": 1_048_576,
        },
        call_shots=10,
        stream_sequence=_StreamSequence(0),
    )

    with pytest.raises(RuntimeError, match="used 2 threads; expected 1"):
        adapter.sample(10)


def test_recompiled_adapters_share_the_task_stream_sequence() -> None:
    """Recompiling one Sinter task cannot repeat an in-process stream ID."""
    sequence = _StreamSequence(12)
    first_fake = FakeSymftCountsSampler()
    second_fake = FakeSymftCountsSampler()
    first = CompiledSymftSinterSampler(
        sampler=first_fake,
        sampler_info={
            "backend": "batch",
            "threads": 8,
            "sample_chunk_shots": 2,
        },
        call_shots=10,
        stream_sequence=sequence,
    )
    second = CompiledSymftSinterSampler(
        sampler=second_fake,
        sampler_info={
            "backend": "batch",
            "threads": 8,
            "sample_chunk_shots": 2,
        },
        call_shots=10,
        stream_sequence=sequence,
    )

    first.sample(4)
    second.sample(4)

    assert first_fake.calls == [(4, 12)]
    assert second_fake.calls == [(4, 13)]


def test_resume_validation_and_next_stream_use_custom_counts() -> None:
    """Persisted stream keys select the next unused per-task stream."""
    source_text = REFERENCE_PATH.read_text(encoding="utf-8")
    tasks = build_tasks(source_text)
    task = tasks[0]
    stat = _task_stat(
        task,
        shots=20,
        errors=2,
        discards=3,
        stream_ids=(0, 1, 4),
    )

    validate_resume_stats([stat], tasks)
    starts = next_stream_ids([stat], tasks)

    assert starts[task.strong_id()] == 5
    assert all(
        value == 0
        for strong_id, value in starts.items()
        if strong_id != task.strong_id()
    )


def test_resume_allows_other_names_and_rejects_reused_names() -> None:
    """Shared CSV rows are isolated by name and same-name drift is rejected."""
    source_text = REFERENCE_PATH.read_text(encoding="utf-8")
    alpha_tasks = build_tasks(source_text, "alpha")
    beta_tasks = build_tasks(source_text, "beta")
    alpha_stat = _task_stat(
        alpha_tasks[0],
        shots=10,
        errors=1,
        discards=2,
        stream_ids=(0,),
    )
    beta_stat = _task_stat(
        beta_tasks[0],
        shots=10,
        errors=1,
        discards=2,
        stream_ids=(0,),
    )

    validate_resume_stats([alpha_stat, beta_stat], beta_tasks)

    changed_alpha_tasks = build_tasks(source_text + "\n", "alpha")
    with pytest.raises(RuntimeError, match="already identifies different"):
        validate_resume_stats([alpha_stat], changed_alpha_tasks)

    conflicting_stat = sinter.TaskStats(
        strong_id="different-framework-id",
        decoder="different-decoder",
        json_metadata={"circuit_name": "beta", "experiment": "other"},
        shots=10,
        errors=1,
        discards=0,
        seconds=0.001,
    )
    with pytest.raises(RuntimeError, match="already identifies different"):
        validate_resume_stats([conflicting_stat], beta_tasks)


def test_resume_accepts_compatible_unselected_noise_levels() -> None:
    """Historical same-circuit points remain valid outside the requested subset."""
    source_text = REFERENCE_PATH.read_text(encoding="utf-8")
    historical_task = build_tasks(
        source_text,
        "alpha",
        noise_levels=(0.001,),
    )[0]
    historical_stat = _task_stat(
        historical_task,
        shots=10,
        errors=1,
        discards=2,
        stream_ids=(0,),
    )

    resume_levels = _resume_noise_levels(
        [historical_stat],
        "alpha",
        (0.004,),
    )
    resume_tasks = build_tasks(source_text, "alpha", resume_levels)

    assert resume_levels == (0.001, 0.004)
    validate_resume_stats([historical_stat], resume_tasks)

    changed_tasks = build_tasks(source_text + "\n", "alpha", resume_levels)
    with pytest.raises(RuntimeError, match="already identifies different"):
        validate_resume_stats([historical_stat], changed_tasks)


def test_resume_rejects_pre_circuit_name_statistics() -> None:
    """Schema-one experiment rows must be converted before shared resume."""
    source_text = REFERENCE_PATH.read_text(encoding="utf-8")
    task = build_tasks(source_text)[0]
    metadata = dict(task.json_metadata)
    del metadata["circuit_name"]
    metadata["schema_version"] = 1
    old_stat = sinter.TaskStats(
        strong_id="pre-schema-id",
        decoder=str(task.decoder),
        json_metadata=metadata,
        shots=10,
        errors=1,
        discards=2,
        seconds=0.001,
        custom_counts=collections.Counter(
            {f"{STREAM_COUNT_PREFIX}0": 1}
        ),
    )

    with pytest.raises(RuntimeError, match="predate circuit_name"):
        validate_resume_stats([old_stat], build_tasks(source_text))


def test_sinter_collect_resumes_without_repeating_streams(tmp_path: Path) -> None:
    """A complete rerun adds nothing, while a raised cap starts at stream two.

    :param tmp_path: Temporary directory supplied by pytest.
    """
    task = _fixed_task()
    stats_path = tmp_path / STATS_FILENAME

    first = sinter.collect(
        num_workers=1,
        tasks=[task],
        save_resume_filepath=stats_path,
        max_shots=20,
        custom_decoders={"fixed": FixedSampler(0)},
    )
    first_text = stats_path.read_text(encoding="utf-8")
    second = sinter.collect(
        num_workers=1,
        tasks=[task],
        save_resume_filepath=stats_path,
        max_shots=20,
        custom_decoders={"fixed": FixedSampler(2)},
    )
    second_text = stats_path.read_text(encoding="utf-8")
    third = sinter.collect(
        num_workers=1,
        tasks=[task],
        save_resume_filepath=stats_path,
        max_shots=30,
        custom_decoders={"fixed": FixedSampler(2)},
    )

    assert first[0].shots == 20
    assert second[0].shots == 20
    assert second_text == first_text
    assert stats_path.read_text(encoding="utf-8") != second_text
    assert third[0].shots == 30
    assert third[0].custom_counts[f"{STREAM_COUNT_PREFIX}0"] == 1
    assert third[0].custom_counts[f"{STREAM_COUNT_PREFIX}1"] == 1
    assert third[0].custom_counts[f"{STREAM_COUNT_PREFIX}2"] == 1


def test_sinter_collect_keeps_shared_tasks_independent(tmp_path: Path) -> None:
    """Two named tasks stop and resume independently in one Sinter CSV.

    :param tmp_path: Temporary directory supplied by pytest.
    """
    stats_path = tmp_path / STATS_FILENAME
    alpha = _fixed_task("alpha")
    beta = _fixed_task("beta")

    sinter.collect(
        num_workers=1,
        tasks=[alpha],
        save_resume_filepath=stats_path,
        max_shots=20,
        custom_decoders={"fixed": FixedSampler(0)},
    )
    sinter.collect(
        num_workers=1,
        tasks=[beta],
        save_resume_filepath=stats_path,
        max_shots=10,
        custom_decoders={"fixed": FixedSampler(0)},
    )
    sinter.collect(
        num_workers=1,
        tasks=[alpha],
        save_resume_filepath=stats_path,
        max_shots=30,
        custom_decoders={"fixed": FixedSampler(2)},
    )
    stats = {
        stat.json_metadata["case"]: stat
        for stat in sinter.read_stats_from_csv_files(stats_path)
    }

    assert stats["alpha"].shots == 30
    assert stats["beta"].shots == 10
    assert stats["alpha"].custom_counts == collections.Counter(
        {
            f"{STREAM_COUNT_PREFIX}0": 1,
            f"{STREAM_COUNT_PREFIX}1": 1,
            f"{STREAM_COUNT_PREFIX}2": 1,
        }
    )
    assert stats["beta"].custom_counts == collections.Counter(
        {f"{STREAM_COUNT_PREFIX}0": 1}
    )


def test_cli_defaults_and_overrides_reference_identity(tmp_path: Path) -> None:
    """Every command accepts a reference and derives or overrides its name.

    :param tmp_path: Temporary directory supplied by pytest.
    """
    parser = build_parser()
    custom_reference = tmp_path / "other-reference.stim"
    stats_path = tmp_path / STATS_FILENAME

    validate_args = parser.parse_args(["validate"])
    smoke_args = parser.parse_args(
        [
            "smoke",
            "--reference",
            str(custom_reference),
            "--circuit-name",
            "published-label",
        ]
    )
    run_args = parser.parse_args(
        [
            "run",
            "--reference",
            str(custom_reference),
            "--stats",
            str(stats_path),
            "--noise-levels",
            "0.001",
            "0.004",
            "--cuda",
        ]
    )

    assert validate_args.reference == REFERENCE_PATH
    assert validate_args.noise_levels == DEFAULT_NOISE_LEVELS
    assert resolve_circuit_name(
        validate_args.reference,
        validate_args.circuit_name,
    ) == DEFAULT_CIRCUIT_NAME
    assert resolve_circuit_name(
        smoke_args.reference,
        smoke_args.circuit_name,
    ) == "published-label"
    assert resolve_circuit_name(
        run_args.reference,
        run_args.circuit_name,
    ) == "other-reference"
    assert run_args.stats == stats_path
    assert run_args.noise_levels == [0.001, 0.004]
    assert smoke_args.cuda is False
    assert run_args.cuda is True


def test_run_requires_a_stats_path() -> None:
    """Production sampling cannot silently select a per-reference CSV."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run"])


def test_smoke_sample_uses_sinter_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke sampling displays Sinter progress and returns no summary rows.

    :param monkeypatch: Pytest fixture used to replace collection.
    """
    collect = Mock(return_value=[])
    monkeypatch.setattr(sinter, "collect", collect)

    result = smoke_sample(REFERENCE_PATH, 100, "smoke")

    assert result is None
    assert collect.call_args.kwargs["print_progress"] is True
    assert collect.call_args.kwargs["max_shots"] == 100
    assert len(collect.call_args.kwargs["tasks"]) == 2
