"""Tests for the Sinter-backed SymFT cultivation framework."""

from __future__ import annotations

import collections
import json
import math
import re
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import sinter
import stim
import symft

from cliffordea.sim import msc_framework
from cliffordea.sim.msc_framework import (
    DEFAULT_CIRCUIT_NAME,
    DEFAULT_NOISE_LEVELS,
    PLOT_ONLY_AGGREGATE_KEY,
    PLOT_PROVENANCE_KEY,
    REFERENCE_PATH,
    STATS_FILENAME,
    TASK_SCHEMA_VERSION,
    SymftSinterSampler,
    CompiledSymftSinterSampler,
    _StreamSequence,
    _resume_noise_levels,
    aggregate_stats_for_plot,
    build_tasks,
    inspect_circuit,
    make_variant_text,
    read_plot_stats,
    sampler_settings,
    smoke_sample,
    validate_all_variants,
    validate_resume_stats,
)
from cliffordea.sim.run_simulation import (
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
    """Produce fixed-size batches with no custom result counts."""

    def handles_throttling(self) -> bool:
        """Disable Sinter's adaptive throttling for deterministic tests.

        :param self: Fixed compiled sampler.
        :return: True because the fake controls its own batch size.
        """
        return True

    def sample(self, suggested_shots: int) -> sinter.AnonTaskStats:
        """Return one fixed-size batch for Sinter resume tests.

        :param self: Fixed compiled sampler.
        :param suggested_shots: Maximum shots requested by Sinter.
        :return: Deterministic anonymous task statistics.
        """
        shots = min(suggested_shots, 10)
        return sinter.AnonTaskStats(shots=shots, errors=1)


class FixedSampler(sinter.Sampler):
    """Compile deterministic samplers for Sinter resume tests."""

    def compiled_sampler_for_task(
        self,
        task: sinter.Task,
    ) -> sinter.CompiledSampler:
        """Create a deterministic sampler for a resume test task.

        :param self: Fixed sampler factory.
        :param task: Sinter task being compiled.
        :return: Deterministic compiled sampler.
        """
        return FixedCompiledSampler()


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
) -> sinter.TaskStats:
    """Create one persisted statistic matching a task.

    :param task: Task supplying identity and metadata.
    :param shots: Attempted shot count.
    :param errors: Logical-error count.
    :param discards: Detector-rejected shot count.
    :return: Sinter statistic compatible with resume validation.
    """
    return sinter.TaskStats(
        strong_id=task.strong_id(),
        decoder=str(task.decoder),
        json_metadata=task.json_metadata,
        shots=shots,
        errors=errors,
        discards=discards,
        seconds=0.001,
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
    :return: Raw Sinter statistic with an empty custom-count field.
    """
    return sinter.TaskStats(
        strong_id=strong_id,
        decoder=decoder,
        json_metadata=metadata,
        shots=shots,
        errors=errors,
        discards=discards,
        seconds=0.125,
    )


def _change_reference_circuit(reference_text: str) -> str:
    """Change one physical error probability in a reference circuit.

    :param reference_text: Original noisy cultivation circuit text.
    :return: Circuit text representing a distinct sampling distribution.
    """
    return reference_text.replace("X_ERROR(0.001)", "X_ERROR(0.002)", 1)


def test_s_reference_generates_exact_existing_t_and_s_circuits() -> None:
    """The S source swaps back to the exact previously sampled T circuits."""
    source_text = REFERENCE_PATH.read_text(encoding="utf-8")

    rows = validate_all_variants(source_text)

    assert len(rows) == 12
    assert all("circuit_sha256" not in row for row in rows)
    assert not re.search(r"(?m)^T(?:_DAG)? ", source_text)
    assert re.search(r"(?m)^S(?:_DAG)? ", source_text)
    for noise_level in DEFAULT_NOISE_LEVELS:
        assert re.search(
            r"(?m)^T(?:_DAG)? ",
            make_variant_text(source_text, noise_level, "T"),
        )
        assert not re.search(
            r"(?m)^T(?:_DAG)? ",
            make_variant_text(source_text, noise_level, "S"),
        )


@pytest.mark.parametrize(
    "filename",
    (
        "d3a6_inject+cultivate_p1e-3.stim",
        "d3a6f2_inject+cultivate_p1e-3.stim",
        "d5a19_inject+cultivate_p1e-3.stim",
        "d5a19f13_inject+cultivate_p1e-3.stim",
        "d5a19r4_inject+cultivate_p1e-3.stim",
        "d5a19r4f13_inject+cultivate_p1e-3.stim",
    ),
)
@pytest.mark.parametrize("variant", ("T", "S"))
def test_noiseless_t_variant_detectors_are_deterministic(
    filename: str,
    variant: str,
) -> None:
    """All noiseless T-state shots must agree on every raw detector parity."""
    reference_path = REFERENCE_PATH.with_name(filename)
    t_circuit_text = make_variant_text(
        reference_text=reference_path.read_text(encoding="utf-8"),
        noise_level=0.0,
        variant=variant,
    )

    detector_samples = symft.Circuit(t_circuit_text).sample_detectors(
        shots=256,
        seed=123,
    )
    detection_events = detector_samples ^ detector_samples[0]
    varying_detector_indices = np.flatnonzero(
        np.any(detection_events, axis=0)
    ).tolist()

    assert varying_detector_indices == []


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
        noise_level = float(task.json_metadata["noise_level"])
        variant = str(task.json_metadata["variant"])
        assert task.json_metadata == {
            "schema_version": TASK_SCHEMA_VERSION,
            "decoder_version": symft.__version__,
            "noise_level": noise_level,
            "variant": variant,
            "circuit_name": DEFAULT_CIRCUIT_NAME,
            "sampler": sampler_settings(),
        }
        assert task.json_metadata["circuit_name"] == DEFAULT_CIRCUIT_NAME
        circuit = task.circuit
        assert type(circuit) is stim.Circuit
        assert task.detector_error_model == circuit.detector_error_model(
            decompose_errors=False,
            approximate_disjoint_errors=True,
        )


def test_distance_five_tasks_allow_nongraphlike_detector_errors() -> None:
    """D5 tasks retain undecomposed hyperedges instead of failing construction."""
    reference_path = REFERENCE_PATH.with_name(
        "d5a19_inject+cultivate_p1e-3.stim"
    )

    tasks = build_tasks(
        reference_path.read_text(encoding="utf-8"),
        reference_path.stem,
        noise_levels=(0.002,),
    )

    assert len(tasks) == 2
    for task in tasks:
        dem = task.detector_error_model
        assert type(dem) is stim.DetectorErrorModel
        assert dem.num_detectors == 107
        assert dem.num_observables == 1


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


def test_plot_aggregation_ignores_nonidentity_metadata() -> None:
    """Only the declared identity fields split points; provenance stays available."""
    source_text = REFERENCE_PATH.read_text(encoding="utf-8")
    cpu_metadata = json.loads(json.dumps(build_tasks(source_text)[0].json_metadata))
    cuda_metadata = json.loads(
        json.dumps(build_tasks(source_text, cuda=True)[0].json_metadata)
    )
    cpu_metadata["decoder_version"] = "0.1.1"
    cuda_metadata["decoder_version"] = "0.1.0"
    cuda_metadata["sampler"]["batch"] = not cpu_metadata["sampler"]["batch"]
    cuda_metadata["sampler"]["observable"] = 1
    cuda_metadata["sampler"]["postselect_detectors"] = False
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
    assert provenance["decoder_versions"] == ["0.1.0", "0.1.1"]
    assert {
        json.dumps(configuration, sort_keys=True)
        for configuration in provenance["sampler_configurations"]
    } == {
        json.dumps(cpu_metadata["sampler"], sort_keys=True),
        json.dumps(cuda_metadata["sampler"], sort_keys=True),
    }


def test_plot_aggregation_keeps_identity_or_decoder_differences_separate() -> None:
    """Every variable metadata identity field and the decoder split points."""
    source_text = REFERENCE_PATH.read_text(encoding="utf-8")
    baseline = json.loads(json.dumps(build_tasks(source_text)[0].json_metadata))
    cases = [
        ("circuit_name", "different-name", None),
        ("noise_level", 0.123, None),
        ("variant", "different-variant", None),
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


def test_read_plot_stats_pools_csvs_without_changing_them_and_cannot_resume(
    tmp_path: Path,
) -> None:
    """Separate CSVs pool without altering storage, and cannot become resume data."""
    source_text = REFERENCE_PATH.read_text(encoding="utf-8")
    metadata = json.loads(json.dumps(build_tasks(source_text)[0].json_metadata))
    raw_stat = _plot_source_stat(
        "raw",
        metadata,
    )
    other_metadata = json.loads(json.dumps(metadata))
    other_metadata["decoder_version"] = "another-version"
    other_stat = _plot_source_stat(
        "other",
        other_metadata,
        shots=250,
        errors=9,
    )
    stats_path = tmp_path / STATS_FILENAME
    other_stats_path = tmp_path / "other-stats.csv"
    stats_path.write_text(
        sinter.CSV_HEADER + "\n" + raw_stat.to_csv_line(),
        encoding="utf-8",
    )
    other_stats_path.write_text(
        sinter.CSV_HEADER + "\n" + other_stat.to_csv_line(),
        encoding="utf-8",
    )
    before = stats_path.read_text(encoding="utf-8")
    other_before = other_stats_path.read_text(encoding="utf-8")

    pooled = read_plot_stats(stats_path, other_stats_path)

    assert stats_path.read_text(encoding="utf-8") == before
    assert other_stats_path.read_text(encoding="utf-8") == other_before
    assert len(pooled) == 1
    assert pooled[0].shots == 350
    assert pooled[0].errors == 12
    with pytest.raises(RuntimeError, match="plot-only aggregated"):
        validate_resume_stats(pooled, build_tasks(source_text))


@pytest.mark.parametrize("schema_version", [1, 2, 3])
def test_plot_aggregation_rejects_obsolete_schemas(schema_version: int) -> None:
    """Plot aggregation accepts schema four only, without legacy fallbacks."""
    source_text = REFERENCE_PATH.read_text(encoding="utf-8")
    metadata = dict(build_tasks(source_text)[0].json_metadata)
    metadata["schema_version"] = schema_version

    with pytest.raises(RuntimeError, match="requires schema_version=4"):
        aggregate_stats_for_plot([_plot_source_stat("old", metadata)])


def test_plot_aggregation_requires_decoder_version() -> None:
    """Schema-four plotting rejects rows missing the decoder version field."""
    source_text = REFERENCE_PATH.read_text(encoding="utf-8")
    metadata = dict(build_tasks(source_text)[0].json_metadata)
    del metadata["decoder_version"]

    with pytest.raises(RuntimeError, match="missing 'decoder_version'"):
        aggregate_stats_for_plot([_plot_source_stat("missing", metadata)])


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
    """The adapter advances streams while emitting no custom result counts."""
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
    assert first.custom_counts == collections.Counter()
    assert second.shots == 4
    assert second.custom_counts == collections.Counter()


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

    assert stat.custom_counts == collections.Counter()


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


def test_factory_seeds_each_task_and_shares_recompiled_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each task gets entropy once and keeps its sequence across recompiles.

    :param monkeypatch: Pytest fixture replacing entropy and SymFT compilation.
    """
    source_text = REFERENCE_PATH.read_text(encoding="utf-8")
    tasks = build_tasks(source_text, noise_levels=(0.001,))
    first_fake = FakeSymftCountsSampler()
    second_fake = FakeSymftCountsSampler()
    recompiled_fake = FakeSymftCountsSampler()
    sampler_info = {
        "backend": "batch",
        "threads": 8,
        "sample_chunk_shots": 2,
    }
    compile_sampler = Mock(
        side_effect=[
            (first_fake, sampler_info),
            (second_fake, sampler_info),
            (recompiled_fake, sampler_info),
        ]
    )
    randbits = Mock(side_effect=[12, 42])
    monkeypatch.setattr(msc_framework, "compile_counts_sampler", compile_sampler)
    monkeypatch.setattr(msc_framework.secrets, "randbits", randbits)
    factory = SymftSinterSampler(source_text, call_shots=10)

    factory.compiled_sampler_for_task(tasks[0]).sample(4)
    factory.compiled_sampler_for_task(tasks[1]).sample(4)
    factory.compiled_sampler_for_task(tasks[0]).sample(4)

    assert first_fake.calls == [(4, 12)]
    assert second_fake.calls == [(4, 42)]
    assert recompiled_fake.calls == [(4, 13)]
    assert [item.args for item in randbits.call_args_list] == [(64,), (64,)]


def test_stream_sequence_uses_entropy_and_wraps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default streams use 64-bit entropy and explicit maximum seeds wrap."""
    randbits = Mock(return_value=123)
    monkeypatch.setattr(msc_framework.secrets, "randbits", randbits)

    entropy_sequence = _StreamSequence(None)
    wrapping_sequence = _StreamSequence(2**64 - 1)

    assert entropy_sequence.take() == 123
    assert entropy_sequence.take() == 124
    assert wrapping_sequence.take() == 2**64 - 1
    assert wrapping_sequence.take() == 0
    randbits.assert_called_once_with(64)


@pytest.mark.parametrize("seed", [-1, 2**64])
def test_programmatic_sampler_rejects_invalid_seed(seed: int) -> None:
    """Direct Python callers cannot select a stream outside 64-bit range.

    :param seed: Invalid boundary value supplied to the sampler factory.
    """
    source_text = REFERENCE_PATH.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match=r"range\(2\*\*64\)"):
        SymftSinterSampler(source_text, call_shots=10, seed=seed)


def test_resume_validation_accepts_empty_custom_counts() -> None:
    """Schema-four resume rows need no persisted RNG or thread counters."""
    source_text = REFERENCE_PATH.read_text(encoding="utf-8")
    tasks = build_tasks(source_text)
    task = tasks[0]
    stat = _task_stat(
        task,
        shots=20,
        errors=2,
        discards=3,
    )

    validate_resume_stats([stat], tasks)
    assert stat.custom_counts == collections.Counter()


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
    )
    beta_stat = _task_stat(
        beta_tasks[0],
        shots=10,
        errors=1,
        discards=2,
    )

    validate_resume_stats([alpha_stat, beta_stat], beta_tasks)

    changed_alpha_tasks = build_tasks(
        _change_reference_circuit(source_text),
        "alpha",
    )
    with pytest.raises(RuntimeError, match="already identifies different"):
        validate_resume_stats([alpha_stat], changed_alpha_tasks)

    conflicting_stat = sinter.TaskStats(
        strong_id="different-framework-id",
        decoder="different-decoder",
        json_metadata={
            "schema_version": TASK_SCHEMA_VERSION,
            "circuit_name": "beta",
        },
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
    )

    resume_levels = _resume_noise_levels(
        [historical_stat],
        "alpha",
        (0.004,),
    )
    resume_tasks = build_tasks(source_text, "alpha", resume_levels)

    assert resume_levels == (0.001, 0.004)
    validate_resume_stats([historical_stat], resume_tasks)

    changed_tasks = build_tasks(
        _change_reference_circuit(source_text),
        "alpha",
        resume_levels,
    )
    with pytest.raises(RuntimeError, match="already identifies different"):
        validate_resume_stats([historical_stat], changed_tasks)


@pytest.mark.parametrize("schema_version", [1, 2, 3])
def test_resume_rejects_obsolete_schema(schema_version: int) -> None:
    """Collection rejects pre-schema-four statistics instead of migrating."""
    source_text = REFERENCE_PATH.read_text(encoding="utf-8")
    task = build_tasks(source_text)[0]
    metadata = dict(task.json_metadata)
    metadata["schema_version"] = schema_version
    old_stat = sinter.TaskStats(
        strong_id="obsolete-schema-id",
        decoder=str(task.decoder),
        json_metadata=metadata,
        shots=10,
        errors=1,
        discards=2,
        seconds=0.001,
    )

    with pytest.raises(RuntimeError, match="start a new CSV"):
        validate_resume_stats([old_stat], build_tasks(source_text))


def test_sinter_collect_resumes_counts_without_custom_state(tmp_path: Path) -> None:
    """A complete rerun adds nothing while a raised cap adds an empty-count row.

    :param tmp_path: Temporary directory supplied by pytest.
    """
    task = _fixed_task()
    stats_path = tmp_path / STATS_FILENAME

    first = sinter.collect(
        num_workers=1,
        tasks=[task],
        save_resume_filepath=stats_path,
        max_shots=20,
        custom_decoders={"fixed": FixedSampler()},
    )
    first_text = stats_path.read_text(encoding="utf-8")
    second = sinter.collect(
        num_workers=1,
        tasks=[task],
        save_resume_filepath=stats_path,
        max_shots=20,
        custom_decoders={"fixed": FixedSampler()},
    )
    second_text = stats_path.read_text(encoding="utf-8")
    third = sinter.collect(
        num_workers=1,
        tasks=[task],
        save_resume_filepath=stats_path,
        max_shots=30,
        custom_decoders={"fixed": FixedSampler()},
    )

    assert first[0].shots == 20
    assert second[0].shots == 20
    assert second_text == first_text
    assert stats_path.read_text(encoding="utf-8") != second_text
    assert third[0].shots == 30
    assert third[0].custom_counts == collections.Counter()
    assert all(
        line.endswith(",")
        for line in stats_path.read_text(encoding="utf-8").splitlines()[1:]
    )


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
        custom_decoders={"fixed": FixedSampler()},
    )
    sinter.collect(
        num_workers=1,
        tasks=[beta],
        save_resume_filepath=stats_path,
        max_shots=10,
        custom_decoders={"fixed": FixedSampler()},
    )
    sinter.collect(
        num_workers=1,
        tasks=[alpha],
        save_resume_filepath=stats_path,
        max_shots=30,
        custom_decoders={"fixed": FixedSampler()},
    )
    stats = {
        stat.json_metadata["case"]: stat
        for stat in sinter.read_stats_from_csv_files(stats_path)
    }

    assert stats["alpha"].shots == 30
    assert stats["beta"].shots == 10
    assert stats["alpha"].custom_counts == collections.Counter()
    assert stats["beta"].custom_counts == collections.Counter()


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
    seeded_smoke_args = parser.parse_args(["smoke", "--seed", "0"])
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
            "--seed",
            str(2**64 - 1),
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
    assert smoke_args.seed is None
    assert seeded_smoke_args.seed == 0
    assert run_args.cuda is True
    assert run_args.seed == 2**64 - 1

    for invalid_seed in ("-1", str(2**64)):
        with pytest.raises(SystemExit):
            parser.parse_args(["smoke", "--seed", invalid_seed])


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

    result = smoke_sample(REFERENCE_PATH, 100, "smoke", seed=123)

    assert result is None
    assert collect.call_args.kwargs["print_progress"] is True
    assert collect.call_args.kwargs["max_shots"] == 100
    assert len(collect.call_args.kwargs["tasks"]) == 2
    sampler = collect.call_args.kwargs["custom_decoders"]["symft_counts"]
    assert isinstance(sampler, SymftSinterSampler)
    assert sampler.seed == 123
