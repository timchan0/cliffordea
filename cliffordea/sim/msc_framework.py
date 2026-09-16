"""Sinter-backed machinery for the SymFT S-versus-T cultivation study."""

from __future__ import annotations

import collections
import hashlib
import json
import math
import secrets
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import sinter
import stim
import symft

from cliffordea.sim.circuit_text import (
    replace_noise_level,
    swap_t_and_s_gates,
)


BASE_DIR = Path(__file__).resolve().parent
REFERENCE_PATH = (
    BASE_DIR
    / "stim_files"
    / "d3a6_inject+cultivate_p1e-3.stim"
)
DEFAULT_CIRCUIT_NAME = REFERENCE_PATH.stem
DEFAULT_RESULTS_DIR = BASE_DIR / "results"
REFERENCE_NOISE_LEVEL = 0.001
DEFAULT_NOISE_LEVELS = (0.001, 0.002, 0.003, 0.005, 0.007, 0.01)
VARIANTS = ("T", "S")

SINTER_WORKERS = 1
THREADS = 8
BATCH_SIZE = 1024
SAMPLE_CHUNK_SHOTS = 0
# CPU is the portable default; ``--cuda`` selects GPU sampling per invocation.
DEFAULT_CUDA = False
CUDA_MODE = "gpu"
CUDA_SHOTS_PER_LAUNCH = 0
CUDA_THREADS_PER_BLOCK = 0

DEFAULT_TARGET_ERRORS = 100
DEFAULT_MAX_SHOTS = 1_000_000_000
DEFAULT_CALL_SHOTS = 10_000_000
DEFAULT_SMOKE_SHOTS = 100_000

DECODER_NAME = "symft_counts"
TASK_SCHEMA_VERSION = 4
STATS_FILENAME = "stats.csv"
STREAM_ID_MODULUS = 1 << 64
PLOT_ONLY_AGGREGATE_KEY = "plot_only_aggregate"
PLOT_PROVENANCE_KEY = "plot_provenance"


def _plot_identity_metadata(stat: sinter.TaskStats) -> dict[str, Any]:
    """Extract metadata that defines an aggregated plotting data point.

    :param stat: Raw Sinter statistic to normalize for plotting.
    :return: Canonical metadata used as the plotting aggregate identity.
    :raises RuntimeError: If the statistic lacks required cultivation metadata.
    """
    metadata = stat.json_metadata
    if not isinstance(metadata, dict):
        raise RuntimeError("plot aggregation requires dictionary JSON metadata")
    schema_version = metadata.get("schema_version")
    if schema_version != TASK_SCHEMA_VERSION:
        raise RuntimeError(
            "plot aggregation requires "
            f"schema_version={TASK_SCHEMA_VERSION}, got {schema_version!r}"
        )
    sampler = metadata.get("sampler")
    if not isinstance(sampler, dict):
        raise RuntimeError("plot aggregation requires sampler JSON metadata")
    if "decoder_version" not in metadata:
        raise RuntimeError(
            "plot aggregation metadata is missing 'decoder_version'"
        )
    try:
        return {
            "schema_version": metadata["schema_version"],
            "circuit_name": metadata["circuit_name"],
            "noise_level": metadata["noise_level"],
            "variant": metadata["variant"],
        }
    except KeyError as error:
        raise RuntimeError(
            f"plot aggregation metadata is missing {error.args[0]!r}"
        ) from error


def aggregate_stats_for_plot(
    stats: Iterable[sinter.TaskStats],
) -> list[sinter.TaskStats]:
    """Pool raw CPU/CUDA counts into plotting-only physical data points.

    This intentionally ignores decoder version and sampler settings while
    retaining them in ``plot_provenance``. Its return value must never be used
    as a Sinter resume file.

    :param stats: Raw version- and backend-specific Sinter statistics.
    :return: Deterministically ordered plotting-only pooled statistics.
    :raises RuntimeError: If a source statistic lacks required metadata.
    """
    grouped: dict[str, list[sinter.TaskStats]] = collections.defaultdict(list)
    group_metadata: dict[str, dict[str, Any]] = {}
    for stat in stats:
        metadata = _plot_identity_metadata(stat)
        identity = {"decoder": stat.decoder, "metadata": metadata}
        key = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        grouped[key].append(stat)
        group_metadata[key] = metadata

    aggregated = []
    for key in sorted(grouped):
        members = grouped[key]
        versions = sorted(
            {
                str(member.json_metadata["decoder_version"])
                for member in members
            }
        )
        sampler_configurations = sorted(
            {
                json.dumps(
                    member.json_metadata["sampler"],
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for member in members
            }
        )
        metadata = {
            **group_metadata[key],
            PLOT_ONLY_AGGREGATE_KEY: True,
            PLOT_PROVENANCE_KEY: {
                "decoder_versions": versions,
                "sampler_configurations": [
                    json.loads(configuration)
                    for configuration in sampler_configurations
                ],
            },
        }
        aggregated.append(
            sinter.TaskStats(
                strong_id=f"plot-{sha256_text(key)}",
                decoder=members[0].decoder,
                json_metadata=metadata,
                shots=sum(member.shots for member in members),
                errors=sum(member.errors for member in members),
                discards=sum(member.discards for member in members),
                seconds=sum(member.seconds for member in members),
                custom_counts=collections.Counter(),
            )
        )
    return aggregated


def read_plot_stats(*paths: Path) -> list[sinter.TaskStats]:
    """Read raw Sinter statistics and derive plotting-only pooled statistics.

    :param paths: Raw append-only Sinter CSVs to read without modifying them.
    :return: Pooled statistics suitable only for plotting.
    """
    return aggregate_stats_for_plot(
        stat
        for path in paths
        for stat in read_sinter_stats(path)
    )


def sha256_text(text: str) -> str:
    """Hash text using its UTF-8 representation.

    :param text: Circuit or metadata text to hash.
    :return: Lowercase SHA-256 hexadecimal digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_variant_text(
    reference_text: str,
    noise_level: float,
    variant: str,
) -> str:
    """Create one in-memory noisy S- or T-state circuit.

    :param reference_text: Authoritative S-state circuit at noise 0.001.
    :param noise_level: Replacement physical noise strength.
    :param variant: State variant, either ``"S"`` or ``"T"``.
    :return: Transformed SymFT circuit text.
    """
    text = replace_noise_level(
        reference_text,
        REFERENCE_NOISE_LEVEL,
        noise_level,
    )
    if variant == "T":
        text = swap_t_and_s_gates(text)
    return text


def inspect_circuit(circuit_text: str) -> dict[str, int]:
    """Parse a cultivation circuit and report its variable-size shape.

    :param circuit_text: SymFT circuit text to inspect.
    :return: Qubit, measurement, detector, and observable counts.
    :raises RuntimeError: If the circuit does not define observable 0.
    """
    circuit = symft.Circuit(circuit_text)
    metadata = {
        "num_qubits": int(circuit.num_qubits),
        "num_measurements": int(circuit.num_measurements),
        "num_detectors": int(circuit.num_detectors),
        "num_observables": int(circuit.num_observables),
    }
    observable_indices = {int(item["index"]) for item in circuit.observables}
    if 0 not in observable_indices:
        raise RuntimeError("cultivation circuit does not define observable 0")
    return metadata


def sampler_settings(cuda: bool = DEFAULT_CUDA) -> dict[str, Any]:
    """Return the compatibility-sensitive SymFT sampler configuration.

    :return: JSON-serializable settings included in every Sinter task ID.
    """
    settings = {
        "batch": True,
        "observable": 0,
        "postselect_detectors": True,
        # CUDA is launched by one host worker and chooses its own batch size.
        "threads": 1 if cuda else THREADS,
        "batch_size": 0 if cuda else BATCH_SIZE,
        "sample_chunk_shots": SAMPLE_CHUNK_SHOTS,
        "cuda": cuda,
    }
    if cuda:
        settings.update(
            {
                "cuda_mode": CUDA_MODE,
                "shots_per_launch": CUDA_SHOTS_PER_LAUNCH,
                "threads_per_block": CUDA_THREADS_PER_BLOCK,
            }
        )
    return settings


def validate_seed(seed: int | None) -> int | None:
    """Validate one optional unsigned 64-bit sampling seed.

    :param seed: Explicit seed, or None to request system entropy.
    :return: The unchanged valid seed.
    :raises ValueError: If the seed is outside ``range(2**64)``.
    """
    if seed is not None and seed not in range(STREAM_ID_MODULUS):
        raise ValueError("seed must be in range(2**64)")
    return seed


def task_metadata(
    noise_level: float,
    variant: str,
    circuit_name: str,
    cuda: bool = DEFAULT_CUDA,
) -> dict[str, Any]:
    """Build the metadata that identifies one sampled SymFT distribution.

    :param noise_level: Physical noise strength of the circuit.
    :param variant: State variant, either ``"S"`` or ``"T"``.
    :param circuit_name: Human-readable reference-circuit identifier.
    :param cuda: Whether this task uses SymFT's CUDA counts backend.
    :return: JSON-compatible metadata used by Sinter's strong task ID.
    """
    return {
        "schema_version": TASK_SCHEMA_VERSION,
        "decoder_version": symft.__version__,
        "noise_level": noise_level,
        "variant": variant,
        "circuit_name": circuit_name,
        "sampler": sampler_settings(cuda),
    }


def validate_all_variants(
    reference_text: str,
    noise_levels: Sequence[float] = DEFAULT_NOISE_LEVELS,
    variants: Sequence[str] = VARIANTS,
) -> list[dict[str, Any]]:
    """Validate every requested gate/noise configuration.

    :param reference_text: Authoritative S-state circuit at noise 0.001.
    :param noise_levels: Physical noise strengths to validate.
    :param variants: State variants to validate.
    :return: Metadata rows in ascending-noise canonical-variant order.
    """
    selected_variants = _normalize_variants(variants)
    rows = []
    for noise_level in sorted(set(noise_levels)):
        for variant in selected_variants:
            text = make_variant_text(reference_text, noise_level, variant)
            rows.append(
                {
                    "noise_level": noise_level,
                    "variant": variant,
                    **inspect_circuit(text),
                }
            )
    return rows


def _make_proxy_circuit(reference_text: str, noise_level: float) -> stim.Circuit:
    """Create the Stim-compatible S circuit used to identify Sinter tasks.

    :param reference_text: Authoritative S-state circuit at noise 0.001.
    :param noise_level: Physical noise strength of the proxy circuit.
    :return: Clifford S circuit accepted by Stim.
    """
    return stim.Circuit(make_variant_text(reference_text, noise_level, "S"))


def build_tasks(
    reference_text: str,
    circuit_name: str = DEFAULT_CIRCUIT_NAME,
    noise_levels: Sequence[float] = DEFAULT_NOISE_LEVELS,
    cuda: bool = DEFAULT_CUDA,
    variants: Sequence[str] = VARIANTS,
) -> list[sinter.Task]:
    """Build Sinter tasks for every requested noise level and variant.

    The task circuit is the Stim-compatible S circuit. The custom sampler uses
    the task metadata to construct and sample the actual S or T circuit.

    :param reference_text: Authoritative S-state circuit at noise 0.001.
    :param circuit_name: Human-readable reference-circuit identifier.
    :param noise_levels: Physical noise strengths to sample.
    :param cuda: Whether tasks identify CUDA-backed SymFT sampling.
    :param variants: State variants to sample.
    :return: Tasks ordered by descending noise and canonical variant order.
    """
    selected_variants = _normalize_variants(variants)
    tasks = []
    for noise_level in sorted(set(noise_levels), reverse=True):
        proxy_circuit = _make_proxy_circuit(reference_text, noise_level)
        proxy_dem = proxy_circuit.detector_error_model(
            decompose_errors=False,
            approximate_disjoint_errors=True,
        )
        for variant in selected_variants:
            tasks.append(
                sinter.Task(
                    circuit=proxy_circuit,
                    detector_error_model=proxy_dem,
                    decoder=DECODER_NAME,
                    json_metadata=task_metadata(
                        noise_level,
                        variant,
                        circuit_name,
                        cuda,
                    ),
                )
            )
    return tasks


def read_sinter_stats(path: Path) -> list[sinter.TaskStats]:
    """Read and aggregate a Sinter resume CSV when it exists.

    :param path: Sinter CSV path.
    :return: One aggregate statistic per strong task ID.
    """
    if not path.exists():
        return []
    return sinter.read_stats_from_csv_files(path)


def _metadata_values_for_circuit(
    stats: Iterable[sinter.TaskStats],
    circuit_name: str,
    key: str,
) -> list[Any]:
    """Read one metadata field from persisted rows for a circuit name.

    :param stats: Aggregated statistics loaded from the shared resume CSV.
    :param circuit_name: Circuit dataset whose rows should be selected.
    :param key: Metadata field to read from each selected row.
    :return: Present field values from matching dictionary metadata.
    """
    values = []
    for stat in stats:
        metadata = stat.json_metadata
        if (
            isinstance(metadata, dict)
            and metadata.get("circuit_name") == circuit_name
            and key in metadata
        ):
            values.append(metadata[key])
    return values


def _resume_noise_levels(
    stats: Iterable[sinter.TaskStats],
    circuit_name: str,
    noise_levels: Sequence[float],
) -> tuple[float, ...]:
    """Include persisted same-circuit levels when checking resume metadata.

    :param stats: Aggregated statistics loaded from the shared resume CSV.
    :param circuit_name: Circuit dataset whose rows must remain compatible.
    :param noise_levels: Physical noise strengths requested by this run.
    :return: Sorted union of requested and persisted physical noise strengths.
    """
    result = set(noise_levels)
    result.update(
        float(value)
        for value in _metadata_values_for_circuit(
            stats,
            circuit_name,
            "noise_level",
        )
    )
    return tuple(sorted(result))


def _normalize_variants(variants: Sequence[str]) -> tuple[str, ...]:
    """Deduplicate and canonically order a nonempty variant selection.

    :param variants: Requested cultivation state variants.
    :return: Selected variants in the order declared by ``VARIANTS``.
    :raises ValueError: If no variant is selected or a variant is unknown.
    """
    selected = set(variants)
    unknown = selected - set(VARIANTS)
    if unknown:
        raise ValueError(
            f"unknown cultivation variants: {sorted(unknown)!r}; "
            f"expected values from {VARIANTS!r}"
        )
    normalized = tuple(variant for variant in VARIANTS if variant in selected)
    if not normalized:
        raise ValueError("at least one cultivation variant must be selected")
    return normalized


def _resume_variants(
    stats: Iterable[sinter.TaskStats],
    circuit_name: str,
    variants: Sequence[str],
) -> tuple[str, ...]:
    """Include persisted same-circuit variants during resume validation.

    :param stats: Aggregated statistics loaded from the shared resume CSV.
    :param circuit_name: Circuit dataset whose rows must remain compatible.
    :param variants: State variants requested by this run.
    :return: Canonically ordered union of requested and persisted variants.
    """
    result = list(variants)
    result.extend(
        str(value)
        for value in _metadata_values_for_circuit(
            stats,
            circuit_name,
            "variant",
        )
    )
    return _normalize_variants(result)


def validate_resume_stats(
    stats: Iterable[sinter.TaskStats],
    tasks: Iterable[sinter.Task],
) -> None:
    """Check persisted statistics selected by the current circuit name.

    :param stats: Aggregated statistics loaded from the resume CSV.
    :param tasks: Current Sinter task matrix.
    :return: None.
    :raises RuntimeError: If selected statistics are incompatible or obsolete.
    """
    tasks_by_id = {task.strong_id(): task for task in tasks}
    selected_task = next(iter(tasks_by_id.values()))
    circuit_name = str(selected_task.json_metadata["circuit_name"])
    for stat in stats:
        metadata = stat.json_metadata
        if not isinstance(metadata, dict):
            continue
        if metadata.get(PLOT_ONLY_AGGREGATE_KEY) is True:
            raise RuntimeError(
                "plot-only aggregated statistics cannot be used for resume"
            )
        stat_circuit_name = metadata.get("circuit_name")
        if stat_circuit_name is not None and stat_circuit_name != circuit_name:
            continue
        if metadata.get("schema_version") != TASK_SCHEMA_VERSION and (
            stat_circuit_name == circuit_name
            or (stat_circuit_name is None and stat.decoder == DECODER_NAME)
        ):
            raise RuntimeError(
                "existing SymFT statistics use obsolete "
                f"schema_version={metadata.get('schema_version')!r}; "
                "start a new CSV before collecting"
            )
        if stat_circuit_name is None:
            continue
        task = tasks_by_id.get(stat.strong_id)
        if task is None:
            raise RuntimeError(
                f"circuit_name={circuit_name!r} already identifies different "
                "circuits or sampler settings; choose another circuit name"
            )
        if (
            stat.decoder != task.decoder
            or stat.json_metadata != task.json_metadata
        ):
            raise RuntimeError(
                "persisted task metadata disagrees with its strong ID"
            )


def compile_counts_sampler(
    circuit_text: str,
    cuda: bool = DEFAULT_CUDA,
) -> tuple[Any, dict[str, Any]]:
    """Compile and verify one high-throughput SymFT counts sampler.

    :param circuit_text: Actual S or T circuit to sample.
    :param cuda: Whether to compile SymFT's CUDA counts backend.
    :return: Compiled SymFT counts sampler and normalized sampler information.
    :raises RuntimeError: If SymFT does not honor the locked configuration.
    """
    inspect_circuit(circuit_text)
    circuit = symft.Circuit(circuit_text)
    sampler = circuit.compile_counts_sampler(**sampler_settings(cuda))
    info = dict(sampler.info)
    expected = {
        "backend": "cuda" if cuda else "batch",
        "threads": 1 if cuda else THREADS,
        "batch_size": 0 if cuda else BATCH_SIZE,
        "detector_postselection": True,
        "observable": 0,
    }
    observed = {key: info[key] for key in expected}
    if observed != expected:
        raise RuntimeError(
            f"SymFT sampler configuration mismatch: {observed}; "
            f"expected {expected}"
        )
    return sampler, info


class _StreamSequence:
    """Allocate one entropy-seeded sequence of SymFT stream identifiers."""

    def __init__(self, seed: int | None) -> None:
        """Initialize a task-local stream sequence.

        :param self: Stream sequence being initialized.
        :param seed: First stream identifier, or None to use system entropy.
        :return: None.
        """
        self.next_stream_id = secrets.randbits(64) if seed is None else seed

    def take(self) -> int:
        """Return the next stream identifier and advance the sequence.

        :param self: Task-local stream sequence.
        :return: Newly allocated stream identifier.
        """
        stream_id = self.next_stream_id
        self.next_stream_id = (stream_id + 1) % STREAM_ID_MODULUS
        return stream_id


class CompiledSymftSinterSampler(sinter.CompiledSampler):
    """Adapt one compiled SymFT counts sampler to Sinter's statistics API."""

    def __init__(
        self,
        sampler: Any,
        sampler_info: Mapping[str, Any],
        call_shots: int,
        stream_sequence: _StreamSequence,
    ) -> None:
        """Initialize a reusable sampler for one Sinter task.

        :param self: Compiled adapter being initialized.
        :param sampler: Compiled SymFT counts sampler.
        :param sampler_info: Normalized SymFT sampler configuration.
        :param call_shots: Maximum attempted shots in one persisted call.
        :param stream_sequence: Stream sequence shared across recompilations.
        :return: None.
        """
        self.sampler = sampler
        self.sampler_info = sampler_info
        self.call_shots = call_shots
        self.stream_sequence = stream_sequence

    def handles_throttling(self) -> bool:
        """Tell Sinter that this adapter enforces its own shot-call cap.

        :param self: Compiled SymFT adapter.
        :return: True so Sinter does not wrap the adapter in a ramp limiter.
        """
        return True

    def sample(self, suggested_shots: int) -> sinter.AnonTaskStats:
        """Sample one capped SymFT batch and translate its aggregate counts.

        :param self: Compiled SymFT adapter.
        :param suggested_shots: Maximum shots currently requested by Sinter.
        :return: Sinter-compatible aggregate task statistics.
        :raises RuntimeError: If SymFT returns inconsistent counts or threads.
        """
        shots = min(suggested_shots, self.call_shots)
        stream_id = self.stream_sequence.take()
        result = self.sampler.sample(shots=shots, stream_id=stream_id)

        returned_shots = int(result["shots"])
        accepted = int(result["accepted"])
        discarded = int(result["discarded"])
        logical_errors = int(result["logical_errors"])
        active_threads = int(result["active_threads"])
        if returned_shots != shots:
            raise RuntimeError(
                f"SymFT returned {returned_shots} shots after requesting {shots}"
            )
        if accepted + discarded != returned_shots:
            raise RuntimeError("SymFT count invariant failed")
        if logical_errors > accepted:
            raise RuntimeError("SymFT logical-error count exceeds accepted shots")

        if self.sampler_info["backend"] == "cuda":
            expected_active_threads = 1
        else:
            sample_chunk_shots = int(self.sampler_info["sample_chunk_shots"])
            expected_active_threads = min(
                int(self.sampler_info["threads"]),
                math.ceil(returned_shots / sample_chunk_shots),
            )
        if active_threads != expected_active_threads:
            raise RuntimeError(
                f"SymFT used {active_threads} threads; expected "
                f"{expected_active_threads}"
            )

        sample_seconds = float(result["timing"]["sample_s"])
        return sinter.AnonTaskStats(
            shots=returned_shots,
            errors=logical_errors,
            discards=discarded,
            seconds=sample_seconds,
        )


class SymftSinterSampler(sinter.Sampler):
    """Compile SymFT samplers for Sinter tasks backed by an S reference."""

    def __init__(
        self,
        reference_text: str,
        call_shots: int,
        seed: int | None = None,
        cuda: bool = DEFAULT_CUDA,
    ) -> None:
        """Initialize the factory shared with Sinter's worker process.

        :param self: Sampler factory being initialized.
        :param reference_text: Authoritative S-state circuit text.
        :param call_shots: Maximum attempted shots per SymFT call.
        :param seed: Initial stream identifier, or None to use system entropy.
        :param cuda: Whether to compile SymFT's CUDA counts backend.
        :return: None.
        """
        self.reference_text = reference_text
        self.call_shots = call_shots
        self.seed = validate_seed(seed)
        self.cuda = cuda
        self.stream_sequences: dict[str, _StreamSequence] = {}

    def compiled_sampler_for_task(
        self,
        task: sinter.Task,
    ) -> sinter.CompiledSampler:
        """Compile the actual metadata-selected SymFT circuit for a task.

        :param self: SymFT sampler factory.
        :param task: Sinter task whose metadata selects the actual circuit.
        :return: Compiled SymFT-to-Sinter adapter.
        """
        metadata = task.json_metadata
        circuit_text = make_variant_text(
            self.reference_text,
            float(metadata["noise_level"]),
            str(metadata["variant"]),
        )
        sampler, info = compile_counts_sampler(circuit_text, cuda=self.cuda)
        strong_id = task.strong_id()
        stream_sequence = self.stream_sequences.get(strong_id)
        if stream_sequence is None:
            stream_sequence = _StreamSequence(self.seed)
            self.stream_sequences[strong_id] = stream_sequence
        return CompiledSymftSinterSampler(
            sampler=sampler,
            sampler_info=info,
            call_shots=self.call_shots,
            stream_sequence=stream_sequence,
        )


def collect_stats(
    *,
    reference_path: Path,
    stats_path: Path,
    circuit_name: str,
    target_errors: int,
    max_shots: int,
    call_shots: int,
    print_progress: bool,
    noise_levels: Sequence[float] = DEFAULT_NOISE_LEVELS,
    cuda: bool = DEFAULT_CUDA,
    variants: Sequence[str] = VARIANTS,
    seed: int | None = None,
) -> list[sinter.TaskStats]:
    """Collect or resume selected S/T points through Sinter.

    :param reference_path: Authoritative S-state reference circuit.
    :param stats_path: Shared Sinter save-and-resume CSV.
    :param circuit_name: Human-readable reference-circuit identifier.
    :param target_errors: Logical-error stopping target per task.
    :param max_shots: Attempted-shot safety cap per task.
    :param call_shots: Maximum attempted shots per SymFT call.
    :param print_progress: Whether Sinter should print progress to stderr.
    :param noise_levels: Physical noise strengths to collect.
    :param cuda: Whether to run the CUDA counts backend.
    :param variants: State variants to collect.
    :param seed: Initial stream identifier, or None to use system entropy.
    :return: Aggregated statistics for the current requested tasks.
    """
    reference_text = reference_path.read_text(encoding="utf-8")
    validate_all_variants(reference_text, noise_levels, variants)
    tasks = build_tasks(
        reference_text,
        circuit_name,
        noise_levels,
        cuda,
        variants,
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    existing_stats = read_sinter_stats(stats_path)
    resume_tasks = build_tasks(
        reference_text,
        circuit_name,
        _resume_noise_levels(existing_stats, circuit_name, noise_levels),
        cuda,
        _resume_variants(existing_stats, circuit_name, variants),
    )
    validate_resume_stats(existing_stats, resume_tasks)
    sampler = SymftSinterSampler(
        reference_text=reference_text,
        call_shots=call_shots,
        seed=seed,
        cuda=cuda,
    )
    collected = sinter.collect(
        num_workers=SINTER_WORKERS,
        tasks=tasks,
        save_resume_filepath=stats_path,
        max_shots=max_shots,
        max_errors=target_errors,
        custom_decoders={DECODER_NAME: sampler},
        print_progress=print_progress,
    )
    expected_ids = {task.strong_id() for task in tasks}
    return [stat for stat in collected if stat.strong_id in expected_ids]


def smoke_sample(
    reference_path: Path,
    shots: int,
    circuit_name: str = DEFAULT_CIRCUIT_NAME,
    cuda: bool = DEFAULT_CUDA,
    variants: Sequence[str] = VARIANTS,
    seed: int | None = None,
) -> None:
    """Run a small non-persisted Sinter collection at noise 0.001.

    :param reference_path: Authoritative S-state reference circuit.
    :param shots: Attempted shots for each state variant.
    :param circuit_name: Human-readable reference-circuit identifier.
    :param cuda: Whether to run the CUDA counts backend.
    :param variants: State variants to sample.
    :param seed: Initial stream identifier, or None to use system entropy.
    :return: None.
    """
    reference_text = reference_path.read_text(encoding="utf-8")
    tasks = [
        task
        for task in build_tasks(
            reference_text,
            circuit_name,
            cuda=cuda,
            variants=variants,
        )
        if float(task.json_metadata["noise_level"]) == REFERENCE_NOISE_LEVEL
    ]
    sampler = SymftSinterSampler(
        reference_text=reference_text,
        call_shots=shots,
        seed=seed,
        cuda=cuda,
    )
    sinter.collect(
        num_workers=SINTER_WORKERS,
        tasks=tasks,
        max_shots=shots,
        custom_decoders={DECODER_NAME: sampler},
        print_progress=True,
    )
