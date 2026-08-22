"""Synthesize flag detectors for low-order malignant fault configurations.

The synthesis model deliberately separates three concepts:

* an abstract fault is a bundle of physical error-event trajectories;
* a candidate flag is a binary symplectic detecting region in circuit
  spacetime; and
* a malignant configuration is covered only when every choice of one
  physical realization per abstract fault has a nonzero selected flag
  signature.

Physical realizations are not expanded into configuration products eagerly.
The counterexample separator partitions each fault bundle by the response
signatures of the currently selected candidates and adds a concrete witness
only when the current solution misses one of those products.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
import heapq
import json
import math
from pathlib import Path
import time
from typing import Literal

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_array
import stim

from cliffordep.combinators import FaultCombinator
from cliffordep.noiseless_circuit_tools import split_by_ticks
from cliffordep.type_aliases import ErrorEvent, LogicalTriple, PauliMask


PauliBasis = Literal["X", "Z"]
CandidateSource = Literal["direct-envelope", "nearby-envelope", "single-seed-fallback"]
ObjectiveName = Literal["flag_qubits", "added_cnot_layers", "flag_cnot_count"]


@dataclass(frozen=True, slots=True)
class SymplecticPauli:
    """An unsigned Pauli represented by binary X and Z support masks."""

    x: int = 0
    z: int = 0

    def anticommutes(self, other: "SymplecticPauli") -> bool:
        """Return whether the binary symplectic inner product is one."""
        return bool(
            ((self.x & other.z).bit_count()
             + (self.z & other.x).bit_count())
            & 1
        )

    @property
    def support(self) -> tuple[int, ...]:
        """Return the sorted qubit support."""
        mask = self.x | self.z
        return tuple(index for index in range(mask.bit_length()) if mask >> index & 1)

    @property
    def weight(self) -> int:
        """Return the Pauli weight."""
        return (self.x | self.z).bit_count()


@dataclass(frozen=True, slots=True)
class PhysicalFaultRealization:
    """One physical error event and its propagated spacetime trajectory."""

    ordinal: int
    event: ErrorEvent = field(compare=False, repr=False)
    trajectory: tuple[SymplecticPauli, ...] = field(compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class RealizationClass:
    """Realizations that selected detecting regions cannot distinguish."""

    response_signature: int
    realization_ordinals: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FaultBundle:
    """All physical error events represented by one abstract fault."""

    fault_index: int
    realizations: tuple[PhysicalFaultRealization, ...]


@dataclass(frozen=True, slots=True)
class FaultEffectClassification:
    """Data-restricted packed effect weights for one abstract fault."""

    fault_index: int
    data_effect: PauliMask
    x_data_mask: int
    z_data_mask: int
    data_pauli_weight: int
    x_data_weight: int
    z_data_weight: int
    is_targetable_hook: bool


@dataclass(frozen=True, slots=True)
class CandidatePoolRestrictions:
    """Finite search restrictions used to generate a candidate pool."""

    maximum_duration: int
    maximum_interactions: int
    nearby_boundary_radius: int
    includes_single_seed_fallback: bool
    boundary_range: tuple[int, int]


@dataclass(frozen=True, slots=True)
class FlagCandidate:
    """A valid pure-Z detecting region implementable by one flag lifecycle.

    ``interactions`` lists the nonzero boundary terms of the detecting
    region. Each pair consists of a boundary index and the qubits on which
    the pure-Z boundary Pauli is supported. The ``basis`` name describes the
    detecting region: a Z region physically detects X/Y errors using a
    ``|0>`` flag, circuit-to-flag CNOTs, and a Z measurement.
    """

    identifier: str
    basis: PauliBasis
    interactions: tuple[tuple[int, tuple[int, ...]], ...]
    response_mask: int
    source: CandidateSource = "direct-envelope"

    @property
    def start_boundary(self) -> int:
        return self.interactions[0][0]

    @property
    def end_boundary(self) -> int:
        return self.interactions[-1][0]

    @property
    def interaction_count(self) -> int:
        return sum(len(qubits) for _, qubits in self.interactions)

    def responds_to(self, realization_ordinal: int) -> bool:
        return bool(self.response_mask >> realization_ordinal & 1)


@dataclass(frozen=True, slots=True)
class FlagSynthesisProblem:
    """A finite candidate-pool instance of the flag-synthesis problem."""

    circuit: stim.Circuit = field(compare=False, repr=False)
    fault_bundles: tuple[FaultBundle, ...]
    malignant_configurations: tuple[tuple[int, ...], ...]
    candidates: tuple[FlagCandidate, ...]
    realization_count: int
    fault_classifications: tuple[FaultEffectClassification, ...] = ()
    target_fault_indices: tuple[int, ...] = ()
    candidate_pool_restrictions: CandidatePoolRestrictions | None = None

    def bundle(self, fault_index: int) -> FaultBundle:
        """Return a fault bundle by its combinator index."""
        for bundle in self.fault_bundles:
            if bundle.fault_index == fault_index:
                return bundle
        raise KeyError(fault_index)

    def refine_fault(
            self,
            fault_index: int,
            candidate_indices: Sequence[int],
    ) -> tuple[RealizationClass, ...]:
        """Coarse-grain a fault by responses of the requested candidates.

        The fault starts as one bundle. It is split only when at least one
        requested region gives different responses to its realizations.
        """
        classes: dict[int, list[int]] = defaultdict(list)
        for realization in self.bundle(fault_index).realizations:
            signature = 0
            for signature_index, candidate_index in enumerate(candidate_indices):
                if self.candidates[candidate_index].responds_to(realization.ordinal):
                    signature |= 1 << signature_index
            classes[signature].append(realization.ordinal)
        return tuple(
            RealizationClass(signature, tuple(ordinals))
            for signature, ordinals in sorted(classes.items())
        )


@dataclass(frozen=True, slots=True)
class SynthesisLimits:
    """Wall-clock limits for a synthesis run."""

    milp_time_limit_seconds: float = 600
    total_time_limit_seconds: float = 3600
    mip_relative_gap: float = 0
    maximum_refinement_rounds: int = 100


@dataclass(frozen=True, slots=True)
class MilpRun:
    """Metrics from one HiGHS invocation."""

    objective: ObjectiveName
    elapsed_seconds: float
    status: int
    message: str
    node_count: int | None
    mip_gap: float | None
    objective_value: float | None


@dataclass(frozen=True, slots=True)
class SynthesisMetrics:
    """Candidate, refinement, and solver metrics."""

    candidate_count: int
    realization_count: int
    configuration_count: int
    refinement_rounds: int
    witness_constraint_count: int
    elapsed_seconds: float
    milp_runs: tuple[MilpRun, ...]


@dataclass(frozen=True, slots=True)
class ScheduledInteraction:
    """One scheduled flag-data CNOT."""

    boundary: int
    layer: int
    candidate_index: int
    flag_index: int
    data_index: int


@dataclass(frozen=True, slots=True)
class FlagCircuitSolution:
    """A verified-within-the-model flag selection and schedule."""

    selected_candidate_indices: tuple[int, ...]
    flag_count: int
    added_cnot_layers: int
    interaction_count: int
    candidate_to_flag: tuple[tuple[int, int], ...]
    scheduled_interactions: tuple[ScheduledInteraction, ...]
    objective_order: tuple[ObjectiveName, ...]
    optimal_within_candidate_pool: bool
    metrics: SynthesisMetrics


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Result of noisy malignant-configuration verification."""

    max_order: int
    cultivated_state: str
    malignant_configurations: tuple[tuple[int, ...], ...]
    elapsed_seconds: float

    @property
    def passed(self) -> bool:
        return not self.malignant_configurations


class FlagSynthesisError(RuntimeError):
    """Raised when the candidate pool cannot produce a feasible circuit."""


def extract_malignant_configurations(
        kept_effects: Mapping[
            str,
            Sequence[Mapping[PauliMask, LogicalTriple]],
        ],
        *,
        cultivated_state: str = "T",
        orders: Iterable[int] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Extract accepted configurations with non-unit logical fidelity."""
    effects_by_order = kept_effects[cultivated_state]
    selected_orders = range(len(effects_by_order)) if orders is None else orders
    configurations: list[tuple[int, ...]] = []
    for order in selected_orders:
        for accept_probability, logical_fidelity, effect_configurations in (
                effects_by_order[order].values()):
            if accept_probability and not math.isclose(float(logical_fidelity), 1.0):
                configurations.extend(effect_configurations)
    return tuple(sorted(configurations, key=lambda item: (len(item), item)))


def classify_fault_effect(
        *,
        fault_index: int,
        effect_mask: PauliMask,
        source_qubit_count: int,
        data_indices: Sequence[int],
) -> FaultEffectClassification:
    """Restrict a packed fault effect and classify its data X hook weight.

    :param fault_index: The dense abstract-fault index for diagnostics.
    :param effect_mask: The full packed effect with X bits below Z bits.
    :param source_qubit_count: The width of each packed support half.
    :param data_indices: Circuit-qubit indices retained in compact data order.
    :return: The compact data masks, component weights, and hook decision.
    """
    x_data_mask = 0
    z_data_mask = 0
    for data_index, circuit_index in enumerate(data_indices):
        x_data_mask |= ((effect_mask >> circuit_index) & 1) << data_index
        z_data_mask |= (
            (effect_mask >> (source_qubit_count + circuit_index)) & 1
        ) << data_index
    data_qubit_count = len(data_indices)
    return FaultEffectClassification(
        fault_index=fault_index,
        data_effect=x_data_mask | (z_data_mask << data_qubit_count),
        x_data_mask=x_data_mask,
        z_data_mask=z_data_mask,
        data_pauli_weight=(x_data_mask | z_data_mask).bit_count(),
        x_data_weight=x_data_mask.bit_count(),
        z_data_weight=z_data_mask.bit_count(),
        is_targetable_hook=x_data_mask.bit_count() > 1,
    )


def build_flag_synthesis_problem(
        *,
        combinator: FaultCombinator,
        malignant_configurations: Iterable[Iterable[int]],
        data_indices: Sequence[int],
        maximum_candidate_interactions: int = 8,
        maximum_candidate_duration: int = 10,
        minimum_candidate_duration: int = 1,
        nearby_boundary_radius: int = 1,
        include_single_seed_fallback: bool = False,
        candidate_boundary_range: tuple[int, int] | None = None,
) -> FlagSynthesisProblem:
    """Build a nuisance-aware, hook-directed Z-only synthesis problem.

    Every fault in each malignant configuration remains a physical-realization
    bundle, but only faults with multi-qubit data X support seed candidates.

    :param combinator: The abstract faults and their physical error events.
    :param malignant_configurations: Low-order abstract-fault configurations.
    :param data_indices: Original data-qubit indices used for hook classification.
    :param maximum_candidate_interactions: Maximum CNOT count per lifecycle.
    :param maximum_candidate_duration: Maximum boundary span per lifecycle.
    :param minimum_candidate_duration: Minimum boundary span per lifecycle.
    :param nearby_boundary_radius: Endpoint displacement around direct envelopes.
    :param include_single_seed_fallback: Whether to add general valid Z paths.
    :param candidate_boundary_range: Optional inclusive range of usable boundaries.
    :return: The local realization bundles and finite Z-candidate pool.
    """
    configurations = tuple(sorted(
        {tuple(sorted(configuration)) for configuration in malignant_configurations},
        key=lambda item: (len(item), item),
    ))
    if not configurations:
        raise ValueError("At least one malignant configuration is required.")
    relevant_faults = tuple(sorted({index for c in configurations for index in c}))
    noiseless_circuit = combinator.circuit.noiseless_circuit
    classifications = tuple(
        classify_fault_effect(
            fault_index=fault_index,
            effect_mask=combinator.indexed_faults[fault_index][1],
            source_qubit_count=noiseless_circuit.num_qubits,
            data_indices=data_indices,
        )
        for fault_index in relevant_faults
    )
    classification_by_fault = {
        classification.fault_index: classification
        for classification in classifications
    }
    for configuration in configurations:
        if any(
                classification_by_fault[fault_index].is_targetable_hook
                for fault_index in configuration):
            continue
        effects = "; ".join(
            f"fault {fault_index}: data_effect=0x{classification_by_fault[fault_index].data_effect:x}, "
            f"X={classification_by_fault[fault_index].x_data_weight}, "
            f"Z={classification_by_fault[fault_index].z_data_weight}, "
            f"total={classification_by_fault[fault_index].data_pauli_weight}"
            for fault_index in configuration
        )
        raise FlagSynthesisError(
            f"Malignant configuration {configuration} contains no targetable "
            f"multi-X hook fault ({effects})."
        )
    target_fault_indices = tuple(
        classification.fault_index
        for classification in classifications
        if classification.is_targetable_hook
    )
    ordinal = 0
    bundles: list[FaultBundle] = []
    for fault_index in relevant_faults:
        realizations = []
        for event in sorted(combinator.index_to_events[fault_index], key=_error_event_key):
            realizations.append(PhysicalFaultRealization(
                ordinal=ordinal,
                event=event,
                trajectory=_make_fault_trajectory(noiseless_circuit, event),
            ))
            ordinal += 1
        bundles.append(FaultBundle(fault_index, tuple(realizations)))
    layers = split_by_ticks(noiseless_circuit)
    boundary_range = (
        (1, len(layers) - 2)
        if candidate_boundary_range is None
        else candidate_boundary_range
    )
    candidates = _generate_hook_candidates(
        circuit=noiseless_circuit,
        fault_bundles=bundles,
        target_fault_indices=target_fault_indices,
        maximum_interactions=maximum_candidate_interactions,
        maximum_duration=maximum_candidate_duration,
        minimum_duration=minimum_candidate_duration,
        nearby_boundary_radius=nearby_boundary_radius,
        include_single_seed_fallback=include_single_seed_fallback,
        boundary_range=boundary_range,
    )
    if not candidates:
        raise FlagSynthesisError("No valid Z detecting regions respond to the hook faults.")
    return FlagSynthesisProblem(
        circuit=noiseless_circuit,
        fault_bundles=tuple(bundles),
        malignant_configurations=configurations,
        candidates=candidates,
        realization_count=ordinal,
        fault_classifications=classifications,
        target_fault_indices=target_fault_indices,
        candidate_pool_restrictions=CandidatePoolRestrictions(
            maximum_duration=maximum_candidate_duration,
            maximum_interactions=maximum_candidate_interactions,
            nearby_boundary_radius=nearby_boundary_radius,
            includes_single_seed_fallback=include_single_seed_fallback,
            boundary_range=boundary_range,
        ),
    )


def synthesize_flag_circuit(
        problem: FlagSynthesisProblem,
        *,
        limits: SynthesisLimits = SynthesisLimits(),
) -> FlagCircuitSolution:
    """Synthesize a flag-count-first circuit using lazy witness refinement."""
    started = time.monotonic()
    witness_masks = {
        _canonical_witness_mask(problem, configuration)
        for configuration in problem.malignant_configurations
    }
    milp_runs: list[MilpRun] = []
    last_stage_runs: tuple[MilpRun, ...] = ()

    for refinement_round in range(1, limits.maximum_refinement_rounds + 1):
        _check_total_time(started, limits)
        primary, primary_run = _solve_milp(
            problem,
            witness_masks,
            objective="flag_qubits",
            limits=limits,
            run_started=started,
        )
        milp_runs.append(primary_run)
        selected = _selected_candidates(primary, len(problem.candidates))
        new_witnesses = _find_zero_response_witnesses(problem, selected)
        if new_witnesses:
            witness_masks.update(new_witnesses)
            continue

        flag_count = int(round(primary[len(problem.candidates)]))
        secondary, secondary_run = _solve_milp(
            problem,
            witness_masks,
            objective="added_cnot_layers",
            fixed_flag_count=flag_count,
            limits=limits,
            run_started=started,
        )
        milp_runs.append(secondary_run)
        selected = _selected_candidates(secondary, len(problem.candidates))
        new_witnesses = _find_zero_response_witnesses(problem, selected)
        if new_witnesses:
            witness_masks.update(new_witnesses)
            continue

        layer_count = int(round(sum(secondary[len(problem.candidates) + 1:])))
        tertiary, tertiary_run = _solve_milp(
            problem,
            witness_masks,
            objective="flag_cnot_count",
            fixed_flag_count=flag_count,
            fixed_layer_count=layer_count,
            limits=limits,
            run_started=started,
        )
        milp_runs.append(tertiary_run)
        selected = _selected_candidates(tertiary, len(problem.candidates))
        new_witnesses = _find_zero_response_witnesses(problem, selected)
        if new_witnesses:
            witness_masks.update(new_witnesses)
            continue

        last_stage_runs = (primary_run, secondary_run, tertiary_run)
        candidate_to_flag = _color_candidate_intervals(problem, selected)
        scheduled = _schedule_selected_interactions(
            problem,
            selected,
            dict(candidate_to_flag),
        )
        actual_layer_count = 0
        if scheduled:
            actual_layer_count = sum(
                1 + max(interaction.layer for interaction in scheduled
                        if interaction.boundary == boundary)
                for boundary in sorted({item.boundary for item in scheduled})
            )
        if actual_layer_count != layer_count:
            raise AssertionError(
                f"MILP predicted {layer_count} CNOT layers but scheduling used "
                f"{actual_layer_count}."
            )
        elapsed = time.monotonic() - started
        metrics = SynthesisMetrics(
            candidate_count=len(problem.candidates),
            realization_count=problem.realization_count,
            configuration_count=len(problem.malignant_configurations),
            refinement_rounds=refinement_round,
            witness_constraint_count=len(witness_masks),
            elapsed_seconds=elapsed,
            milp_runs=tuple(milp_runs),
        )
        return FlagCircuitSolution(
            selected_candidate_indices=selected,
            flag_count=flag_count,
            added_cnot_layers=layer_count,
            interaction_count=sum(
                problem.candidates[index].interaction_count for index in selected
            ),
            candidate_to_flag=candidate_to_flag,
            scheduled_interactions=scheduled,
            objective_order=("flag_qubits", "added_cnot_layers", "flag_cnot_count"),
            optimal_within_candidate_pool=all(run.status == 0 for run in last_stage_runs),
            metrics=metrics,
        )
    raise FlagSynthesisError(
        f"Exceeded {limits.maximum_refinement_rounds} witness-refinement rounds."
    )


def greedy_verified_candidate_indices(
        problem: FlagSynthesisProblem,
) -> tuple[tuple[int, ...], set[int]]:
    """Construct a realization-verified cover without branch-and-bound.

    The heuristic strongly prefers candidates that do not increase the
    current maximum interval overlap. It returns both the selected indices and
    every physical witness encountered, which can be used to build a smaller
    but known-feasible MILP candidate pool.
    """
    witness_masks = {
        _canonical_witness_mask(problem, configuration)
        for configuration in problem.malignant_configurations
    }
    candidate_count = len(problem.candidates)
    boundary_count = len(split_by_ticks(problem.circuit)) + 1
    active = np.zeros((candidate_count, boundary_count), dtype=np.int8)
    durations = np.empty(candidate_count, dtype=int)
    interaction_counts = np.empty(candidate_count, dtype=int)
    for index, candidate in enumerate(problem.candidates):
        active[
            index,
            max(0, candidate.start_boundary - 1):candidate.end_boundary + 1,
        ] = 1
        durations[index] = candidate.end_boundary - candidate.start_boundary
        interaction_counts[index] = candidate.interaction_count
    selected_mask = np.zeros(candidate_count, dtype=bool)
    active_counts = np.zeros(boundary_count, dtype=int)
    ordered_witnesses = sorted(witness_masks)
    coverage = np.asarray([
        [
            (candidate.response_mask & witness).bit_count() & 1
            for candidate in problem.candidates
        ]
        for witness in ordered_witnesses
    ], dtype=bool)
    while True:
        uncovered = ~np.any(coverage[:, selected_mask], axis=1) \
            if np.any(selected_mask) else np.ones(len(coverage), dtype=bool)
        while np.any(uncovered):
            covered_counts = coverage[uncovered].sum(axis=0)
            current_q = int(active_counts.max(initial=0))
            next_q = np.max(active_counts + active, axis=1)
            delta_q = next_q - current_q
            scores = covered_counts / (1 + 8 * delta_q)
            scores[(covered_counts == 0) | selected_mask] = -1
            maximum_score = scores.max(initial=-1)
            if maximum_score < 0:
                raise FlagSynthesisError(
                    "The candidate pool cannot cover the current physical witnesses."
                )
            tied = np.flatnonzero(scores == maximum_score)
            best_index = min(
                map(int, tied),
                key=lambda index: (
                    -int(covered_counts[index]),
                    int(delta_q[index]),
                    int(durations[index]),
                    int(interaction_counts[index]),
                    index,
                ),
            )
            selected_mask[best_index] = True
            active_counts += active[best_index]
            uncovered &= ~coverage[:, best_index]

        selected = tuple(map(int, np.flatnonzero(selected_mask)))
        new_witnesses = _find_zero_response_witnesses(problem, selected)
        unseen = new_witnesses - witness_masks
        if unseen:
            witness_masks.update(unseen)
            new_ordered = sorted(unseen)
            new_coverage = np.asarray([
                [
                    (candidate.response_mask & witness).bit_count() & 1
                    for candidate in problem.candidates
                ]
                for witness in new_ordered
            ], dtype=bool)
            coverage = np.concatenate((coverage, new_coverage), axis=0)
            ordered_witnesses.extend(new_ordered)
            continue
        break

    selected_set = set(map(int, np.flatnonzero(selected_mask)))
    changed = True
    while changed:
        changed = False
        for candidate_index in sorted(selected_set, reverse=True):
            trial = tuple(sorted(selected_set - {candidate_index}))
            if not _find_zero_response_witnesses(problem, trial):
                selected_set.remove(candidate_index)
                changed = True
    return tuple(sorted(selected_set)), witness_masks


def shortlist_flag_synthesis_problem(
        problem: FlagSynthesisProblem,
        *,
        maximum_candidates: int = 300,
        alternatives_per_witness: int = 3,
) -> FlagSynthesisProblem:
    """Reduce a large candidate pool while retaining a verified feasible seed."""
    if len(problem.candidates) <= maximum_candidates:
        return problem
    seed, witnesses = greedy_verified_candidate_indices(problem)
    retained = set(seed)
    candidate_coverage_counts = [0] * len(problem.candidates)
    candidate_preference_counts = [0] * len(problem.candidates)
    for witness in witnesses:
        covering = [
            index for index, candidate in enumerate(problem.candidates)
            if (candidate.response_mask & witness).bit_count() & 1
        ]
        for index in covering:
            candidate_coverage_counts[index] += 1
        covering.sort(key=lambda index: (
            problem.candidates[index].end_boundary
            - problem.candidates[index].start_boundary,
            problem.candidates[index].interaction_count,
            problem.candidates[index].start_boundary,
            index,
        ))
        for preference, index in enumerate(
                covering[:alternatives_per_witness], start=1):
            candidate_preference_counts[index] += alternatives_per_witness + 1 - preference

    globally_ranked = sorted(
        range(len(problem.candidates)),
        key=lambda index: (
            -candidate_preference_counts[index],
            -candidate_coverage_counts[index],
            problem.candidates[index].end_boundary
            - problem.candidates[index].start_boundary,
            problem.candidates[index].interaction_count,
            index,
        ),
    )
    for index in globally_ranked:
        if len(retained) >= maximum_candidates:
            break
        retained.add(index)
    retained_candidates = tuple(problem.candidates[index] for index in sorted(retained))
    return FlagSynthesisProblem(
        circuit=problem.circuit,
        fault_bundles=problem.fault_bundles,
        malignant_configurations=problem.malignant_configurations,
        candidates=retained_candidates,
        realization_count=problem.realization_count,
        fault_classifications=problem.fault_classifications,
        target_fault_indices=problem.target_fault_indices,
        candidate_pool_restrictions=problem.candidate_pool_restrictions,
    )


def build_solution_from_candidate_indices(
        problem: FlagSynthesisProblem,
        candidate_indices: Iterable[int],
        *,
        elapsed_seconds: float = 0,
) -> FlagCircuitSolution:
    """Schedule an externally selected, realization-verified candidate set."""
    selected = tuple(sorted(set(candidate_indices)))
    witnesses = _find_zero_response_witnesses(problem, selected)
    if witnesses:
        raise FlagSynthesisError(
            f"The proposed selection misses {len(witnesses)} malignant configurations."
        )
    candidate_to_flag = _color_candidate_intervals(problem, selected)
    scheduled = _schedule_selected_interactions(
        problem,
        selected,
        dict(candidate_to_flag),
    )
    layer_count = sum(
        1 + max(item.layer for item in scheduled if item.boundary == boundary)
        for boundary in sorted({item.boundary for item in scheduled})
    ) if scheduled else 0
    flag_count = 1 + max((color for _, color in candidate_to_flag), default=-1)
    return FlagCircuitSolution(
        selected_candidate_indices=selected,
        flag_count=flag_count,
        added_cnot_layers=layer_count,
        interaction_count=sum(
            problem.candidates[index].interaction_count for index in selected
        ),
        candidate_to_flag=candidate_to_flag,
        scheduled_interactions=scheduled,
        objective_order=("flag_qubits", "added_cnot_layers", "flag_cnot_count"),
        optimal_within_candidate_pool=False,
        metrics=SynthesisMetrics(
            candidate_count=len(problem.candidates),
            realization_count=problem.realization_count,
            configuration_count=len(problem.malignant_configurations),
            refinement_rounds=0,
            witness_constraint_count=0,
            elapsed_seconds=elapsed_seconds,
            milp_runs=(),
        ),
    )


def build_flagged_circuit(
        *,
        base_circuit: stim.Circuit,
        problem: FlagSynthesisProblem,
        solution: FlagCircuitSolution,
        first_flag_index: int | None = None,
        boundary_map: Callable[[int], int] | None = None,
        coordinate_column: float = 9,
) -> tuple[stim.Circuit, tuple[int, ...]]:
    """Insert a synthesized selection into a noiseless Stim circuit."""
    if first_flag_index is None:
        first_flag_index = base_circuit.num_qubits
    map_boundary = (lambda boundary: boundary) if boundary_map is None else boundary_map
    flag_indices = tuple(first_flag_index + offset for offset in range(solution.flag_count))
    candidate_to_color = dict(solution.candidate_to_flag)
    selected = solution.selected_candidate_indices
    if any(problem.candidates[index].basis != "Z" for index in selected):
        raise FlagSynthesisError(
            "Synthesized flags must use pure-Z detecting regions."
        )
    last_candidate_end_by_color = {
        color: max(
            problem.candidates[index].end_boundary
            for index in selected if candidate_to_color[index] == color
        )
        for color in range(solution.flag_count)
    }

    remapped_interactions: list[ScheduledInteraction] = []
    for interaction in solution.scheduled_interactions:
        remapped_interactions.append(ScheduledInteraction(
            boundary=map_boundary(interaction.boundary),
            layer=interaction.layer,
            candidate_index=interaction.candidate_index,
            flag_index=first_flag_index + interaction.flag_index,
            data_index=interaction.data_index,
        ))
    interactions_by_boundary: dict[int, list[ScheduledInteraction]] = defaultdict(list)
    for interaction in remapped_interactions:
        interactions_by_boundary[interaction.boundary].append(interaction)

    starts_by_boundary: dict[int, list[int]] = defaultdict(list)
    ends_by_boundary: dict[int, list[int]] = defaultdict(list)
    for candidate_index in selected:
        candidate = problem.candidates[candidate_index]
        starts_by_boundary[map_boundary(candidate.start_boundary)].append(candidate_index)
        ends_by_boundary[map_boundary(candidate.end_boundary)].append(candidate_index)

    coordinate_prefix = stim.Circuit()
    for offset, flag_index in enumerate(flag_indices):
        coordinate_prefix.append(
            "QUBIT_COORDS", [flag_index], [coordinate_column, offset]
        )
    layers = split_by_ticks(base_circuit)
    added_measurements_by_boundary = {
        boundary: len(candidate_indices)
        for boundary, candidate_indices in ends_by_boundary.items()
    }
    layers = _rewrite_original_record_targets(
        layers,
        added_measurements_by_boundary=added_measurements_by_boundary,
    )
    layer_prefixes = [stim.Circuit() for _ in layers]
    layer_suffixes = [stim.Circuit() for _ in layers]
    boundary_zero_preparations = stim.Circuit()
    final_boundary_measurements = stim.Circuit()

    for boundary, candidate_indices in starts_by_boundary.items():
        preparations = (
            boundary_zero_preparations if boundary == 0
            else layer_suffixes[boundary - 1]
        )
        flag_qubits = [
            first_flag_index + candidate_to_color[index]
            for index in sorted(candidate_indices)
        ]
        if flag_qubits:
            preparations.append("R", flag_qubits)

    for boundary, candidate_indices in ends_by_boundary.items():
        measurements = (
            final_boundary_measurements if boundary == len(layers)
            else layer_prefixes[boundary]
        )
        for candidate_index in sorted(candidate_indices):
            color = candidate_to_color[candidate_index]
            flag_index = first_flag_index + color
            is_reused = (
                problem.candidates[candidate_index].end_boundary
                < last_candidate_end_by_color[color]
            )
            measurement = "MR" if is_reused else "M"
            measurements.append(measurement, [flag_index])
            measurements.append(
                "DETECTOR",
                [stim.target_rec(-1)],
                [coordinate_column, color, boundary],
            )

    result = coordinate_prefix
    for boundary in range(len(layers) + 1):
        if boundary == 0 and boundary_zero_preparations:
            result += boundary_zero_preparations
            result.append("TICK")

        boundary_interactions = interactions_by_boundary.get(boundary, ())
        for added_layer in sorted({item.layer for item in boundary_interactions}):
            targets: list[int] = []
            for item in sorted(
                    (i for i in boundary_interactions if i.layer == added_layer),
                    key=lambda i: (i.flag_index, i.data_index)):
                targets.extend((item.data_index, item.flag_index))
            result.append("CX", targets)
            result.append("TICK")

        if boundary == len(layers):
            result += final_boundary_measurements
            continue
        result += layer_prefixes[boundary]
        result += layers[boundary]
        result += layer_suffixes[boundary]
        if boundary < len(layers) - 1:
            result.append("TICK")
    assert_z_basis_flag_construction(result, flag_indices=flag_indices)
    return result, flag_indices


def assert_z_basis_flag_construction(
        circuit: stim.Circuit,
        *,
        flag_indices: Iterable[int],
) -> None:
    """Assert that named flags use only the intended X-hook construction.

    :param circuit: The built Stim circuit containing the flag lifecycles.
    :param flag_indices: Qubits that must be ``|0>``/Z-basis flag targets.
    :return: None.
    """
    flags = set(flag_indices)
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            raise ValueError("Flag synthesis does not support REPEAT blocks.")
        targets = instruction.targets_copy()
        touched_flags = {
            target.qubit_value
            for target in targets
            if target.qubit_value in flags
        }
        if not touched_flags:
            continue
        if instruction.name in {"RX", "MX", "MRX"}:
            raise FlagSynthesisError(
                f"Flag qubits {sorted(touched_flags)} use {instruction.name}."
            )
        if instruction.name == "CX":
            for control, target in zip(targets[::2], targets[1::2], strict=True):
                if control.qubit_value in flags:
                    raise FlagSynthesisError(
                        f"Flag qubit {control.qubit_value} controls a circuit CNOT."
                    )
                if target.qubit_value not in flags:
                    raise FlagSynthesisError(
                        "A synthesized flag CNOT must target a flag qubit."
                    )


def insertion_layer_counts(
        problem: FlagSynthesisProblem,
        solution: FlagCircuitSolution,
) -> dict[int, int]:
    """Return the number of layers inserted at each original boundary."""
    cnot_layers: dict[int, int] = defaultdict(int)
    for interaction in solution.scheduled_interactions:
        cnot_layers[interaction.boundary] = max(
            cnot_layers[interaction.boundary],
            interaction.layer + 1,
        )
    return dict(cnot_layers)


def update_boundary_map_after_insertion(
        *,
        old_boundary_map: Mapping[int, int],
        insertion_counts: Mapping[int, int],
) -> dict[int, int]:
    """Update an inner-to-full boundary map after parallel insertions.

    The same inserted flag layers are assumed to have been placed at boundary
    ``b`` of the source and boundary ``old_boundary_map[b]`` of the target.
    New boundaries inside an inserted block are mapped one-for-one.
    """
    source_boundaries = sorted(old_boundary_map)
    source_inserted_before = 0
    result: dict[int, int] = {}
    target_insertions = sorted(
        (old_boundary_map[boundary], count)
        for boundary, count in insertion_counts.items()
    )
    for boundary in source_boundaries:
        target_boundary = old_boundary_map[boundary]
        target_inserted_before = sum(
            count for insertion_boundary, count in target_insertions
            if insertion_boundary < target_boundary
        )
        source_start = boundary + source_inserted_before
        target_start = target_boundary + target_inserted_before
        count = insertion_counts.get(boundary, 0)
        for offset in range(count + 1):
            result[source_start + offset] = target_start + offset
        source_inserted_before += count
    return result


def assert_flag_detectors_deterministic(
        circuit: stim.Circuit,
        *,
        coordinate_prefix: tuple[float, ...] = (9,),
        expected_count: int | None = None,
) -> None:
    """Use Stim's reverse detector-flow analysis on newly added detectors."""
    filtered = stim.Circuit()
    matching_count = 0
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            raise ValueError("Flag synthesis does not support REPEAT blocks.")
        if instruction.name != "DETECTOR":
            filtered.append(instruction)
            continue
        coordinates = instruction.gate_args_copy()
        if tuple(coordinates[:len(coordinate_prefix)]) == coordinate_prefix:
            filtered.append(instruction)
            matching_count += 1
    if expected_count is not None and matching_count != expected_count:
        raise FlagSynthesisError(
            f"Expected {expected_count} flag detectors at {coordinate_prefix}, "
            f"found {matching_count}."
        )
    targets = [
        stim.target_relative_detector_id(index)
        for index in range(matching_count)
    ]
    regions = filtered.detecting_regions(targets=targets)
    missing = [target for target in targets if target not in regions]
    if missing:
        raise FlagSynthesisError(f"New flag detectors are nondeterministic: {missing}")


def verify_flag_solution(
        *,
        circuit: stim.Circuit,
        logical_analyzer,
        max_order: int = 4,
        cultivated_state: str = "T",
        noise_level: float = 1e-3,
        print_progress: bool = False,
        maximum_malignant_configurations: int | None = None,
) -> VerificationResult:
    """Re-enumerate a noisy flagged circuit and return malignant faults."""
    from cliffordep import noise

    started = time.monotonic()
    noisy = noise.uniformly_depolarize(circuit, noise_level=noise_level)
    combinator = FaultCombinator(noisy, print_progress=print_progress)
    malignant = find_malignant_configurations(
        combinator=combinator,
        logical_analyzer=logical_analyzer,
        max_order=max_order,
        cultivated_state=cultivated_state,
        maximum_configurations=maximum_malignant_configurations,
        print_progress=print_progress,
    )
    return VerificationResult(
        max_order=max_order,
        cultivated_state=cultivated_state,
        malignant_configurations=malignant,
        elapsed_seconds=time.monotonic() - started,
    )


def find_malignant_configurations(
        *,
        combinator: FaultCombinator,
        logical_analyzer,
        max_order: int,
        cultivated_state: str = "T",
        maximum_configurations: int | None = None,
        print_progress: bool = False,
) -> tuple[tuple[int, ...], ...]:
    """Stream postselectable configurations and retain logical failures.

    Unlike :meth:`FaultCombinator.get_kept_effects`, this verifier can stop
    after a requested batch of counterexamples instead of materializing every
    accepted effect.

    :param combinator: The indexed circuit faults to combine.
    :param logical_analyzer: The analyzer supplying linear prechecks and
        logical-state classifications.
    :param max_order: The largest configuration weight to inspect.
    :param cultivated_state: The logical state whose failures are retained.
    :param maximum_configurations: The optional counterexample batch size.
    :param print_progress: Whether to report enumeration progress by order.
    :return: The malignant configurations ordered by weight and fault indices.
    """
    from cliffordep.combinators.fault_combinator import (
        _iter_zero_syndrome_configurations,
        _make_logical_analysis_cache,
        _make_pauli_mask_restrictor,
    )

    restrict_effect = _make_pauli_mask_restrictor(
        data_indices=logical_analyzer.DATA_INDICES,
        source_qubit_count=combinator.circuit.noisy_circuit.num_qubits,
    )
    detector_count = combinator.circuit.noisy_circuit.num_detectors
    extended_syndromes: list[int] = []
    data_effects: list[int] = []
    for circuit_syndrome, full_effect in combinator.indexed_faults:
        data_effect = restrict_effect(full_effect)
        precheck_syndrome = logical_analyzer.linear_precheck_syndrome(
            data_effect,
        )
        extended_syndromes.append(
            circuit_syndrome | (precheck_syndrome << detector_count)
        )
        data_effects.append(data_effect)
    analyze = _make_logical_analysis_cache(
        logical_analyzer=logical_analyzer,
        cultivated_states=(cultivated_state,),
        maxsize=262_144,
    )

    malignant: list[tuple[int, ...]] = []
    for order in range(max_order + 1):
        visited = 0
        for data_effect, fault_indices in _iter_zero_syndrome_configurations(
                syndromes=extended_syndromes,
                effects=data_effects,
                order=order):
            visited += 1
            ((accept_probability, logical_fidelity),) = analyze(data_effect)
            if accept_probability and not math.isclose(logical_fidelity, 1.0):
                malignant.append(fault_indices)
                if (
                    maximum_configurations is not None
                    and len(malignant) >= maximum_configurations
                ):
                    analyze.cache_clear()
                    return tuple(sorted(malignant, key=lambda item: (len(item), item)))
        if print_progress:
            print(
                f"    order {order}: visited {visited} undetected configurations; "
                f"found {len(malignant)} malignant configurations so far.",
                flush=True,
            )
    analyze.cache_clear()
    return tuple(sorted(malignant, key=lambda item: (len(item), item)))


def write_solution_artifacts(
        *,
        directory: str | Path,
        solution_id: str,
        problem: FlagSynthesisProblem,
        solution: FlagCircuitSolution,
        inner_circuit: stim.Circuit,
        full_circuit: stim.Circuit | None = None,
        verification: VerificationResult | None = None,
) -> dict[str, Path]:
    """Write Stim circuits and a reproducibility manifest."""
    output_directory = Path(directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    inner_path = output_directory / f"d5a19_{solution_id}.inner.stim"
    manifest_path = output_directory / f"d5a19_{solution_id}.json"
    inner_circuit.to_file(inner_path)
    paths = {"inner_circuit": inner_path, "manifest": manifest_path}
    if full_circuit is not None:
        full_path = output_directory / f"d5a19_{solution_id}.full.stim"
        full_circuit.to_file(full_path)
        paths["full_circuit"] = full_path

    selected_candidates = [
        problem.candidates[index] for index in solution.selected_candidate_indices
    ]
    manifest = {
        "schema_version": 1,
        "solution_id": solution_id,
        "objective": list(solution.objective_order),
        "optimality_scope": "finite_candidate_pool",
        "globally_optimal": False,
        "flag_count": solution.flag_count,
        "added_cnot_layers": solution.added_cnot_layers,
        "flag_cnot_count": solution.interaction_count,
        "optimal_within_candidate_pool": solution.optimal_within_candidate_pool,
        "candidate_pool_restrictions": (
            None
            if problem.candidate_pool_restrictions is None
            else asdict(problem.candidate_pool_restrictions)
        ),
        "selected_candidates": [
            {
                "index": index,
                "identifier": candidate.identifier,
                "basis": candidate.basis,
                "interactions": candidate.interactions,
                "source": candidate.source,
            }
            for index, candidate in zip(
                solution.selected_candidate_indices,
                selected_candidates,
                strict=True,
            )
        ],
        "candidate_to_flag": solution.candidate_to_flag,
        "metrics": asdict(solution.metrics),
        "verified_through_order": (
            verification.max_order
            if verification is not None and verification.passed
            else None
        ),
        "verification": None if verification is None else asdict(verification),
        "files": {name: path.name for name, path in paths.items() if name != "manifest"},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return paths


def _rewrite_original_record_targets(
        layers: Sequence[stim.Circuit],
        *,
        added_measurements_by_boundary: Mapping[int, int],
) -> list[stim.Circuit]:
    """Keep original rec targets stable after inserting flag measurements."""
    original_to_new_measurement: dict[int, int] = {}
    original_measurement_count = 0
    added_measurement_count = 0
    for layer_index, layer in enumerate(layers):
        added_measurement_count += added_measurements_by_boundary.get(layer_index, 0)
        for instruction in layer:
            if isinstance(instruction, stim.CircuitRepeatBlock):
                raise ValueError("Flag synthesis does not support REPEAT blocks.")
            for offset in range(instruction.num_measurements):
                original_index = original_measurement_count + offset
                original_to_new_measurement[original_index] = (
                    original_index + added_measurement_count
                )
            original_measurement_count += instruction.num_measurements

    rewritten_layers: list[stim.Circuit] = []
    original_measurement_count = 0
    added_measurement_count = 0
    for layer_index, layer in enumerate(layers):
        added_measurement_count += added_measurements_by_boundary.get(layer_index, 0)
        rewritten = stim.Circuit()
        for instruction in layer:
            if isinstance(instruction, stim.CircuitRepeatBlock):
                raise ValueError("Flag synthesis does not support REPEAT blocks.")
            targets = instruction.targets_copy()
            if any(target.is_measurement_record_target for target in targets):
                new_current_count = original_measurement_count + added_measurement_count
                targets = [
                    stim.target_rec(
                        original_to_new_measurement[
                            original_measurement_count + target.value
                        ] - new_current_count
                    ) if target.is_measurement_record_target else target
                    for target in targets
                ]
            rewritten.append(
                instruction.name,
                targets,
                instruction.gate_args_copy(),
            )
            original_measurement_count += instruction.num_measurements
        rewritten_layers.append(rewritten)
    return rewritten_layers


def _make_fault_trajectory(
        circuit: stim.Circuit,
        event: ErrorEvent,
) -> tuple[SymplecticPauli, ...]:
    layers = split_by_ticks(circuit)
    timeslice, name, targets = event
    states = [SymplecticPauli() for _ in range(len(layers) + 1)]
    pauli = stim.PauliString(circuit.num_qubits)
    if name == "E":
        for target in targets:
            pauli[target.value] = target.pauli_type
    elif not name.startswith("M"):
        target, = targets
        pauli[target.value] = name[0]
    states[timeslice + 1] = _pauli_to_symplectic(pauli)
    for layer_index in range(timeslice + 1, len(layers)):
        pauli = _propagate_error_through_layer(pauli, layers[layer_index])
        states[layer_index + 1] = _pauli_to_symplectic(pauli)
    return tuple(states)


def _propagate_error_through_layer(
        pauli: stim.PauliString,
        layer: stim.Circuit,
) -> stim.PauliString:
    result = pauli
    for instruction in layer:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            raise ValueError("Flag synthesis does not support REPEAT blocks.")
        data = stim.gate_data(instruction.name)
        if data.is_unitary:
            result = result.after(instruction)
        if data.is_reset:
            for target in instruction.targets_copy():
                if target.qubit_value is not None:
                    result[target.qubit_value] = "I"
    return result


def precompute_x_trajectory_masks(
        realizations: Sequence[PhysicalFaultRealization],
        *,
        boundary_count: int,
        qubit_count: int,
) -> tuple[tuple[int, ...], ...]:
    """Index realization X support by circuit boundary and qubit.

    :param realizations: Problem-local physical trajectories with dense ordinals.
    :param boundary_count: Number of spacetime boundaries in each trajectory.
    :param qubit_count: Number of circuit qubits represented by each trajectory.
    :return: Boundary-major masks whose bits identify responding realizations.
    """
    masks = [[0] * qubit_count for _ in range(boundary_count)]
    for realization in realizations:
        realization_bit = 1 << realization.ordinal
        for boundary, pauli in enumerate(realization.trajectory):
            x_support = pauli.x
            while x_support:
                least_bit = x_support & -x_support
                qubit = least_bit.bit_length() - 1
                masks[boundary][qubit] |= realization_bit
                x_support ^= least_bit
    return tuple(tuple(row) for row in masks)


def candidate_response_mask(
        interactions: Sequence[tuple[int, Sequence[int]]],
        x_trajectory_masks: Sequence[Sequence[int]],
) -> int:
    """Compute a pure-Z candidate response by XORing endpoint term masks.

    :param interactions: Detecting-region boundary terms and qubit supports.
    :param x_trajectory_masks: Precomputed X-trajectory masks by boundary/qubit.
    :return: The realization bitset with odd binary symplectic response.
    """
    response_mask = 0
    for boundary, support in interactions:
        for qubit in support:
            response_mask ^= x_trajectory_masks[boundary][qubit]
    return response_mask


def _generate_hook_candidates(
        *,
        circuit: stim.Circuit,
        fault_bundles: Sequence[FaultBundle],
        target_fault_indices: Sequence[int],
        maximum_interactions: int,
        maximum_duration: int,
        minimum_duration: int,
        nearby_boundary_radius: int,
        include_single_seed_fallback: bool,
        boundary_range: tuple[int, int],
) -> tuple[FlagCandidate, ...]:
    """Generate staged pure-Z paths directed by target hook trajectories.

    :param circuit: The noiseless circuit whose boundaries paths traverse.
    :param fault_bundles: All target and nuisance realization bundles.
    :param target_fault_indices: Multi-X faults allowed to seed candidates.
    :param maximum_interactions: Maximum endpoint support size in total.
    :param maximum_duration: Maximum lifecycle span in circuit boundaries.
    :param minimum_duration: Minimum lifecycle span in circuit boundaries.
    :param nearby_boundary_radius: Direct-envelope endpoint variation radius.
    :param include_single_seed_fallback: Whether to add general Z path seeds.
    :param boundary_range: Inclusive usable start/end boundary range.
    :return: The deduplicated Z-only candidate pool in stable order.
    """
    layers = split_by_ticks(circuit)
    trajectories = tuple(
        realization
        for bundle in fault_bundles
        for realization in bundle.realizations
    )
    target_fault_set = set(target_fault_indices)
    target_bundles = tuple(
        bundle for bundle in fault_bundles
        if bundle.fault_index in target_fault_set
    )
    target_realization_mask = sum(
        1 << realization.ordinal
        for bundle in target_bundles
        for realization in bundle.realizations
    )
    x_trajectory_masks = precompute_x_trajectory_masks(
        trajectories,
        boundary_count=len(layers) + 1,
        qubit_count=circuit.num_qubits,
    )
    minimum_boundary, maximum_boundary = boundary_range
    raw: dict[
        tuple[tuple[int, tuple[int, ...]], ...],
        tuple[CandidateSource, int],
    ] = {}
    source_priority = {
        "direct-envelope": 0,
        "nearby-envelope": 1,
        "single-seed-fallback": 2,
    }

    def add_interactions(
            interactions: tuple[tuple[int, tuple[int, ...]], ...],
            source: CandidateSource,
    ) -> None:
        """Retain a bounded path that responds to at least one hook realization.

        :param interactions: The two pure-Z endpoint support terms.
        :param source: The staged family that produced the path.
        :return: None.
        """
        start = interactions[0][0]
        end = interactions[-1][0]
        duration = end - start
        if not minimum_duration <= duration <= maximum_duration:
            return
        if sum(len(support) for _, support in interactions) > maximum_interactions:
            return
        response_mask = candidate_response_mask(interactions, x_trajectory_masks)
        if not response_mask & target_realization_mask:
            return
        previous = raw.get(interactions)
        if previous is None or source_priority[source] < source_priority[previous[0]]:
            raw[interactions] = source, response_mask

    for bundle in target_bundles:
        hook_boundaries = []
        for realization in bundle.realizations:
            hook_boundaries.append(next(
                boundary
                for boundary, pauli in enumerate(realization.trajectory)
                if pauli.x.bit_count() > 1
            ))
        direct_start = max(
            minimum_boundary,
            min(realization.event[0] for realization in bundle.realizations),
        )
        direct_end = min(maximum_boundary, max(hook_boundaries))
        bundle_realization_mask = sum(
            1 << realization.ordinal for realization in bundle.realizations
        )
        for start_shift in range(-nearby_boundary_radius, nearby_boundary_radius + 1):
            for end_shift in range(-nearby_boundary_radius, nearby_boundary_radius + 1):
                start = direct_start + start_shift
                end = direct_end + end_shift
                if not minimum_boundary <= start < end <= maximum_boundary:
                    continue
                source: CandidateSource = (
                    "direct-envelope"
                    if start_shift == end_shift == 0
                    else "nearby-envelope"
                )
                for end_qubit in range(circuit.num_qubits):
                    if not x_trajectory_masks[end][end_qubit] & bundle_realization_mask:
                        continue
                    pauli = _single_basis_pauli(
                        circuit.num_qubits,
                        end_qubit,
                        "Z",
                    )
                    valid = True
                    for layer_index in range(end - 1, start - 1, -1):
                        propagated = _propagate_region_through_layer(
                            pauli,
                            layers[layer_index],
                            forwards=False,
                        )
                        if propagated is None:
                            valid = False
                            break
                        pauli = propagated
                    if not valid:
                        continue
                    start_support = _pure_basis_support(pauli, "Z")
                    if start_support is not None:
                        add_interactions(
                            ((start, start_support), (end, (end_qubit,))),
                            source,
                        )

    if include_single_seed_fallback:
        for start in range(minimum_boundary, maximum_boundary):
            maximum_end = min(maximum_boundary, start + maximum_duration)
            for qubit in range(circuit.num_qubits):
                pauli = _single_basis_pauli(circuit.num_qubits, qubit, "Z")
                for end in range(start + 1, maximum_end + 1):
                    propagated = _propagate_region_through_layer(
                        pauli,
                        layers[end - 1],
                        forwards=True,
                    )
                    if propagated is None:
                        break
                    pauli = propagated
                    end_support = _pure_basis_support(pauli, "Z")
                    if end_support is not None:
                        add_interactions(
                            ((start, (qubit,)), (end, end_support)),
                            "single-seed-fallback",
                        )

        for end in range(minimum_boundary + 1, maximum_boundary + 1):
            minimum_start = max(minimum_boundary, end - maximum_duration)
            for qubit in range(circuit.num_qubits):
                pauli = _single_basis_pauli(circuit.num_qubits, qubit, "Z")
                for start in range(end - 1, minimum_start - 1, -1):
                    propagated = _propagate_region_through_layer(
                        pauli,
                        layers[start],
                        forwards=False,
                    )
                    if propagated is None:
                        break
                    pauli = propagated
                    start_support = _pure_basis_support(pauli, "Z")
                    if start_support is not None:
                        add_interactions(
                            ((start, start_support), (end, (qubit,))),
                            "single-seed-fallback",
                        )

    return tuple(
        FlagCandidate(
            identifier=_candidate_identifier("Z", interactions),
            basis="Z",
            interactions=interactions,
            response_mask=response_mask,
            source=source,
        )
        for interactions, (source, response_mask) in sorted(raw.items())
    )


def _candidate_response_mask(
        basis: PauliBasis,
        interactions: tuple[tuple[int, tuple[int, ...]], ...],
        realizations: Sequence[PhysicalFaultRealization],
) -> int:
    mask = 0
    for realization in realizations:
        response = False
        for boundary, support in interactions:
            term_mask = sum(1 << qubit for qubit in support)
            term = SymplecticPauli(
                x=term_mask if basis == "X" else 0,
                z=term_mask if basis == "Z" else 0,
            )
            response ^= realization.trajectory[boundary].anticommutes(term)
        if response:
            mask |= 1 << realization.ordinal
    return mask


def _propagate_region_through_layer(
        pauli: stim.PauliString,
        layer: stim.Circuit,
        *,
        forwards: bool,
) -> stim.PauliString | None:
    instructions = tuple(layer)
    if not forwards:
        instructions = tuple(reversed(instructions))
    result = pauli
    for instruction in instructions:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            raise ValueError("Flag synthesis does not support REPEAT blocks.")
        data = stim.gate_data(instruction.name)
        if data.is_unitary:
            result = result.after(instruction) if forwards else result.before(instruction)
            continue
        if data.produces_measurements or data.is_reset:
            touched = {
                target.qubit_value for target in instruction.targets_copy()
                if target.qubit_value is not None
            }
            if any(result[index] for index in touched):
                return None
    return result


def _solve_milp(
        problem: FlagSynthesisProblem,
        witness_masks: set[int],
        *,
        objective: ObjectiveName,
        limits: SynthesisLimits,
        run_started: float,
        fixed_flag_count: int | None = None,
        fixed_layer_count: int | None = None,
) -> tuple[np.ndarray, MilpRun]:
    candidate_count = len(problem.candidates)
    boundary_count = len(split_by_ticks(problem.circuit)) + 1
    variable_count = candidate_count + 1 + boundary_count
    q_index = candidate_count
    layer_offset = candidate_count + 1

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_row(coefficients: Iterable[tuple[int, float]], lb: float, ub: float) -> None:
        row = len(lower)
        for column, value in coefficients:
            rows.append(row)
            columns.append(column)
            values.append(value)
        lower.append(lb)
        upper.append(ub)

    for witness_mask in sorted(witness_masks):
        covering = [
            index for index, candidate in enumerate(problem.candidates)
            if (candidate.response_mask & witness_mask).bit_count() & 1
        ]
        if not covering:
            raise FlagSynthesisError(
                "The candidate pool cannot detect a physical realization of a "
                "malignant configuration."
            )
        add_row(((index, 1) for index in covering), 1, np.inf)

    for boundary in range(boundary_count):
        active = [
            index for index, candidate in enumerate(problem.candidates)
            if candidate.start_boundary - 1 <= boundary <= candidate.end_boundary
        ]
        if active:
            add_row(
                (*((index, 1) for index in active), (q_index, -1)),
                -np.inf,
                0,
            )

    endpoints_by_boundary: dict[int, list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
    for candidate_index, candidate in enumerate(problem.candidates):
        for boundary, support in candidate.interactions:
            endpoints_by_boundary[boundary].append((candidate_index, support))

    for boundary, endpoints in endpoints_by_boundary.items():
        layer_index = layer_offset + boundary
        by_data: dict[int, list[int]] = defaultdict(list)
        for candidate_index, support in endpoints:
            add_row(
                ((candidate_index, len(support)), (layer_index, -1)),
                -np.inf,
                0,
            )
            for qubit in support:
                by_data[qubit].append(candidate_index)
        for candidate_indices in by_data.values():
            add_row(
                (*((index, 1) for index in candidate_indices), (layer_index, -1)),
                -np.inf,
                0,
            )

    if fixed_layer_count is not None:
        add_row(
            ((layer_offset + boundary, 1) for boundary in range(boundary_count)),
            fixed_layer_count,
            fixed_layer_count,
        )

    matrix = coo_array(
        (values, (rows, columns)),
        shape=(len(lower), variable_count),
    ).tocsc()
    constraint = LinearConstraint(matrix, np.asarray(lower), np.asarray(upper))
    lower_bounds = np.zeros(variable_count)
    upper_bounds = np.full(variable_count, np.inf)
    upper_bounds[:candidate_count] = 1
    upper_bounds[q_index] = candidate_count
    if fixed_flag_count is not None:
        lower_bounds[q_index] = fixed_flag_count
        upper_bounds[q_index] = fixed_flag_count
    objective_vector = np.zeros(variable_count)
    if objective == "flag_qubits":
        objective_vector[q_index] = 1
    elif objective == "added_cnot_layers":
        objective_vector[layer_offset:] = 1
    elif objective == "flag_cnot_count":
        objective_vector[:candidate_count] = [
            candidate.interaction_count for candidate in problem.candidates
        ]
    else:
        raise ValueError(objective)

    remaining = limits.total_time_limit_seconds - (time.monotonic() - run_started)
    if remaining <= 0:
        raise TimeoutError("Flag synthesis reached its total wall-clock limit.")
    solve_limit = min(limits.milp_time_limit_seconds, remaining)
    solve_started = time.monotonic()
    result = milp(
        c=objective_vector,
        integrality=np.ones(variable_count),
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=constraint,
        options={
            "time_limit": solve_limit,
            "mip_rel_gap": limits.mip_relative_gap,
            "presolve": True,
        },
    )
    elapsed = time.monotonic() - solve_started
    run = MilpRun(
        objective=objective,
        elapsed_seconds=elapsed,
        status=int(result.status),
        message=str(result.message),
        node_count=(None if getattr(result, "mip_node_count", None) is None
                    else int(result.mip_node_count)),
        mip_gap=(None if getattr(result, "mip_gap", None) is None
                 else float(result.mip_gap)),
        objective_value=None if result.fun is None else float(result.fun),
    )
    if result.x is None:
        raise FlagSynthesisError(f"HiGHS did not return a feasible solution: {result.message}")
    return np.asarray(result.x), run


def _find_zero_response_witnesses(
        problem: FlagSynthesisProblem,
        selected_candidate_indices: Sequence[int],
) -> set[int]:
    witnesses: set[int] = set()
    for configuration in problem.malignant_configurations:
        states: dict[int, int] = {0: 0}
        for fault_index in configuration:
            classes = problem.refine_fault(fault_index, selected_candidate_indices)
            next_states: dict[int, int] = {}
            for accumulated_signature, accumulated_mask in states.items():
                for realization_class in classes:
                    resultant = accumulated_signature ^ realization_class.response_signature
                    next_states.setdefault(
                        resultant,
                        accumulated_mask | (1 << realization_class.realization_ordinals[0]),
                    )
            states = next_states
        if 0 in states:
            witnesses.add(states[0])
    return witnesses


def _color_candidate_intervals(
        problem: FlagSynthesisProblem,
        selected_candidate_indices: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    available: list[int] = []
    active: list[tuple[int, int]] = []
    next_color = 0
    result: list[tuple[int, int]] = []
    ordered = sorted(
        selected_candidate_indices,
        key=lambda index: (
            problem.candidates[index].start_boundary,
            problem.candidates[index].end_boundary,
            index,
        ),
    )
    for candidate_index in ordered:
        candidate = problem.candidates[candidate_index]
        while active and active[0][0] + 1 < candidate.start_boundary:
            _, color = heapq.heappop(active)
            heapq.heappush(available, color)
        if available:
            color = heapq.heappop(available)
        else:
            color = next_color
            next_color += 1
        result.append((candidate_index, color))
        heapq.heappush(active, (candidate.end_boundary, color))
    return tuple(sorted(result))


def _schedule_selected_interactions(
        problem: FlagSynthesisProblem,
        selected_candidate_indices: Sequence[int],
        candidate_to_flag: Mapping[int, int],
) -> tuple[ScheduledInteraction, ...]:
    endpoint_edges: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for candidate_index in selected_candidate_indices:
        candidate = problem.candidates[candidate_index]
        flag = candidate_to_flag[candidate_index]
        for boundary, support in candidate.interactions:
            for data in support:
                endpoint_edges[boundary].append((flag, data, candidate_index))

    scheduled: list[ScheduledInteraction] = []
    for boundary, edges in sorted(endpoint_edges.items()):
        colors = _edge_color_bipartite(edges)
        for (flag, data, candidate_index), layer in zip(edges, colors, strict=True):
            scheduled.append(ScheduledInteraction(
                boundary=boundary,
                layer=layer,
                candidate_index=candidate_index,
                flag_index=flag,
                data_index=data,
            ))
    return tuple(sorted(
        scheduled,
        key=lambda item: (item.boundary, item.layer, item.flag_index, item.data_index),
    ))


def _edge_color_bipartite(edges: Sequence[tuple[int, int, int]]) -> tuple[int, ...]:
    """Optimally edge-color a bipartite graph by regular completion."""
    if not edges:
        return ()
    left_values = sorted({left for left, _, _ in edges})
    right_values = sorted({right for _, right, _ in edges})
    size = max(len(left_values), len(right_values))
    left_index = {value: index for index, value in enumerate(left_values)}
    right_index = {value: index for index, value in enumerate(right_values)}
    counts = np.zeros((size, size), dtype=int)
    real_edges: dict[tuple[int, int], deque[int]] = defaultdict(deque)
    for edge_index, (left, right, _) in enumerate(edges):
        pair = (left_index[left], right_index[right])
        counts[pair] += 1
        real_edges[pair].append(edge_index)
    degree = max(
        int(counts.sum(axis=0).max(initial=0)),
        int(counts.sum(axis=1).max(initial=0)),
    )
    left_deficit = degree - counts.sum(axis=1)
    right_deficit = degree - counts.sum(axis=0)
    left_cursor = right_cursor = 0
    while left_cursor < size and right_cursor < size:
        while left_cursor < size and left_deficit[left_cursor] == 0:
            left_cursor += 1
        while right_cursor < size and right_deficit[right_cursor] == 0:
            right_cursor += 1
        if left_cursor == size or right_cursor == size:
            break
        amount = min(left_deficit[left_cursor], right_deficit[right_cursor])
        counts[left_cursor, right_cursor] += amount
        left_deficit[left_cursor] -= amount
        right_deficit[right_cursor] -= amount
    if np.any(counts.sum(axis=0) != degree) or np.any(counts.sum(axis=1) != degree):
        raise AssertionError("Failed to complete bipartite graph to regularity.")

    colors = [-1] * len(edges)
    for color in range(degree):
        cost = np.where(counts > 0, 0, 1)
        left_matching, right_matching = linear_sum_assignment(cost)
        if np.any(cost[left_matching, right_matching]):
            raise AssertionError("Regular bipartite multigraph has no perfect matching.")
        for left, right in zip(left_matching, right_matching, strict=True):
            counts[left, right] -= 1
            queue = real_edges.get((int(left), int(right)))
            if queue:
                colors[queue.popleft()] = color
    if any(color < 0 for color in colors):
        raise AssertionError("A real edge was not assigned a layer.")
    return tuple(colors)


def _canonical_witness_mask(
        problem: FlagSynthesisProblem,
        configuration: Sequence[int],
) -> int:
    mask = 0
    for fault_index in configuration:
        mask |= 1 << problem.bundle(fault_index).realizations[0].ordinal
    return mask


def _selected_candidates(vector: np.ndarray, candidate_count: int) -> tuple[int, ...]:
    return tuple(index for index in range(candidate_count) if vector[index] > 0.5)


def _interval_overlap(
        problem: FlagSynthesisProblem,
        candidate_indices: Iterable[int],
) -> int:
    indices = tuple(candidate_indices)
    if not indices:
        return 0
    return max(
        sum(
            problem.candidates[index].start_boundary - 1 <= boundary
            <= problem.candidates[index].end_boundary
            for index in indices
        )
        for boundary in range(len(split_by_ticks(problem.circuit)) + 1)
    )


def _check_total_time(started: float, limits: SynthesisLimits) -> None:
    if time.monotonic() - started >= limits.total_time_limit_seconds:
        raise TimeoutError("Flag synthesis reached its total wall-clock limit.")


def _single_basis_pauli(
        qubit_count: int,
        qubit: int,
        basis: PauliBasis,
) -> stim.PauliString:
    pauli = stim.PauliString(qubit_count)
    pauli[qubit] = basis
    return pauli


def _pure_basis_support(
        pauli: stim.PauliString,
        basis: PauliBasis,
) -> tuple[int, ...] | None:
    support = []
    expected = 1 if basis == "X" else 3
    for index, value in enumerate(pauli):
        if value == 0:
            continue
        if value != expected:
            return None
        support.append(index)
    return tuple(support) if support else None


def _pauli_to_symplectic(pauli: stim.PauliString) -> SymplecticPauli:
    x = z = 0
    for index, value in enumerate(pauli):
        if value in (1, 2):
            x |= 1 << index
        if value in (2, 3):
            z |= 1 << index
    return SymplecticPauli(x, z)


def _candidate_identifier(
        basis: PauliBasis,
        interactions: tuple[tuple[int, tuple[int, ...]], ...],
) -> str:
    terms = "__".join(
        f"b{boundary:02d}q{'-'.join(map(str, support))}"
        for boundary, support in interactions
    )
    return f"path_{basis}_{terms}"


def _error_event_key(event: ErrorEvent) -> tuple:
    timeslice, name, targets = event
    return (
        timeslice,
        name,
        tuple((target.value, target.pauli_type) for target in targets),
    )


__all__ = [
    "CandidatePoolRestrictions",
    "FaultBundle",
    "FaultEffectClassification",
    "FlagCandidate",
    "FlagCircuitSolution",
    "FlagSynthesisError",
    "FlagSynthesisProblem",
    "MilpRun",
    "PhysicalFaultRealization",
    "RealizationClass",
    "ScheduledInteraction",
    "SymplecticPauli",
    "SynthesisLimits",
    "SynthesisMetrics",
    "VerificationResult",
    "assert_z_basis_flag_construction",
    "build_flag_synthesis_problem",
    "build_flagged_circuit",
    "build_solution_from_candidate_indices",
    "assert_flag_detectors_deterministic",
    "candidate_response_mask",
    "classify_fault_effect",
    "extract_malignant_configurations",
    "find_malignant_configurations",
    "greedy_verified_candidate_indices",
    "insertion_layer_counts",
    "precompute_x_trajectory_masks",
    "synthesize_flag_circuit",
    "shortlist_flag_synthesis_problem",
    "update_boundary_map_after_insertion",
    "verify_flag_solution",
    "write_solution_artifacts",
]
