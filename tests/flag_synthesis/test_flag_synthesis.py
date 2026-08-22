import itertools

import numpy as np
import pytest
import stim

from cliffordep import circuits, noise
import cliffordep.flag_synthesis as flag_synthesis_module
from cliffordep.circuits import distance_5_circuits
from cliffordep.combinators import FaultCombinator
from cliffordep.flag_synthesis import (
    FaultBundle,
    FlagCandidate,
    FlagCircuitSolution,
    FlagSynthesisError,
    FlagSynthesisProblem,
    PhysicalFaultRealization,
    ScheduledInteraction,
    SymplecticPauli,
    SynthesisLimits,
    SynthesisMetrics,
    assert_flag_detectors_deterministic,
    assert_z_basis_flag_construction,
    build_flag_synthesis_problem,
    build_flagged_circuit,
    candidate_response_mask,
    classify_fault_effect,
    extract_malignant_configurations,
    find_flag_lifecycles,
    find_malignant_configurations,
    greedily_prune_flag_lifecycles,
    precompute_x_trajectory_masks,
    remove_flag_lifecycle,
    shortlist_flag_synthesis_problem,
    synthesize_flag_circuit,
    verify_flag_solution,
)
from cliffordep.flag_synthesis import _candidate_response_mask
from cliffordep.flag_synthesis import _color_candidate_intervals
from cliffordep.flag_synthesis import _edge_color_bipartite
from cliffordep.flag_synthesis import _find_zero_response_witnesses
from cliffordep.flag_synthesis import _rewrite_original_record_targets
from cliffordep.logical_analyzers import CliffordLogicalAnalyzer


def _realization(
        ordinal: int,
        trajectory: tuple[SymplecticPauli, ...] = (SymplecticPauli(),),
) -> PhysicalFaultRealization:
    """Build a compact synthetic physical realization for parity tests.

    :param ordinal: The problem-local realization bit position.
    :param trajectory: The symplectic state at each circuit boundary.
    :return: A deterministic synthetic X error-event realization.
    """
    return PhysicalFaultRealization(
        ordinal=ordinal,
        event=(0, "X_ERROR", (stim.GateTarget(0),)),
        trajectory=trajectory,
    )


@pytest.fixture(scope="module")
def d3_synthesis_inputs():
    """Share the enumerated D3 failures across focused integration tests.

    :return: The unflagged D3 double-check circuit wrapper.
    :return: Its Clifford logical analyzer.
    :return: Its mask-native abstract-fault combinator.
    :return: Its four malignant order-two configurations.
    """
    circuit = circuits.D3A6()
    analyzer = CliffordLogicalAnalyzer(
        data_indices=circuit.DATA_INDICES,
        stabilizer_generators=circuit.STABILIZER_GENERATORS_RESTRICTED,
        logical_s=circuit.LOGICAL_S,
    )
    combinator = FaultCombinator(noise.uniformly_depolarize(
        circuit.INNER_CIRCUIT,
        noise_level=1e-3,
    ))
    malignant = find_malignant_configurations(
        combinator=combinator,
        logical_analyzer=analyzer,
        max_order=2,
        cultivated_state="T",
    )
    return circuit, analyzer, combinator, malignant


def test_symplectic_inner_product():
    """Binary symplectic products detect only odd crossed X/Z overlap."""
    x0 = SymplecticPauli(x=1)
    z0 = SymplecticPauli(z=1)
    x1 = SymplecticPauli(x=2)

    assert x0.anticommutes(z0)
    assert z0.anticommutes(x0)
    assert not x0.anticommutes(x1)
    assert not x0.anticommutes(x0)


def test_fault_bundle_is_refined_only_when_candidate_responses_differ():
    """Lazy classes split an abstract bundle only on selected signatures."""
    problem = FlagSynthesisProblem(
        circuit=stim.Circuit(),
        fault_bundles=(FaultBundle(7, (_realization(0), _realization(1))),),
        malignant_configurations=((7,),),
        candidates=(
            FlagCandidate("uniform", "Z", ((0, (0,)), (1, (0,))), 0b11),
            FlagCandidate("splitting", "Z", ((0, (0,)), (1, (0,))), 0b01),
        ),
        realization_count=2,
    )

    uniform_classes = problem.refine_fault(7, (0,))
    split_classes = problem.refine_fault(7, (0, 1))

    assert len(uniform_classes) == 1
    assert uniform_classes[0].realization_ordinals == (0, 1)
    assert {item.realization_ordinals for item in split_classes} == {(0,), (1,)}


def test_lazy_separator_covers_every_physical_realization_product():
    """Synthesis adds witnesses until every realization product is detected."""
    problem = FlagSynthesisProblem(
        circuit=stim.Circuit("I 0\nTICK\nI 0"),
        fault_bundles=(
            FaultBundle(0, (_realization(0), _realization(1))),
            FaultBundle(1, (_realization(2),)),
        ),
        malignant_configurations=((0, 1),),
        candidates=(
            FlagCandidate("a", "Z", ((0, (0,)), (1, (0,))), 0b101),
            FlagCandidate("b", "Z", ((0, (1,)), (1, (1,))), 0b001),
        ),
        realization_count=3,
    )

    solution = synthesize_flag_circuit(
        problem,
        limits=SynthesisLimits(
            milp_time_limit_seconds=5,
            total_time_limit_seconds=15,
        ),
    )

    assert solution.selected_candidate_indices == (0, 1)
    for first, second in itertools.product((0, 1), (2,)):
        assert any(
            problem.candidates[index].responds_to(first)
            ^ problem.candidates[index].responds_to(second)
            for index in solution.selected_candidate_indices
        )


def test_bipartite_edge_coloring_uses_maximum_degree_layers():
    """Endpoint CNOTs schedule in the bipartite graph's optimal layer count."""
    edges = (
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 2),
        (1, 2, 3),
        (2, 2, 4),
    )
    colors = _edge_color_bipartite(edges)

    assert set(colors) == {0, 1}
    for color in set(colors):
        colored_edges = [edge for edge, edge_color in zip(edges, colors) if edge_color == color]
        assert len({edge[0] for edge in colored_edges}) == len(colored_edges)
        assert len({edge[1] for edge in colored_edges}) == len(colored_edges)


def test_build_flagged_circuit_has_deterministic_new_detector():
    """A Z-region candidate builds a deterministic |0>-to-Z flag lifecycle."""
    base = stim.Circuit("""
        R 0
        TICK
        I 0
        TICK
        M 0
    """)
    candidate = FlagCandidate(
        "z_path",
        "Z",
        ((1, (0,)), (2, (0,))),
        response_mask=1,
    )
    problem = FlagSynthesisProblem(
        circuit=base,
        fault_bundles=(FaultBundle(0, (_realization(0),)),),
        malignant_configurations=((0,),),
        candidates=(candidate,),
        realization_count=1,
    )
    metrics = SynthesisMetrics(1, 1, 1, 1, 1, 0, ())
    solution = FlagCircuitSolution(
        selected_candidate_indices=(0,),
        flag_count=1,
        added_cnot_layers=2,
        interaction_count=2,
        candidate_to_flag=((0, 0),),
        scheduled_interactions=(
            ScheduledInteraction(1, 0, 0, 0, 0),
            ScheduledInteraction(2, 0, 0, 0, 0),
        ),
        objective_order=("flag_qubits", "added_cnot_layers", "flag_cnot_count"),
        optimal_within_candidate_pool=True,
        metrics=metrics,
    )

    flagged, flag_indices = build_flagged_circuit(
        base_circuit=base,
        problem=problem,
        solution=solution,
    )
    samples = flagged.compile_detector_sampler(seed=1).sample(100)

    assert flag_indices == (1,)
    assert flagged.get_final_qubit_coordinates()[1] == [9.0, 0.0]
    assert np.all(samples[:, -1] == 0)


def test_inserted_flag_measurement_preserves_original_record_targets():
    """A new earlier flag measurement leaves original detector identities intact."""
    layers = (
        stim.Circuit("M 0"),
        stim.Circuit("M 1\nDETECTOR rec[-1] rec[-2]"),
    )

    rewritten = _rewrite_original_record_targets(
        layers,
        added_measurements_by_boundary={1: 1},
    )

    detector = tuple(rewritten[1])[-1]
    assert [target.value for target in detector.targets_copy()] == [-1, -3]


def test_mask_native_verifier_finds_distance_three_malignant_configurations(
        d3_synthesis_inputs,
):
    """Both mask-native flag inputs recover the four D3 weight-two failures."""
    _, analyzer, combinator, malignant = d3_synthesis_inputs
    kept_effects = combinator.get_kept_effects(
        logical_analyzer=analyzer,
        max_order=2,
        cultivated_states=("T",),
    )

    assert malignant == (
        (54, 150),
        (65, 148),
        (74, 146),
        (111, 148),
    )
    assert extract_malignant_configurations(
        kept_effects,
        cultivated_state="T",
        orders=(2,),
    ) == malignant


def test_fault_effect_restriction_uses_arbitrary_data_index_order():
    """Packed X/Z supports restrict into compact, caller-specified data order."""
    classification = classify_fault_effect(
        fault_index=12,
        effect_mask=(1 << 4) | (1 << 5) | (1 << 7) | (1 << 11),
        source_qubit_count=6,
        data_indices=(4, 1, 5),
    )

    assert classification.data_effect == 0b110101
    assert classification.x_data_mask == 0b101
    assert classification.z_data_mask == 0b110
    assert classification.data_pauli_weight == 3
    assert classification.x_data_weight == 2
    assert classification.z_data_weight == 2
    assert classification.is_targetable_hook


@pytest.mark.parametrize(
    ("effect_mask", "expected_total", "expected_x", "expected_z", "is_hook"),
    (
        (1 << 4, 1, 1, 0, False),
        ((1 << 10) | (1 << 7), 2, 0, 2, False),
        ((1 << 4) | (1 << 1), 2, 2, 0, True),
    ),
)
def test_only_multi_x_data_effects_are_synthesis_targets(
        effect_mask,
        expected_total,
        expected_x,
        expected_z,
        is_hook,
):
    """Weight-one and Z-only effects are nuisances while multi-X effects target."""
    classification = classify_fault_effect(
        fault_index=0,
        effect_mask=effect_mask,
        source_qubit_count=6,
        data_indices=(4, 1),
    )

    assert classification.data_pauli_weight == expected_total
    assert classification.x_data_weight == expected_x
    assert classification.z_data_weight == expected_z
    assert classification.is_targetable_hook is is_hook


def test_nuisance_response_can_cancel_a_hook_response():
    """A hook and nuisance that both flip one flag XOR to an undetected product."""
    problem = FlagSynthesisProblem(
        circuit=stim.Circuit("I 0\nTICK\nI 0"),
        fault_bundles=(
            FaultBundle(0, (_realization(0),)),
            FaultBundle(1, (_realization(1),)),
        ),
        malignant_configurations=((0, 1),),
        candidates=(
            FlagCandidate("cancels", "Z", ((0, (0,)), (1, (0,))), 0b11),
            FlagCandidate("detects", "Z", ((0, (1,)), (1, (1,))), 0b01),
        ),
        realization_count=2,
        target_fault_indices=(0,),
    )

    assert _find_zero_response_witnesses(problem, (0,)) == {0b11}
    solution = synthesize_flag_circuit(
        problem,
        limits=SynthesisLimits(
            milp_time_limit_seconds=5,
            total_time_limit_seconds=15,
        ),
    )
    assert solution.selected_candidate_indices == (1,)


def test_bitmask_response_matches_direct_symplectic_products():
    """Boundary/qubit realization bitsets reproduce direct endpoint products."""
    realizations = (
        _realization(0, (
            SymplecticPauli(),
            SymplecticPauli(x=0b01),
            SymplecticPauli(x=0b11),
        )),
        _realization(1, (
            SymplecticPauli(x=0b10),
            SymplecticPauli(x=0b10, z=0b01),
            SymplecticPauli(x=0b01),
        )),
    )
    interactions = ((0, (1,)), (2, (0, 1)))
    masks = precompute_x_trajectory_masks(
        realizations,
        boundary_count=3,
        qubit_count=2,
    )

    optimized = candidate_response_mask(interactions, masks)
    direct = _candidate_response_mask("Z", interactions, realizations)

    assert optimized == direct


def test_continuous_path_distinguishes_late_realizations():
    """A lifecycle ending before a later event catches only the early realization."""
    realizations = (
        _realization(0, (
            SymplecticPauli(),
            SymplecticPauli(x=1),
            SymplecticPauli(x=1),
        )),
        _realization(1, (
            SymplecticPauli(),
            SymplecticPauli(),
            SymplecticPauli(x=1),
        )),
    )
    masks = precompute_x_trajectory_masks(
        realizations,
        boundary_count=3,
        qubit_count=1,
    )

    assert candidate_response_mask(((0, (0,)), (1, (0,))), masks) == 0b01


def test_closed_lifecycles_include_one_boundary_of_cooldown():
    """A physical flag is reused only after disjoint closed active intervals."""
    problem = FlagSynthesisProblem(
        circuit=stim.Circuit("I 0\nTICK\nI 0\nTICK\nI 0\nTICK\nI 0\nTICK\nI 0"),
        fault_bundles=(),
        malignant_configurations=(),
        candidates=(
            FlagCandidate("first", "Z", ((1, (0,)), (2, (0,))), 0),
            FlagCandidate("cooldown_overlap", "Z", ((3, (0,)), (4, (0,))), 0),
            FlagCandidate("reusable", "Z", ((4, (0,)), (5, (0,))), 0),
        ),
        realization_count=0,
    )

    colors = dict(_color_candidate_intervals(problem, (0, 1, 2)))

    assert colors[0] != colors[1]
    assert colors[0] == colors[2]


def test_observable_record_target_survives_flag_measurement_insertion():
    """An observable continues to reference its original measurement identity."""
    layers = (
        stim.Circuit("M 0"),
        stim.Circuit("OBSERVABLE_INCLUDE(0) rec[-1]"),
    )

    rewritten = _rewrite_original_record_targets(
        layers,
        added_measurements_by_boundary={1: 1},
    )

    observable = tuple(rewritten[1])[0]
    assert [target.value for target in observable.targets_copy()] == [-2]


def test_multiple_insertions_preserve_original_measurement_identities():
    """Per-boundary rewrites account for several inserted measurements at once."""
    layers = (
        stim.Circuit("M 0"),
        stim.Circuit("M 1"),
        stim.Circuit(
            "DETECTOR rec[-1] rec[-2]\n"
            "OBSERVABLE_INCLUDE(0) rec[-2]"
        ),
    )

    rewritten = _rewrite_original_record_targets(
        layers,
        added_measurements_by_boundary={1: 1, 2: 2},
    )

    detector, observable = tuple(rewritten[2])
    assert [target.value for target in detector.targets_copy()] == [-3, -5]
    assert [target.value for target in observable.targets_copy()] == [-5]


def test_lifecycle_removal_preserves_surviving_record_targets():
    """Removing a flag measurement keeps later detector identities intact.

    :return: None.
    """
    circuit = stim.Circuit("""
        QUBIT_COORDS(9, 0) 1
        R 0 1
        M 0
        CX 0 1
        M 1
        DETECTOR(9, 0, 1) rec[-1]
        DETECTOR(0, 0, 0) rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-2]
    """)
    lifecycle, = find_flag_lifecycles(circuit, flag_indices=(1,))

    pruned = remove_flag_lifecycle(
        circuit,
        flag_indices=(1,),
        lifecycle=lifecycle,
    )

    detector, observable = tuple(pruned)[-2:]
    assert pruned.num_measurements == 1
    assert pruned.num_detectors == 1
    assert find_flag_lifecycles(pruned, flag_indices=(1,)) == ()
    assert [target.value for target in detector.targets_copy()] == [-1]
    assert [target.value for target in observable.targets_copy()] == [-1]


def test_greedy_lifecycle_pruning_stops_at_a_verified_fixed_point(monkeypatch):
    """Greedy pruning accepts one safe lifecycle and retains the required one.

    :param monkeypatch: Pytest helper replacing exhaustive verification.
    :return: None.
    """
    circuit = stim.Circuit("""
        QUBIT_COORDS(9, 0) 1
        R 0 1
        CX 0 1
        TICK
        CX 0 1
        MR 1
        DETECTOR(9, 0, 1) rec[-1]
        R 1
        CX 0 1
        TICK
        CX 0 1
        M 1
        DETECTOR(9, 0, 2) rec[-1]
    """)
    monkeypatch.setattr(
        flag_synthesis_module,
        "verify_flag_solution",
        lambda **kwargs: flag_synthesis_module.VerificationResult(
            max_order=kwargs["max_order"],
            cultivated_state=kwargs["cultivated_state"],
            malignant_configurations=(
                () if kwargs["circuit"].num_detectors else ((0,),)
            ),
            elapsed_seconds=0.0,
        ),
    )

    result = greedily_prune_flag_lifecycles(
        inner_circuit=circuit,
        full_circuit=circuit,
        flag_indices=(1,),
        logical_analyzer=object(),
        max_order=2,
    )

    remaining = find_flag_lifecycles(result.inner_circuit, flag_indices=(1,))
    assert result.attempt_count == 2
    assert len(result.removed_lifecycles) == 1
    assert len(remaining) == 1
    assert remaining[0].interaction_count == 2
    assert result.inner_circuit == result.full_circuit


def test_lifecycle_controlling_a_later_flag_is_not_independently_removable():
    """Cross-iteration flag controls prevent unsafe singleton removal.

    :return: None.
    """
    circuit = stim.Circuit("""
        QUBIT_COORDS(9, 0) 1
        QUBIT_COORDS(10, 0) 2
        R 0 1
        CX 0 1
        R 2
        CX 1 2
        TICK
        CX 1 2
        M 2
        DETECTOR(10, 0, 1) rec[-1]
        CX 0 1
        M 1
        DETECTOR(9, 0, 2) rec[-1]
    """)
    lifecycles = find_flag_lifecycles(circuit, flag_indices=(1, 2))
    controlling = next(item for item in lifecycles if item.flag_index == 1)

    assert controlling.controlled_interaction_count == 2
    with pytest.raises(
            FlagSynthesisError,
            match="controlling another synthesized flag",
    ):
        remove_flag_lifecycle(
            circuit,
            flag_indices=(1, 2),
            lifecycle=controlling,
        )


def test_hook_invariant_reports_concrete_nuisance_configuration(
        d3_synthesis_inputs,
):
    """Problem construction reports effects when a configuration has no hook."""
    circuit, _, combinator, _ = d3_synthesis_inputs

    with pytest.raises(
            FlagSynthesisError,
            match=r"Malignant configuration \(146,\).*no targetable multi-X hook.*fault 146",
    ):
        build_flag_synthesis_problem(
            combinator=combinator,
            malignant_configurations=((146,),),
            data_indices=circuit.DATA_INDICES,
        )


def test_d3_z_only_synthesis_eliminates_all_order_two_failures(
        d3_synthesis_inputs,
):
    """The staged Z-only synthesizer restores D3 fault distance through order two."""
    circuit, analyzer, combinator, malignant = d3_synthesis_inputs
    envelope_problem = build_flag_synthesis_problem(
        combinator=combinator,
        malignant_configurations=malignant,
        data_indices=circuit.DATA_INDICES,
    )
    problem = build_flag_synthesis_problem(
        combinator=combinator,
        malignant_configurations=malignant,
        data_indices=circuit.DATA_INDICES,
        include_single_seed_fallback=True,
    )
    problem = shortlist_flag_synthesis_problem(problem)

    assert {candidate.source for candidate in envelope_problem.candidates} <= {
        "direct-envelope",
        "nearby-envelope",
    }
    assert all(candidate.basis == "Z" for candidate in problem.candidates)
    assert set(problem.target_fault_indices) == {54, 65, 74, 111}
    assert {146, 148, 150} <= {
        bundle.fault_index for bundle in problem.fault_bundles
    }

    solution = synthesize_flag_circuit(
        problem,
        limits=SynthesisLimits(
            milp_time_limit_seconds=60,
            total_time_limit_seconds=180,
        ),
    )
    flagged, flag_indices = build_flagged_circuit(
        base_circuit=circuit.INNER_CIRCUIT,
        problem=problem,
        solution=solution,
    )
    assert_z_basis_flag_construction(flagged, flag_indices=flag_indices)
    assert_flag_detectors_deterministic(
        flagged,
        expected_count=len(solution.selected_candidate_indices),
    )

    verification = verify_flag_solution(
        circuit=flagged,
        logical_analyzer=analyzer,
        max_order=2,
        cultivated_state="T",
    )

    assert verification.passed
    assert verification.malignant_configurations == ()


def test_d5_loader_accepts_the_verified_generated_artifact():
    """Verified D5 flags use unique iteration columns in both circuit views."""
    circuit = circuits.D5A19Flagged()
    manifest = circuit.SYNTHESIS_MANIFEST
    iterations = manifest["iterations"]
    pruning = manifest["post_synthesis_lifecycle_pruning"]
    removed_coordinates = {
        tuple(item["detector_coordinates"])
        for item in pruning["removed_lifecycles"]
    }

    assert manifest["verified_through_order"] == 4
    assert len(circuit.FLAG_INDICES) == manifest["total_flag_qubits"]
    assert [item["coordinate_column"] for item in iterations] == [9, 10, 11]
    assert manifest["total_flag_lifecycles"] == 43
    assert manifest["total_flag_cnot_count"] == 138
    assert pruning["pre_pruning_lifecycle_count"] == 48
    assert pruning["post_pruning_lifecycle_count"] == 43
    assert {
        item["candidate_index"] for item in pruning["removed_lifecycles"]
    } == {8, 18, 102, 112, 251}

    expected_coordinates = {}
    first_flag_index = min(circuit.FLAG_INDICES)
    for item in iterations:
        flag_count = item["solution"]["flag_count"]
        for row, flag_index in enumerate(
                range(first_flag_index, first_flag_index + flag_count)):
            expected_coordinates[flag_index] = [
                float(item["coordinate_column"]),
                float(row),
            ]
        first_flag_index += flag_count

    for generated_circuit in (circuit.INNER_CIRCUIT, circuit.CIRCUIT):
        coordinates = generated_circuit.get_final_qubit_coordinates()
        lifecycles = find_flag_lifecycles(
            generated_circuit,
            flag_indices=circuit.FLAG_INDICES,
        )
        flag_cnot_count = sum(
            target.qubit_value in circuit.FLAG_INDICES
            for instruction in generated_circuit
            if instruction.name == "CX"
            for _, target in instruction.target_groups()
        )
        actual_coordinates = {
            flag_index: coordinates[flag_index]
            for flag_index in circuit.FLAG_INDICES
        }
        assert actual_coordinates == expected_coordinates
        assert len({tuple(coords) for coords in actual_coordinates.values()}) == 21
        assert len(lifecycles) == manifest["total_flag_lifecycles"]
        assert flag_cnot_count == manifest["total_flag_cnot_count"]
        assert not removed_coordinates & {
            lifecycle.detector_coordinates for lifecycle in lifecycles
        }
        for item in iterations:
            removed_count = sum(
                record["iteration"] == item["iteration"]
                for record in pruning["removed_lifecycles"]
            )
            assert_flag_detectors_deterministic(
                generated_circuit,
                coordinate_prefix=(item["coordinate_column"],),
                expected_count=(
                    len(item["solution"]["selected_candidates"])
                    - removed_count
                ),
            )


def test_d5_loader_rejects_an_unverified_manifest(tmp_path, monkeypatch):
    """A null verification marker cannot be loaded as a finished D5 solution."""
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "d5a19_checkpoint.json").write_text(
        '{"verified_through_order": null}\n'
    )
    monkeypatch.setattr(distance_5_circuits, "_STIM_FILES_DIR", tmp_path)

    with pytest.raises(ValueError, match="unverified"):
        circuits.D5A19Flagged("checkpoint")
