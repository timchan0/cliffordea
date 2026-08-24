import itertools

import numpy as np
import stim

from cliffordep import circuits, noise
from cliffordep.combinators import FaultCombinator
from cliffordep.flag_synthesis import (
    FaultBundle,
    FlagCandidate,
    FlagCircuitSolution,
    FlagSynthesisProblem,
    PhysicalFaultRealization,
    ScheduledInteraction,
    SymplecticPauli,
    SynthesisLimits,
    SynthesisMetrics,
    build_flagged_circuit,
    extract_malignant_configurations,
    find_malignant_configurations,
    synthesize_flag_circuit,
)
from cliffordep.flag_synthesis import _edge_color_bipartite
from cliffordep.flag_synthesis import _rewrite_original_record_targets
from cliffordep.logical_analyzers import CliffordLogicalAnalyzer


def _realization(ordinal: int) -> PhysicalFaultRealization:
    return PhysicalFaultRealization(
        ordinal=ordinal,
        event=(0, "X_ERROR", (stim.GateTarget(0),)),
        trajectory=(SymplecticPauli(),),
    )


def test_symplectic_inner_product():
    x0 = SymplecticPauli(x=1)
    z0 = SymplecticPauli(z=1)
    x1 = SymplecticPauli(x=2)

    assert x0.anticommutes(z0)
    assert z0.anticommutes(x0)
    assert not x0.anticommutes(x1)
    assert not x0.anticommutes(x0)


def test_fault_bundle_is_refined_only_when_candidate_responses_differ():
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


def test_mask_native_verifier_finds_distance_three_malignant_configurations():
    """Both mask-native flag inputs recover the four D3 weight-two failures."""
    circuit = circuits.Distance3DoubleCheck(ancilla_count=6)
    combinator = FaultCombinator(noise.uniformly_depolarize(
        circuit.INNER_CIRCUIT,
        noise_level=1e-3,
    ))
    analyzer = CliffordLogicalAnalyzer(
        data_indices=circuit.DATA_INDICES,
        stabilizer_generators=circuit.STABILIZER_GENERATORS_RESTRICTED,
        logical_s=circuit.LOGICAL_S,
    )

    malignant = find_malignant_configurations(
        combinator=combinator,
        logical_analyzer=analyzer,
        max_order=2,
        cultivated_state="T",
    )
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
