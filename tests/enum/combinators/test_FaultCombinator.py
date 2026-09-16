from collections import defaultdict
import itertools
import math
import pickle
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
import stim

import cliffordea
from cliffordea.enum.combinators import FaultCombinator
from cliffordea.enum.combinators.fault_combinator import (
    _compile_odds_terms,
    _make_logical_analysis_cache,
    _make_pauli_mask_restrictor,
    _iter_undetected_configurations,
    _sum_benign_malignant_weights,
)
from cliffordea.accept.logical_analyzers import CliffordLogicalAnalyzer, LogicalAnalyzer
from cliffordea.enum.cultivation_circuit import (
    CultivationCircuit,
    mask_to_unsigned_pauli,
)
from cliffordea.accept.pauli import forget_sign
from cliffordea.accept.pauli import pauli_mask


def _make_small_fault_combinator() -> FaultCombinator:
    """Build a combinator containing one- and two-qubit fault processes."""
    return FaultCombinator(stim.Circuit("""
        R 0 1
        DEPOLARIZE1(0.001) 0
        TICK
        CX 0 1
        DEPOLARIZE2(0.001) 0 1
        TICK
        M(0.001) 0 1
        DETECTOR rec[-2]
        DETECTOR rec[-1]
    """))


def test_index_to_events_reuses_indices_recorded_during_construction(
        monkeypatch,
):
    """Rebuilding event objects should not repeat fault propagation."""
    combinator = _make_small_fault_combinator()
    event_count = sum(
        len(group)
        for group in combinator.circuit.group_error_events_by_location().values()
    )
    assert len(combinator._event_fault_indices) == event_count

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Fault propagation was unexpectedly repeated.")

    monkeypatch.setattr(
        combinator.circuit,
        'get_signature_and_effect',
        fail_if_called,
    )
    index_to_events = combinator.index_to_events

    assert sum(map(len, index_to_events.values())) == event_count
    assert set(index_to_events) == set(combinator.index_to_bag)


def test_index_to_events_cache_is_excluded_from_pickle():
    """Cached Stim targets should be rebuilt from picklable integer indices."""
    combinator = _make_small_fault_combinator()
    expected = combinator.index_to_events
    reverse_data = combinator.circuit._reverse_propagation_data
    measurement_locations = (
        combinator.circuit._measurement_locations_by_event
    )
    assert all(
        isinstance(response, int)
        for generator_pair in reverse_data.responses_by_timeslice
        for generator_responses in generator_pair
        for response in generator_responses
    )
    assert all(
        isinstance(measurement_index, int)
        for measurement_index
        in reverse_data.measurement_index_by_event.values()
    )
    assert all(
        isinstance(timeslice, int)
        and all(isinstance(qubit, int) for qubit in measured_qubits)
        for timeslice, measured_qubits
        in reverse_data.measurement_index_by_event
    )
    assert all(
        isinstance(detector_mask, int)
        for detector_mask in reverse_data.measurement_detector_masks
    )
    assert all(
        isinstance(timeslice, int)
        and all(isinstance(qubit, int) for qubit in measured_qubits)
        and isinstance(instruction_index, int)
        and isinstance(target_index, int)
        for (
            timeslice,
            measured_qubits,
        ), (
            instruction_index,
            target_index,
        ) in measurement_locations.items()
    )

    restored = pickle.loads(pickle.dumps(combinator))

    assert 'index_to_events' not in restored.__dict__
    assert restored._event_fault_indices == combinator._event_fault_indices
    assert restored.index_to_events == expected
    assert restored.circuit._reverse_propagation_data == reverse_data
    assert (
        restored.circuit._measurement_locations_by_event
        == measurement_locations
    )


def test_construction_avoids_legacy_event_analysis(monkeypatch):
    """Construct a combinator without decoding masks to arrays and strings."""
    legacy_method = Mock(side_effect=AssertionError(
        'Legacy event analysis was unexpectedly called.',
    ))
    monkeypatch.setattr(
        CultivationCircuit,
        'get_signature_and_effect',
        legacy_method,
    )
    combinator = FaultCombinator(stim.Circuit("""
        R 0 1
        TICK
        CX 0 1
        DEPOLARIZE2(0.001) 0 1
        TICK
        M 0 1
        DETECTOR rec[-2]
        DETECTOR rec[-1]
    """))

    legacy_method.assert_not_called()
    assert combinator.fault_count > 0


def test_visualization_measurement_locations_are_lazily_cached():
    """Only measurement visualization should build and reuse its locations."""
    combinator = _make_small_fault_combinator()
    assert '_measurement_locations_by_event' not in combinator.circuit.__dict__
    original_circuit = combinator.circuit.noiseless_circuit.copy()
    pauli_only_index = next(
        fault_index
        for fault_index, events in combinator.index_to_events.items()
        if all(not name.startswith('M') for _, name, _ in events)
    )
    measurement_index = next(
        fault_index
        for fault_index, events in combinator.index_to_events.items()
        if any(name.startswith('M') for _, name, _ in events)
    )

    combinator._fault_configuration_circuit(
        configuration=(pauli_only_index,),
        probability_increment=0.125,
    )
    assert '_measurement_locations_by_event' not in combinator.circuit.__dict__

    first = combinator._fault_configuration_circuit(
        configuration=(measurement_index,),
        probability_increment=0.125,
    )
    cached_locations = (
        combinator.circuit._measurement_locations_by_event
    )
    second = combinator._fault_configuration_circuit(
        configuration=(measurement_index,),
        probability_increment=0.125,
    )

    assert first == second
    assert combinator.circuit.noiseless_circuit == original_circuit
    assert (
        combinator.circuit._measurement_locations_by_event
        is cached_locations
    )


def test_basis_and_indexed_faults_remain_mask_native_and_readable(capsys):
    """Store masks for computation while decoding only printed introspection.

    :param capsys: Pytest fixture used to capture readable basis output.
    """
    combinator = _make_small_fault_combinator()
    assert all(isinstance(signature, int) for signature in combinator.basis)
    assert all(
        isinstance(effect, int)
        for effects in combinator.basis.values()
        for effect in effects
    )

    expected_by_index = {
        index: (signature, effect)
        for signature, effects in combinator.basis.items()
        for effect, index in effects.items()
    }
    assert combinator._indexed_faults == tuple(
        expected_by_index[index] for index in range(combinator.fault_count)
    )

    sample_index = 0
    sample_signature, sample_effect = expected_by_index[sample_index]
    combinator.print_basis()
    output = capsys.readouterr().out
    signature_text = ''.join(
        '1' if sample_signature >> detector & 1 else '0'
        for detector in range(combinator.circuit.noisy_circuit.num_detectors)
    )
    effect_text = mask_to_unsigned_pauli(
        sample_effect,
        combinator.circuit.noisy_circuit.num_qubits,
    )
    assert signature_text in output
    assert f"  {effect_text}: {sample_index}" in output


def test_mask_analyzer_matches_direct_event_propagation():
    """Recover every small-circuit fault exactly from cached mask analyses."""
    circuit = CultivationCircuit(noisy_circuit=stim.Circuit("""
        R 0 1
        DEPOLARIZE1(0.001) 0
        TICK
        CX 0 1
        DEPOLARIZE2(0.001) 0 1
        TICK
        M(0.001) 0 1
        DETECTOR rec[-2]
        DETECTOR rec[-1]
    """))

    for group in circuit.group_error_events_by_location().values():
        for error_event in group:
            signature, effect = circuit.get_signature_and_effect(error_event)
            expected = (
                sum(
                    bool(value) << index
                    for index, value in enumerate(signature)
                ),
                pauli_mask(effect),
            )
            assert circuit._get_signature_and_effect_masks(error_event) == expected


@pytest.mark.parametrize("qubit_count", range(4))
def test_mask_xor_matches_stim_multiplication_exhaustively(qubit_count: int):
    """Compare mask XOR with every short unsigned Pauli product.

    :param qubit_count: The Pauli-string width for the exhaustive comparison.
    """
    strings = tuple(
        ''.join(paulis)
        for paulis in itertools.product('_XYZ', repeat=qubit_count)
    )
    for left, right in itertools.product(strings, repeat=2):
        expected = forget_sign(stim.PauliString(left) * stim.PauliString(right))
        actual = mask_to_unsigned_pauli(
            pauli_mask(left)
            ^ pauli_mask(right),
            qubit_count,
        )
        assert actual == expected


@pytest.mark.parametrize(
    ("factors", "qubit_count"),
    [
        (
            (
                "_XYZ_XYZ_XYZ_",
                "Y__ZX_XY_Z__X",
                "ZZ_XY___YXX__",
                "X_Y_Z_X_Y_Z_X",
            ),
            13,
        ),
        (
            (
                "_XYZ_XYZ_XYZ_XYZ_XYZ_XYZ_XYZ_XYZ_XYZ_X",
                "Y__ZX_XY_Z__XY__ZX_XY_Z__XY__ZX_XY_Z__X",
                "ZZ_XY___YXX__ZZ_XY___YXX__ZZ_XY___YXX__",
                "X_Y_Z_X_Y_Z_X_Y_Z_X_Y_Z_X_Y_Z_X_Y_Z_X_",
            ),
            38,
        ),
    ],
)
def test_representative_mask_products_match_stim(
        factors: tuple[str, ...],
        qubit_count: int,
):
    """Compare mask XOR with Stim at representative circuit widths.

    :param factors: The unsigned Pauli factors to multiply.
    :param qubit_count: The distance-3 or distance-5 circuit width represented
        by each factor.
    """
    factors = tuple(factor[:qubit_count] for factor in factors)
    assert all(len(factor) == qubit_count for factor in factors)
    expected = forget_sign(math.prod(
        (stim.PauliString(factor) for factor in factors),
        start=stim.PauliString(qubit_count),
    ))
    product_mask = 0
    for factor in factors:
        product_mask ^= pauli_mask(factor)
    assert mask_to_unsigned_pauli(product_mask, qubit_count) == expected


@pytest.mark.parametrize("unsigned_string", ["", "_", "I", "XYZ_", "Y_YXZI"])
def test_pauli_mask_round_trip(unsigned_string: str):
    """Check that packing and decoding preserves each Pauli symbol.

    Identity is canonicalized from ``I`` to ``_`` while ``Y`` must preserve
    both its X and Z support.

    :param unsigned_string: The unsigned Pauli string to round-trip.
    """
    canonical = unsigned_string.replace('I', '_')
    mask = pauli_mask(unsigned_string)
    assert mask_to_unsigned_pauli(mask, len(unsigned_string)) == canonical


def _restrict_pauli_mask(
        mask: int,
        *,
        data_indices: tuple[int, ...],
        source_qubit_count: int,
) -> int:
    """Provide a direct reference implementation of Pauli-mask restriction.

    This test-local implementation is intentionally independent of the
    lookup-table implementation exercised by the production code.

    :param mask: The packed Pauli effect on the source qubits.
    :param data_indices: Source-qubit indices to retain, in output order.
    :param source_qubit_count: The number of qubits represented by each source
        support mask.

    :return restricted_mask: The packed effect on the selected data qubits.
    """
    source_support_mask = (1 << source_qubit_count) - 1
    source_x_mask = mask & source_support_mask
    source_z_mask = (mask >> source_qubit_count) & source_support_mask
    data_x_mask = 0
    data_z_mask = 0
    for data_index, source_index in enumerate(data_indices):
        data_x_mask |= ((source_x_mask >> source_index) & 1) << data_index
        data_z_mask |= ((source_z_mask >> source_index) & 1) << data_index
    return data_x_mask | (data_z_mask << len(data_indices))


def test_pauli_mask_restriction_preserves_requested_order():
    """Check that mask restriction follows the requested data-qubit order.

    The lookup-table implementation is compared with both the expected Pauli
    string and the direct test-local implementation.
    """
    source = "XYZ_YX"
    data_indices = (5, 0, 3, 2)
    restricted_mask = _restrict_pauli_mask(
        pauli_mask(source),
        data_indices=data_indices,
        source_qubit_count=len(source),
    )
    assert mask_to_unsigned_pauli(
        restricted_mask,
        len(data_indices),
    ) == ''.join(source[index] for index in data_indices)
    fast_restrict = _make_pauli_mask_restrictor(
        data_indices=data_indices,
        source_qubit_count=len(source),
    )
    assert fast_restrict(pauli_mask(source)) == restricted_mask


@pytest.mark.parametrize("fault_count", range(5))
def test_meet_in_the_middle_matches_exhaustive_subsets(fault_count: int):
    """Canonical half joins return every zero-signature subset exactly once.

    :param fault_count: The distinct-fault subset size compared with brute force.
    """
    signatures = (0b00, 0b01, 0b01, 0b10, 0b11, 0b00)
    effects = (0b0001, 0b0010, 0b0100, 0b1000, 0b0011, 0b1100)
    expected = []
    for indices in itertools.combinations(range(len(signatures)), fault_count):
        signature = 0
        effect = 0
        for index in indices:
            signature ^= signatures[index]
            effect ^= effects[index]
        if signature == 0:
            expected.append((effect, indices))

    actual = list(_iter_undetected_configurations(
        signatures=signatures,
        effects=effects,
        fault_count=fault_count,
    ))

    assert sorted(actual) == sorted(expected)


class _RecordingAnalyzer(LogicalAnalyzer):
    def __init__(self, data_indices: tuple[int, ...] = ()):
        """Initialize an analyzer that records each logical-analysis request.

        :param data_indices: Source-qubit indices treated as data qubits.
        """
        self.DATA_INDICES = data_indices
        self.calls: list[tuple[str, int]] = []

    def analyze(
            self,
            cultivated_state: str,
            before_transversal: int,
    ) -> tuple[float, float]:
        """Record an analysis call and return deterministic synthetic values.

        :param cultivated_state: The logical state being cultivated.
        :param before_transversal: The packed data-qubit Pauli effect.

        :return: A synthetic acceptance probability and logical fidelity.
        """
        self.calls.append((cultivated_state, before_transversal))
        support_mask = (1 << len(self.DATA_INDICES)) - 1
        return (
            float((before_transversal & support_mask).bit_count() + 1),
            float(cultivated_state == 'S'),
        )


@pytest.mark.parametrize("maxsize", [0, 1, 262_144, None])
def test_logical_analysis_cache_sizes_have_identical_results(maxsize: int | None):
    """Check that cache capacity affects reuse but not analysis results.

    :param maxsize: The disabled, bounded, default-sized, or unbounded cache
        capacity under test.
    """
    analyzer = _RecordingAnalyzer(data_indices=(0, 1))
    analyze_mask = _make_logical_analysis_cache(
        logical_analyzer=analyzer,
        cultivated_states=('S', 'T'),
        maxsize=maxsize,
    )
    access_sequence = (0, 1, 2, 0, 1, 2)
    actual = [analyze_mask(mask) for mask in access_sequence]
    expected = [
        (
            (float(mask.bit_count() + 1), 1.0),
            (float(mask.bit_count() + 1), 0.0),
        )
        for mask in access_sequence
    ]
    assert actual == expected
    cache_info = analyze_mask.cache_info()
    if maxsize is not None:
        assert cache_info.currsize <= maxsize
    if maxsize == 0:
        assert cache_info.hits == 0
    if maxsize == 1:
        assert cache_info.misses == len(access_sequence)
    analyze_mask.cache_clear()


def test_fault_combinator_exposes_only_mask_native_kept_effect_enumeration():
    """The obsolete kept-string API is removed in favor of packed effects.

    The former two-step methods should not remain on either collaborating
    class after the intentional breaking API change.
    """
    assert not hasattr(FaultCombinator, "get_undetected_configurations")
    assert not hasattr(FaultCombinator, "get_kept_strings")
    assert hasattr(FaultCombinator, "get_kept_effects")
    assert not hasattr(LogicalAnalyzer, "get_kept_strings")


def _exhaustive_kept_effects_oracle(
        combinator: FaultCombinator,
        logical_analyzer,
        max_fault_count: int,
        cultivated_states: tuple[str, ...],
):
    """Compute kept effects with an independent exhaustive subset search.

    This oracle explicitly checks every distinct fault subset instead of using
    the production meet-in-the-middle join.

    :param combinator: The fault combinator whose indexed basis is enumerated.
    :param logical_analyzer: The analyzer used to postselect each data effect.
    :param max_fault_count: The largest fault-subset size to enumerate.
    :param cultivated_states: The logical states to analyze.

    :return: Kept-effect results in the same structure as
        :meth:`FaultCombinator.get_kept_effects`.
    """
    faults = tuple(enumerate(combinator._indexed_faults))
    restrict_effect = _make_pauli_mask_restrictor(
        data_indices=logical_analyzer.DATA_INDICES,
        source_qubit_count=combinator.circuit.noisy_circuit.num_qubits,
    )
    result = {state: [] for state in cultivated_states}
    for fault_count in range(max_fault_count + 1):
        configurations_by_state = {
            state: defaultdict(list) for state in cultivated_states
        }
        analyses_by_state = {state: {} for state in cultivated_states}
        for combination in itertools.combinations(faults, fault_count):
            resultant_signature = 0
            full_effect = 0
            for _, (signature, effect) in combination:
                resultant_signature ^= signature
                full_effect ^= effect
            if resultant_signature:
                continue
            data_effect = restrict_effect(full_effect)
            fault_indices = tuple(fault[0] for fault in combination)
            for state in cultivated_states:
                if data_effect not in analyses_by_state[state]:
                    analyses_by_state[state][data_effect] = (
                        logical_analyzer.analyze(state, data_effect)
                    )
                analysis = analyses_by_state[state][data_effect]
                if analysis[0]:
                    configurations_by_state[state][data_effect].append(
                        fault_indices
                    )
        for state in cultivated_states:
            result[state].append({
                effect: (
                    *analyses_by_state[state][effect],
                    configurations,
                )
                for effect, configurations
                in configurations_by_state[state].items()
            })
    return result


class _SyntheticAnalyzer(LogicalAnalyzer):
    def __init__(self):
        """Initialize a deterministic analyzer with nontrivial data ordering.

        Reversing the selected source indices ensures the fused implementation
        and the independent oracle both exercise ordered data restriction.
        """
        self.DATA_INDICES = (2, 0)

    def analyze(
            self,
            cultivated_state: str,
            before_transversal: int,
    ) -> tuple[float, float]:
        """Return synthetic acceptance and fidelity for a packed effect.

        :param cultivated_state: The logical state being cultivated.
        :param before_transversal: The packed data-qubit Pauli effect.

        :return: A deterministic acceptance probability and logical fidelity
            chosen to exercise rejection and state-dependent output.
        """
        support_mask = 0b11
        x_mask = before_transversal & support_mask
        z_mask = (before_transversal >> 2) & support_mask
        if not (x_mask & 1) and z_mask & 1:
            return 0.0, 0.0
        acceptance_probability = 0.5 if x_mask & z_mask else 1.0
        logical_fidelity = float(
            (((x_mask & ~z_mask).bit_count() + (cultivated_state == 'S')) % 2)
            == 0
        )
        return acceptance_probability, logical_fidelity


def _make_synthetic_combinator() -> FaultCombinator:
    """Build a small combinator with three cancelling nonzero signatures.

    The synthetic A, B, and C signature groups satisfy ``A XOR B XOR C = 0``
    and contain enough effects to exercise repeated and three-way selection.

    :return combinator: A minimally initialized combinator suitable for the
        independent exhaustive-oracle comparison.
    """
    indexed_faults = (
        (0b00, pauli_mask("Z_Z")),
        (0b01, pauli_mask("X__")),
        (0b01, pauli_mask("_X_")),
        (0b10, pauli_mask("__Z")),
        (0b10, pauli_mask("Y__")),
        (0b11, pauli_mask("_Y_")),
        (0b11, pauli_mask("ZZ_")),
    )
    basis: defaultdict[int, dict[int, int]] = defaultdict(dict)
    for index, (signature, effect) in enumerate(indexed_faults):
        basis[signature][effect] = index
    combinator = FaultCombinator.__new__(FaultCombinator)
    combinator.basis = dict(basis)
    combinator._indexed_faults = indexed_faults
    combinator.index_to_bag = {
        index: (1, 0, 0) for index in range(len(indexed_faults))
    }
    combinator.circuit = cast(
        CultivationCircuit,
        SimpleNamespace(
            noisy_circuit=SimpleNamespace(num_qubits=3, num_detectors=2),
        ),
    )
    return combinator


def test_progress_bars_stream_without_precounting(monkeypatch):
    """Check that progress is streamed without an analyzer-unaware pre-count.

    :param monkeypatch: Pytest fixture used to replace tqdm with a lightweight
        recording wrapper.
    """
    combinator = _make_synthetic_combinator()
    logical_analyzer = _SyntheticAnalyzer()
    max_fault_count = 3
    progress_calls: list[dict[str, object]] = []

    def recording_tqdm(iterable, **kwargs):
        """Record progress-bar options without adding iteration overhead.

        :param iterable: The configuration iterator that tqdm would wrap.
        :param kwargs: The progress-bar display and total options.

        :return: The original iterable without a rendered progress bar.
        """
        progress_calls.append(kwargs)
        return iterable

    monkeypatch.setattr(
        "cliffordea.enum.combinators.fault_combinator.tqdm",
        recording_tqdm,
    )

    without_progress = combinator.get_kept_effects(
        logical_analyzer=logical_analyzer,
        max_fault_count=max_fault_count,
        cultivated_states=('S', 'T'),
        print_progress=False,
    )
    assert progress_calls == []

    with_progress = combinator.get_kept_effects(
        logical_analyzer=logical_analyzer,
        max_fault_count=max_fault_count,
        cultivated_states=('S', 'T'),
        print_progress=True,
    )
    assert with_progress == without_progress
    assert len(progress_calls) == max_fault_count + 1
    for fault_count, progress_options in enumerate(progress_calls):
        assert progress_options == {
            'desc': f"{fault_count}-fault configurations",
            'unit': "configuration",
            'dynamic_ncols': True,
            'leave': True,
        }


def test_synthetic_three_fault_result_matches_exhaustive_subset_oracle():
    """Compare lazy enumeration with exhaustive three-way cancellation.

    The synthetic basis ensures that an undetected 3-fault configuration
    can arise from three different nonzero signature groups.
    """
    combinator = _make_synthetic_combinator()
    logical_analyzer = _SyntheticAnalyzer()
    cultivated_states = ('S', 'T')
    assert all(
        logical_analyzer.linear_precheck_syndrome(effect) == 0
        for _, effect in combinator._indexed_faults
    )
    actual = combinator.get_kept_effects(
        logical_analyzer=logical_analyzer,
        max_fault_count=3,
        cultivated_states=cultivated_states,
    )
    expected = _exhaustive_kept_effects_oracle(
        combinator,
        logical_analyzer,
        max_fault_count=3,
        cultivated_states=cultivated_states,
    )
    assert actual.keys() == expected.keys()
    for state in cultivated_states:
        assert len(actual[state]) == len(expected[state])
        for actual_effects, expected_effects in zip(
                actual[state],
                expected[state],
                strict=True,
        ):
            assert actual_effects.keys() == expected_effects.keys()
            for effect_mask in actual_effects:
                actual_probability, actual_fidelity, actual_configurations = (
                    actual_effects[effect_mask]
                )
                expected_probability, expected_fidelity, expected_configurations = (
                    expected_effects[effect_mask]
                )
                assert (actual_probability, actual_fidelity) == (
                    expected_probability,
                    expected_fidelity,
                )
                assert sorted(actual_configurations) == sorted(
                    expected_configurations
                )
    effect_mask = pauli_mask('_Z')
    assert (1, 4, 5) in actual['S'][3][effect_mask][2]


def test_d3_cache_sizes_preserve_kept_effects_and_share_configuration_lists():
    """Check cache-size invariance and cross-state configuration sharing.

    Disabled, bounded, default-sized, and unbounded caches must retain equal
    distance-3 results. Effects accepted for both S and T should also refer to
    the same configuration-list object.
    """
    circuit = cliffordea.enum.circuits.DoubleCheck(ancilla_count=6)
    noisy_circuit = cliffordea.enum.noise.uniformly_depolarize(
        circuit.INNER_CIRCUIT,
        noise_level=1e-3,
    )
    combinator = FaultCombinator(noisy_circuit)
    logical_analyzer = CliffordLogicalAnalyzer(
        data_indices=circuit.DATA_INDICES,
        stabilizer_generators=circuit.STABILIZER_GENERATORS,
        logical_s=circuit.LOGICAL_S,
    )
    results_by_cache_size = {
        maxsize: combinator.get_kept_effects(
            logical_analyzer=logical_analyzer,
            max_fault_count=2,
            cultivated_states=('S', 'T'),
            logical_analysis_cache_maxsize=maxsize,
        )
        for maxsize in (0, 1, 262_144, None)
    }
    expected = results_by_cache_size[262_144]
    assert all(result == expected for result in results_by_cache_size.values())

    for fault_count, effects_for_s in enumerate(expected['S']):
        effects_for_t = expected['T'][fault_count]
        for effect in effects_for_s.keys() & effects_for_t.keys():
            assert effects_for_s[effect][2] is effects_for_t[effect][2]


def test_clifford_linear_precheck_preserves_results_and_reduces_enumeration(
        monkeypatch,
):
    """Fold optional Clifford checks into enumeration without changing output.

    :param monkeypatch: Pytest fixture used to count configurations yielded by
        the meet-in-the-middle enumerator.
    """
    circuit = cliffordea.enum.circuits.DoubleCheck(ancilla_count=6)
    noisy_circuit = cliffordea.enum.noise.uniformly_depolarize(
        circuit.INNER_CIRCUIT,
        noise_level=1e-3,
    )
    combinator = FaultCombinator(noisy_circuit)
    yielded_counts = []
    original_iterator = _iter_undetected_configurations

    def recording_iterator(*, signatures, effects, fault_count):
        """Record one fixed-fault-count configuration stream.

        :param signatures: Packed rejection signatures in fault-index order.
        :param effects: Packed data effects in fault-index order.
        :param fault_count: The number of faults being enumerated.
        :return: An iterator over the original configuration stream.
        """
        count = 0
        for configuration in original_iterator(
                signatures=signatures,
                effects=effects,
                fault_count=fault_count,
        ):
            count += 1
            yield configuration
        yielded_counts.append(count)

    monkeypatch.setattr(
        'cliffordea.enum.combinators.fault_combinator.'
        '_iter_undetected_configurations',
        recording_iterator,
    )
    analyzer_arguments = {
        'data_indices': circuit.DATA_INDICES,
        'stabilizer_generators': circuit.STABILIZER_GENERATORS,
        'logical_s': circuit.LOGICAL_S,
    }
    enabled_results = combinator.get_kept_effects(
        logical_analyzer=CliffordLogicalAnalyzer(
            **analyzer_arguments,
            precheck_z_stabilizers=True,
        ),
        max_fault_count=3,
        cultivated_states=('S', 'T'),
    )
    enabled_counts = tuple(yielded_counts)
    yielded_counts.clear()
    disabled_results = combinator.get_kept_effects(
        logical_analyzer=CliffordLogicalAnalyzer(
            **analyzer_arguments,
            precheck_z_stabilizers=False,
        ),
        max_fault_count=3,
        cultivated_states=('S', 'T'),
    )

    assert enabled_results == disabled_results
    assert all(
        enabled <= disabled
        for enabled, disabled in zip(
            enabled_counts,
            yielded_counts,
            strict=True,
        )
    )
    assert any(
        enabled < disabled
        for enabled, disabled in zip(
            enabled_counts,
            yielded_counts,
            strict=True,
        )
    )


def test_full_effects_merge_canonical_configuration_lists():
    """Equal data effects share one list of canonical fault-index tuples.

    Two full-circuit effects that differ only on a discarded qubit should
    contribute to one retained data effect without overwriting either fault.
    """
    combinator = FaultCombinator.__new__(FaultCombinator)
    x_identity = pauli_mask("X_")
    x_z = pauli_mask("XZ")
    combinator.basis = {
        0: {
            x_identity: 0,
            x_z: 1,
        },
    }
    combinator._indexed_faults = ((0, x_identity), (0, x_z))
    combinator.index_to_bag = {0: (1, 0, 0), 1: (1, 0, 0)}
    combinator.circuit = cast(
        CultivationCircuit,
        SimpleNamespace(
            noisy_circuit=SimpleNamespace(num_qubits=2, num_detectors=1),
        ),
    )
    analyzer = _RecordingAnalyzer(data_indices=(0,))

    kept_effects = combinator.get_kept_effects(
        logical_analyzer=analyzer,
        max_fault_count=1,
        cultivated_states=('S', 'T'),
    )

    data_effect = pauli_mask('X')
    configurations_for_s = kept_effects['S'][1][data_effect][2]
    configurations_for_t = kept_effects['T'][1][data_effect][2]
    assert configurations_for_s == [(0,), (1,)]
    assert isinstance(configurations_for_s, list)
    assert all(
        isinstance(configuration, tuple)
        and all(
            left_index < right_index
            for left_index, right_index in itertools.pairwise(configuration)
        )
        for configuration in configurations_for_s
    )
    assert configurations_for_s is configurations_for_t


def test_d3_four_fault_frozen_regression(
        d3_combinator_and_kept_effects_by_analyzer,
):
    """Check frozen distance-3 results through four faults.

    :param d3_combinator_and_kept_effects_by_analyzer: Session-scoped
        combinator and kept-effect results shared with the analyzer-comparison
        tests.
    """
    combinator, _, kept_effects = d3_combinator_and_kept_effects_by_analyzer
    expected_summaries = {
        'S': [
            (1, 1, 1.0, 0.0),
            (2, 2, 2.0, 0.0),
            (6, 34, 6.0, 0.0),
            (40, 1257, 16.0, 24.0),
            (110, 22653, 46.0, 64.0),
        ],
        'T': [
            (1, 1, 1.0, 0.0),
            (2, 2, 2.0, 0.0),
            (12, 41, 2.75, 1.0),
            (133, 1851, 18.5, 23.75),
            (364, 43193, 55.75, 55.5),
        ],
    }
    for state, expected in expected_summaries.items():
        actual = []
        for effects in kept_effects[state]:
            benign_weight, malignant_weight = _sum_benign_malignant_weights(
                effects.values()
            )
            actual.append((
                len(effects),
                sum(len(triple[2]) for triple in effects.values()),
                benign_weight,
                malignant_weight,
            ))
        assert actual == expected

    contribution_summary = combinator.summarize_contributions(kept_effects)
    assert contribution_summary.index.names == [
        'cultivated_state',
        'fault_count',
    ]
    assert ('weight', 'malignant_probability') in contribution_summary
    assert ('weight', 'error_probability') not in contribution_summary

    noise_levels = (1e-4, 1e-3, 1e-2)
    expected_error_rates = {
        'S': (
            1.1745627300415205e-11,
            1.2214229460983108e-08,
            1.6935408918281458e-05,
        ),
        'T': (
            2.17031232425879e-09,
            2.817318122322684e-07,
            1.0696917642722221e-04,
        ),
    }
    for state, expected in expected_error_rates.items():
        actual = combinator.error_rates_per_kept_shot(
            kept_effects[state],
            noise_levels,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0)


def test_compile_odds_terms_combines_equivalent_fault_bags():
    """Equivalent fault bags share one accumulated classification term.

    :return: None.
    """
    kept_effects = [
        {},
        {
            0: (1.0, 0.75, [(0,), (1,), (2,)]),
        },
    ]

    compiled_terms = _compile_odds_terms(
        all_kept_effects=kept_effects,
        fault_index_to_bag_index=(0, 0, 1),
    )

    bag_index_rows, classification_weights = compiled_terms[1]
    assert bag_index_rows.tolist() == [[0], [1]]
    np.testing.assert_array_equal(
        classification_weights,
        [[1.5, 0.5], [0.75, 0.25]],
    )


def test_error_rate_sweep_accepts_generators_and_reports_progress(capsys):
    """A generated sweep returns ordered rates and reports every noise level.

    :param capsys: Pytest fixture used to capture progress output.
    :return: None.
    """
    combinator = _make_small_fault_combinator()
    kept_effects = [
        {0: (1.0, 1.0, [()])},
        {1: (1.0, 0.5, [(0,)])},
    ]
    noise_levels = (noise_level for noise_level in (1e-3, 2e-3))

    error_rates = combinator.error_rates_per_kept_shot(
        kept_effects,
        noise_levels,
        print_progress=True,
    )

    assert error_rates.shape == (2,)
    output = capsys.readouterr().out
    assert output.count('Noise level = ') == 2
    assert output.count('0-fault configurations:') == 2
    assert output.count('1-fault configurations:') == 2
    assert output.count('Benign odds = ') == 4
    assert output.count('Malignant odds = ') == 4


def test_error_rate_sweep_returns_empty_float_array_without_noise_levels():
    """An empty sweep returns an empty floating-point result.

    :return: None.
    """
    combinator = _make_small_fault_combinator()

    error_rates = combinator.error_rates_per_kept_shot([], ())

    assert error_rates.shape == (0,)
    assert error_rates.dtype == np.float64


def test_d5_four_fault_documented_weights_and_enumeration_counts(monkeypatch):
    """Check distance-5 weights and prechecked enumeration through four faults.

    :param monkeypatch: Pytest fixture used to record configurations yielded by
        the production meet-in-the-middle enumerator.
    """
    enumeration_counts = []
    original_iterator = _iter_undetected_configurations

    def recording_iterator(*, signatures, effects, fault_count):
        """Record configurations yielded at one fault count.

        :param signatures: Packed rejection signatures in fault-index order.
        :param effects: Packed data effects in fault-index order.
        :param fault_count: The number of faults being enumerated.
        :return: An iterator over the original configuration stream.
        """
        count = 0
        for configuration in original_iterator(
                signatures=signatures,
                effects=effects,
                fault_count=fault_count,
        ):
            count += 1
            yield configuration
        enumeration_counts.append(count)

    monkeypatch.setattr(
        'cliffordea.enum.combinators.fault_combinator.'
        '_iter_undetected_configurations',
        recording_iterator,
    )
    circuit = cliffordea.enum.circuits.DoubleCheck(distance=5, ancilla_count=19)
    noisy_circuit = cliffordea.enum.noise.uniformly_depolarize(
        circuit.INNER_CIRCUIT,
        noise_level=1e-3,
    )
    combinator = FaultCombinator(noisy_circuit)
    logical_analyzer = CliffordLogicalAnalyzer(
        data_indices=circuit.DATA_INDICES,
        stabilizer_generators=circuit.STABILIZER_GENERATORS,
        logical_s=circuit.LOGICAL_S,
    )
    kept_effects = combinator.get_kept_effects(
        logical_analyzer=logical_analyzer,
        max_fault_count=4,
        cultivated_states=('S', 'T'),
    )
    expected_weights = {
        'S': [
            (1.0, 0.0),
            (2.0, 0.0),
            (4.0, 0.0),
            (20.0, 0.0),
            (100.0, 0.0),
        ],
        'T': [
            (1.0, 0.0),
            (2.0, 0.0),
            (1.75, 0.0),
            (7.25, 0.1875),
            (50.3125, 6.203125),
        ],
    }
    for state, expected in expected_weights.items():
        assert [
            _sum_benign_malignant_weights(effects.values())
            for effects in kept_effects[state]
        ] == expected
    assert enumeration_counts == [1, 21, 493, 14259, 351145]
