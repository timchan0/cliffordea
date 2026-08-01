from collections import Counter, defaultdict
import itertools
import math
from types import SimpleNamespace
from typing import cast

import pytest
import stim

import cliffordep
from cliffordep.combinators import FaultCombinator
from cliffordep.combinators.fault_combinator import (
    _make_logical_analysis_cache,
    _make_pauli_mask_restrictor,
    _pauli_mask_to_unsigned_string,
    _sum_logical_weights,
    _unsigned_pauli_string_to_mask,
)
from cliffordep.logical_analyzers import CliffordLogicalAnalyzer, LogicalAnalyzer
from cliffordep.noisy_circuit_tools import CultivationCircuit
from cliffordep.pauli_string_tools import forget_sign


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
        actual = _pauli_mask_to_unsigned_string(
            _unsigned_pauli_string_to_mask(left)
            ^ _unsigned_pauli_string_to_mask(right),
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
        product_mask ^= _unsigned_pauli_string_to_mask(factor)
    assert _pauli_mask_to_unsigned_string(product_mask, qubit_count) == expected


@pytest.mark.parametrize("unsigned_string", ["", "_", "I", "XYZ_", "Y_YXZI"])
def test_pauli_mask_round_trip(unsigned_string: str):
    """Check that packing and decoding preserves each Pauli symbol.

    Identity is canonicalized from ``I`` to ``_`` while ``Y`` must preserve
    both its X and Z support.

    :param unsigned_string: The unsigned Pauli string to round-trip.
    """
    canonical = unsigned_string.replace('I', '_')
    mask = _unsigned_pauli_string_to_mask(unsigned_string)
    assert _pauli_mask_to_unsigned_string(mask, len(unsigned_string)) == canonical


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
        _unsigned_pauli_string_to_mask(source),
        data_indices=data_indices,
        source_qubit_count=len(source),
    )
    assert _pauli_mask_to_unsigned_string(
        restricted_mask,
        len(data_indices),
    ) == ''.join(source[index] for index in data_indices)
    fast_restrict = _make_pauli_mask_restrictor(
        data_indices=data_indices,
        source_qubit_count=len(source),
    )
    assert fast_restrict(_unsigned_pauli_string_to_mask(source)) == restricted_mask


def test_repeated_syndrome_uses_distinct_fault_indices():
    """Check that one syndrome group cannot reuse a fault index.

    Selecting two faults from a three-fault group should produce exactly the
    three ordinary combinations of distinct indices.
    """
    combinator = FaultCombinator.__new__(FaultCombinator)
    syndrome = (True,)
    combinator._mask_basis = {
        syndrome: (
            (_unsigned_pauli_string_to_mask("X"), 0),
            (_unsigned_pauli_string_to_mask("Y"), 1),
            (_unsigned_pauli_string_to_mask("Z"), 2),
        ),
    }

    configurations = list(
        combinator._iter_configurations_for_syndrome_counter(
            Counter({syndrome: 2})
        )
    )

    assert len(configurations) == 3
    assert {indices for _, indices in configurations} == {
        (0, 1),
        (0, 2),
        (1, 2),
    }
    assert all(len(indices) == len(set(indices)) for _, indices in configurations)


def test_order_zero_yields_identity_configuration():
    """Check the base case of lazy configuration enumeration.

    Order zero should yield exactly the empty fault-index tuple with the packed
    identity effect.
    """
    combinator = FaultCombinator.__new__(FaultCombinator)
    combinator._mask_basis = {(False,): ((1, 0),)}
    assert list(combinator._iter_undetected_configurations_for_order(0)) == [
        (0, ())
    ]


def test_syndrome_group_recursion_is_lazy(monkeypatch):
    """Check that requesting one result does not consume later combinations.

    :param monkeypatch: Pytest fixture used to replace
        :func:`itertools.combinations` with an iterator that fails if a second
        item is requested.
    """
    combinator = FaultCombinator.__new__(FaultCombinator)
    syndrome = (False,)
    combinator._mask_basis = {syndrome: ((1, 0), (2, 1))}
    original_combinations = itertools.combinations

    def guarded_combinations(iterable, count):
        """Yield one combination and fail if enumeration continues eagerly.

        :param iterable: The fault entries from which combinations are drawn.
        :param count: The number of entries in each combination.

        :yield: The first combination produced by :mod:`itertools`.
        """
        iterator = original_combinations(iterable, count)
        yield next(iterator)
        raise AssertionError("The remaining combinations were consumed eagerly.")

    monkeypatch.setattr(
        "cliffordep.combinators.fault_combinator.itertools.combinations",
        guarded_combinations,
    )
    generator = combinator._iter_configurations_for_syndrome_counter(
        Counter({syndrome: 1})
    )
    assert next(generator) == (1, (0,))


class _RecordingAnalyzer(LogicalAnalyzer):
    def __init__(self, data_indices: tuple[int, ...] = ()):
        """Initialize an analyzer that records each logical-analysis request.

        :param data_indices: Source-qubit indices treated as data qubits.
        """
        self.DATA_INDICES = data_indices
        self.calls: list[tuple[str, str]] = []

    def analyze(
            self,
            cultivated_state: str,
            before_transversal: str,
    ) -> tuple[float, float]:
        """Record an analysis call and return deterministic synthetic values.

        :param cultivated_state: The logical state being cultivated.
        :param before_transversal: The unsigned data-qubit Pauli effect.

        :return: A synthetic acceptance probability and logical fidelity.
        """
        self.calls.append((cultivated_state, before_transversal))
        return (
            float(before_transversal.count('X') + 1),
            float(cultivated_state == 'S'),
        )


@pytest.mark.parametrize("maxsize", [0, 1, 262_144, None])
def test_logical_analysis_cache_sizes_have_identical_results(maxsize: int | None):
    """Check that cache capacity affects reuse but not analysis results.

    :param maxsize: The disabled, bounded, default-sized, or unbounded cache
        capacity under test.
    """
    analyzer = _RecordingAnalyzer()
    analyze_mask = _make_logical_analysis_cache(
        logical_analyzer=analyzer,
        cultivated_states=('S', 'T'),
        data_qubit_count=2,
        maxsize=maxsize,
    )
    access_sequence = (0, 1, 2, 0, 1, 2)
    actual = [analyze_mask(mask) for mask in access_sequence]
    expected = [
        (
            (float(_pauli_mask_to_unsigned_string(mask, 2).count('X') + 1), 1.0),
            (float(_pauli_mask_to_unsigned_string(mask, 2).count('X') + 1), 0.0),
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


def test_fault_combinator_owns_kept_string_enumeration():
    """Check that only ``FaultCombinator`` exposes the fused enumeration API.

    The former two-step methods should not remain on either collaborating
    class after the intentional breaking API change.
    """
    assert not hasattr(FaultCombinator, "get_undetected_configurations")
    assert not hasattr(LogicalAnalyzer, "get_kept_strings")


def _exhaustive_kept_strings_oracle(
        combinator: FaultCombinator,
        logical_analyzer,
        max_order: int,
        cultivated_states: tuple[str, ...],
):
    """Compute kept strings with an independent exhaustive subset search.

    This oracle uses Stim multiplication and explicitly checks the syndrome of
    every fault subset, avoiding the recursive enumeration and packed-effect
    multiplication used by :meth:`FaultCombinator.get_kept_strings`.

    :param combinator: The fault combinator whose indexed basis is enumerated.
    :param logical_analyzer: The analyzer used to postselect each data effect.
    :param max_order: The largest fault-subset size to enumerate.
    :param cultivated_states: The logical states to analyze.

    :return kept_strings: Kept-string results in the same per-state, per-order
        structure as :meth:`FaultCombinator.get_kept_strings`.
    """
    fault_by_index = {
        index: (syndrome, effect)
        for syndrome, effect_to_index in combinator.basis.items()
        for effect, index in effect_to_index.items()
    }
    faults = tuple(sorted(fault_by_index.items()))
    detector_count = len(next(iter(combinator.basis)))
    result = {state: [] for state in cultivated_states}
    for order in range(max_order + 1):
        configurations_by_state = {
            state: defaultdict(set) for state in cultivated_states
        }
        analyses_by_state = {state: {} for state in cultivated_states}
        for combination in itertools.combinations(faults, order):
            resultant_syndrome = tuple(
                sum(fault[1][0][detector] for fault in combination) % 2
                for detector in range(detector_count)
            )
            if any(resultant_syndrome):
                continue
            full_effect = math.prod(
                (stim.PauliString(fault[1][1]) for fault in combination),
                start=stim.PauliString(
                    combinator.circuit.noisy_circuit.num_qubits
                ),
            )
            full_effect_string = forget_sign(full_effect)
            data_effect = ''.join(
                full_effect_string[index]
                for index in logical_analyzer.DATA_INDICES
            )
            fault_indices = frozenset(fault[0] for fault in combination)
            for state in cultivated_states:
                if data_effect not in analyses_by_state[state]:
                    analyses_by_state[state][data_effect] = (
                        logical_analyzer.analyze(state, data_effect)
                    )
                analysis = analyses_by_state[state][data_effect]
                if analysis[0]:
                    configurations_by_state[state][data_effect].add(
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
            before_transversal: str,
    ) -> tuple[float, float]:
        """Return synthetic acceptance and fidelity for an unsigned effect.

        :param cultivated_state: The logical state being cultivated.
        :param before_transversal: The unsigned data-qubit Pauli effect.

        :return: A deterministic acceptance probability and logical fidelity
            chosen to exercise rejection and state-dependent output.
        """
        if before_transversal.startswith('Z'):
            return 0.0, 0.0
        acceptance_probability = 0.5 if 'Y' in before_transversal else 1.0
        logical_fidelity = float(
            (before_transversal.count('X') + (cultivated_state == 'S')) % 2 == 0
        )
        return acceptance_probability, logical_fidelity


def _make_synthetic_combinator() -> FaultCombinator:
    """Build a small combinator with three cancelling nonzero syndromes.

    The synthetic A, B, and C syndrome groups satisfy ``A XOR B XOR C = 0``
    and contain enough effects to exercise repeated and three-way selection.

    :return combinator: A minimally initialized combinator suitable for the
        independent exhaustive-oracle comparison.
    """
    indexed_faults = (
        ((False, False), "Z_Z", 0),
        ((True, False), "X__", 1),
        ((True, False), "_X_", 2),
        ((False, True), "__Z", 3),
        ((False, True), "Y__", 4),
        ((True, True), "_Y_", 5),
        ((True, True), "ZZ_", 6),
    )
    basis: defaultdict[tuple[bool, ...], dict[str, int]] = defaultdict(dict)
    for syndrome, effect, index in indexed_faults:
        basis[syndrome][effect] = index
    combinator = FaultCombinator.__new__(FaultCombinator)
    combinator.basis = dict(basis)
    combinator._mask_basis = {
        syndrome: tuple(
            (_unsigned_pauli_string_to_mask(effect), index)
            for effect, index in effects.items()
        )
        for syndrome, effects in combinator.basis.items()
    }
    combinator.circuit = cast(
        CultivationCircuit,
        SimpleNamespace(noisy_circuit=SimpleNamespace(num_qubits=3)),
    )
    return combinator


def test_synthetic_order_three_matches_exhaustive_subset_oracle():
    """Compare lazy enumeration with exhaustive three-way cancellation.

    The synthetic basis ensures that an undetected order-three configuration
    can arise from three different nonzero syndrome groups.
    """
    combinator = _make_synthetic_combinator()
    logical_analyzer = _SyntheticAnalyzer()
    cultivated_states = ('S', 'T')
    actual = combinator.get_kept_strings(
        logical_analyzer=logical_analyzer,
        max_order=3,
        cultivated_states=cultivated_states,
    )
    expected = _exhaustive_kept_strings_oracle(
        combinator,
        logical_analyzer,
        max_order=3,
        cultivated_states=cultivated_states,
    )
    assert actual == expected
    assert frozenset({1, 4, 5}) in actual['S'][3]['_Z'][2]


def test_d3_cache_sizes_preserve_kept_strings_and_share_configuration_sets():
    """Check cache-size invariance and cross-state configuration sharing.

    Disabled, bounded, default-sized, and unbounded caches must retain equal
    distance-3 results. Effects accepted for both S and T should also refer to
    the same configuration-set object.
    """
    circuit = cliffordep.circuits.D3DoubleCatCheckA6()
    noisy_circuit = cliffordep.noise.uniformly_depolarize(
        circuit.INNER_CIRCUIT,
        noise_level=1e-3,
    )
    combinator = FaultCombinator(noisy_circuit)
    logical_analyzer = CliffordLogicalAnalyzer(
        data_indices=circuit.DATA_INDICES,
        stabilizer_generators=circuit.STABILIZER_GENERATORS_RESTRICTED,
        logical_s=circuit.LOGICAL_S,
    )
    results_by_cache_size = {
        maxsize: combinator.get_kept_strings(
            logical_analyzer=logical_analyzer,
            max_order=2,
            cultivated_states=('S', 'T'),
            logical_analysis_cache_maxsize=maxsize,
        )
        for maxsize in (0, 1, 262_144, None)
    }
    expected = results_by_cache_size[262_144]
    assert all(result == expected for result in results_by_cache_size.values())

    for order, strings_for_s in enumerate(expected['S']):
        strings_for_t = expected['T'][order]
        for effect in strings_for_s.keys() & strings_for_t.keys():
            assert strings_for_s[effect][2] is strings_for_t[effect][2]


def test_full_effects_with_same_data_restriction_merge_configurations():
    """Check that equal data restrictions merge their fault configurations.

    Two full-circuit effects that differ only on a discarded qubit should
    contribute to one retained data effect without overwriting either fault.
    """
    combinator = FaultCombinator.__new__(FaultCombinator)
    zero_syndrome = (False,)
    combinator._mask_basis = {
        zero_syndrome: (
            (_unsigned_pauli_string_to_mask("X_"), 0),
            (_unsigned_pauli_string_to_mask("XZ"), 1),
        ),
    }
    combinator.circuit = cast(
        CultivationCircuit,
        SimpleNamespace(noisy_circuit=SimpleNamespace(num_qubits=2)),
    )
    analyzer = _RecordingAnalyzer(data_indices=(0,))

    kept_strings = combinator.get_kept_strings(
        logical_analyzer=analyzer,
        max_order=1,
        cultivated_states=('S', 'T'),
    )

    configurations_for_s = kept_strings['S'][1]['X'][2]
    configurations_for_t = kept_strings['T'][1]['X'][2]
    assert configurations_for_s == {frozenset({0}), frozenset({1})}
    assert configurations_for_s is configurations_for_t


def test_d3_order_four_frozen_regression(
        d3_combinator_and_kept_strings_by_analyzer,
):
    """Check frozen distance-3 summaries and logical error rates through order four.

    :param d3_combinator_and_kept_strings_by_analyzer: Session-scoped
        combinator and kept-string results shared with the analyzer-comparison
        tests.
    """
    combinator, _, kept_strings = d3_combinator_and_kept_strings_by_analyzer
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
        for strings in kept_strings[state]:
            identity_weight, error_weight = _sum_logical_weights(
                strings.values()
            )
            actual.append((
                len(strings),
                sum(len(triple[2]) for triple in strings.values()),
                identity_weight,
                error_weight,
            ))
        assert actual == expected

    assert combinator.error_rate_per_kept_shot(
        kept_strings['S'], 1e-3
    ) == pytest.approx(1.2214229460983106e-08)
    assert combinator.error_rate_per_kept_shot(
        kept_strings['T'], 1e-3
    ) == pytest.approx(2.817318122322684e-07)


def test_d5_through_order_three_documented_weights():
    """Check documented distance-5 S/T weights through order three.

    This integration regression exercises the real distance-5 circuit while
    remaining substantially cheaper than its order-four workflow.
    """
    circuit = cliffordep.circuits.D5DoubleCatCheckA19()
    noisy_circuit = cliffordep.noise.uniformly_depolarize(
        circuit.INNER_CIRCUIT,
        noise_level=1e-3,
    )
    combinator = FaultCombinator(noisy_circuit)
    logical_analyzer = CliffordLogicalAnalyzer(
        data_indices=circuit.DATA_INDICES,
        stabilizer_generators=circuit.STABILIZER_GENERATORS_RESTRICTED,
        logical_s=circuit.LOGICAL_S,
    )
    kept_strings = combinator.get_kept_strings(
        logical_analyzer=logical_analyzer,
        max_order=3,
        cultivated_states=('S', 'T'),
    )
    expected_weights = {
        'S': [(1.0, 0.0), (2.0, 0.0), (4.0, 0.0), (20.0, 0.0)],
        'T': [(1.0, 0.0), (2.0, 0.0), (1.75, 0.0), (7.25, 0.1875)],
    }
    for state, expected in expected_weights.items():
        assert [
            _sum_logical_weights(strings.values())
            for strings in kept_strings[state]
        ] == expected
