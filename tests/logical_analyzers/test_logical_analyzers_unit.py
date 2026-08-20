import pytest
import stim

from cliffordep.logical_analyzers import (
    CliffordLogicalAnalyzer,
    LogicalAnalyzer,
    SuperpositionLogicalAnalyzer,
    _pauli_masks_and_j_power,
    LOGICAL_COEFFICIENTS,
)
from cliffordep import circuits
from cliffordep.combinators.fault_combinator import (
    _make_pauli_mask_restrictor,
    _unsigned_pauli_string_to_mask,
)
from cliffordep.pauli_string_tools import forget_sign

class TestPauliMasksAndJPower:
    
    def test_uses_x_then_z_convention(self):
        """Check that mask conversion matches the i^phi X(x)Z(z) convention."""
        assert _pauli_masks_and_j_power(stim.PauliString("+Y")) == (0b1, 0b1, 1)
        assert _pauli_masks_and_j_power(stim.PauliString("-Y")) == (0b1, 0b1, 3)


class TestCliffordLogicalAnalyzer:

    @pytest.fixture
    def bitflip_repetition_code(self):
        return CliffordLogicalAnalyzer(
            data_indices=(0, 1, 2),
            stabilizer_generators={
                'X': (stim.PauliString("ZZ_"), stim.PauliString("_ZZ")),
                'Z': (),
            },
            logical_s=stim.Circuit("S_DAG 0 1 2"),
        )

    def test_accept_probability_includes_logical_coefficients(self, bitflip_repetition_code: CliffordLogicalAnalyzer):
        """Check that a non-identity logical fibre contributes correctly to acceptance."""
        unencoded_error = stim.Tableau.from_circuit(stim.Circuit("""
            H 2
            SWAP 0 2
        """))
        assert bitflip_repetition_code._get_accept_probability(unencoded_error, LOGICAL_COEFFICIENTS['maximally_mixed']) == 0.5
        assert bitflip_repetition_code._get_accept_probability(unencoded_error, LOGICAL_COEFFICIENTS['+']) == 1.0


    def test_accept_probability_rejects_negative_zero_column(self, bitflip_repetition_code: CliffordLogicalAnalyzer):
        """Check that a zero rho column with negative sign forces rejection."""
        unencoded_error = stim.Tableau.from_named_gate("X") + stim.Tableau(2)
        assert bitflip_repetition_code._get_accept_probability(unencoded_error, LOGICAL_COEFFICIENTS['maximally_mixed']) == 0.0


class TestDistance3:
    
    @pytest.fixture(params=[CliffordLogicalAnalyzer, SuperpositionLogicalAnalyzer])
    def analyzer(self, request) -> LogicalAnalyzer:
        class_ = request.param
        circuit = circuits.D3A6()
        return class_(
            data_indices=circuit.DATA_INDICES,
            stabilizer_generators=circuit.STABILIZER_GENERATORS_RESTRICTED,
            logical_s=circuit.LOGICAL_S,
        )


    def test_x_tensor_n(self, analyzer: LogicalAnalyzer):
        effect_mask = _unsigned_pauli_string_to_mask(
            forget_sign(analyzer.X_TENSOR_N),
        )
        assert analyzer.analyze('T', effect_mask) == (1.0, True)


    def test_abort(self, analyzer: LogicalAnalyzer):
        full_effect = '___Z_X_XX____'
        restrict_effect = _make_pauli_mask_restrictor(
            data_indices=analyzer.DATA_INDICES,
            source_qubit_count=len(full_effect),
        )
        data_effect = restrict_effect(
            _unsigned_pauli_string_to_mask(full_effect),
        )
        assert analyzer.analyze('T', data_effect) == (0.0, False)


    def test_nontrivial_acceptance_probability(self, analyzer: LogicalAnalyzer):
        """Check a distance-3 example whose trivial-syndrome probability is 1/4."""
        effect_mask = _unsigned_pauli_string_to_mask('YX_X___')
        assert analyzer.analyze('T', effect_mask) == (0.25, False)


    def test_clifford_tableau_is_encoding_circuit(self):
        """Check that the stabilizer-overlap analyzer fixes the logical X frame."""
        circuit = circuits.D3A6()
        analyzer = CliffordLogicalAnalyzer(
            data_indices=circuit.DATA_INDICES,
            stabilizer_generators=circuit.STABILIZER_GENERATORS_RESTRICTED,
            logical_s=circuit.LOGICAL_S,
        )
        stabilizer_generators = (
            *circuit.STABILIZER_GENERATORS_RESTRICTED['X'],
            *circuit.STABILIZER_GENERATORS_RESTRICTED['Z'],
        )
        for index, stabilizer_generator in enumerate(stabilizer_generators):
            assert analyzer.encoder.z_output(index) == stabilizer_generator
        assert analyzer.encoder.z_output(analyzer.stabilizer_rank) == analyzer.Z_TENSOR_N
        assert analyzer.encoder.x_output(analyzer.stabilizer_rank) == analyzer.X_TENSOR_N


    def test_encoding_tableau_supports_multiple_logical_qubits(self):
        """Check that the encoding-tableau helper handles a two-logical-qubit frame."""
        stabilizers = (stim.PauliString("Z__"),)
        logical_zs = (
            stim.PauliString("_Z_"),
            stim.PauliString("__Z"),
        )
        logical_xs = (
            stim.PauliString("ZX_"),
            stim.PauliString("__X"),
        )
        logical_zero_tableau = stim.Tableau.from_stabilizers((*stabilizers, *logical_zs))
        assert not logical_zero_tableau.x_output(0).commutes(logical_xs[0])

        tableau = CliffordLogicalAnalyzer._get_encoding_tableau(
            stabilizers=stabilizers,
            logical_zs=logical_zs,
            logical_xs=logical_xs,
        )
        assert tableau.x_output(0).commutes(logical_xs[0])
        assert tableau.z_output(0) == stabilizers[0]
        for logical_index, (logical_z, logical_x) in enumerate(
                zip(logical_zs, logical_xs, strict=True),
                start=len(stabilizers),
        ):
            assert tableau.z_output(logical_index) == logical_z
            assert tableau.x_output(logical_index) == logical_x


    @pytest.mark.parametrize("restricted", [
        "_______",
        "XXXXXXX",
        "YX_X___",
        "_X__XX_",
        "Z______",
        "XYZXYZX",
    ])
    def test_clifford_matches_pauli_sum(self, restricted: str):
        """Compare the stabilizer-overlap analyzer to the existing pauli sum analyzer."""
        circuit = circuits.D3A6()
        clifford_analyzer = CliffordLogicalAnalyzer(
            data_indices=circuit.DATA_INDICES,
            stabilizer_generators=circuit.STABILIZER_GENERATORS_RESTRICTED,
            logical_s=circuit.LOGICAL_S,
        )
        pauli_sum_analyzer = SuperpositionLogicalAnalyzer(
            data_indices=circuit.DATA_INDICES,
            stabilizer_generators=circuit.STABILIZER_GENERATORS_RESTRICTED,
            logical_s=circuit.LOGICAL_S,
        )
        effect_mask = _unsigned_pauli_string_to_mask(restricted)
        clifford_probability, clifford_fidelity = clifford_analyzer.analyze(
            'T', effect_mask,
        )
        pauli_sum_probability, pauli_sum_fidelity = pauli_sum_analyzer.analyze(
            'T', effect_mask,
        )
        assert clifford_probability == pauli_sum_probability
        if clifford_probability > 0:
            assert clifford_fidelity == pauli_sum_fidelity


class TestDistance5:
    
    @pytest.fixture
    def analyzer(self):
        circuit = circuits.D5A19()
        return CliffordLogicalAnalyzer(
            data_indices=circuit.DATA_INDICES,
            stabilizer_generators=circuit.STABILIZER_GENERATORS_RESTRICTED,
            logical_s=circuit.LOGICAL_S,
        )


    def test_x_tensor_n(self, analyzer: CliffordLogicalAnalyzer):
        effect_mask = _unsigned_pauli_string_to_mask(
            forget_sign(analyzer.X_TENSOR_N),
        )
        assert analyzer.analyze('T', effect_mask) == (1.0, True)


    def test_abort(self, analyzer: CliffordLogicalAnalyzer):
        full_effect = 'X__X_X_X_X_X_XX_Y_X_X_X_X_X__X_XX_X_X_'
        restrict_effect = _make_pauli_mask_restrictor(
            data_indices=analyzer.DATA_INDICES,
            source_qubit_count=len(full_effect),
        )
        data_effect = restrict_effect(
            _unsigned_pauli_string_to_mask(full_effect),
        )
        assert analyzer.analyze('T', data_effect) == (0.0, False)


    @pytest.mark.parametrize("restricted", [
        "XX_X__X___Y________",
        "X__________________",
        "Y__________________",
        "Z__________________",
    ])
    def test_z_stabilizer_precheck_matches_general_path(self, restricted: str):
        circuit = circuits.D5A19()
        enabled_analyzer = CliffordLogicalAnalyzer(
            data_indices=circuit.DATA_INDICES,
            stabilizer_generators=circuit.STABILIZER_GENERATORS_RESTRICTED,
            logical_s=circuit.LOGICAL_S,
            precheck_z_stabilizers=True,
        )
        disabled_analyzer = CliffordLogicalAnalyzer(
            data_indices=circuit.DATA_INDICES,
            stabilizer_generators=circuit.STABILIZER_GENERATORS_RESTRICTED,
            logical_s=circuit.LOGICAL_S,
            precheck_z_stabilizers=False,
        )
        effect_mask = _unsigned_pauli_string_to_mask(restricted)
        assert enabled_analyzer.analyze(
            'T', effect_mask,
        ) == disabled_analyzer.analyze('T', effect_mask)
        assert disabled_analyzer.linear_precheck_syndrome(effect_mask) == 0


@pytest.mark.parametrize("precheck_z_stabilizers", [True, False])
def test_factored_transversal_matches_general_path_exhaustively(
        precheck_z_stabilizers: bool,
):
    """All D3 Pauli effects agree for every supported diagonal state.

    :param precheck_z_stabilizers: Whether both analyzers apply the preliminary
        pure-Z stabilizer checks.
    :return: None.
    """
    circuit = circuits.D3A6()
    analyzer_arguments = {
        'data_indices': circuit.DATA_INDICES,
        'stabilizer_generators': circuit.STABILIZER_GENERATORS_RESTRICTED,
        'logical_s': circuit.LOGICAL_S,
        'precheck_z_stabilizers': precheck_z_stabilizers,
    }
    factored_analyzer = CliffordLogicalAnalyzer(
        **analyzer_arguments,
        factor_transversal_errors=True,
    )
    general_analyzer = CliffordLogicalAnalyzer(
        **analyzer_arguments,
        factor_transversal_errors=False,
    )
    effect_count = 1 << (2 * len(circuit.DATA_INDICES))

    for cultivated_state in ('T', 'S', 'Z'):
        for effect_mask in range(effect_count):
            assert factored_analyzer.analyze(
                cultivated_state,
                effect_mask,
            ) == general_analyzer.analyze(cultivated_state, effect_mask)


@pytest.mark.parametrize(
    ("maxsize", "evicts"),
    [(0, True), (1, True), (4_096, False), (None, False)],
)
def test_transversal_structure_cache_respects_maxsize(
        maxsize: int | None,
        evicts: bool,
):
    """Cache bounds control reuse without changing the built structures.

    :param maxsize: Maximum number of structures retained by the analyzer.
    :param evicts: Whether requesting a second X support removes the first.
    :return: None.
    """
    circuit = circuits.D3A6()
    analyzer = CliffordLogicalAnalyzer(
        data_indices=circuit.DATA_INDICES,
        stabilizer_generators=circuit.STABILIZER_GENERATORS_RESTRICTED,
        logical_s=circuit.LOGICAL_S,
        precheck_z_stabilizers=False,
        transversal_structure_cache_maxsize=maxsize,
    )

    first_structure = analyzer._get_transversal_structure('T', 0b1)
    repeated_structure = analyzer._get_transversal_structure('T', 0b1)
    if maxsize == 0:
        assert repeated_structure is not first_structure
    else:
        assert repeated_structure is first_structure

    analyzer._get_transversal_structure('T', 0b10)
    rebuilt_structure = analyzer._get_transversal_structure('T', 0b1)
    assert (rebuilt_structure is not first_structure) is evicts
    if maxsize is not None:
        assert len(analyzer._transversal_structure_cache) <= maxsize


def test_z_variants_share_transversal_structure():
    """Effects with equal X support reuse one cached acceptance structure.

    :return: None.
    """
    circuit = circuits.D3A6()
    analyzer = CliffordLogicalAnalyzer(
        data_indices=circuit.DATA_INDICES,
        stabilizer_generators=circuit.STABILIZER_GENERATORS_RESTRICTED,
        logical_s=circuit.LOGICAL_S,
        precheck_z_stabilizers=False,
    )
    effect_x = 0b101
    analyzer.analyze('T', effect_x)
    first_structure, = analyzer._transversal_structure_cache.values()

    effect_z = 0b010 << len(circuit.DATA_INDICES)
    analyzer.analyze('T', effect_x | effect_z)

    repeated_structure, = analyzer._transversal_structure_cache.values()
    assert repeated_structure is first_structure


@pytest.mark.parametrize(
    ("factor_transversal_errors", "supports_xz_factorization"),
    [(False, True), (True, False)],
)
def test_inapplicable_factoring_uses_general_path(
        factor_transversal_errors: bool,
        supports_xz_factorization: bool,
        monkeypatch: pytest.MonkeyPatch,
):
    """Disabled or unsupported factorization falls back without caching.

    :param factor_transversal_errors: Whether the candidate analyzer enables
        the optional factored path.
    :param supports_xz_factorization: Whether its T transversal advertises the
        diagonal factorization capability.
    :param monkeypatch: Pytest helper that restores the shared gate capability
        after the test.
    :return: None.
    """
    circuit = circuits.D3A6()
    analyzer_arguments = {
        'data_indices': circuit.DATA_INDICES,
        'stabilizer_generators': circuit.STABILIZER_GENERATORS_RESTRICTED,
        'logical_s': circuit.LOGICAL_S,
    }
    candidate = CliffordLogicalAnalyzer(
        **analyzer_arguments,
        factor_transversal_errors=factor_transversal_errors,
    )
    monkeypatch.setattr(
        candidate.LOGICAL['T'],
        'supports_xz_factorization',
        supports_xz_factorization,
    )
    general = CliffordLogicalAnalyzer(
        **analyzer_arguments,
        factor_transversal_errors=False,
    )
    effect_mask = _unsigned_pauli_string_to_mask('YX_X___')

    assert candidate.analyze('T', effect_mask) == general.analyze(
        'T',
        effect_mask,
    )
    assert not candidate._transversal_structure_cache
