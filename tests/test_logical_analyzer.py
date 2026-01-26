import pytest

from cliffordep.logical_analyzers import LogicalAnalyzer, TableauLogicalAnalyzer, SuperpositionLogicalAnalyzer
from cliffordep import circuits
from cliffordep.pauli_string_tools import forget_sign


class TestDistance3:
    
    @pytest.fixture(params=[TableauLogicalAnalyzer, SuperpositionLogicalAnalyzer])
    def analyzer(self, request) -> LogicalAnalyzer:
        class_ = request.param
        circuit = circuits.D3DoubleCatCheckA6()
        return class_(
            data_indices=circuit.DATA_INDICES,
            stabilizer_generators=circuit.STABILIZER_GENERATORS_RESTRICTED,
            logical_s=circuit.LOGICAL_S,
        )


    def test_x_tensor_n(self, analyzer: LogicalAnalyzer):
        ps = forget_sign(analyzer.X_TENSOR_N)
        assert analyzer.analyze('T', ps) == (1.0, True)


    def test_abort(self, analyzer: LogicalAnalyzer):
        restricted = analyzer.restrict_to_data('___Z_X_XX____')
        assert analyzer.analyze('T', restricted) == (0.0, False)


class TestDistance5:
    
    @pytest.fixture
    def analyzer(self):
        circuit = circuits.D5DoubleCatCheckA19()
        return TableauLogicalAnalyzer(
            data_indices=circuit.DATA_INDICES,
            stabilizer_generators=circuit.STABILIZER_GENERATORS_RESTRICTED,
            logical_s=circuit.LOGICAL_S,
        )


    def test_x_tensor_n(self, analyzer: TableauLogicalAnalyzer):
        ps = forget_sign(analyzer.X_TENSOR_N)
        assert analyzer.analyze('T', ps) == (1.0, True)


    def test_abort(self, analyzer: TableauLogicalAnalyzer):
        ps = analyzer.restrict_to_data('X__X_X_X_X_X_XX_Y_X_X_X_X_X__X_XX_X_X_')
        assert analyzer.analyze('T', ps) == (0.0, False)