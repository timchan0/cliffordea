import pytest
import stim

from cliffordep.combinators import FaultSourceCombinator
from cliffordep.noisy_circuit_tools import CultivationCircuit


@pytest.fixture
def d3_double_cat_check_grouped_by_source(noisy_d3_double_cat_check: CultivationCircuit):
    """Output of `group_faults_by_source` for the distance-3 double cat check circuit."""
    return noisy_d3_double_cat_check.group_faults_by_source()


@pytest.fixture
def d3_double_cat_check_fault_source_combinator(noisy_d3_double_cat_check: CultivationCircuit) -> FaultSourceCombinator:
    """Fault source combinator for the distance-3 double cat check circuit."""
    combinator = FaultSourceCombinator(noisy_d3_double_cat_check)
    return combinator


@pytest.fixture
def multi_qubit_detector_circuit():
    return CultivationCircuit(
        noisy_circuit=stim.Circuit("""
            CX 0 1
            TICK
            MZ 0 1
            DETECTOR rec[-2]
            DETECTOR rec[-1]
        """),
        data_indices= (0, 1),
        stabilizer_generators=(stim.PauliString(),),
        logical_x=stim.PauliString(),
        logical_z=stim.PauliString(),
    )