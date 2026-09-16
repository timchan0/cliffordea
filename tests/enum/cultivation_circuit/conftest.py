import pytest
import stim

from cliffordea.enum.combinators import DisjointFaultCombinator
from cliffordea.enum.cultivation_circuit import CultivationCircuit


@pytest.fixture
def d3_double_cat_check_grouped_by_location(noisy_d3_double_cat_check: CultivationCircuit):
    """Output of `group_error_events_by_location` for the distance-3 double-check circuit."""
    return noisy_d3_double_cat_check.group_error_events_by_location()


@pytest.fixture
def d3_double_cat_check_disjoint_fault_combinator(noisy_d3_double_cat_check_circuit: stim.Circuit):
    """Disjoint fault combinator for the distance-3 double-check circuit."""
    combinator = DisjointFaultCombinator(noisy_d3_double_cat_check_circuit)
    return combinator


@pytest.fixture
def multi_qubit_detector_circuit():
    return CultivationCircuit(noisy_circuit=stim.Circuit("""
            CX 0 1
            TICK
            MZ 0 1
            DETECTOR rec[-2]
            DETECTOR rec[-1]
        """))
