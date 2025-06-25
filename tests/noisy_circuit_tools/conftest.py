import pytest
import stim

from cliffordep import CultivationCircuit
from cliffordep.type_aliases import EffectMap


@pytest.fixture
def d3_double_cat_check_grouped_by_source(noisy_d3_double_cat_check: CultivationCircuit):
    """Output of `_group_faults_by_source` for the distance-3 double cat check circuit."""
    return noisy_d3_double_cat_check._group_faults_by_source()


@pytest.fixture
def d3_double_cat_check_effect_maps(noisy_d3_double_cat_check: CultivationCircuit) -> dict[tuple[bool, ...], EffectMap]:
    """Output of `group_faults_by_effect` for the distance-3 double cat check circuit."""
    effect_maps = noisy_d3_double_cat_check.group_faults_by_effect()
    return effect_maps


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