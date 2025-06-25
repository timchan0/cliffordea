import numpy as np
import numpy.typing as npt
import pytest
import stim

import cliffordep
from cliffordep.brute_force import BruteCultivationCircuit
from cliffordep.type_aliases import Fault, FaultSource


@pytest.fixture
def noisy_d3_double_cat_check_circuit():
    """Noisy version of the distance-3 double cat check circuit."""
    circuit = cliffordep.circuits.D3_DOUBLE_CAT_CHECK
    noisy_circuit = cliffordep.noise.uniformly_depolarize(circuit, noise_level=1e-3)
    # remove last layer of depolarizing noise
    noisy_circuit = noisy_circuit[:-1]
    return noisy_circuit


@pytest.fixture
def noisy_d3_double_cat_check(noisy_d3_double_cat_check_circuit: stim.Circuit):
    """Noisy version of the distance-3 double cat check circuit."""
    return cliffordep.CultivationCircuit(
        noisy_d3_double_cat_check_circuit,
        data_indices=cliffordep.circuits.D3_DOUBLE_CAT_CHECK_DATA_INDICES,
        stabilizer_generators=cliffordep.circuits.D3_DOUBLE_CAT_CHECK_STABILIZER_GENERATORS,
        logical_x=cliffordep.circuits.D3_DOUBLE_CAT_CHECK_LOGICAL_X,
        logical_z=cliffordep.circuits.D3_DOUBLE_CAT_CHECK_LOGICAL_Z,
    )


@pytest.fixture
def d3_double_cat_check_group_brute(noisy_d3_double_cat_check_brute: BruteCultivationCircuit) -> dict[
    FaultSource, dict[Fault, tuple[npt.NDArray[np.bool_], str]]]:
    """Output of `brute_force.group_faults_by_source` for the distance-3 double cat check circuit."""
    effect_maps = noisy_d3_double_cat_check_brute.group_faults_by_source()
    return effect_maps


@pytest.fixture
def noisy_d3_double_cat_check_brute(noisy_d3_double_cat_check_circuit: stim.Circuit):
    """Noisy version of the distance-3 double cat check circuit for brute-force analysis."""
    return BruteCultivationCircuit(
        noisy_d3_double_cat_check_circuit,
        data_indices=cliffordep.circuits.D3_DOUBLE_CAT_CHECK_DATA_INDICES,
        stabilizer_generators=cliffordep.circuits.D3_DOUBLE_CAT_CHECK_STABILIZER_GENERATORS,
        logical_x=cliffordep.circuits.D3_DOUBLE_CAT_CHECK_LOGICAL_X,
        logical_z=cliffordep.circuits.D3_DOUBLE_CAT_CHECK_LOGICAL_Z,
    )