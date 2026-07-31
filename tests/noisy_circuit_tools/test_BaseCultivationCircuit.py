import pytest
import stim

from cliffordep.noisy_circuit_tools import CultivationCircuit
from cliffordep.type_aliases import ErrorEvent, ErrorLocation


def get_key(qubits_affected: list[int]):
    """Key function for sorting error events."""
    def key(error_event: ErrorEvent):
        timeslice, name, targets = error_event
        if name == 'E':
            assert len(qubits_affected) == 2
            string = ''.join(f'{target.pauli_type}' for target in targets)
        else:
            assert len(targets) == 1
            pauli_type, *_ = name
            _target, = targets
            qubit_value = _target.qubit_value
            if len(qubits_affected) == 1:
                string = f'{pauli_type}'
            elif len(qubits_affected) == 2:
                if qubit_value == qubits_affected[0]:
                    string = f'{pauli_type}I'
                else:
                    string = f'I{pauli_type}'
            else:
                raise ValueError("Error event affects more than 2 qubits.")
        return (timeslice, string)
    return key

def test_group_error_events_by_location(d3_double_cat_check_grouped_by_location: dict[ErrorLocation, list[ErrorEvent]]):
    """Test each group is consistently sorted."""
    for group in d3_double_cat_check_grouped_by_location.values():
        qubits_affected = sorted(set(target.value for _, _, targets in group for target in targets))
        _key = get_key(qubits_affected)
        assert group == sorted(group, key=_key)


@pytest.mark.parametrize('measurement_name', ('M', 'MX', 'MY', 'MR', 'MRX', 'MRY'))
def test_group_single_qubit_measurement_error(measurement_name: str):
    """A noisy one-qubit measurement contributes exactly one outcome-flip event.

    The event keeps the original Stim instruction name, including the ``R`` in
    measure-reset gates, so downstream tools know whether the qubit is reset.
    """
    circuit = CultivationCircuit(stim.Circuit(f"""
        {measurement_name}(0.001) 0
        DETECTOR rec[-1]
    """))
    target = stim.GateTarget(0)
    event = (0, measurement_name, (target,))

    assert circuit.group_error_events_by_location() == {event: [event]}
