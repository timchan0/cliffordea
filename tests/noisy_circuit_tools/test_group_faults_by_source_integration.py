import stim

from cliffordep.type_aliases import Fault, FaultSource


class TestD3DoubleCatCheck:
    """Tests on the `d3_double_cat_check` circuit."""


    def test_correct_length(
            self,
            noisy_d3_double_cat_check_circuit: stim.Circuit,
            d3_double_cat_check_grouped_by_source: dict[FaultSource, set[Fault]],
    ):
        """Test returns a dictionary with the correct key count."""
        assert type(d3_double_cat_check_grouped_by_source) is dict

        noisy_gates = {'DEPOLARIZE1', 'DEPOLARIZE2', 'Z_ERROR', 'MX'}
        count = sum(len(instruction.target_groups())
                    for instruction in noisy_d3_double_cat_check_circuit
                    if isinstance(instruction, stim.CircuitInstruction)
                    and instruction.name in noisy_gates)
        assert len(d3_double_cat_check_grouped_by_source) == count