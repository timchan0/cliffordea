import abc
from collections import Counter
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from cliffordep.noisy_circuit_tools import CultivationCircuit


class Combinator(abc.ABC):
    """Class to find all combinations of faults that lead to trivial syndrome.
    
    Instance attributes:
    * `circuit` the `CultivationCircuit` to analyze.
    """


    @abc.abstractmethod
    def __init__(
            self,
            circuit: 'CultivationCircuit',
            restrict_to_data: bool = True,
            print_progress: bool = False,
    ):
        """Input:
        * `circuit` the `CultivationCircuit` to analyze.
        * `restrict_to_data` whether to restrict all effects to only the data qubits.
        * `print_progress` whether to print progress.
        """
        self.circuit = circuit


    def get_undetected_fault_combinations(
            self,
            max_order: int,
            print_progress: bool = False,
    ):
        """Find all combinations of faults up to `max_order` that have trivial syndrome.

        Input:
        * `max_order` the maximum order of probability to consider.
        * `print_progress` whether to print progress.

        Output:
        * A list whose kth entry is a map
        from each effect to a counter of denominators.
        Each denominator divides (noise level)^k to equal
        the probability an instance of that undetected combination of k faults occurs.
        """
        return [self._get_undetected_fault_combinations_for_length(
            length, print_progress) for length in range(max_order + 1)]
    

    def get_kept_strings(
            self,
            fault_combinations: list[dict[str, Counter[int]]],
            cultivated_state: Literal['T', 'S', 'Z'] = 'T',
            print_progress: bool = False,
    ) -> list[tuple[
        dict[str, tuple[float, Counter[int]]],
        dict[str, tuple[float, Counter[int]]],
    ]]:
        """Return all the information needed to reconstruct the logical error probability for any noise level.
        
        Input:
        * `fault_combinations` the output of `get_undetected_fault_combinations`.
        * `cultivated_state` the target logical state cultivated.
        * `print_progress` whether to print progress.

        Output:
        * A list of pairs, one for each order. Each pair contains:
            - `identity_strings` a map from each effect that leads to logical identity to a pair containing:
                - the probability they are not rejected,
                - a counter of denominators.
            - `error_strings` ditto for effects that lead to a logical error.
        """
        if print_progress:
            print(f"{cultivated_state} state cultivation:")
        result = [self._get_kept_strings(
                combinations_of_order,
                cultivated_state=cultivated_state,
                order=order if print_progress else None,
            ) for order, combinations_of_order in enumerate(fault_combinations)]
        return result
    

    def _get_kept_strings(
            self,
            combinations_of_order: dict[str, Counter[int]],
            cultivated_state: Literal['T', 'S', 'Z'] = 'T',
            order: None | int = None,
    ) -> tuple[
        dict[str, tuple[float, Counter[int]]],
        dict[str, tuple[float, Counter[int]]],
    ]:
        """Group postselected effects by whether they lead to identity or error.

        Helper for `get_kept_strings`.

        Input:
        * `combinations_of_order` a map from each effect to a counter of denominators.
        Each denominator divides (noise level)^order to equal
        the probability an instance of that undetected combination occurs.
        * `cultivated_state` the target logical state cultivated.
        * `order` an optional parameter used only for printing progress.

        Output:
        * `identity_strings` a map from each effect that leads to logical identity,
        to the probability it is not rejected and a counter of denominators.
        * `error_strings` ditto for effects that lead to a logical error.
        """
        identity_strings: dict[str, tuple[float, Counter[int]]] = {}
        error_strings: dict[str, tuple[float, Counter[int]]] = {}
        for data_string, denominators in combinations_of_order.items():
            logical_vector = self.circuit.string_to_logical_vector(cultivated_state, data_string)
            # TODO: check if `LogicalVector.transfer_xy_to_iz` is unitary. If so, move from above method into below conditional block.
            if logical_vector.probability_mass:
                # TODO: use `logical_error_probability` instead of `is_logical_error`
                if logical_vector.is_logical_error:
                    error_strings[data_string] = (logical_vector.probability_mass, denominators)
                else:
                    identity_strings[data_string] = (logical_vector.probability_mass, denominators)
        if order is not None:
            print(f'For order {order}, {len(identity_strings)} ({len(error_strings)}) effects are stabilized and lead to identity (error).')
        return identity_strings, error_strings


    @abc.abstractmethod
    def _get_undetected_fault_combinations_for_length(
            self,
            length: int,
            print_progress: bool = False,
    ) -> dict[str, Counter[int]]:
        """Find all combinations of `length` faults that have trivial syndrome.
        
        Helper for `self.get_undetected_fault_combinations()`.
        
        Input:
        * `length` the length of combinations to find.
        * `print_progress` whether to print progress.

        Output:
        * a map from each effect to a counter of denominators.
        Each denominator divides (noise level)^length to equal
        the probability an instance of that undetected combination occurs.
        """