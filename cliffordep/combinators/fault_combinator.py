from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from functools import cached_property
import itertools
import math

import stim
import pandas as pd

from cliffordep.combinators._base import BaseFaultCombinator
from cliffordep.noisy_circuit_tools import CultivationCircuit
from cliffordep.combinators.fault_combinator_exclusive import get_trivial_syndrome_combinations
from cliffordep.pauli_string_tools import forget_sign, LogicalVector
from cliffordep.type_aliases import FaultBag, ErrorEvent
from cliffordep import noiseless_circuit_tools


class FaultCombinator(BaseFaultCombinator):
    """Group faults by their syndrome then effect,
    where effects are pure Clifford strings.
    
    Extends `BaseFaultCombinator`.
    """
    
    def __init__(
            self,
            circuit: CultivationCircuit,
            print_progress: bool = False,
    ):
        _basis: defaultdict[
            tuple[bool, ...], dict[str, int]
        ] = defaultdict(dict)
        _index_to_bag: defaultdict[int, list[int]] = defaultdict(lambda: [0, 0, 0])
        for (_, process_name, _), group in circuit.group_error_events_by_location().items():
            process_class = self._classify(process_name)
            for error_event in group:
                syndrome, effect = circuit.get_syndrome_and_effect(error_event)
                index = _basis[tuple(syndrome)].setdefault(effect, len(_index_to_bag))
                _index_to_bag[index][process_class] += 1
        self.basis: dict[tuple[bool, ...], dict[str, int]] = dict(_basis) # type: ignore
        self.index_to_bag: dict[int, FaultBag] = { # type: ignore
            index: tuple(bag) for index, bag in _index_to_bag.items()}
        if print_progress:
            print(f"Finished enumerating all faults. {str(self)}")
        super().__init__(circuit, print_progress=print_progress)
        self.circuit: CultivationCircuit

    
    @cached_property
    def index_to_events(self):
        """A map from fault index to the set of error events it represents.
        Not used for computation, just for introspection.
        """
        map_: defaultdict[int, set[ErrorEvent]] = defaultdict(set)
        for group in self.circuit.group_error_events_by_location().values():
            for error_event in group:
                syndrome, effect = self.circuit.get_syndrome_and_effect(error_event)
                index = self.basis[tuple(syndrome)][effect]
                map_[index].add(error_event)
        return dict(map_)


    def get_undetected_configurations( # type: ignore
            self,
            max_order: int,
            print_progress: bool = False,
    ) -> list[dict[str, set[frozenset[int]]]]:
        return super().get_undetected_configurations(max_order, print_progress=print_progress) # type: ignore


    def _get_undetected_configurations_for_length(
            self,
            length: int,
            print_progress: bool = False
    ) -> dict[str, set[frozenset[int]]]:
        result: defaultdict[str, set[frozenset[int]]] = defaultdict(set)
        if length == 0:
            result['_'*self.circuit.noisy_circuit.num_qubits].add(frozenset())
        else:
            trivial_syndrome_combos = get_trivial_syndrome_combinations(self.basis.keys(), length)
            for syndrome_counter in trivial_syndrome_combos:
                options = self._syndrome_counter_to_options(syndrome_counter)
                for segment_product in itertools.product(*(option.items() for option in options)):
                    # note: this is probably just as fast as iterating through `segment_product` once
                    # segment_product never empty
                    product_effect = forget_sign(math.prod(
                        stim.PauliString(effect) for effect, _ in segment_product)) # type: ignore
                    candidates = itertools.product(*(
                        segments for _, segments in segment_product))
                    for candidate in candidates:
                        flattened = frozenset(itertools.chain.from_iterable(candidate))
                        result[product_effect].add(flattened)
        if print_progress:
            print(f"    {length}, leading to {len(result)} distinct errors.")
        return dict(result)
    

    def _syndrome_counter_to_options(
            self,
            syndrome_counter: Counter[tuple[bool, ...]],
    ) -> list[dict[str, set[frozenset[int]]]]:
        """Get configuration segments for each syndrome based on the counts in `syndrome_counter`.
        
        Input:
        * `syndrome_counter` dictates for each syndrome
        how many faults in `self.basis[syndrome]` should appear in the configuration.
        E.g. `{syndrome1: 2, syndrome2: 1}`.

        Output:
        * `options` a list of options, one for each item in `syndrome_counter`
        e.g. `[option1, option2]`.
        Each option is a map from effect to a set of frozen sets
        of fault indices whose combination produces that effect.
        e.g. `{effect1: {frozenset1, frozenset2}, effect2: {frozenset3}}`
        where e.g. `frozenset1 = {1, 2}`.
        """
        options: list[dict[str, set[frozenset[int]]]] = []
        for syndrome, count in syndrome_counter.items():
            effects_to_segments = self._get_effect_to_segments(syndrome, count)
            options.append(effects_to_segments)
        return options
    

    def _get_effect_to_segments(
        self,
        syndrome: tuple[bool, ...],
        length: int,
    ):
        """Get a map from effect to the configurations with that resultant effect.
        
        Input:
        * `syndrome` defines the set of faults to take combinations from.
        * `length` the length of configurations to consider.

        Output:
        * a map from effect to a set of configurations.
        Each configuration is a frozen set of fault indices
        with that resultant effect.
        """
        result: defaultdict[str, set[frozenset[int]]] = defaultdict(set)
        fault_set = self.basis[syndrome].items()
        fault_combos = itertools.combinations(fault_set, length)
        for fault_combo in fault_combos:
            # e.g. fault_combo = (('effect1', 1), ('effect2', 2)) and is never empty
            product_effect = forget_sign(math.prod(
                stim.PauliString(effect) for effect, _ in fault_combo)) # type: ignore
            index_combo: frozenset[int] = frozenset(
                index for _, index in fault_combo)
            result[product_effect].add(index_combo)
        return dict(result)
    

    def get_index_to_odds(self, noise_level: float):
        index_to_odds: dict[int, float] = {}
        for index, bag in self.index_to_bag.items():
            decay_factor = math.prod(
                (1-2*self._decomposed_probability(process_class, noise_level))**count
                for process_class, count in enumerate(bag)
            )
            prob = (1 - decay_factor) / 2
            index_to_odds[index] = prob / (1 - prob)
        return index_to_odds
    

    def summarize_contributions(
            self,
            all_kept_strings: dict[str, list[dict[str, tuple[LogicalVector, set[frozenset[int]]]]]],
    ):
        """Summarize contribution of each order to overall probability.
        
        Input:
        * `all_kept_strings` a map from cultivated state to an output of `get_kept_strings`.

        Output:
        * A DataFrame whose rows are of the form (cultivated_state, order)
        whose columns are [identity_entropy, error_entropy, conditional_probability].
        """
        dict_: dict[tuple[str, int], tuple[float, float]] = {}
        for state, strings_for_state in all_kept_strings.items():
            for order, kept_strings in enumerate(strings_for_state):
                i_entropy, e_entropy = 0, 0
                for logical_vector, set_of_configurations in kept_strings.values():
                    entropy = sum(math.prod(self._bag_to_entropy(
                        self.index_to_bag[index]) for index in config) for config in set_of_configurations)
                    i_entropy += logical_vector.probability_of('I') * entropy
                    e_entropy += logical_vector.probability_of('Z') * entropy
                dict_[state, order] = (i_entropy, e_entropy)
        data = pd.DataFrame(dict_).T
        data.index.set_names(['cultivated_state', 'order'], inplace=True)
        data.columns.set_names(['logical_error'], inplace=True)
        data['conditional_probability'] = data[1] / (data[0] + data[1])
        return data
    
    
    @staticmethod
    def _bag_to_entropy(bag: FaultBag) -> float:
        """Convert a fault bag to its entropy contribution.
        
        Input:
        * `bag` a `FaultBag`.

        Output:
        * The probability of the fault bag in terms of the noise level,
        as the noise level tends to zero.
        """
        a, b, c = bag
        return a + b/3 + c/15
    

    def visualize_fault_configurations(
            self,
            configurations: Sequence[Iterable[int]],
            probability_increment: float = 0.1,
            diagram_type: str = 'timeline',
            **kwargs_for_diagram,
    ):
        """Visualize multiple fault configurations one by one.
        
        Input:
        * `configurations` a sequence of fault configurations.
        Each fault configuration is a frozen set of fault indices.
        * `probability_increment` the increment in error probability for each fault in the configuration.
        I.e. all error events of the kth fault have probability `k*probability_increment`.
        This is used to distinguish error events belonging to different faults.
        * `diagram_type` the type of diagram to produce. See `stim.Circuit.diagram()` for options.
        * `**kwargs_for_diagram` other keyword arguments for `stim.Circuit.diagram()`.

        Output:
        * An interactive widget that displays one configuration at a time.
        For each configuration, shows a diagram of the circuit where each fault in the configuration
        is represented by all the error events that correspond to it.
        """
        from ipywidgets import interact, BoundedIntText
        @interact(configuration=BoundedIntText(
                value=0,
                min=0,
                max=len(configurations)-1,
                step=1,
        ))
        def f(configuration: int):
            circuit = self.circuit.noiseless_circuit
            for k, fault_index in enumerate(configurations[configuration]):
                circuit = noiseless_circuit_tools.insert_error_events(
                    circuit=circuit,
                    error_events=self.index_to_events[fault_index],
                    probability=k*probability_increment,
                )
            return circuit.diagram(type=diagram_type, **kwargs_for_diagram)
        return f