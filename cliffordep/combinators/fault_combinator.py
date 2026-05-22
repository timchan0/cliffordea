from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from functools import cache, cached_property
import itertools
import math

import stim
import pandas as pd

from cliffordep.noisy_circuit_tools import CultivationCircuit
from cliffordep.combinators._base import Combinator, get_trivial_syndrome_combinations
from cliffordep.pauli_string_tools import forget_sign
from cliffordep.type_aliases import FaultBag, ErrorEvent, LogicalTriple
from cliffordep import noiseless_circuit_tools


class FaultCombinator(Combinator):
    """Group faults by their syndrome then effect,
    where all faults are independent and effects are pure Pauli sums.

    Extends `Combinator`.

    Instance attributes:
    * `circuit` the noisy circuit to analyze.
    * `basis` a map from each syndrome to another map from each effect to a fault index.
    * `index_to_bag` a map from each fault index to its fault bag.
    """
    
    def __init__(self, noisy_circuit, print_progress=False):
        _circuit = CultivationCircuit(noisy_circuit=noisy_circuit)
        _basis: defaultdict[
            tuple[bool, ...], dict[str, int]
        ] = defaultdict(dict)
        _index_to_bag: defaultdict[int, list[int]] = defaultdict(lambda: [0, 0, 0])
        for (_, process_name, _), group in _circuit.group_error_events_by_location().items():
            process_class = self._classify(process_name)
            for error_event in group:
                syndrome, effect = _circuit.get_syndrome_and_effect(error_event)
                index = _basis[tuple(syndrome)].setdefault(effect, len(_index_to_bag))
                _index_to_bag[index][process_class] += 1
        self.basis: dict[tuple[bool, ...], dict[str, int]] = dict(_basis) # type: ignore
        self.index_to_bag: dict[int, FaultBag] = { # type: ignore
            index: tuple(bag) for index, bag in _index_to_bag.items()}
        if print_progress:
            print(f"Finished enumerating all faults. {str(self)}")
        self.circuit = _circuit

    def __repr__(self):
        return f"{self.__class__.__name__}({self.circuit})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__} with {self.fault_count} faults distributed among {self.syndrome_count} syndromes."


    def error_rate_per_kept_shot(
            self,
            all_string_leads: list[dict[str, LogicalTriple]],
            noise_level: float,
            print_progress: bool = False,
    ) -> float:
        """Calculate the logical error rate per kept shot for a given noise level.

        Input:
        * `all_string_leads` the output of `get_kept_strings`.
        * `noise_level` the noise level to analyze.
        * `print_progress` whether to print progress.
        """
        index_to_odds = self.get_index_to_odds(noise_level)
        identity_odds, error_odds = 0, 0
        for degree, strings in enumerate(all_string_leads):
            i_odds, e_odds = _sum_odds(strings.values(), index_to_odds)
            if print_progress:
                print(f'O(p^{degree}) events:')
                print(f'Identity odds = {i_odds}')
                print(f'Error odds = {e_odds}')
            identity_odds += i_odds
            error_odds += e_odds
        return error_odds / (identity_odds + error_odds)
    

    def get_index_to_odds(self, noise_level: float) -> dict[int, float]:
        """Get a map from fault index to the odds of it flipping."""
        index_to_odds: dict[int, float] = {}
        for index, bag in self.index_to_bag.items():
            decay_factor = math.prod(
                (1-2*self._decomposed_probability(process_class, noise_level))**count
                for process_class, count in enumerate(bag)
            )
            prob = (1 - decay_factor) / 2
            index_to_odds[index] = prob / (1 - prob)
        return index_to_odds


    def get_undetected_configurations(
            self,
            max_order: int,
            print_progress: bool = False,
    ) -> list[dict[str, set[frozenset[int]]]]:
        """Find all combinations of faults up to `max_order` that have trivial syndrome.

        Input:
        * `max_order` the maximum order of probability to consider.
        * `print_progress` whether to print progress.

        Output:
        * A list whose kth entry is a map
        from each effect to a set of frozen sets of fault indices.
        """
        if print_progress:
            print("Found undetected configurations of length...")
        return [self._get_undetected_configurations_for_length(
            length, print_progress) for length in range(max_order + 1)]

    @property
    def fault_count(self) -> int:
        return sum(len(faults) for faults in self.basis.values())

    
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


    @cache
    def index_to_syndrome_and_effect(self, index: int) -> tuple[tuple[bool, ...], str]:
        """Get the syndrome and effect corresponding to a fault index.

        Input:
        * `index` the fault index.

        Output:
        * `syndrome` the syndrome of the fault.
        * `effect` the effect of the fault.
        """
        for syndrome, effect_dict in self.basis.items():
            for effect, effect_index in effect_dict.items():
                if effect_index == index:
                    return syndrome, effect
        raise ValueError(f"Index {index} not found in basis.")

    
    def print_basis(self):
        """Print the syndrome, effect, index, and fault bag of all faults."""
        lines = []
        for syndrome, effect_dict in self.basis.items():
            lines.append(''.join('1' if detector else '0' for detector in syndrome))
            for effect, index in effect_dict.items():
                bag = self.index_to_bag[index]
                bag_str = bag if type(bag[0]) is int else tuple(dict(b) for b in bag) # type: ignore
                lines.append(f"  {effect}: {index}, {bag_str}")
        print('\n'.join(lines))
    

    def summarize_contributions(
            self,
            all_kept_strings: dict[str, list[dict[str, LogicalTriple]]],
    ):
        """Summarize contribution of each degree to overall probability.
        
        Input:
        * `all_kept_strings` a map from cultivated state to an output of `get_kept_strings`.

        Output:
        * A DataFrame whose rows are of the form (cultivated_state, degree)
        whose columns are [identity_entropy, error_entropy, conditional_probability].
        """
        dict_: dict[tuple[str, int], tuple[float, float]] = {}
        for state, strings_for_state in all_kept_strings.items():
            for degree, strings in enumerate(strings_for_state):
                i_entropy, e_entropy = 0, 0
                for accept_probability, logical_fidelity, set_of_configurations in strings.values():
                    entropy = sum(math.prod(self._bag_to_entropy(
                        self.index_to_bag[index]) for index in config) for config in set_of_configurations)
                    i_entropy += accept_probability * logical_fidelity * entropy
                    e_entropy += accept_probability * (1-logical_fidelity) * entropy
                dict_[state, degree] = (i_entropy, e_entropy)
        data = pd.DataFrame(dict_).T
        data.index.set_names(['cultivated_state', 'degree'], inplace=True)
        data.columns.set_names(['logical_error'], inplace=True)
        data['conditional_probability'] = data[1] / (data[0] + data[1])
        return data

    
    @property
    def syndrome_count(self) -> int:
        return len(self.basis)
    

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


    @staticmethod
    def _classify(process_name: str) -> int:
        """Classify an error process by how many error events it can make.

        Input:
        * `process_name` the name of the Stim gate that gives rise to error events.
        This can be 'DEPOLARIZE1', 'DEPOLARIZE2', 'X_ERROR', 'Y_ERROR', 'Z_ERROR', 'MX', 'MY', 'MZ'.

        Output:
        * An integer indicating the type of error process:
            * 0 if it makes only 1 error event (X_ERROR, Y_ERROR, Z_ERROR, MX, MY, MZ).
            * 1 if it makes 3 error events (DEPOLARIZE1).
            * 2 if it makes 15 error events (DEPOLARIZE2).
        """
        if process_name in {'X_ERROR', 'Y_ERROR', 'Z_ERROR', 'MX', 'MY', 'MZ'}:
            return 0
        elif process_name == 'DEPOLARIZE1':
            return 1
        elif process_name == 'DEPOLARIZE2':
            return 2
        else:
            raise ValueError(f"Unknown error process: {process_name}")


    @staticmethod
    def _decomposed_probability(
            class_: int,
            noise_level: float,
    ) -> float:
        """Return the probability of the independent events of an error location of class `class_`.

        Input:
        * `class_` the class of the error location.
        It can be 0, 1, or 2.
        This depends on the number of independent events
        that sequentially compose to equal the error location.
        * `noise_level` a float in [0, 1].
        """
        p = noise_level
        if class_ == 0:
            return p
        elif class_ == 1:
            return 1/2 - math.sqrt(9 - 12*p)/6
        elif class_ == 2:
            return -15**(7/8)*(15 - 16*p)**(1/8)/30 + 1/2
        else:
            raise ValueError(f"Invalid `class_`: {class_}. Only 0, 1, and 2 are supported.")
    

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


    def _get_undetected_configurations_for_length(
            self,
            length: int,
            print_progress: bool = False
    ) -> dict[str, set[frozenset[int]]]:
        """Find all combinations of `length` faults that have trivial syndrome.

        Helper for `self.get_undetected_configurations()`.

        Input:
        * `length` the length of combinations to find.
        * `print_progress` whether to print progress.

        Output:
        * a map from each effect to a set of frozen sets of fault indices.
        """
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


def _sum_odds(
        logical_triples: Iterable[LogicalTriple],
        index_to_odds: dict[int, float],
    ) -> tuple[float, float]:
    """Sum the odds of all configurations in `vector_combo_pairs`.

    Input:
    * `logical_triples` an iterable of triples, each containing:
        - an acceptance probability,
        - a logical fidelity,
        - a set of frozen sets of fault indices that defines the combination.
    * `index_to_odds` a map from each fault index to the odds of it flipping.
    """
    i_odds, e_odds = 0, 0
    for accept_probability, logical_fidelity, set_of_configurations in logical_triples:
        prob = sum(math.prod(index_to_odds[index] for index in combo) for combo in set_of_configurations)
        i_odds += accept_probability * logical_fidelity * prob
        e_odds += accept_probability * (1-logical_fidelity) * prob
    return i_odds, e_odds