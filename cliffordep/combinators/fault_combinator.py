from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from functools import cache, cached_property, lru_cache
import itertools
import math
from typing import Literal

import pandas as pd
import stim
from tqdm.auto import tqdm

from cliffordep.noisy_circuit_tools import CultivationCircuit
from cliffordep.combinators._base import Combinator
from cliffordep.logical_analyzers import LogicalAnalyzer
from cliffordep.type_aliases import (
    ErrorEvent,
    FaultBag,
    LogicalTriple,
    PauliMask,
    SyndromeMask,
)
from cliffordep import noiseless_circuit_tools
from cliffordep.constants import ONE_QUBIT_ERROR_EVENTS


CultivatedState = Literal['T', 'S', 'Z']
LogicalAnalysis = tuple[float, float]
PauliBasis = Literal['X', 'Z']
PauliGenerator = tuple[int, int, PauliBasis]
MaskAnalysis = tuple[int, int]


DiagramType = Literal[
    'timeline-text',
    'timeline-svg',
    'timeline-svg-html',
    'timeline-3d',
    'timeline-3d-html',
    'detslice-text',
    'detslice-svg',
    'detslice-svg-html',
    'matchgraph-svg',
    'matchgraph-svg-html',
    'matchgraph-3d',
    'matchgraph-3d-html',
    'timeslice-svg',
    'timeslice-svg-html',
    'detslice-with-ops-svg',
    'detslice-with-ops-svg-html',
    'interactive',
    'interactive-html',
]
"""Stim diagram type."""


def _decompose_error_event(error_event: ErrorEvent) -> tuple[PauliGenerator, ...]:
    """Decompose a non-measurement Pauli event into X/Z generators.

    :param error_event: The physical Pauli error event to decompose.
    :return generators: Elementary ``(timeslice, qubit, basis)`` generators
        whose unsigned product equals the original event.
    """
    timeslice, name, targets = error_event
    if name.startswith('M'):
        raise ValueError('Measurement errors are not Pauli-generator events.')

    if name == 'E':
        components = tuple(
            (target.value, target.pauli_type) for target in targets
        )
    else:
        target, = targets
        components = ((target.value, name[0]),)

    generators: list[PauliGenerator] = []
    for qubit, pauli_type in components:
        if pauli_type == 'X':
            generators.append((timeslice, qubit, 'X'))
        elif pauli_type == 'Y':
            generators.extend((
                (timeslice, qubit, 'X'),
                (timeslice, qubit, 'Z'),
            ))
        elif pauli_type == 'Z':
            generators.append((timeslice, qubit, 'Z'))
        else:
            raise ValueError(f'Unsupported Pauli component: {pauli_type!r}.')
    return tuple(generators)


def _make_error_event_analyzer(
        circuit: CultivationCircuit,
) -> Callable[[ErrorEvent], MaskAnalysis]:
    """Create a mask-valued event analyzer with a local generator cache.

    :param circuit: The cultivation circuit through which errors are propagated.
    :return analyze_error_event: A callable that maps a physical error event to
        its integer syndrome and effect masks.
    """
    @cache
    def analyze_generator(
            timeslice: int,
            qubit: int,
            basis: PauliBasis,
    ) -> MaskAnalysis:
        """Propagate and cache one elementary Pauli generator.

        :param timeslice: The circuit timeslice where the generator occurs.
        :param qubit: The qubit on which the generator acts.
        :param basis: Whether the generator is an X or Z error.
        :return analysis: The propagated syndrome and effect as integer masks.
        """
        generator_event = (
            timeslice,
            f'{basis}_ERROR',
            (stim.GateTarget(qubit),),
        )
        syndrome, effect = circuit.get_pauli_error_syndrome_and_effect(
            generator_event,
        )
        return (
            _bools_to_mask(syndrome),
            _unsigned_pauli_string_to_mask(effect),
        )

    def analyze_error_event(error_event: ErrorEvent) -> MaskAnalysis:
        """Analyze one physical event using cached generators when possible.

        :param error_event: The physical error event to propagate.
        :return analysis: The event's syndrome and effect as integer masks.
        """
        _, name, _ = error_event
        if name.startswith('M'):
            syndrome = circuit.get_measurement_error_syndrome(error_event)
            return _bools_to_mask(syndrome), 0

        syndrome_mask = 0
        effect_mask = 0
        for generator in _decompose_error_event(error_event):
            generator_syndrome_mask, generator_effect_mask = analyze_generator(
                *generator,
            )
            syndrome_mask ^= generator_syndrome_mask
            effect_mask ^= generator_effect_mask
        return syndrome_mask, effect_mask

    return analyze_error_event


def _bools_to_mask(values: Iterable[bool]) -> int:
    """Pack a little-endian boolean sequence into an integer mask.

    :param values: Boolean coefficients ordered from least-significant bit.
    :return mask: The packed integer mask.
    """
    return sum(bool(value) << index for index, value in enumerate(values))


def _iter_zero_syndrome_configurations(
        *,
        syndromes: Sequence[SyndromeMask],
        effects: Sequence[PauliMask],
        order: int,
) -> Iterator[tuple[PauliMask, tuple[int, ...]]]:
    """Join canonical fault-subset halves with equal XOR syndromes.

    :param syndromes: Packed syndrome masks in fault-index order.
    :param effects: Packed data-effect masks in fault-index order.
    :param order: The number of distinct fault indices in each configuration.
    :return: An iterator over combined effects and sorted fault-index tuples.
    """
    if order == 0:
        yield 0, ()
        return

    left_order = order // 2
    right_order = order - left_order
    fault_indices = range(len(syndromes))
    left_halves: defaultdict[
        SyndromeMask,
        list[tuple[tuple[int, ...], PauliMask]],
    ] = defaultdict(list)
    for indices in itertools.combinations(fault_indices, left_order):
        syndrome = 0
        effect = 0
        for index in indices:
            syndrome ^= syndromes[index]
            effect ^= effects[index]
        left_halves[syndrome].append((indices, effect))

    for right_indices in itertools.combinations(fault_indices, right_order):
        right_syndrome = 0
        right_effect = 0
        for index in right_indices:
            right_syndrome ^= syndromes[index]
            right_effect ^= effects[index]
        for left_indices, left_effect in left_halves.get(right_syndrome, ()):
            if not left_indices or left_indices[-1] < right_indices[0]:
                yield left_effect ^ right_effect, left_indices + right_indices


class FaultCombinator(Combinator):
    """Group faults by their syndrome then effect,
    where all faults are independent and effects are pure Pauli sums.

    Extends `Combinator`.
    """
    
    def __init__(self, noisy_circuit, print_progress=False):
        _circuit = CultivationCircuit(noisy_circuit=noisy_circuit)
        _basis: defaultdict[SyndromeMask, dict[PauliMask, int]] = defaultdict(dict)
        _index_to_bag: defaultdict[int, list[int]] = defaultdict(lambda: [0, 0, 0])
        event_fault_indices: list[int] = []
        analyze_error_event = _make_error_event_analyzer(_circuit)

        for (_, process_name, _), group in _circuit.group_error_events_by_location().items():
            process_class = self._classify(process_name)
            for error_event in group:
                syndrome_mask, effect_mask = analyze_error_event(error_event)
                index = _basis[syndrome_mask].setdefault(
                    effect_mask,
                    len(_index_to_bag),
                )
                _index_to_bag[index][process_class] += 1
                event_fault_indices.append(index)
        self.basis: dict[SyndromeMask, dict[PauliMask, int]] = dict(_basis)
        """Packed syndrome and Pauli masks mapped to dense fault indices."""
        self.index_to_bag: dict[int, FaultBag] = { # type: ignore
            index: tuple(bag) for index, bag in _index_to_bag.items()}
        """A map from each fault index to its fault bag."""
        indexed_faults: list[tuple[SyndromeMask, PauliMask]] = [
            (0, 0) for _ in self.index_to_bag
        ]
        for syndrome_mask, effect_to_index in self.basis.items():
            for effect_mask, index in effect_to_index.items():
                indexed_faults[index] = syndrome_mask, effect_mask
        self._indexed_faults = tuple(indexed_faults)
        """Syndrome and full Pauli effect masks in fault-index order."""
        self._event_fault_indices = tuple(event_fault_indices)
        """The fault index of each deterministically enumerated error event."""
        if print_progress:
            print(f"Finished enumerating all faults. {str(self)}")
        self.circuit = _circuit
        """The noisy circuit to analyze."""

    def __getstate__(self):
        """Return picklable state without the cached Stim error events."""
        state = self.__dict__.copy()
        state.pop('index_to_events', None)
        return state

    def __repr__(self):
        return f"{self.__class__.__name__}({self.circuit})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__} with {self.fault_count} faults distributed among {self.syndrome_count} syndromes."


    def error_rate_per_kept_shot(
            self,
            all_kept_effects: list[dict[PauliMask, LogicalTriple]],
            noise_level: float,
            print_progress: bool = False,
    ) -> float:
        """Calculate the logical error rate per kept shot for a given noise level.

        :param all_kept_effects: One state's output from `get_kept_effects`.
        :param noise_level: The noise level to analyze.
        :param print_progress: Whether to print progress.
        """
        index_to_odds = self.get_index_to_odds(noise_level)
        identity_odds, error_odds = 0, 0
        for degree, effects in enumerate(all_kept_effects):
            i_odds, e_odds = _sum_odds(effects.values(), index_to_odds)
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


    def get_kept_effects(
            self,
            *,
            logical_analyzer: LogicalAnalyzer,
            max_order: int,
            cultivated_states: tuple[CultivatedState, ...] = ('T',),
            logical_analysis_cache_maxsize: int | None = 262_144,
            print_progress: bool = False,
    ) -> dict[str, list[dict[PauliMask, LogicalTriple]]]:
        """Enumerate circuit-undetected, postselected packed Pauli effects.

        :param self: The fault combinator whose indexed faults are enumerated.
        :param max_order: The maximum order of probability to consider.
        :param logical_analyzer: The analyzer used for stabilizer postselection
            and logical-fidelity calculations.
        :param cultivated_states: The logical states to analyze together.
            States share enumeration and logical-analysis cache entries.
        :param logical_analysis_cache_maxsize: Maximum number of data-only
            Pauli effects retained in the logical-analysis LRU cache.
            Use `None` for an unbounded cache or `0` to disable caching.
        :param print_progress: Whether to print progress.

        :return: A map from each requested state to a list whose kth entry maps
            each accepted packed data-only Pauli effect to its
            acceptance probability, logical fidelity, and fault configurations.
        """
        data_indices = logical_analyzer.DATA_INDICES
        full_qubit_count = self.circuit.noisy_circuit.num_qubits
        restrict_effect_mask = _make_pauli_mask_restrictor(
            data_indices=data_indices,
            source_qubit_count=full_qubit_count,
        )
        analyze_mask = _make_logical_analysis_cache(
            logical_analyzer=logical_analyzer,
            cultivated_states=cultivated_states,
            maxsize=logical_analysis_cache_maxsize,
        )
        detector_count = self.circuit.noisy_circuit.num_detectors
        extended_syndromes: list[SyndromeMask] = []
        data_effects: list[PauliMask] = []
        for circuit_syndrome, full_effect in self._indexed_faults:
            data_effect = restrict_effect_mask(full_effect)
            precheck_syndrome = logical_analyzer.linear_precheck_syndrome(
                data_effect,
            )
            extended_syndromes.append(
                circuit_syndrome | (precheck_syndrome << detector_count)
            )
            data_effects.append(data_effect)

        result: dict[str, list[dict[PauliMask, LogicalTriple]]] = {
            state: [] for state in cultivated_states
        }

        for order in range(max_order + 1):
            configurations_by_mask: dict[
                PauliMask, set[frozenset[int]]
            ] = {}
            triples_by_state: dict[
                str, dict[PauliMask, LogicalTriple]
            ] = {state: {} for state in cultivated_states}
            visited_configuration_count = 0

            configuration_iterator: Iterable[
                tuple[PauliMask, tuple[int, ...]]
            ] = _iter_zero_syndrome_configurations(
                syndromes=extended_syndromes,
                effects=data_effects,
                order=order,
            )
            if print_progress:
                configuration_iterator = tqdm(
                    configuration_iterator,
                    desc=f"Order {order}",
                    unit="configuration",
                    dynamic_ncols=True,
                    leave=True,
                )

            for data_effect_mask, fault_indices in configuration_iterator:
                visited_configuration_count += 1
                analyses = analyze_mask(data_effect_mask)
                if not any(accept_probability for accept_probability, _ in analyses):
                    continue

                configurations = configurations_by_mask.setdefault(data_effect_mask, set())
                configurations.add(frozenset(fault_indices))
                for state, (accept_probability, logical_fidelity) in zip(
                        cultivated_states, analyses, strict=True):
                    if accept_probability and data_effect_mask not in triples_by_state[state]:
                        triples_by_state[state][data_effect_mask] = (
                            accept_probability,
                            logical_fidelity,
                            configurations,
                        )

            for state in cultivated_states:
                result[state].append(triples_by_state[state])

            if print_progress:
                retained_configuration_count = sum(
                    len(configurations)
                    for configurations in configurations_by_mask.values()
                )
                print(
                    f"    {order}: visited {visited_configuration_count} configurations; "
                    f"kept {retained_configuration_count} configurations "
                    f"across {len(configurations_by_mask)} data effects."
                )
                for state in cultivated_states:
                    identity_weight, error_weight = _sum_logical_weights(
                        result[state][order].values()
                    )
                    print(
                        f"        {state}: accepted effect logical weight: "
                        f"{identity_weight} identity, {error_weight} error."
                    )

        if print_progress:
            cache_info = analyze_mask.cache_info()
            print(
                "Logical analysis cache: "
                f"hits={cache_info.hits}, misses={cache_info.misses}, "
                f"maxsize={cache_info.maxsize}, currsize={cache_info.currsize}."
            )
        analyze_mask.cache_clear()
        return result

    @property
    def fault_count(self) -> int:
        return len(self._indexed_faults)

    
    @cached_property
    def index_to_events(self):
        """A map from fault index to the set of error events it represents.
        Not used for computation, just for introspection.
        """
        map_: defaultdict[int, set[ErrorEvent]] = defaultdict(set)
        error_events = (
            error_event
            for group in self.circuit.group_error_events_by_location().values()
            for error_event in group
        )
        for index, error_event in zip(
                self._event_fault_indices, error_events, strict=True):
            map_[index].add(error_event)
        return dict(map_)

    
    def print_basis(self):
        """Print readable syndromes, effects, indices, and fault bags.

        :param self: The fault combinator whose basis is displayed.
        :return: None.
        """
        lines = []
        detector_count = self.circuit.noisy_circuit.num_detectors
        qubit_count = self.circuit.noisy_circuit.num_qubits
        for syndrome_mask, effect_dict in self.basis.items():
            lines.append(''.join(
                '1' if syndrome_mask >> detector_index & 1 else '0'
                for detector_index in range(detector_count)
            ))
            for effect_mask, index in effect_dict.items():
                bag = self.index_to_bag[index]
                effect = _pauli_mask_to_unsigned_string(effect_mask, qubit_count)
                bag_str = bag
                lines.append(f"  {effect}: {index}, {bag_str}")
        print('\n'.join(lines))
    

    def summarize_contributions(
            self,
            all_kept_effects: dict[
                str,
                list[dict[PauliMask, LogicalTriple]],
            ],
    ):
        """Summarize contribution of each degree to overall probability.
        
        :param all_kept_effects: Results returned by `get_kept_effects`.

        :return summary:
            A DataFrame whose rows are of the form (cultivated_state, degree)
            whose columns are [('configuration_count', 'benign'), ('configuration_count', 'malignant'),
            ('weight', 'benign'), ('weight', 'malignant'), ('weight', 'error_probability')].
        """
        dict_: dict[tuple[str, int], tuple[int, int, float, float]] = {}
        for state, effects_for_state in all_kept_effects.items():
            for degree, effects in enumerate(effects_for_state):
                i_entropy, e_entropy = 0, 0
                i_count, e_count = 0, 0
                for accept_probability, logical_fidelity, set_of_configurations in effects.values():
                    entropy = sum(math.prod(self._bag_to_entropy(self.index_to_bag[index])
                        for index in config) for config in set_of_configurations)
                    i_entropy += accept_probability * logical_fidelity * entropy
                    e_entropy += accept_probability * (1-logical_fidelity) * entropy
                    if accept_probability > 0:
                        if logical_fidelity == 1:
                            i_count += len(set_of_configurations)
                        else:
                            e_count += len(set_of_configurations)
                dict_[state, degree] = (i_count, e_count, i_entropy, e_entropy)
        data = pd.DataFrame(dict_).T
        data.index.set_names(['cultivated_state', 'degree'], inplace=True)
        data.columns = pd.MultiIndex.from_product([
            ['configuration_count', 'weight'], ['benign', 'malignant']])
        data['weight', 'error_probability'] = data['weight', 'malignant'] / \
            (data['weight', 'benign'] + data['weight', 'malignant'])
        return data.astype({
            ('configuration_count', 'benign'): int,
            ('configuration_count', 'malignant'): int,
        })

    
    @property
    def syndrome_count(self) -> int:
        return len(self.basis)
    

    def visualize_fault_configurations(
            self,
            configurations: Sequence[Iterable[int]],
            probability_increment: float = 0.1,
            diagram_type: DiagramType = 'timeline-text',
            **kwargs_for_diagram,
    ):
        """Visualize multiple fault configurations one by one.
        
        :param configurations: A sequence of fault configurations.
            Each fault configuration is a frozen set of fault indices.
        :param probability_increment: The increment in error probability for each fault in the configuration.
            I.e. all error events realizing the kth fault have probability `k*probability_increment`.
            This is used to distinguish error events realizing different faults.
        :param diagram_type: The type of diagram to produce. See `stim.Circuit.diagram()` for options.
        :param **kwargs_for_diagram: Other keyword arguments for `stim.Circuit.diagram()`.

        :return: An interactive widget that displays one configuration at a time.
            For each configuration, shows a diagram of the circuit where each fault in the configuration
            is represented by all the error events that realize it.
        """
        from ipywidgets import interact, BoundedIntText
        @interact(configuration=BoundedIntText(
                value=0,
                min=0,
                max=len(configurations)-1,
                step=1,
        ))
        def f(configuration: int):
            circuits = noiseless_circuit_tools.split_by_ticks(
                self.circuit.noiseless_circuit
            )
            for k, fault_index in enumerate(configurations[configuration]):
                noiseless_circuit_tools._insert_error_events_into_slices(
                    circuits=circuits,
                    error_events=self.index_to_events[fault_index],
                    probability=k*probability_increment,
                )
            circuit = noiseless_circuit_tools.compose_slices(circuits)
            return circuit.diagram(type=diagram_type, **kwargs_for_diagram)
        return f
    
    
    @staticmethod
    def _bag_to_entropy(bag: FaultBag) -> float:
        """Convert a fault bag to its entropy contribution.
        
        :param bag: A `FaultBag`.

        :return entropy:
            The probability of the fault bag in terms of the noise level,
            as the noise level tends to zero.
        """
        a, b, c = bag
        return a + b/3 + c/15


    @staticmethod
    def _classify(process_name: str) -> int:
        """Classify an error process by how many error events it can make.

        :param process_name: The name of the Stim gate that gives rise to error events.
            This can be 'DEPOLARIZE1', 'DEPOLARIZE2',
            or a member of `ONE_QUBIT_ERROR_EVENTS`.

        :return:
            An integer indicating the type of error process:

                * 0 if `process_name` is in `ONE_QUBIT_ERROR_EVENTS`.
                * 1 if it makes 3 error events (DEPOLARIZE1).
                * 2 if it makes 15 error events (DEPOLARIZE2).
        """
        if process_name in ONE_QUBIT_ERROR_EVENTS:
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

        :param class_: the class of the error location.
            It can be 0, 1, or 2.
            This depends on the number of independent events
            that sequentially compose to equal the error location.
        :param noise_level: a float in [0, 1].

        :return: The probability of the independent events.
        """
        if class_ == 0:
            return noise_level
        elif class_ == 1:
            return 1/2 - math.sqrt(9 - 12*noise_level)/6
        elif class_ == 2:
            return -15**(7/8)*(15 - 16*noise_level)**(1/8)/30 + 1/2
        else:
            raise ValueError(f"Invalid `class_`: {class_}. Only 0, 1, and 2 are supported.")
    

def _unsigned_pauli_string_to_mask(unsigned_string: str) -> int:
    """Pack an unsigned Pauli string into one integer.

    The lower ``n`` bits encode X support and the next ``n`` bits encode Z
    support. Consequently, ``Y`` sets both corresponding bits and unsigned
    Pauli multiplication is integer XOR.

    :param unsigned_string: A signless Pauli string using ``_`` or ``I`` for
        identity and ``X``, ``Y``, or ``Z`` for nonidentity Paulis.

    :return mask: The packed X/Z-support mask.
    """
    x_mask = 0
    z_mask = 0
    for index, pauli in enumerate(unsigned_string):
        bit = 1 << index
        if pauli in ('X', 'Y'):
            x_mask |= bit
        if pauli in ('Z', 'Y'):
            z_mask |= bit
    return x_mask | (z_mask << len(unsigned_string))


def _pauli_mask_to_unsigned_string(mask: int, qubit_count: int) -> str:
    """Decode a packed Pauli mask into a canonical unsigned string.

    :param mask: An integer whose lower ``qubit_count`` bits encode X support
        and whose next ``qubit_count`` bits encode Z support.
    :param qubit_count: The number of qubits represented by each support mask.

    :return unsigned_string: The decoded string, using ``_`` for identity.
    """
    support_mask = (1 << qubit_count) - 1
    x_mask = mask & support_mask
    z_mask = (mask >> qubit_count) & support_mask
    paulis = []
    for index in range(qubit_count):
        x = bool(x_mask & (1 << index))
        z = bool(z_mask & (1 << index))
        paulis.append('Y' if x and z else 'X' if x else 'Z' if z else '_')
    return ''.join(paulis)


def _make_pauli_mask_restrictor(
        *,
        data_indices: tuple[int, ...],
        source_qubit_count: int,
) -> Callable[[int], int]:
    """Build a fast packed-mask restriction function.

    The returned function selects source qubits in ``data_indices`` order and
    repacks them into contiguous X and Z support masks. Byte lookup tables are
    precomputed once for repeated restrictions using the same qubit layout.

    :param data_indices: Source-qubit indices to retain, in the desired output
        order.
    :param source_qubit_count: The number of qubits represented by each source
        support mask.

    :return restrict: A function mapping a packed source effect to its packed,
        data-only effect.
    """
    data_qubit_count = len(data_indices)
    source_support_mask = (1 << source_qubit_count) - 1
    byte_tables: list[tuple[int, tuple[int, ...]]] = []
    for source_shift in range(0, source_qubit_count, 8):
        table = []
        for byte in range(256):
            restricted_byte = 0
            for data_index, source_index in enumerate(data_indices):
                if source_shift <= source_index < source_shift + 8:
                    source_bit = (byte >> (source_index - source_shift)) & 1
                    restricted_byte |= source_bit << data_index
            table.append(restricted_byte)
        byte_tables.append((source_shift, tuple(table)))

    def restrict(mask: int) -> int:
        """Restrict and repack one source Pauli mask.

        :param mask: A packed Pauli effect on ``source_qubit_count`` qubits.

        :return data_mask: The packed effect on the selected data qubits, in
            ``data_indices`` order.
        """
        source_x_mask = mask & source_support_mask
        source_z_mask = (mask >> source_qubit_count) & source_support_mask
        data_x_mask = 0
        data_z_mask = 0
        for source_shift, table in byte_tables:
            data_x_mask |= table[(source_x_mask >> source_shift) & 0xff]
            data_z_mask |= table[(source_z_mask >> source_shift) & 0xff]
        return data_x_mask | (data_z_mask << data_qubit_count)

    return restrict


def _make_logical_analysis_cache(
        *,
        logical_analyzer: LogicalAnalyzer,
        cultivated_states: tuple[CultivatedState, ...],
        maxsize: int | None,
):
    """Build a cached logical-analysis function for one enumeration run.

    On a cache miss, the packed data effect is analyzed for every cultivated
    state. Keeping the state tuple in the closure means the effect alone forms
    the cache key.

    :param logical_analyzer: The analyzer used to compute acceptance
        probabilities and logical fidelities.
    :param cultivated_states: The logical states to analyze together on each
        cache miss.
    :param maxsize: The maximum number of effects retained by the LRU cache.
        Use ``None`` for an unbounded cache or ``0`` to disable caching.

    :return analyze_mask: An LRU-wrapped function mapping a packed data effect
        to one logical-analysis pair per cultivated state. The wrapper also
        exposes ``cache_info`` and ``cache_clear``.
    """
    @lru_cache(maxsize=maxsize)
    def analyze_mask(data_effect_mask: PauliMask) -> tuple[LogicalAnalysis, ...]:
        """Analyze one packed data effect for every requested state.

        :param data_effect_mask: The packed Pauli effect on the data qubits.

        :return analyses: Acceptance-probability and logical-fidelity pairs in
            the same order as ``cultivated_states``.
        """
        return tuple(
            logical_analyzer.analyze(state, data_effect_mask)
            for state in cultivated_states
        )

    return analyze_mask


def _sum_logical_weights(
        logical_triples: Iterable[LogicalTriple],
) -> tuple[float, float]:
    """Sum acceptance-weighted logical identity and error contributions.

    :param logical_triples: Logical-analysis results containing an acceptance
        probability, logical fidelity, and set of fault configurations for
        each retained effect. Configuration multiplicity is not included in
        this diagnostic sum.

    :return: The total identity contribution followed by the total logical
        error contribution.
    """
    identity_weight = 0.0
    error_weight = 0.0
    for accept_probability, logical_fidelity, _ in logical_triples:
        identity_weight += accept_probability * logical_fidelity
        error_weight += accept_probability * (1 - logical_fidelity)
    return identity_weight, error_weight


def _sum_odds(
        logical_triples: Iterable[LogicalTriple],
        index_to_odds: dict[int, float],
    ) -> tuple[float, float]:
    """Sum the odds of all configurations in `vector_combo_pairs`.

    :param logical_triples:
        An iterable of triples, each containing:

            * an acceptance probability,
            * a logical fidelity,
            * a set of frozen sets of fault indices that defines the combination.
    :param index_to_odds: A map from each fault index to the odds of it flipping.
    """
    i_odds, e_odds = 0, 0
    for accept_probability, logical_fidelity, set_of_configurations in logical_triples:
        prob = sum(math.prod(index_to_odds[index] for index in combo) for combo in set_of_configurations)
        i_odds += accept_probability * logical_fidelity * prob
        e_odds += accept_probability * (1-logical_fidelity) * prob
    return i_odds, e_odds
