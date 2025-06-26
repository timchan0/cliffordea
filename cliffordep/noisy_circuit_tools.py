"""Module for enumerating faults in noisy stim circuits."""

from collections import defaultdict, Counter
from collections.abc import Iterable, Sequence
import itertools
from functools import cached_property
import math
from typing import Literal

import numpy as np
import numpy.typing as npt
import stim

from cliffordep.noiseless_circuit_tools import split_by_ticks
from cliffordep.constants import DEPOLARIZE2_FAULTS
from cliffordep.type_aliases import Fault, FaultSource, EffectMap
from cliffordep.pauli_string_tools import unsigned_str, push_through_transversal


def fault_count(source_name: str) -> int:
    """Return the number of faults a fault source can make.

    Input:
    * `source_name` the name of the Stim gate that gives rise to faults.
    This can be 'DEPOLARIZE1', 'DEPOLARIZE2', 'X_ERROR', 'Y_ERROR', 'Z_ERROR', 'MX', 'MY', 'MZ'.

    Output:
    * The number of faults the fault source can make.
    This indicates the noise strength p divided by the probability of each fault.
    """
    if source_name in {'X_ERROR', 'Y_ERROR', 'Z_ERROR', 'MX', 'MY', 'MZ'}:
        return 1
    elif source_name == 'DEPOLARIZE1':
        return 3
    elif source_name == 'DEPOLARIZE2':
        return 15
    else:
        raise ValueError(f"Unknown fault source: {source_name}")
    

class CultivationCircuit:
    """A class representing a noisy stim circuit with methods to analyze faults.
    
    Instance attributes:
    * `noisy_circuit` a `stim.Circuit` annotated with noise.
    * `DATA_INDICES` a set of indices corresponding to the data qubits.
    * `STABILIZER_GENERATORS` the generators of the stabilizer group.
    * `LOGICAL_X` a Pauli string representing a logical X operator.
    * `LOGICAL_Z` ditto for Z.
    """


    def __init__(
            self,
            noisy_circuit: stim.Circuit,
            data_indices: Sequence[int],
            stabilizer_generators: Iterable[stim.PauliString],
            logical_x: stim.PauliString,
            logical_z: stim.PauliString,
    ) -> None:
        self.noisy_circuit = noisy_circuit
        self.DATA_INDICES = data_indices
        self.STABILIZER_GENERATORS = stabilizer_generators
        self.LOGICAL_X = logical_x
        self.LOGICAL_Z = logical_z


    @cached_property
    def noiseless_circuit(self):
        return self.noisy_circuit.without_noise()


    def get_undetected_fault_combinations(self, max_order: int, print_progress: bool = False):
        """Find all combinations of faults up to `max_order` that have trivial syndrome.

        Input:
        * `max_order` the maximum order of probability to consider event account for.
        * `print_progress` whether to print progress.

        Output:
        * A list whose kth entry is a map
        from each effect to a counter of k-tuples of denominators.
        Each denominator represents a fault such that if all in the tuple occur,
        the syndrome will be trivial.
        """
        effect_maps = self.group_faults_by_effect()
        if print_progress:
            print(f"Finished enumerating all faults. Found {len(effect_maps)} distinct syndromes.")
        return [self._get_undetected_fault_combinations_for_length(
            effect_maps, length, print_progress) for length in range(max_order + 1)]


    def get_kept_strings(
            self,
            fault_combinations: list[dict[str, Counter[tuple[int, ...]]]],
            cultivated_state: Literal['T', 'S', 'Z'] = 'T',
            print_progress: bool = False,
    ) -> list[tuple[
        dict[str, tuple[float, Counter[tuple[int, ...]]]],
        dict[str, tuple[float, Counter[tuple[int, ...]]]],
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
                - a counter of ordered denominator tuples.
            - `error_strings` ditto for effects that lead to a logical error.
        """
        return [self._get_kept_strings(combinations_of_order, cultivated_state, order if print_progress else None)
                for order, combinations_of_order in enumerate(fault_combinations)]


    def group_faults_by_effect(
            self,
            restrict_to_data: bool = True,
    ) -> dict[tuple[bool, ...], EffectMap]:
        """Group all possible faults in a noisy circuit by their syndrome then effect.

        Input:
        * `restrict_to_data` whether to restrict all effects to only the data qubits.

        Output:
        * `effect_maps` maps each syndrome to an `EffectMap` containing at its lowest level
        faults that cause that syndrome.
        """
        _faults: defaultdict[
            tuple[bool, ...], defaultdict[str, Counter[FaultSource]]
        ] = defaultdict(lambda: defaultdict(Counter))
        for source, group in self._group_faults_by_source().items():
            for fault in group:
                syndrome, effect = self._get_syndrome_and_effect(fault)
                tuple_syndrome: tuple[bool] = tuple(syndrome)
                if restrict_to_data:
                    effect = ''.join(effect[index] for index in self.DATA_INDICES)
                _faults[tuple_syndrome][effect][source] += 1
        return {syndrome: dict(effect_map) for syndrome, effect_map in _faults.items()}


    def _get_syndrome_and_effect(self, fault: Fault):
        """Get the syndrome and the resultant Pauli string after inserting a fault.
        
        Input:
        * `fault` the fault to analyze.

        Output:
        * `syndrome` a tuple of booleans representing the syndrome, where each boolean
        indicates whether the corresponding detector has been flipped.
        * `effect` the effect of the fault when propagated to the end of the circuit,
        as an unsigned Pauli string.
        """
        timeslice, name, targets = fault
        pauli_string = _fault_to_pauli_string(
            qubit_count=self.noisy_circuit.num_qubits,
            name=name,
            targets=targets,
        )
        syndrome: npt.NDArray[np.bool_] = np.zeros(self.noisy_circuit.num_detectors, dtype=bool)
        if name.startswith('M'):
            # pauli_string is identity
            for target in targets:
                for detector in self._measurement_to_detectors[timeslice, target.value]:
                    syndrome[detector] ^= True
        else:
            remaining_layers = self._noiseless_layers[timeslice+1:]
            for k, layer in enumerate(remaining_layers, start=1):
                for instruction in layer:
                    if isinstance(instruction, stim.CircuitRepeatBlock):
                        raise ValueError("There is a REPEAT block in the circuit.")
                    data = stim.gate_data(instruction.name)
                    if produces_measurements:=data.produces_measurements:
                        anticommuting_paulis = _get_anticommuting_paulis(instruction.name)
                        for target in instruction.targets_copy():
                            if pauli_string[target.value] in anticommuting_paulis:
                                for detector in self._measurement_to_detectors[timeslice+k, target.value]:
                                    syndrome[detector] ^= True
                    if is_reset:=data.is_reset:
                        for target in instruction.targets_copy():
                            pauli_string[target.value] = 'I'
                    if not (produces_measurements or is_reset):
                        pauli_string = pauli_string.after(instruction)
        
        return syndrome, unsigned_str(pauli_string)


    @cached_property
    def _measurement_to_detectors(self) -> dict[tuple[int, int], set[int]]:
        """A map from each measurement in `self.noisy_circuit` indexed by (timeslice, qubit value)
        to a set of indices of the detectors it flips.
        """
        measurement_to_detectors: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
        measurement_indices: list[tuple[int, int]] = []
        detector_count: int = 0
        timeslice: int = 0
        for instruction in self.noisy_circuit:
            if isinstance(instruction, stim.CircuitRepeatBlock):
                raise ValueError("REPEAT blocks not handled in this mapping.")
            elif (name:=instruction.name) == "TICK":
                timeslice += 1
            elif stim.gate_data(name).produces_measurements:
                for target in instruction.targets_copy():
                    measurement_indices.append((timeslice, target.value))
            elif name == "DETECTOR":
                for target in instruction.targets_copy():
                    measurement_to_detectors[measurement_indices[target.value]].add(detector_count)
                detector_count += 1
        return dict(measurement_to_detectors)
    

    @cached_property
    def _noiseless_layers(self) -> list[stim.Circuit]:
        """A list of stim.Circuit objects, one for each timeslice,
        representing the noiseless layers of the circuit.
        """
        return split_by_ticks(self.noisy_circuit.without_noise())


    def _group_faults_by_source(self):
        """Group all possible faults in a noisy circuit by their source Stim gate.

        Output:
        * A map from a `FaultSource` to a set `Fault`s.`
        """
        _faults: defaultdict[FaultSource, set[Fault]] = defaultdict(set)
        for timeslice, layer in enumerate(split_by_ticks(self.noisy_circuit)):
            for instruction in layer:
                if isinstance(instruction, stim.CircuitRepeatBlock):
                    raise ValueError("There is a REPEAT block in the circuit.")
                for basis in ('X', 'Y', 'Z'):
                    if instruction.name in {f'{basis}_ERROR', f'M{basis}'}:
                        for target in instruction.targets_copy():
                            _faults[timeslice, instruction.name, (target,)].add(
                                (timeslice, instruction.name, (target,)))
                    elif instruction.name == 'DEPOLARIZE1':
                        for target in instruction.targets_copy():
                            _faults[timeslice, instruction.name, (target,)].add(
                                (timeslice, f'{basis}_ERROR', (target,)))
                if instruction.name == 'DEPOLARIZE2':
                    for target_1, target_2 in instruction.target_groups():
                        group = _faults[timeslice, instruction.name, (target_1, target_2)]
                        for fault in DEPOLARIZE2_FAULTS:
                            match fault:
                                case ('I', basis):
                                    group.add((timeslice, f'{basis}_ERROR', (target_2,)))
                                case (basis, 'I'):
                                    group.add((timeslice, f'{basis}_ERROR', (target_1,)))
                                case (basis_1, basis_2):
                                    group.add((timeslice, 'E', (
                                        stim.target_pauli(target_1.value, basis_1),
                                        stim.target_pauli(target_2.value, basis_2),
                                    )))
        return dict(_faults)


    def _get_kept_strings(
            self,
            combinations_of_order: dict[str, Counter[tuple[int, ...]]],
            cultivated_state: Literal['T', 'S', 'Z'] = 'T',
            order: None | int = None,
    ) -> tuple[
        dict[str, tuple[float, Counter[tuple[int, ...]]]],
        dict[str, tuple[float, Counter[tuple[int, ...]]]],
    ]:
        """Group postselected effects by whether they lead to identity or error.

        Helper for `get_kept_strings`.

        Input:
        * `combinations_of_order` a map from each effect to a counter of tuples of denominators.
        Each denominator represents a fault such that if all in the tuple occur,
        the syndrome will be trivial.
        * `cultivated_state` the target logical state cultivated.
        * `order` an optional parameter used only for printing progress.

        Output:
        * `identity_strings` a map from each effect that leads to logical identity,
        to the probability they are not rejected and a counter of ordered denominator tuples.
        * `error_strings` ditto for effects that lead to a logical error.
        """
        identity_strings: dict[str, tuple[float, Counter[tuple[int, ...]]]] = {}
        error_strings: dict[str, tuple[float, Counter[tuple[int, ...]]]] = {}
        for data_string, denominators in combinations_of_order.items():
            clifford = push_through_transversal(stim.PauliString(data_string), gate=cultivated_state)
            clifford.postselect_from_stabilizers(self.STABILIZER_GENERATORS)
            logical_vector = clifford.get_logical_amplitudes(self.LOGICAL_X, self.LOGICAL_Z)
            # TODO: check if `transfer_xy_to_iz` is unitary. If so, swap with next line.
            logical_vector.transfer_xy_to_iz(logical_state=cultivated_state)
            if logical_vector.probability_mass:
                # TODO: use `logical_error_probability` instead of `is_logical_error`
                if logical_vector.is_logical_error:
                    error_strings[data_string] = (logical_vector.probability_mass, denominators)
                else:
                    identity_strings[data_string] = (logical_vector.probability_mass, denominators)
        if order is not None:
            print(f'For order {order}, {len(identity_strings)} ({len(error_strings)}) effects are stabilized and lead to identity (error).')
        return identity_strings, error_strings
    
    
    def _get_undetected_fault_combinations_for_length(
            self,
            effect_maps: dict[tuple[bool, ...], EffectMap],
            length: int,
            print_progress: bool = False,
    ) -> dict[str, Counter[tuple[int, ...]]]:
        """Find all combinations of `length` faults from `effect_maps` that have trivial syndrome.
        
        Helper for `get_undetected_fault_combinations`.
        
        Input:
        * `effect_maps` the output of `CultivationCircuit.group_faults_by_effect`.
        * `length` the length of combinations to find.
        * `print_progress` whether to print progress.

        Output:
        * a map from each effect to a counter of `length`-tuples of denominators.
        Each denominator represents a fault such that if all in the tuple occur,
        the syndrome will be trivial.
        """
        data_qubit_count = len(self.DATA_INDICES)
        result: defaultdict[str, Counter[tuple[int, ...]]] = defaultdict(Counter)
        if print_progress:
            print(f"Finding undetected combinations of length {length}...")
        if length == 0:
            first_effect_map, *_ = effect_maps.values()
            first_effect, *_ = first_effect_map.keys()
            result['_'*len(first_effect)][()] += 1
        else:
            trivial_syndrome_combos = self._get_trivial_syndrome_combinations(effect_maps.keys(), length)
            for syndrome_counter in trivial_syndrome_combos:
                options = _syndrome_counter_to_options(effect_maps, syndrome_counter, data_qubit_count)
                for segment_product in itertools.product(*(option.items() for option in options)):
                    product_effect = unsigned_str(math.prod(
                        (stim.PauliString(effect) for effect, _ in segment_product),
                        start=stim.PauliString(data_qubit_count)
                    ))
                    nested_candidates = itertools.product(*(valid_segments for _, valid_segments in segment_product))
                    for nested_candidate in nested_candidates:
                        candidate = itertools.chain(*nested_candidate)
                        _update_undetected_combinations(result, product_effect, candidate)
        if print_progress:
            print(f"Done. They lead to {len(result)} distinct effects.")
        return dict(result)
    
    
    @staticmethod
    def _get_trivial_syndrome_combinations(
            syndromes: Iterable[tuple[bool, ...]],
            length: int,
    ) -> list[Counter[tuple[bool, ...]]]:
        """Find all combinations of `length` syndromes that result in a trivial syndrome.
        
        Input:
        * `syndromes` an iterable of syndromes.
        * `length` the length of combinations to find.
        """
        if length == 0:
            return [Counter()]
        trivial_combos: list[Counter[tuple[bool, ...]]] = []
        combos = itertools.combinations_with_replacement(syndromes, length)
        for combo in combos:
            resultant_syndrome: npt.NDArray[np.int_] = np.array(combo).sum(axis=0) % 2
            if not any(resultant_syndrome):
                trivial_combos.append(Counter(combo))
        return trivial_combos


def restrict_to_data(
        strings: dict[str, Counter[tuple[int, ...]]],
        data_indices: Iterable[int],
        min_weight: int = 0,
):
    """Restrict `strings` to only the data qubits and optionally filter by weight.

    Input:
    * `strings` maps each Pauli string to a Counter of denominator tuples.
    * `data_indices` a set of indices corresponding to the data qubits.
    * `min_weight` the minimum weight on the data qubits of the Pauli strings to consider.

    Output:
    * a map from each Pauli string,
    restricted to the data qubits,
    to a Counter of denominator tuples.
    If `min_weight` is set,
    only include Pauli strings with that weight or higher on the data qubits.
    """
    filtered: defaultdict[str, Counter[tuple[int, ...]]] = defaultdict(Counter)
    for string, denominators in strings.items():
        data_string = stim.PauliString(string[index] for index in data_indices)
        if data_string.weight >= min_weight:
            filtered[unsigned_str(data_string)].update(denominators)
    return dict(filtered)


def _fault_to_pauli_string(qubit_count: int, name: str, targets: tuple[stim.GateTarget, ...]):
    """Convert a fault to a stim.PauliString.
    
    Input:
    * `qubit_count` the number of qubits in the Pauli string.
    * `name` the name of the fault,
    which can be 'E', 'X_ERROR', 'Y_ERROR', 'Z_ERROR', 'MX', 'MY', 'MZ'.
    The measurement faults do not affect the Pauli string.
    * `targets` a tuple of stim.GateTarget objects representing the qubits the fault acts on.

    Output:
    * A stim.PauliString representing the fault.
    """
    pauli_string = stim.PauliString(qubit_count)
    if name == 'E':
        for target in targets:
            pauli_string[target.value] = target.pauli_type
    elif (basis := name[0]) != 'M':
        (target,) = targets
        pauli_string[target.value] = basis
    return pauli_string


def _get_anticommuting_paulis(name: str):
    """Get the set of Paulis that anticommute with the measurement given by `name`."""
    if 'X' in name:
        return {2, 3}
    elif 'Y' in name:
        return {1, 3}
    elif 'Z' in name or name in {'M', 'MR'}:
        return {1, 2}
    else:
        raise NotImplementedError


# ALL FUNCTIONS BELOW PERTAIN TO `undetected_combinations`


def _syndrome_counter_to_options(
        effect_maps: dict[tuple[bool, ...], EffectMap],
        syndrome_counter: Counter[tuple[bool, ...]],
        qubit_count: int,
):
    """Get valid fault combination segments for each syndrome based on the counts in `syndrome_counter`.
    
    Input:
    * `effect_maps` the output of `group_faults_by_effect`.
    * `syndrome_counter` dictates for each syndrome
    how many fault sources in `effect_maps[syndrome]` should appear in the fault combination.
    E.g. `{syndrome1: 2, syndrome2: 1}`.

    Output:
    * `options` a list of options, one for each item in `syndrome_counter`
    e.g. `[option1, option2]`.
    Each option is a map from effect to a list of valid segments with that effect.
    e.g. `{effect1: [segment1, segment2], effect2: [segment3]}`
    where `segment1 = [(source1, 1), (source2, 1)]`.
    """
    options: list[dict[str, list[Sequence[tuple[FaultSource, int]]]]] = []
    for syndrome, count in syndrome_counter.items():
        effect_map = effect_maps[syndrome]
        product_effects_to_valid_segments = _get_product_effect_to_valid_segments(
            effect_map, count, qubit_count)
        options.append(product_effects_to_valid_segments)
    return options


def _get_product_effect_to_valid_segments(
    effect_map: EffectMap,
    length: int,
    qubit_count: int,
):
    """Get a map from product effect to all valid segments of undetected fault combinations.
    
    Input:
    * `effect_map` maps each effect to a map from each fault source
    to the number of its faults that cause that syndrome and effect.
    E.g. `{effect1: {source1: 1, source2: 1, source3: 2}, effect2: {source4: 1}}`.
    * `length` the length of valid segments to consider.

    Output:
    * a map from product effect to a list of valid segments.
    Each valid segment is a sequence of pairs each containing:
        - a unique fault source
        - the number of its faults that can be used as part of the undetected combination.
    """
    result: defaultdict[str, list[Sequence[tuple[FaultSource, int]]]] = defaultdict(list)
    effect_combos = itertools.combinations_with_replacement(effect_map.keys(), length)
    for effect_combo in effect_combos:
        # e.g. effect_combo = ('effect1', 'effect1', 'effect2')
        counter = Counter(effect_combo)
        # e.g. counter = {'effect1': 2, 'effect2': 1}
        product_effect = unsigned_str(math.prod(
            (stim.PauliString(effect) for effect, count in counter.items() if count % 2),
            start=stim.PauliString(qubit_count),
        ))
        options = _effect_counter_to_options(effect_map, counter)
        valid_segments: list[Sequence[tuple[FaultSource, int]]] = []
        for fault_product in itertools.product(*options):
            flattened = list(itertools.chain(*fault_product))
            _process_candidate_segment(valid_segments, flattened)
        if valid_segments:
            result[product_effect] += valid_segments
    
    # ### # TODO: store segments as frozensets to avoid duplicates
    # for segments in result.values():
    #     unique_segments: set[frozenset[FaultSource]] = set()
    #     for segment in segments:
    #         frozen_sources = frozenset(source for source, _ in segment)
    #         if frozen_sources in unique_segments:
    #             print(f"Duplicate segment found: {segment}")
    #         else:
    #             unique_segments.add(frozen_sources)
    # ###
    
    return dict(result)


def _effect_counter_to_options(effect_map: EffectMap, counter: Counter[str]):
    """Get fault source combinations for each effect based on the counts in `counter`.
    
    Input:
    * `effect_map` maps each effect to a map from each fault source
    to the number of its faults that cause that syndrome and effect.
    E.g. `{effect1: {source1: 1, source2: 1, source3: 2}, effect2: {source4: 1}}`.
    * `counter` dictates for each effect and source map in `effect_map`,
    how many sources in source map should appear in the fault combination.
    E.g. `{effect1: 2, effect2: 1}`.

    Output:
    * `options` a list of options, one for each item in `counter`
    e.g. `[option1, option2]`.
    Each option is an iterable of source combinations
    e.g. `[((source1, 1), (source2, 1)), ((source1, 1), (source3, 2)), ((source2, 1), (source3, 2))]`.
    Each source combination is a combination of `count` fault sources in `effect_map[effect]`.
    """
    options: list[itertools.combinations[tuple[tuple[FaultSource, int], ...]]] = []
    for effect, count in counter.items():
        option = itertools.combinations(effect_map[effect].items(), count)
        # without replacement because each item in effect_map[effect]
        # is a (fault source, fault count) pair and each error event must have distinct sources
        # e.g. effect_map[effect] = {'source1': 1, 'source2': 1, 'source3': 2}
        options.append(option)
    return options


def _process_candidate_segment(
        valid_segments: list[Sequence[tuple[FaultSource, int]]],
        candidate: Sequence[tuple[FaultSource, int]],
):
    """Update `valid_segments` with a candidate segment of an undetected combination of faults.

    Input:
    * `valid_segments` a list of valid segments.
    Each (valid) segment is a sequence of pairs each containing:
        - a (unique) fault source
        - the number of its faults that can be used as part of the undetected combination.
    * `candidate` the segment to consider adding
    e.g. `[(source1, 1), (source2, 1), (source3, 2)]`.

    Side effect:
    * Update `valid_segments` with `candidate` if it is valid i.e. comprises distinct fault sources.
    """
    # check for duplicate sources
    sources: set[FaultSource] = set()
    for source, _ in candidate:
        if source in sources:
            return
        sources.add(source)
    valid_segments.append(candidate)


def _update_undetected_combinations(
        undetected_combinations: defaultdict[str, Counter[tuple[int, ...]]],
        effect: str,
        candidate: Iterable[tuple[FaultSource, int]],
):
    """Update `undetected_combinations` with a candidate undetected combination of faults.
    
    Input:
    * `undetected_combinations` maps each effect to a counter of denominators for each combination.
    * `effect` the resultant (unsigned) Pauli string of the candidate combination.
    * `candidate` a sequence (whose length equals that of the candidate combination)
    of pairs each containing:
        - a fault source
        - count the number of its faults that can be used as part of the undetected combination.

    Side effect:
    * Update `undetected_combinations` with the undetected combination of faults if it is valid
    i.e. comprises distinct fault sources.
    """
    # check for duplicate sources
    sources: set[FaultSource] = set()
    unsorted_denominators: list[int] = []
    tot = 1
    for source, count in candidate:
        if source in sources:
            return
        _, name, _ = source
        unsorted_denominators.append(fault_count(name))
        tot *= count
        sources.add(source)
    denominators = tuple(sorted(unsorted_denominators))
    undetected_combinations[effect][denominators] += tot