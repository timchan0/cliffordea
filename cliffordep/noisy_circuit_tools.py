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
from cliffordep.combinators import Combinator


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


    def get_syndrome_and_effect(self, fault: Fault):
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
        pauli_string = self._fault_to_pauli_string(name=name, targets=targets)
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
                        anticommuting_paulis = self._get_anticommuting_paulis(instruction.name)
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


    def _fault_to_pauli_string(self, name: str, targets: tuple[stim.GateTarget, ...]):
        """Convert a fault to a stim.PauliString.
        
        Input:
        * `name` the name of the fault,
        which can be 'E', 'X_ERROR', 'Y_ERROR', 'Z_ERROR', 'MX', 'MY', 'MZ'.
        The measurement faults do not affect the Pauli string.
        * `targets` a tuple of stim.GateTarget objects representing the qubits the fault acts on.

        Output:
        * A stim.PauliString representing the fault.
        """
        pauli_string = stim.PauliString(self.noisy_circuit.num_qubits)
        if name == 'E':
            for target in targets:
                pauli_string[target.value] = target.pauli_type
        elif (basis := name[0]) != 'M':
            (target,) = targets
            pauli_string[target.value] = basis
        return pauli_string


    @staticmethod
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


    def group_faults_by_source(self):
        """Group all possible faults in a noisy circuit by their source Stim gate.

        Output:
        * A map from a `FaultSource` to a set `Fault`s.
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


    def string_to_logical_vector(self, cultivated_state: Literal['T', 'S', 'Z'], data_string: str):
        clifford = push_through_transversal(stim.PauliString(data_string), gate=cultivated_state)
        clifford.postselect_from_stabilizers(self.STABILIZER_GENERATORS)
        logical_vector = clifford.get_logical_amplitudes(self.LOGICAL_X, self.LOGICAL_Z)
        logical_vector.transfer_xy_to_iz(logical_state=cultivated_state)
        return logical_vector
    

def get_trivial_syndrome_combinations(
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


class FaultSourceCombinator(Combinator):
    """Group all possible faults in a noisy circuit by their syndrome then effect.

    Extends `Combinator`.
    
    Instance attributes:
    * `circuit` the noisy circuit to analyze.
    * `basis` a map from each syndrome to an `EffectMap` containing at its lowest level
    faults that cause that syndrome.
    """

    def __init__(
            self,
            circuit: CultivationCircuit,
            restrict_to_data: bool = True,
            print_progress: bool = False,
    ) -> None:
        _basis: defaultdict[
            tuple[bool, ...], defaultdict[str, Counter[FaultSource]]
        ] = defaultdict(lambda: defaultdict(Counter))
        for source, group in circuit.group_faults_by_source().items():
            for fault in group:
                syndrome, effect = circuit.get_syndrome_and_effect(fault)
                tuple_syndrome: tuple[bool] = tuple(syndrome)
                if restrict_to_data:
                    effect = ''.join(effect[index] for index in circuit.DATA_INDICES)
                _basis[tuple_syndrome][effect][source] += 1
        self.basis = {syndrome: dict(effect_map) for syndrome, effect_map in _basis.items()}
        if print_progress:
            print(f"Finished enumerating all faults. Found {self.syndrome_count} distinct syndromes.")
        super().__init__(circuit, restrict_to_data, print_progress)

    @property
    def syndrome_count(self) -> int:
        return len(self.basis)
    
    def keys(self): return self.basis.keys()
    
    def values(self): return self.basis.values()
    
    def items(self): return self.basis.items()
    

    @classmethod
    def error_rate_per_kept_shot(
            cls,
            all_string_leads: list[tuple[
                dict[str, tuple[float, Counter[int]]],
                dict[str, tuple[float, Counter[int]]],
            ]],
            noise_level: float,
            print_progress: bool = False,
    ) -> float:
        """Calculate the logical error rate per kept shot for a given noise level.
        
        Input:
        * `all_string_leads` the output of `get_kept_strings`.
        * `noise_level` the noise level to analyze.
        * `print_progress` whether to print progress.
        """
        p_odds = noise_level / (1 - noise_level)
        identity_odds, error_odds = 0, 0
        for length, (identity_strings, error_strings) in enumerate(all_string_leads):
            single_odds = p_odds**length
            i_odds = single_odds * cls._get_normalized_counts(identity_strings.values())
            e_odds = single_odds * cls._get_normalized_counts(error_strings.values())
            identity_odds += i_odds
            error_odds += e_odds
            if print_progress:
                print(f'O(p^{length}) events:')
                print(f'Identity odds = {i_odds}')
                print(f'Error odds = {e_odds}')
        return error_odds / (identity_odds + error_odds)


    def _get_undetected_fault_combinations_for_length(
            self,
            length: int,
            print_progress: bool = False,
    ) -> dict[str, Counter[int]]:
        result: defaultdict[str, Counter[int]] = defaultdict(Counter)
        if print_progress:
            print(f"Finding undetected combinations of length {length}...")
        if length == 0:
            first_effect_map, *_ = self.values()
            first_effect, *_ = first_effect_map.keys()
            result['_'*len(first_effect)][1] += 1
        else:
            trivial_syndrome_combos = get_trivial_syndrome_combinations(
                self.keys(), length)
            for syndrome_counter in trivial_syndrome_combos:
                options = self._syndrome_counter_to_options(syndrome_counter, len(self.circuit.DATA_INDICES))
                for segment_product in itertools.product(*(option.items() for option in options)):
                    # note: this is just as fast as iterating through `segment_product` once
                    product_effect = unsigned_str(math.prod(
                        (stim.PauliString(effect) for effect, _ in segment_product),
                        start=stim.PauliString(len(self.circuit.DATA_INDICES))))
                    candidates = itertools.product(*(
                        valid_segments.items() for _, valid_segments in segment_product))
                    for candidate in candidates:
                        self._process_candidate(result, product_effect, candidate)
        if print_progress:
            print(f"Done. They lead to {len(result)} distinct effects.")
        return dict(result)
    

    def _syndrome_counter_to_options(
            self,
            syndrome_counter: Counter[tuple[bool, ...]],
            qubit_count: int,
    ):
        """Get valid fault combination segments for each syndrome based on the counts in `syndrome_counter`.
        
        Input:
        * `syndrome_counter` dictates for each syndrome
        how many fault sources in `self.basis[syndrome]` should appear in the fault combination.
        E.g. `{syndrome1: 2, syndrome2: 1}`.

        Output:
        * `options` a list of options, one for each item in `syndrome_counter`
        e.g. `[option1, option2]`.
        Each option is a map from effect to a counter of valid segments with that effect.
        e.g. `{effect1: {segment1: 3, segment2: 1}, effect2: {segment3: 5}}`
        where e.g. `segment1 = {source1, source2}`.
        """
        options: list[dict[str, Counter[frozenset[FaultSource]]]] = []
        for syndrome, count in syndrome_counter.items():
            effect_map = self.basis[syndrome]
            effects_to_valid_segments = self._get_effect_to_valid_segments(
                effect_map, count, qubit_count)
            options.append(effects_to_valid_segments)
        return options
    

    @classmethod
    def _get_effect_to_valid_segments(
        cls,
        effect_map: EffectMap,
        length: int,
        qubit_count: int,
    ):
        """Get a map from effect to the number of valid fault combinations with that resultant effect.
        
        Input:
        * `effect_map` maps each effect to a map from each fault source
        to the number of its faults that cause that syndrome and effect.
        E.g. `{effect1: {source1: 1, source2: 1, source3: 2}, effect2: {source4: 1}}`.
        * `length` the length of valid fault combinations to consider.

        Output:
        * a map from effect to a counter of fault source combinations.
        Each fault source combination is a set of unique fault sources.
        Each count is the number of fault combinations from that fault source combination
        with that resultant effect.
        """
        result: defaultdict[str, Counter[frozenset[FaultSource]]] = defaultdict(Counter)
        effect_combos = itertools.combinations_with_replacement(effect_map.keys(), length)
        for effect_combo in effect_combos:
            # e.g. effect_combo = ('effect1', 'effect1', 'effect2')
            counter = Counter(effect_combo)
            # e.g. counter = {'effect1': 2, 'effect2': 1}
            product_effect = unsigned_str(math.prod(
                (stim.PauliString(effect) for effect, count in counter.items() if count % 2),
                start=stim.PauliString(qubit_count),
            ))
            options = cls._effect_counter_to_options(effect_map, counter)
            valid_segments: Counter[frozenset[FaultSource]] = Counter()
            for fault_product in itertools.product(*options):
                flattened = itertools.chain(*fault_product)
                cls._process_candidate_segment(valid_segments, flattened)
            if valid_segments:
                result[product_effect].update(valid_segments)
        return dict(result)
    

    @staticmethod
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
        Each source combination is a combination of `count` distinct fault sources in `effect_map[effect]`.
        """
        options: list[itertools.combinations[tuple[tuple[FaultSource, int], ...]]] = []
        for effect, count in counter.items():
            option = itertools.combinations(effect_map[effect].items(), count)
            # without replacement because each item in effect_map[effect]
            # is a (fault source, fault count) pair and each error event must have distinct sources
            # e.g. effect_map[effect] = {'source1': 1, 'source2': 1, 'source3': 2}

            # TODO: see if making `option` the following form is faster:
            # `[({source1, source2}, 1), ({source1, source3}, 2), ({source2, source3}, 2)]`
            options.append(option)
        return options
    

    @staticmethod
    def _process_candidate_segment(
            valid_segments: Counter[frozenset[FaultSource]],
            candidate: Iterable[tuple[FaultSource, int]],
    ):
        """Update `valid_segments` with a candidate segment of an undetected combination of faults.

        Input:
        * `valid_segments` a counter of valid segments.
        Each (valid) segment is a frozen set of (unique) fault sources.
        Each count is the number of fault combinations from that fault source combination
        that can be used as part of the undetected combination.
        * `candidate` the segment to consider
        e.g. `[(source1, 1), (source2, 1), (source3, 2)]`.

        Side effect:
        * Update `valid_segments` with `candidate` if it is valid i.e. comprises distinct fault sources.
        """
        sources: set[FaultSource] = set()
        counts: int = 1
        for source, count in candidate:
            if source in sources:  # check for duplicate sources
                return
            sources.add(source)
            counts *= count
        valid_segments[frozenset(sources)] += counts


    @staticmethod
    def _process_candidate(
            undetected_combinations: defaultdict[str, Counter[int]],
            effect: str,
            candidate: Iterable[tuple[frozenset[FaultSource], int]],
    ):
        """Update `undetected_combinations` with a candidate undetected combination of faults.
        
        Input:
        * `undetected_combinations` the output of `CultivationCircuit._get_undetected_fault_combinations_for_length`.
        * `effect` the resultant (unsigned) Pauli string of the candidate combination.
        * `candidate` an iterable of segments (whose total length equals that of the candidate combination).
        Each segment is a pair containing:
            - a frozenset of fault sources,
            - the number of fault combinations that can be used as part of the undetected combination.
        E.g. `((frozenset({source1, source2}), 1), (frozenset({source3}), 2))`
        represents a candidate combination
        of faults that has the resultant effect `effect` and is made up of two segments
        where the first segment is made up of two distinct fault sources and the second segment is made
        up of one fault source.

        Side effect:
        * Update `undetected_combinations` with the undetected combination of faults if it is valid
        i.e. comprises distinct fault sources.
        """
        current_sources: set[FaultSource] = set()
        current_count: int = 1
        denominators: int = 1
        for sources, count in candidate:
            if not current_sources.isdisjoint(sources):  # check for duplicate sources
                return
            current_sources.update(sources)
            current_count *= count
            denominators *= math.prod(fault_count(name) for _, name, _ in sources)
        undetected_combinations[effect][denominators] += current_count
    

    @classmethod
    def _get_normalized_counts(cls, probability_counter_pairs: Iterable[tuple[float, Counter[int]]]) -> float:
        return sum(
            probability_kept * cls._counter_to_normalized_total(counter)
            for probability_kept, counter in probability_counter_pairs
        )
    
    @staticmethod
    def _counter_to_normalized_total(counter: Counter[int]) -> float:
        return sum(count / denominator for denominator, count in counter.items())


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