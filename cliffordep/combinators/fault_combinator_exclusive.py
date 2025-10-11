import itertools
from collections import Counter, defaultdict
from collections.abc import Iterable
import math

import numpy as np
import numpy.typing as npt
import stim

from cliffordep.noisy_circuit_tools import CultivationCircuit
from cliffordep.pauli_string_tools import forget_sign
from cliffordep.type_aliases import EffectMap, ErrorLocation
from cliffordep.combinators._base import BaseExclusiveCombinator


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


def error_event_count(process_name: str) -> int:
    """Return the number of error events an error process can make.

    Input:
    * `process_name` the name of the Stim gate that gives rise to faults.
    This can be 'DEPOLARIZE1', 'DEPOLARIZE2', 'X_ERROR', 'Y_ERROR', 'Z_ERROR', 'MX', 'MY', 'MZ'.

    Output:
    * The number of error events the error process can make.
    This indicates the noise strength divided by the probability of each error event.
    """
    if process_name in {'X_ERROR', 'Y_ERROR', 'Z_ERROR', 'MX', 'MY', 'MZ'}:
        return 1
    elif process_name == 'DEPOLARIZE1':
        return 3
    elif process_name == 'DEPOLARIZE2':
        return 15
    else:
        raise ValueError(f"Unknown error process: {process_name}")


class FaultCombinatorExclusive(BaseExclusiveCombinator):
    """Group all possible faults in a noisy circuit by their syndrome then effect then error location.

    Extends `BaseExclusiveCombinator`.
    Finds undetected fault combinations first by iterating through all
    syndrome combinations and recording which ones are trivial.
    Each syndrome corresponds to a set of error locations,
    so for each trivial syndrome combination,
    this combinator finds all error location combinations
    that correspond to this syndrome combination.

    Additional instance attributes:
    * `basis` a map from each syndrome to an `EffectMap` containing at its lowest level
    faults that cause that syndrome.
    """

    def __init__(
            self,
            circuit: CultivationCircuit,
            print_progress: bool = False,
    ) -> None:
        _basis: defaultdict[
            tuple[bool, ...], defaultdict[str, Counter[ErrorLocation]]
        ] = defaultdict(lambda: defaultdict(Counter))
        for error_location, group in circuit.group_error_events_by_location().items():
            for error_event in group:
                syndrome, effect = circuit.get_syndrome_and_effect(error_event)
                tuple_syndrome = tuple(syndrome)
                _basis[tuple_syndrome][effect][error_location] += 1
        self.basis = {syndrome: dict(effect_map) for syndrome, effect_map in _basis.items()}
        if print_progress:
            print(f"Finished enumerating all faults. Found {self.syndrome_count} distinct syndromes.")
        super().__init__(circuit, print_progress)

    @property
    def syndrome_count(self) -> int:
        return len(self.basis)

    def keys(self): return self.basis.keys()

    def values(self): return self.basis.values()

    def items(self): return self.basis.items()


    def _get_undetected_configurations_for_length(
            self,
            length: int,
            print_progress: bool = False,
    ) -> dict[str, Counter[int]]:
        result: defaultdict[str, Counter[int]] = defaultdict(Counter)
        if print_progress:
            print(f"Finding undetected configurations of length {length}...")
        if length == 0:
            result['_'*self.circuit.noisy_circuit.num_qubits][1] += 1
        else:
            trivial_syndrome_combos = get_trivial_syndrome_combinations(
                self.keys(), length)
            for syndrome_counter in trivial_syndrome_combos:
                options = self._syndrome_counter_to_options(syndrome_counter)
                for segment_product in itertools.product(*(option.items() for option in options)):
                    # note: this is just as fast as iterating through `segment_product` once
                    # segment_product never empty
                    product_effect = forget_sign(math.prod(
                        stim.PauliString(effect) for effect, _ in segment_product)) # type: ignore
                    candidates = itertools.product(*(
                        valid_segments.items() for _, valid_segments in segment_product))
                    for candidate in candidates:
                        self._process_candidate(result, product_effect, candidate)
        if print_progress:
            print(f"Done. They lead to {len(result)} distinct errors.")
        return dict(result)


    def _syndrome_counter_to_options(self, syndrome_counter: Counter[tuple[bool, ...]]):
        """Get valid fault combination segments for each syndrome based on the counts in `syndrome_counter`.

        Input:
        * `syndrome_counter` dictates for each syndrome
        how many error locations in `self.basis[syndrome]` should appear in the fault combination.
        E.g. `{syndrome1: 2, syndrome2: 1}`.

        Output:
        * `options` a list of options, one for each item in `syndrome_counter`
        e.g. `[option1, option2]`.
        Each option is a map from effect to a counter of valid segments with that effect.
        e.g. `{effect1: {segment1: 3, segment2: 1}, effect2: {segment3: 5}}`
        where e.g. `segment1 = {location1, location2}`.
        """
        options: list[dict[str, Counter[frozenset[ErrorLocation]]]] = []
        for syndrome, count in syndrome_counter.items():
            effects_to_valid_segments = self._get_effect_to_valid_segments(syndrome, count)
            options.append(effects_to_valid_segments)
        return options


    def _get_effect_to_valid_segments(
        self,
        syndrome: tuple[bool, ...],
        length: int,
    ):
        """Get a map from effect to the number of valid fault combinations with that resultant effect.

        Input:
        * `syndrome` defines the set of faults to take combinations from.
        * `length` the length of valid fault combinations to consider.

        Output:
        * a map from effect to a counter of error location combinations.
        Each error location combination is a set of unique error locations.
        Each count is the number of error event combinations from that error location combination
        with that resultant effect.
        """
        result: defaultdict[str, Counter[frozenset[ErrorLocation]]] = defaultdict(Counter)
        effect_map = self.basis[syndrome]
        effect_combos = itertools.combinations_with_replacement(effect_map.keys(), length)
        for effect_combo in effect_combos:
            # e.g. effect_combo = ('effect1', 'effect1', 'effect2') and is never empty
            counter = Counter(effect_combo)
            # e.g. counter = {'effect1': 2, 'effect2': 1}
            product_effect = forget_sign(math.prod(
                (stim.PauliString(effect) for effect, count in counter.items() if count % 2),
                start=stim.PauliString(self.circuit.noisy_circuit.num_qubits),
            ))
            options = self._effect_counter_to_options(effect_map, counter)
            valid_segments: Counter[frozenset[ErrorLocation]] = Counter()
            for fault_product in itertools.product(*options):
                flattened = itertools.chain(*fault_product)
                self._process_candidate_segment(valid_segments, flattened)
            if valid_segments:
                result[product_effect].update(valid_segments)
        return dict(result)


    @staticmethod
    def _effect_counter_to_options(effect_map: EffectMap, counter: Counter[str]):
        """Get error location combinations for each effect based on the counts in `counter`.

        Input:
        * `effect_map` maps each effect to a map from each error location
        to the number of its faults that cause that syndrome and effect.
        E.g. `{effect1: {location1: 1, location2: 1, source3: 2}, effect2: {source4: 1}}`.
        * `counter` dictates for each effect and source map in `effect_map`,
        how many sources in source map should appear in the fault combination.
        E.g. `{effect1: 2, effect2: 1}`.

        Output:
        * `options` a list of options, one for each item in `counter`
        e.g. `[option1, option2]`.
        Each option is an iterable of source combinations
        e.g. `[((location1, 1), (location2, 1)), ((location1, 1), (source3, 2)), ((location2, 1), (source3, 2))]`.
        Each source combination is a combination of `count` distinct error locations in `effect_map[effect]`.
        """
        options: list[itertools.combinations[tuple[tuple[ErrorLocation, int], ...]]] = []
        for effect, count in counter.items():
            option = itertools.combinations(effect_map[effect].items(), count)
            # without replacement because each item in effect_map[effect]
            # is a (error location, fault count) pair and each option must have distinct error locations
            # e.g. effect_map[effect] = {'location1': 1, 'location2': 1, 'source3': 2}

            # TODO: see if making `option` the following form is faster:
            # `[({location1, location2}, 1), ({location1, source3}, 2), ({location2, source3}, 2)]`
            options.append(option)
        return options


    @staticmethod
    def _process_candidate_segment(
            valid_segments: Counter[frozenset[ErrorLocation]],
            candidate: Iterable[tuple[ErrorLocation, int]],
    ):
        """Update `valid_segments` with a candidate segment of an undetected combination of faults.

        Input:
        * `valid_segments` a counter of valid segments.
        Each (valid) segment is a frozen set of (unique) error locations.
        Each count is the number of fault combinations from that error location combination
        that can be used as part of the undetected combination.
        * `candidate` the segment to consider
        e.g. `[(location1, 1), (location2, 1), (source3, 2)]`.

        Side effect:
        * Update `valid_segments` with `candidate` if it is valid i.e. comprises distinct error locations.
        """
        sources: set[ErrorLocation] = set()
        counts: int = 1
        for source, count in candidate:
            if source in sources:  # check for duplicate sources
                return
            sources.add(source)
            counts *= count
        valid_segments[frozenset(sources)] += counts


    @staticmethod
    def _process_candidate(
            undetected_configurations: defaultdict[str, Counter[int]],
            effect: str,
            candidate: Iterable[tuple[frozenset[ErrorLocation], int]],
    ):
        """Update `undetected_configurations` with a candidate undetected combination of faults.

        Input:
        * `undetected_configurations` the output of `self._get_undetected_configurations_for_length`.
        * `effect` the resultant (unsigned) Pauli string of the candidate combination.
        * `candidate` an iterable of segments (whose total length equals that of the candidate combination).
        Each segment is a pair containing:
            - a frozenset of error locations,
            - the number of fault combinations that can be used as part of the undetected combination.
        E.g. `((frozenset({location1, location2}), 1), (frozenset({source3}), 2))`
        represents a candidate combination
        of faults that has the resultant effect `effect` and is made up of two segments
        where the first segment is made up of two distinct error locations and the second segment is made
        up of one error location.

        Side effect:
        * Update `undetected_configurations` with the undetected combination of faults if it is valid
        i.e. comprises distinct error locations.
        """
        current_sources: set[ErrorLocation] = set()
        current_count: int = 1
        denominators: int = 1
        for sources, count in candidate:
            if not current_sources.isdisjoint(sources):  # check for duplicate sources
                return
            current_sources.update(sources)
            current_count *= count
            denominators *= math.prod(error_event_count(name) for _, name, _ in sources)
        undetected_configurations[effect][denominators] += current_count