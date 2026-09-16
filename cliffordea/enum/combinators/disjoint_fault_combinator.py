import itertools
from collections import Counter, defaultdict
from collections.abc import Iterable
import math

import stim

from cliffordea.enum.cultivation_circuit import CultivationCircuit
from cliffordea.accept.pauli import forget_sign
from cliffordea.enum.types import EffectMap, ErrorLocation
from cliffordea.enum.combinators._base import (
    BaseDisjointCombinator,
    error_event_count,
    get_zero_signature_combinations,
)


class DisjointFaultCombinator(BaseDisjointCombinator):
    """Group disjoint error events by signature, effect, and location.

    Extends `BaseDisjointCombinator`.
    Finds undetected fault combinations first by iterating through all
    detector-signature combinations and recording which ones XOR to zero.
    Each signature corresponds to a set of error locations,
    so for each zero-signature combination,
    this combinator finds all error location combinations
    that correspond to this signature combination.
    """

    def __init__(self, noisy_circuit, print_progress=False):
        _circuit = CultivationCircuit(noisy_circuit=noisy_circuit)
        _basis: defaultdict[
            tuple[bool, ...], defaultdict[str, Counter[ErrorLocation]]
        ] = defaultdict(lambda: defaultdict(Counter))
        for error_location, group in _circuit.group_error_events_by_location().items():
            for error_event in group:
                signature, effect = _circuit.get_signature_and_effect(error_event)
                tuple_signature = tuple(signature)
                _basis[tuple_signature][effect][error_location] += 1
        self.basis = {signature: dict(effect_map) for signature, effect_map in _basis.items()}
        """Map signatures and effects to their error-event realizations.
        """
        if print_progress:
            print(f"Finished enumerating all faults. Found {self.signature_count} distinct signatures.")
        self.circuit = _circuit
        """The noisy circuit to analyze."""

    def __repr__(self):
        return f"{self.__class__.__name__}({self.circuit})"

    @property
    def signature_count(self) -> int:
        return len(self.basis)

    def keys(self): return self.basis.keys()

    def values(self): return self.basis.values()

    def items(self): return self.basis.items()


    def _get_undetected_configurations_for_fault_count(
            self,
            fault_count: int,
            print_progress: bool = False,
    ) -> dict[str, Counter[int]]:
        result: defaultdict[str, Counter[int]] = defaultdict(Counter)
        if print_progress:
            print(f"Finding undetected {fault_count}-fault configurations...")
        if fault_count == 0:
            result['_'*self.circuit.noisy_circuit.num_qubits][1] += 1
        else:
            zero_signature_combinations = get_zero_signature_combinations(
                self.keys(), fault_count)
            for signature_counter in zero_signature_combinations:
                options = self._signature_counter_to_options(signature_counter)
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
            print(f"Done. They have {len(result)} distinct resultant effects.")
        return dict(result)


    def _signature_counter_to_options(self, signature_counter: Counter[tuple[bool, ...]]):
        """Get valid fault segments for each detector signature.

        :param signature_counter: A counter dictating for each signature
            how many error locations in `self.basis[signature]` should appear
            in the fault configuration. E.g. `{signature1: 2, signature2: 1}`.

        :return options:
            A list of options, one for each item in `signature_counter`
            e.g. `[option1, option2]`.
            Each option is a map from effect to a counter of valid segments with that effect.
            e.g. `{effect1: {segment1: 3, segment2: 1}, effect2: {segment3: 5}}`
            where e.g. `segment1 = {location1, location2}`.
        """
        options: list[dict[str, Counter[frozenset[ErrorLocation]]]] = []
        for signature, count in signature_counter.items():
            effects_to_valid_segments = self._get_effect_to_valid_segments(signature, count)
            options.append(effects_to_valid_segments)
        return options


    def _get_effect_to_valid_segments(
        self,
        signature: tuple[bool, ...],
        fault_count: int,
    ):
        """Get a map from effect to the number of valid fault combinations with that resultant effect.

        :param signature: Defines the set of faults to take combinations from.
        :param fault_count: The number of faults in each combination.

        :return:
            A map from effect to a counter of error location combinations.
            Each error location combination is a set of unique error locations.
            Each count is the number of error event combinations from that error location combination
            with that resultant effect.
        """
        result: defaultdict[str, Counter[frozenset[ErrorLocation]]] = defaultdict(Counter)
        effect_map = self.basis[signature]
        effect_combos = itertools.combinations_with_replacement(
            effect_map.keys(),
            fault_count,
        )
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

        :param effect_map: A map from each effect to a map from each error location
            to the number of disjoint error events that realize the fault.
            E.g. `{effect1: {location1: 1, location2: 1, location3: 2},
            effect2: {location4: 1}}`.
        :param counter: A counter dictating for each effect and location map in
            `effect_map`,
            how many error locations in the location map should appear in the
            fault configuration.
            E.g. `{effect1: 2, effect2: 1}`.

        :return options:
            A list of options, one for each item in `counter`
            e.g. `[option1, option2]`.
            Each option is an iterable of error-location combinations, where
            each integer is an error-event realization count.
        """
        options: list[itertools.combinations[tuple[tuple[ErrorLocation, int], ...]]] = []
        for effect, count in counter.items():
            option = itertools.combinations(effect_map[effect].items(), count)
            # without replacement because each item in effect_map[effect]
            # is an (error location, realization count) pair and each option
            # must have distinct error locations.

            # TODO: see if making `option` the following form is faster:
            # `[({location1, location2}, 1), ({location1, location3}, 2)]`
            options.append(option)
        return options


    @staticmethod
    def _process_candidate_segment(
            valid_segments: Counter[frozenset[ErrorLocation]],
            candidate: Iterable[tuple[ErrorLocation, int]],
    ):
        """Update `valid_segments` with a candidate segment of an undetected combination of faults.

        :param valid_segments: A counter of valid segments.
            Each (valid) segment is a frozen set of (unique) error locations.
            Each count is the number of fault combinations from that error location combination
            that can be used as part of the undetected combination.
        :param candidate: The segment to consider
            e.g. `[(location1, 1), (location2, 1), (location3, 2)]`.

        Side effect:
        * Update `valid_segments` with `candidate` if it is valid i.e. comprises distinct error locations.
        """
        error_locations: set[ErrorLocation] = set()
        realization_count = 1
        for error_location, count in candidate:
            if error_location in error_locations:
                return
            error_locations.add(error_location)
            realization_count *= count
        valid_segments[frozenset(error_locations)] += realization_count


    @staticmethod
    def _process_candidate(
            undetected_configurations: defaultdict[str, Counter[int]],
            effect: str,
            candidate: Iterable[tuple[frozenset[ErrorLocation], int]],
    ):
        """Update `undetected_configurations` with a candidate undetected combination of faults.

        :param undetected_configurations: The output of
            `self._get_undetected_configurations_for_fault_count`.
        :param effect: The resultant (unsigned) Pauli string of the candidate combination.
        :param candidate: Segments whose total fault count equals that of the
            candidate configuration.
            Each segment is a pair containing:

                * a frozenset of error locations,
                * the number of fault combinations that can be used as part of the undetected combination.

            E.g. `((frozenset({location1, location2}), 1), (frozenset({location3}), 2))`
            represents a candidate combination
            of faults that has the resultant effect `effect` and is made up of two segments
            where the first segment is made up of two distinct error locations and the second segment is made
            up of one error location.

        Side effect:
        * Update `undetected_configurations` with the undetected combination of faults if it is valid
        i.e. comprises distinct error locations.
        """
        current_locations: set[ErrorLocation] = set()
        realization_count = 1
        probability_denominator = 1
        for error_locations, count in candidate:
            if not current_locations.isdisjoint(error_locations):
                return
            current_locations.update(error_locations)
            realization_count *= count
            probability_denominator *= math.prod(
                error_event_count(name)
                for _, name, _ in error_locations
            )
        undetected_configurations[effect][probability_denominator] += (
            realization_count
        )
