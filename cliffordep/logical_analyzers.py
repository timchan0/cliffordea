import abc
from collections import Counter
from collections.abc import Mapping, Iterable, Sequence
import itertools
from typing import Literal, overload, override

import stim

from cliffordep.pauli_string_tools import PUSH_THROUGH_MAP, PauliSum, tensor_paulis, split_sign
from cliffordep.type_aliases import LogicalTriple


_SIGN_TO_J_POWER: dict[complex, Literal[0, 1, 2, 3]] = {
    (1+0j): 0,
    (0+1j): 1,
    (-1+0j): 2,
    (0-1j): 3,
}


class _TransversalGate:
    """A transversal implementation of physical Z, S, T gates or their inverses.
    
    To conjugate a Pauli string by this transversal gate,
    call the instance on the unsigned Pauli string;
    the output is a `stim.Tableau`.
    """

    _PUSH_THROUGH_S: dict[str, str] = {
        '_': 'I',
        'X': 'Y',
        'Y': 'X',
        'Z': 'Z',
    }

    _PUSH_THROUGH_Z: dict[str, str] = {
        '_': 'I',
        'X': 'X',
        'Y': 'Y',
        'Z': 'Z',
    }

    _PUSH_THROUGH_MAP: dict[str, dict[str, str]] = {
        'T': {
            '_': 'I',
            'X': 'H_XY',
            'Y': 'H_NXY',
            'Z': 'Z',
        },
        'S': _PUSH_THROUGH_S,
        'Z': _PUSH_THROUGH_Z,
        'T_DAG': {
            '_': 'I',
            'X': 'H_NXY',
            'Y': 'H_XY',
            'Z': 'Z',
        },
        'S_DAG': _PUSH_THROUGH_S,
        'Z_DAG': _PUSH_THROUGH_Z,
    }

    def __init__(self, physical_gates: Iterable[str]):
        self.PHYSICAL_GATES = tuple(physical_gates)

    def __call__(self, pauli_string: str) -> stim.Tableau:
        """
        Push an _unsigned_ Pauli string through the transversal gate.
        
        :param pauli_string: The unsigned Pauli string to be conjugated.
        :return tableau: The resulting tableau after conjugation.
        :raises ValueError: If the length of the Pauli string
            does not match the length of the transversal gate.
        """
        return sum((stim.Tableau.from_named_gate(self._PUSH_THROUGH_MAP[physical_gate][pauli])
            for physical_gate, pauli in zip(self.PHYSICAL_GATES, pauli_string, strict=True)),
            start=stim.Tableau(0))
    
    def __repr__(self) -> str:
        return f"_TransversalGate({self.PHYSICAL_GATES})"
    
    def conjugate(self, pauli_string: str):
        """
        Push an _unsigned_ Pauli string through the transversal gate.
        
        :param pauli_string: The unsigned Pauli string to be conjugated.
        :return clifford_string: The resulting Pauli sum after conjugation.
        :raises ValueError: If the length of the Pauli string
            does not match the length of the transversal gate.
        """
        options: list[tuple[str, ...]] = [
            PUSH_THROUGH_MAP[physical_gate][pauli] for physical_gate, pauli
            in zip(self.PHYSICAL_GATES, pauli_string, strict=True)
        ]
        terms: dict[str, complex] = {}
        for pauli_tuple in itertools.product(*options):
            sign, child = split_sign(tensor_paulis(*pauli_tuple))
            terms[child] = sign
        return PauliSum(terms)


class LogicalAnalyzer(abc.ABC):
    """Base class for analyzing logical errors."""

    def __init__(
            self,
            data_indices: tuple[int, ...],
            logical_s: stim.Circuit,
    ) -> None:
        """
        :param data_indices: A tuple of integers representing the data qubit indices in ascending order.
        :param logical_s: A transversal implementation of the logical S gate,
            where indices are in terms of all the physical qubits.
        """
        # TODO: consider lowercasing these attributes since they are not really constants.
        self.DATA_INDICES = data_indices
        """A tuple of integers representing the data qubit indices in ascending order."""
        self.X_TENSOR_N = stim.PauliString('X'*len(data_indices))
        """A Pauli string representing X applied to all data qubits.
        Represents a logical X operator."""
        self.Z_TENSOR_N = stim.PauliString('Z'*len(data_indices))
        """A Pauli string representing Z applied to all data qubits.
        Represents a logical Z operator."""
        
        _majority_indices: set[int] = set()
        for instruction in logical_s:
            if isinstance(instruction, stim.CircuitRepeatBlock):
                raise ValueError("There is a REPEAT block in the circuit.")
            elif instruction.name == 'S':
                _majority_indices.update(target.value for target in instruction.targets_copy())
            elif instruction.name != 'S_DAG':
                raise ValueError(f"Unexpected gate {instruction.name} in transversal S gate.")
        self.LOGICAL: dict[str, _TransversalGate] = {}
        """A map from logical gate name to a transversal implementation that conjugates Paulis to Cliffords."""
        for letter in ('Z', 'S', 'T'):
            self.LOGICAL[letter] = _TransversalGate(
                letter if qubit_index in _majority_indices
                else f'{letter}_DAG' for qubit_index in data_indices
            )
            self.LOGICAL[f'{letter}_DAG'] = _TransversalGate(
                f'{letter}_DAG' if qubit_index in _majority_indices
                else letter for qubit_index in data_indices
            )

    def restrict_to_data(self, unsigned_string: str):
        return ''.join(unsigned_string[index] for index in self.DATA_INDICES)

    @abc.abstractmethod
    def analyze(
            self,
            cultivated_state: Literal['T', 'S', 'Z'],
            before_transversal: str,
    ) -> tuple[float, float]:
        """Compute the acceptance probability and logical fidelity of a logical state affected by error.

        :param cultivated_state: The cultivated state, either 'T', 'S', or 'Z'.
        :param before_transversal: A string representing the Pauli error without sign,
            restricted to the data qubits.
        :return acceptance_probability: The probability the resulting state yields a trivial syndrome
            when all stabilizer generators are noiselessly measured.
        :return logical_fidelity: The fidelity of the resulting state to the target logical state.
        """

    @overload
    def get_kept_strings(
            self,
            configurations: list[dict[str, Counter[int]]],
            cultivated_state: Literal['T', 'S', 'Z'] = 'T',
            print_progress: bool = False,
    ) -> list[dict[str, tuple[float, float, Counter[int]]]]:
        pass
    @overload
    def get_kept_strings(
            self,
            configurations: list[dict[str, set[frozenset[int]]]],
            cultivated_state: Literal['T', 'S', 'Z'] = 'T',
            print_progress: bool = False,
    ) -> list[dict[str, LogicalTriple]]:
        pass
    def get_kept_strings(
            self,
            configurations,
            cultivated_state: Literal['T', 'S', 'Z'] = 'T',
            print_progress: bool = False,
    ):
        """Return all the information needed to reconstruct the logical error probability for any noise level.

        :param configurations: the output of the `get_undetected_configurations()` method
            from `BaseFaultCombinator` or `BaseExclusiveCombinator`.
        :param cultivated_state: the target logical state cultivated.
        :param print_progress: whether to print progress.

        :return kept_strings: A list of maps, one for each degree. Each one maps from each error
            (as an unsigned Pauli string) to a pair containing:
            
            1. the resulting logical vector,
            2. a counter of denominators OR a set of fault configurations
            (each represented by a frozen set of fault indices).
        """
        if print_progress:
            print(f"{cultivated_state} state cultivation:")
        result = [self._get_kept_strings(
                combinations_of_order,
                cultivated_state=cultivated_state,
                order=order if print_progress else None,
            ) for order, combinations_of_order in enumerate(configurations)]
        return result

    @overload
    def _get_kept_strings(
            self,
            combinations_of_order: dict[str, Counter[int]],
            cultivated_state: Literal['T', 'S', 'Z'] = 'T',
            order: None | int = None,
    ) -> dict[str, tuple[float, float, Counter[int]]]:
        pass
    @overload
    def _get_kept_strings(
            self,
            combinations_of_order: dict[str, set[frozenset[int]]],
            cultivated_state: Literal['T', 'S', 'Z'] = 'T',
            order: None | int = None,
    ) -> dict[str, LogicalTriple]:
        pass
    def _get_kept_strings(
            self,
            combinations_of_order: Mapping[str, Counter[int] | set[frozenset[int]]],
            cultivated_state: Literal['T', 'S', 'Z'] = 'T',
            order: None | int = None,
    ) -> Mapping[str, tuple[float, float, Counter[int] | set[frozenset[int]]]]:
        """Compute the logical vector for each postselected error.

        Helper for `get_kept_strings`.

        :param combinations_of_order: A map from each effect to a counter of denominators
            OR a map from each effect to a set of frozen sets of fault indices.
            Each denominator divides (noise level)^order to equal
            the probability an instance of that undetected combination occurs.
        :param cultivated_state: The target logical state cultivated.
        :param order: An optional parameter used only for printing progress.

        :return kept_strings_of_order: A map from each error (as an unsigned Pauli string)
            to a pair containing:
            
            1. the resulting logical vector,
            2. a counter of denominators OR a set of fault configurations
            (each represented by a frozen set of fault indices).
        """
        strings: dict[str, tuple[float, float, Counter[int] | set[frozenset[int]]]] = {}
        for pauli_string, denominators in combinations_of_order.items():
            restricted_pauli_string = self.restrict_to_data(pauli_string)
            accept_probability, logical_fidelity = self.analyze(cultivated_state, restricted_pauli_string)
            if accept_probability:
                strings[restricted_pauli_string] = (accept_probability, logical_fidelity, denominators)
        if order is not None:
            identity_weight = 0
            error_weight = 0
            for accept_probability, logical_fidelity, _ in strings.values():
                identity_weight += accept_probability * logical_fidelity
                error_weight += accept_probability * (1-logical_fidelity)
            print(f'    {order}, {identity_weight} ({error_weight}) errors are kept and lead to identity (error).')
        return strings


def _pauli_masks_and_j_power(pauli_string: stim.PauliString) -> tuple[int, int, int]:
    """Convert a Pauli string to integer X/Z masks and an i-power.

    :param pauli_string: A signed Pauli string P.
    :return x_mask: Integer bitmask for the X component.
    :return z_mask: Integer bitmask for the Z component.
    :return j_power: Power such that P = 1j^j_power [X string] [Z string].
    """
    xs, zs = pauli_string.to_numpy(bit_packed=True)
    x_mask = int.from_bytes(xs.tobytes(), 'little')
    z_mask = int.from_bytes(zs.tobytes(), 'little')
    j_power = (_SIGN_TO_J_POWER[pauli_string.sign] + (x_mask & z_mask).bit_count()) % 4
    return x_mask, z_mask, j_power


def _multiply_pauli_masks(
        left: tuple[int, int, int],
        right: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Multiply Paulis represented by integer masks and i-powers.

    :param left: Tuple ``(x_mask, z_mask, j_power)`` representing the left Pauli.
    :param right: Tuple ``(x_mask, z_mask, j_power)`` representing the right Pauli.
    :return product: Tuple ``(x_mask, z_mask, j_power)`` representing their product.
    """
    left_x_mask, left_z_mask, left_j_power = left
    right_x_mask, right_z_mask, right_j_power = right
    commutation_power = 2 * ((left_z_mask & right_x_mask).bit_count() & 1)
    return (
        left_x_mask ^ right_x_mask,
        left_z_mask ^ right_z_mask,
        (left_j_power + right_j_power + commutation_power) % 4,
    )


class CliffordLogicalAnalyzer(LogicalAnalyzer):
    """Analyzes Clifford errors using unencoded stabilizer overlap."""

    @override
    def __init__(
        self,
        data_indices: tuple[int, ...],
        stabilizer_generators: dict[str, tuple[stim.PauliString, ...]],
        logical_s: stim.Circuit,
        precheck_z_stabilizers: bool = True,
    ):
        """
        :param data_indices: A tuple of integers representing the data qubit indices in ascending order.
        :param stabilizer_generators: The generators of the CSS code stabilizer group
            restricted to data qubits, categorized by basis.
        :param logical_s: A transversal implementation of the logical S gate,
            where indices are in terms of all the physical qubits.
        :param precheck_z_stabilizers: Whether to early reject errors that
            anticommute with a pure Z stabilizer before
            pushing the error through the layer of Z/S/T gates.
            This pre-check is valid because the considered errors
            (anti)commute with a Z stabilizer before the Z/S/T gates iff they
            (anti)commute with the same stabilizer after.
        """
        super().__init__(data_indices, logical_s)
        self.precheck_z_stabilizers = precheck_z_stabilizers
        """Whether to analyze the Z stabilizer generators before the layer of Z/S/T gates."""
        stabilizer_generator_tuple = (
            *stabilizer_generators['X'],
            *stabilizer_generators['Z'],
        )
        self.stabilizer_rank = len(stabilizer_generator_tuple)
        """The number of independent stabilizer generators."""
        self.encoder = self._get_encoding_tableau(
            stabilizers=stabilizer_generator_tuple,
            logical_zs=(self.Z_TENSOR_N,),
            logical_xs=(self.X_TENSOR_N,),
        )
        """The encoding circuit for the stabilizer code."""
        self.unencoder = self.encoder.inverse()
        """The unencoding circuit for the stabilizer code."""
        self.unencoded_stabilizer_generators = {
            basis: tuple(self.unencoder(generator) for generator in generator_list)
            for basis, generator_list in stabilizer_generators.items()
        }
        """The unencoded generators of the CSS code stabilizer group restricted to data qubits, by basis."""
        self.qubit_count = len(data_indices)
        """The number of data qubits."""
        self.unencoded_logical_mask = (
            ((1 << self.qubit_count) - 1) ^ ((1 << self.stabilizer_rank) - 1)
        )
        """Bitmask selecting the unencoded logical qubits."""

    @staticmethod
    def _get_encoding_tableau(
            stabilizers: Sequence[stim.PauliString],
            logical_zs: Sequence[stim.PauliString],
            logical_xs: Sequence[stim.PauliString],
    ) -> stim.Tableau:
        """Construct an encoding tableau from a stabilizer-code Pauli frame.

        Stim can complete the all-zero logical-state stabilizers into an arbitrary
        tableau. This method keeps the resulting destabilizers for the stabilizer
        generators, adjusts them to commute with the chosen logical X operators,
        and then rebuilds a full encoding tableau whose final X and Z outputs are
        the code's logical X and logical Z operators.

        :param stabilizers: Stabilizer generators for the code.
        :param logical_zs: Logical Z operators, one for each logical qubit.
        :param logical_xs: Logical X operators, one for each logical qubit.
        :return tableau: A full encoding tableau for the stabilizer code.
        :raises ValueError: If the logical X and Z operator counts differ.
            Stim may also raise ``ValueError`` if the supplied Pauli frame does
            not satisfy the required commutation relationships.
        """
        logical_zero_generators = [*stabilizers, *logical_zs]
        logical_zero_tableau = stim.Tableau.from_stabilizers(logical_zero_generators)
        destabilizers = [
            logical_zero_tableau.x_output(index)
            for index, _ in enumerate(stabilizers)
        ]
        for index, destabilizer in enumerate(destabilizers):
            adjusted_destabilizer = destabilizer
            for logical_z, logical_x in zip(logical_zs, logical_xs, strict=True):
                if not adjusted_destabilizer.commutes(logical_x):
                    adjusted_destabilizer *= logical_z
            destabilizers[index] = adjusted_destabilizer
        return stim.Tableau.from_conjugated_generators(
            xs=[*destabilizers, *logical_xs],
            zs=logical_zero_generators,
        )

    def analyze(self, cultivated_state, before_transversal):
        before_transversal_pauli = stim.PauliString(before_transversal)
        
        # The logical fidelity of a logical X eigenstate suffering from error `before_transversal`.
        logical_fidelity = self.X_TENSOR_N.commutes(before_transversal_pauli)
        
        if self.precheck_z_stabilizers:
            # Under Z-preserving transversal gates, any pre-existing pure
            # Z-stabilizer syndrome persists through the transversal layer.
            unencoded_before = self.unencoder(before_transversal_pauli)
            for z_generator in self.unencoded_stabilizer_generators['Z']:
                if not z_generator.commutes(unencoded_before):
                    return 0.0, logical_fidelity
        after_transversal = self.LOGICAL[cultivated_state](before_transversal)
        unencoded_error = self.unencoder * after_transversal * self.encoder
        accept_probability = self._get_accept_probability(unencoded_error)
        
        return accept_probability, logical_fidelity

    def _get_accept_probability(self, unencoded_error: stim.Tableau) -> float:
        """Calculate the trivial-syndrome probability from stabilizer overlap.

        :param unencoded_error: The Clifford error after passing through the inverse
            of the code's encoder.
        :return accept_probability: The probability that all noiseless stabilizer
            generator measurements return +1.
        """
        row_basis: dict[int, int] = {}
        pauli_basis: dict[int, tuple[int, int, int]] = {}

        for unencoded_generator in itertools.chain(
                self.unencoded_stabilizer_generators['X'],
                () if self.precheck_z_stabilizers
                else self.unencoded_stabilizer_generators['Z'],
        ):
            transformed_generator = unencoded_error(unencoded_generator)
            if transformed_generator == unencoded_generator:
                continue
            if transformed_generator == -unencoded_generator:
                return 0.0

            row, pauli_product = self._get_row_and_pauli_product(transformed_generator)
            if row == 0:
                if pauli_product[2] == 2:
                    return 0.0
                continue

            while row:
                pivot = row.bit_length() - 1
                if pivot not in row_basis:
                    row_basis[pivot] = row
                    pauli_basis[pivot] = pauli_product
                    break
                row ^= row_basis[pivot]
                pauli_product = _multiply_pauli_masks(pauli_product, pauli_basis[pivot])

            if row == 0 and pauli_product[2] == 2:
                return 0.0

        return 2.0 ** (-len(row_basis))

    def _get_row_and_pauli_product(
            self,
            pauli_string: stim.PauliString,
    ) -> tuple[int, tuple[int, int, int]]:
        """Construct the row used for stabilizer-overlap rank reduction.

        The row records all X support and the Z support on unencoded logical qubits.

        :param pauli_string: A signed Pauli string in unencoded coordinates.
        :return row: Integer bitmask representing support outside the unencoded
            stabilizer-Z subgroup.
        :return pauli_product: Tuple ``(x_mask, z_mask, j_power)`` representing
            ``pauli_string``.
        """
        x_mask, z_mask, j_power = _pauli_masks_and_j_power(pauli_string)
        logical_z_support = z_mask & self.unencoded_logical_mask
        outside_support = x_mask | (logical_z_support << self.qubit_count)
        return outside_support, (x_mask, z_mask, j_power)


class SuperpositionLogicalAnalyzer(LogicalAnalyzer):
    """Analyzes Clifford errors as superpositions of Paulis."""

    def __init__(
            self,
            data_indices: tuple[int, ...],
            stabilizer_generators: dict[str, tuple[stim.PauliString, ...]],
            logical_s: stim.Circuit,
    ) -> None:
        """
        :param data_indices: A tuple of integers representing the data qubit indices in ascending order.
        :param stabilizer_generators: The generators of the stabilizer group restricted to data qubits.
        :param logical_s: A transversal implementation of the logical S gate,
            where indices are in terms of all the physical qubits.
        """
        super().__init__(data_indices, logical_s)
        self.STABILIZER_GENERATORS = tuple(
            g for list_ in stabilizer_generators.values() for g in list_)
        """The generators of the stabilizer group restricted to data qubits."""

    def analyze(self, cultivated_state, before_transversal):
        clifford = self.LOGICAL[cultivated_state].conjugate(before_transversal)
        clifford.postselect_from_stabilizers(self.STABILIZER_GENERATORS)
        logical_vector = clifford.get_logical_amplitudes(self.X_TENSOR_N, self.Z_TENSOR_N)
        logical_vector.transfer_xy_to_iz(logical_state=cultivated_state)
        acceptance_probability = logical_vector.probability_mass
        logical_fidelity = logical_vector.probability_of('I')/acceptance_probability if acceptance_probability else 0.0
        return acceptance_probability, logical_fidelity
