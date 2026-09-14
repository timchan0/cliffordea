import abc
from collections import OrderedDict
from collections.abc import Mapping, Iterable, Sequence
from dataclasses import dataclass
import itertools
from typing import Literal, override

import stim

from cliffordea.pauli_string_tools import PUSH_THROUGH_MAP, PauliSum, tensor_paulis, split_sign
from cliffordea.type_aliases import PauliMask, SyndromeMask


LogicalCoefficients = Mapping[int, float]
"""Sparse Pauli-basis coefficients for a logical density operator.

Each key is a nonnegative integer encoding a phaseless logical Pauli on k
logical qubits. The low k bits encode X support and the high k bits encode
Z support. For k=1, the masks are ``0b00`` for I, ``0b01`` for X, ``0b10``
for Z, and ``0b11`` for Y.
"""


_SIGN_TO_J_POWER: dict[complex, Literal[0, 1, 2, 3]] = {
    (1+0j): 0,
    (0+1j): 1,
    (-1+0j): 2,
    (0-1j): 3,
}

_PAULI_VALUE_TO_CHAR = ('_', 'X', 'Z', 'Y')
"""Unsigned Pauli character indexed by packed X/Z bits."""


_I_STATE_LOGICAL_COEFFICIENTS: LogicalCoefficients = {0b00: 1.0, 0b11: 1.0}
_MINUS_STATE_LOGICAL_COEFFICIENTS: LogicalCoefficients = {0b00: 1.0, 0b01: -1.0}


LOGICAL_COEFFICIENTS: dict[str, LogicalCoefficients] = {
    '0': {0b00: 1.0, 0b10: 1.0},
    '1': {0b00: 1.0, 0b10: -1.0},
    '+': {0b00: 1.0, 0b01: 1.0},
    '-': _MINUS_STATE_LOGICAL_COEFFICIENTS,
    'i': _I_STATE_LOGICAL_COEFFICIENTS,
    '-i': {0b00: 1.0, 0b11: -1.0},
    'T': {0b00: 1.0, 0b01: 2**-0.5, 0b11: 2**-0.5},
    'S': _I_STATE_LOGICAL_COEFFICIENTS,
    'Z': _MINUS_STATE_LOGICAL_COEFFICIENTS,
    'maximally_mixed': {0b00: 1.0},
}
"""One-logical-qubit Pauli-basis coefficients by logical state.

These maps contain the nonzero coefficients alpha_L in
``|Psi><Psi| = 2^-k sum_L alpha_L L``. Raw ket labels denote the Pauli
eigenstates stabilized by their corresponding signed Paulis. The key ``S`` is
an alias for ``i``, and ``Z`` is an alias for ``-``.
"""


PhaseConstraint = tuple[int, int]
"""A physical X-support phase mask followed by its base sign bit."""

LogicalAcceptanceTerm = tuple[float, int, int]
"""A logical coefficient, physical phase mask, and base sign bit."""


@dataclass(frozen=True, slots=True)
class _AcceptanceStructure:
    """Store phase-parametrized data for one acceptance calculation.

    :param zero_constraints: Sign constraints from transformed generators with
        zero rows.
    :param kernel_constraints: Sign constraints from dependencies between
        nonzero rows.
    :param logical_terms: Signed logical-coefficient contributions.
    :param rank: Rank controlling the power-of-two acceptance suppression.
    """

    zero_constraints: tuple[PhaseConstraint, ...]
    kernel_constraints: tuple[PhaseConstraint, ...]
    logical_terms: tuple[LogicalAcceptanceTerm, ...]
    rank: int


class _TransversalGate:
    """A transversal implementation of physical Z, S, T gates or their inverses.
    
    To conjugate a Pauli by this transversal gate,
    call the instance on its packed X/Z-support mask;
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

    _DIAGONAL_PHYSICAL_GATES = frozenset({
        'T', 'T_DAG', 'S', 'S_DAG', 'Z', 'Z_DAG',
    })
    """Physical gates for which X/Z error factorization is valid."""

    def __init__(self, physical_gates: Iterable[str]):
        self.PHYSICAL_GATES = tuple(physical_gates)
        self.supports_xz_factorization = all(
            gate in self._DIAGONAL_PHYSICAL_GATES
            for gate in self.PHYSICAL_GATES
        )
        """Whether Z factors commute through every physical gate."""

    def __call__(self, pauli_mask: PauliMask) -> stim.Tableau:
        """Push a packed unsigned Pauli through the transversal gate.

        :param self: The transversal gate implementation.
        :param pauli_mask: The packed Pauli to conjugate.
        :return: The conjugated Clifford tableau.
        """
        qubit_count = len(self.PHYSICAL_GATES)
        return sum((
            stim.Tableau.from_named_gate(self._PUSH_THROUGH_MAP[
                physical_gate
            ][_PAULI_VALUE_TO_CHAR[
                ((pauli_mask >> qubit_index) & 1)
                | (((pauli_mask >> (qubit_count + qubit_index)) & 1) << 1)
            ]])
            for qubit_index, physical_gate in enumerate(self.PHYSICAL_GATES)
        ), start=stim.Tableau(0))
    
    def __repr__(self) -> str:
        return f"_TransversalGate({self.PHYSICAL_GATES})"
    
    def conjugate(self, pauli_mask: PauliMask):
        """Expand a packed Pauli conjugated by the transversal gate.

        :param self: The transversal gate implementation.
        :param pauli_mask: The packed Pauli to conjugate.
        :return: The resulting Pauli superposition.
        """
        qubit_count = len(self.PHYSICAL_GATES)
        options: list[tuple[str, ...]] = [
            PUSH_THROUGH_MAP[physical_gate][_PAULI_VALUE_TO_CHAR[
                ((pauli_mask >> qubit_index) & 1)
                | (((pauli_mask >> (qubit_count + qubit_index)) & 1) << 1)
            ]]
            for qubit_index, physical_gate in enumerate(self.PHYSICAL_GATES)
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

    def linear_precheck_syndrome(
            self,
            before_transversal: PauliMask,
    ) -> SyndromeMask:
        """Return optional linear checks required for postselection.

        :param self: The logical analyzer supplying the checks.
        :param before_transversal: The packed data-qubit Pauli effect.
        :return: The packed check outcomes, or zero when none are supplied.
        """
        return 0

    @abc.abstractmethod
    def analyze(
            self,
            cultivated_state: Literal['T', 'S', 'Z'],
            before_transversal: PauliMask,
    ) -> tuple[float, float]:
        """Compute acceptance and fidelity for a packed Pauli effect.

        :param self: The logical analyzer evaluating the effect.
        :param cultivated_state: The cultivated state, either 'T', 'S', or 'Z'.
        :param before_transversal: The packed data-qubit Pauli effect.
        :return acceptance_probability: The probability the resulting state yields a trivial syndrome
            when all stabilizer generators are noiselessly measured.
        :return logical_fidelity: The fidelity of the resulting state to the target logical state.
        """

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
        factor_transversal_errors: bool = True,
        transversal_structure_cache_maxsize: int | None = 4_096,
    ):
        """
        :param self: The Clifford logical analyzer being initialized.
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
        :param factor_transversal_errors: Whether to reuse X-dependent
            acceptance structures and apply Z-dependent phases separately for
            supported diagonal transversal gates.
        :param transversal_structure_cache_maxsize: Maximum number of
            X-dependent acceptance structures to retain. Use a positive
            integer for a bounded cache, ``None`` for an unbounded cache, or
            ``0`` to disable retention without disabling factorization.
        :return: None.
        """
        super().__init__(data_indices, logical_s)
        self.precheck_z_stabilizers = precheck_z_stabilizers
        """Whether to analyze the Z stabilizer generators before the layer of Z/S/T gates."""
        self.factor_transversal_errors = factor_transversal_errors
        """Whether supported transversal errors use X/Z factorization."""
        self.transversal_structure_cache_maxsize = (
            transversal_structure_cache_maxsize
        )
        """Maximum retained X-dependent acceptance structures."""
        self._transversal_structure_cache: OrderedDict[
            tuple[str, PauliMask, bool],
            _AcceptanceStructure,
        ] = OrderedDict()
        """Least-recently-used X-dependent acceptance structures."""
        self.x_stabilizer_rank = len(stabilizer_generators['X'])
        """The number of independent X-basis stabilizer generators."""
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
        self.qubit_count = len(data_indices)
        """The number of data qubits."""
        self.support_mask = (1 << self.qubit_count) - 1
        """Bitmask selecting one support half of a packed data Pauli."""
        self.z_precheck_generators = tuple(
            _pauli_masks_and_j_power(generator)[:2]
            for generator in stabilizer_generators['Z']
        )
        """X/Z masks of stabilizers used by the optional linear precheck."""
        self.logical_qubit_count = self.qubit_count - self.stabilizer_rank
        """The number of encoded logical qubits."""
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

    def linear_precheck_syndrome(
            self,
            before_transversal: PauliMask,
    ) -> SyndromeMask:
        """Return commutation outcomes for optional Z-stabilizer checks.

        :param self: The Clifford logical analyzer supplying the checks.
        :param before_transversal: The packed data-qubit Pauli effect.
        :return: The packed Z-stabilizer syndrome when enabled, otherwise zero.
        """
        if not self.precheck_z_stabilizers:
            return 0
        effect_x = before_transversal & self.support_mask
        effect_z = (before_transversal >> self.qubit_count) & self.support_mask
        return sum(
            (
                (
                    (effect_x & stabilizer_z).bit_count()
                    + (effect_z & stabilizer_x).bit_count()
                ) & 1
            ) << stabilizer_index
            for stabilizer_index, (stabilizer_x, stabilizer_z)
            in enumerate(self.z_precheck_generators)
        )

    def analyze(self, cultivated_state, before_transversal):
        effect_z = (before_transversal >> self.qubit_count) & self.support_mask
        logical_fidelity = not (effect_z.bit_count() & 1)
        if self.linear_precheck_syndrome(before_transversal):
            return 0.0, logical_fidelity
        transversal_gate = self.LOGICAL[cultivated_state]
        if (
                self.factor_transversal_errors
                and transversal_gate.supports_xz_factorization):
            effect_x = before_transversal & self.support_mask
            structure = self._get_transversal_structure(
                cultivated_state,
                effect_x,
            )
            accept_probability = self._evaluate_acceptance_structure(
                structure,
                effect_z,
            )
            return accept_probability, logical_fidelity

        after_transversal = transversal_gate(before_transversal)
        unencoded_error = self.unencoder * after_transversal * self.encoder
        accept_probability = self._get_accept_probability(
            unencoded_error,
            LOGICAL_COEFFICIENTS[cultivated_state],
        )
        
        return accept_probability, logical_fidelity

    def _get_transversal_structure(
            self,
            cultivated_state: Literal['T', 'S', 'Z'],
            effect_x: PauliMask,
    ) -> _AcceptanceStructure:
        """Return a cached X-dependent transversal acceptance structure.

        :param self: The analyzer supplying the code and transversal gate.
        :param cultivated_state: The logical state whose transversal gate and
            logical coefficients are used.
        :param effect_x: X support of the Pauli before the transversal layer.
        :return: The reusable acceptance structure for the X support.
        """
        maxsize = self.transversal_structure_cache_maxsize
        if maxsize == 0:
            return self._build_transversal_structure(
                cultivated_state,
                effect_x,
            )

        key = cultivated_state, effect_x, self.precheck_z_stabilizers
        try:
            structure = self._transversal_structure_cache.pop(key)
        except KeyError:
            structure = self._build_transversal_structure(
                cultivated_state,
                effect_x,
            )
        self._transversal_structure_cache[key] = structure
        if (
                maxsize is not None
                and len(self._transversal_structure_cache) > maxsize):
            self._transversal_structure_cache.popitem(last=False)
        return structure

    def _build_transversal_structure(
            self,
            cultivated_state: Literal['T', 'S', 'Z'],
            effect_x: PauliMask,
    ) -> _AcceptanceStructure:
        """Build the acceptance structure for one transversal X support.

        :param self: The analyzer supplying the code and transversal gate.
        :param cultivated_state: The logical state whose transversal gate and
            logical coefficients are used.
        :param effect_x: X support of the Pauli before the transversal layer.
        :return: The phase-parametrized acceptance structure.
        """
        base_error = self.LOGICAL[cultivated_state](effect_x)
        unencoded_base_error = self.unencoder * base_error * self.encoder
        checked_stabilizer_count = (
            self.x_stabilizer_rank if self.precheck_z_stabilizers
            else self.stabilizer_rank
        )
        columns: list[tuple[int, int, int]] = []
        phase_masks: list[int] = []
        for stabilizer_index in range(checked_stabilizer_count):
            transformed_generator = unencoded_base_error.inverse_z_output(
                stabilizer_index,
            )
            columns.append(_pauli_masks_and_j_power(transformed_generator))
            physical_generator = self.encoder(transformed_generator)
            physical_x_mask, _, _ = _pauli_masks_and_j_power(
                physical_generator,
            )
            phase_masks.append(physical_x_mask)
        return self._build_acceptance_structure(
            columns=columns,
            phase_masks=phase_masks,
            logical_coefficients=LOGICAL_COEFFICIENTS[cultivated_state],
        )

    def _get_accept_probability(
            self,
            unencoded_error: stim.Tableau,
            logical_coefficients: LogicalCoefficients,
    ) -> float:
        """Calculate the trivial-syndrome probability from stabilizer overlap.

        This constructs columns from ``F^dag Z_i F``,
        omits zero columns after checking their phase,
        row-reduces the remaining matrix ``S``,
        and sums over all logical Pauli coefficients whose fibres intersect ``im(S)``.

        :param unencoded_error: The Clifford error after passing through the inverse
            of the code's encoder.
        :param logical_coefficients: The nonzero Pauli-basis coefficients of the
            logical density operator, keyed by compact logical Pauli mask.
        :return accept_probability: The probability that all noiseless stabilizer
            generator measurements return +1.
        """
        checked_stabilizer_count = (
            self.x_stabilizer_rank if self.precheck_z_stabilizers
            else self.stabilizer_rank
        )
        columns: list[tuple[int, int, int]] = []
        for stabilizer_index in range(checked_stabilizer_count):
            transformed_generator = unencoded_error.inverse_z_output(stabilizer_index)
            column = _pauli_masks_and_j_power(transformed_generator)
            x_mask, z_mask, _ = column
            if self._rho(x_mask, z_mask) == 0:
                if self._chi(1, (column,)):
                    return 0.0
                continue
            columns.append(column)
        structure = self._build_acceptance_structure(
            columns=columns,
            phase_masks=(0,) * len(columns),
            logical_coefficients=logical_coefficients,
        )
        return self._evaluate_acceptance_structure(structure, 0)

    def _build_acceptance_structure(
            self,
            *,
            columns: Sequence[tuple[int, int, int]],
            phase_masks: Sequence[PauliMask],
            logical_coefficients: LogicalCoefficients,
    ) -> _AcceptanceStructure:
        """Precompute row and sign data for an acceptance calculation.

        :param self: The analyzer defining the unencoded code coordinates.
        :param columns: Transformed stabilizer generators represented by X/Z
            masks and powers of ``1j``.
        :param phase_masks: Physical X supports whose overlap with a deferred Z
            factor flips each corresponding column sign.
        :param logical_coefficients: Nonzero Pauli-basis coefficients of the
            logical input state.
        :return: The phase-parametrized acceptance structure.
        """
        zero_constraints: list[PhaseConstraint] = []
        retained_columns: list[tuple[int, int, int]] = []
        retained_phase_masks: list[PauliMask] = []
        rows: list[int] = []
        for column, phase_mask in zip(columns, phase_masks, strict=True):
            x_mask, z_mask, _ = column
            row = self._rho(x_mask, z_mask)
            if row == 0:
                zero_constraints.append((
                    phase_mask,
                    self._chi(1, (column,)),
                ))
                continue
            retained_columns.append(column)
            retained_phase_masks.append(phase_mask)
            rows.append(row)

        # Row-reduce S over GF(2). row_basis[p] is an image vector whose
        # leading bit is p. source_basis[p] is the corresponding source vector:
        # a bitmask over retained columns whose image is row_basis[p].
        # If a new column reduces to zero, source records a kernel relation.
        # We keep source vectors, rather than Pauli products, so we can later
        # evaluate chi(source) and solve S u = rho(I^r tensor L).
        row_basis: dict[int, int] = {}
        source_basis: dict[int, int] = {}
        kernel_basis: list[int] = []
        kernel_constraints: list[PhaseConstraint] = []
        for column_index, column in enumerate(rows):
            row = column
            source = 1 << column_index
            while row:
                pivot = row.bit_length() - 1
                if pivot not in row_basis:
                    row_basis[pivot] = row
                    source_basis[pivot] = source
                    break
                row ^= row_basis[pivot]
                # Keep the source combination in sync with the row operation.
                source ^= source_basis[pivot]
            if row == 0:
                kernel_basis.append(source)

        for kernel_vector in kernel_basis:
            phase_mask = self._xor_selected_masks(
                kernel_vector,
                retained_phase_masks,
            )
            kernel_constraints.append((
                phase_mask,
                self._chi(kernel_vector, retained_columns),
            ))

        logical_terms: list[LogicalAcceptanceTerm] = []
        for logical_pauli_mask, coefficient in logical_coefficients.items():
            if coefficient == 0:
                continue
            target = self._get_logical_rho(logical_pauli_mask)
            solution = self._try_solve(row_basis, source_basis, target)
            if solution is not None:
                phase_mask = self._xor_selected_masks(
                    solution,
                    retained_phase_masks,
                )
                logical_terms.append((
                    coefficient,
                    phase_mask,
                    self._chi(solution, retained_columns),
                ))

        return _AcceptanceStructure(
            zero_constraints=tuple(zero_constraints),
            kernel_constraints=tuple(kernel_constraints),
            logical_terms=tuple(logical_terms),
            rank=len(row_basis),
        )

    @staticmethod
    def _xor_selected_masks(
            source: int,
            masks: Sequence[PauliMask],
    ) -> PauliMask:
        """Combine masks selected by a source-vector bitmask.

        :param source: Bitmask selecting entries from ``masks``.
        :param masks: Masks in source-vector bit order.
        :return: XOR of the selected masks.
        """
        combined_mask = 0
        for mask_index, mask in enumerate(masks):
            if source & (1 << mask_index):
                combined_mask ^= mask
        return combined_mask

    @staticmethod
    def _evaluate_acceptance_structure(
            structure: _AcceptanceStructure,
            effect_z: PauliMask,
    ) -> float:
        """Evaluate a precomputed acceptance structure for one Z support.

        :param structure: Precomputed structural constraints and logical terms.
        :param effect_z: Deferred physical Z support controlling column signs.
        :return: Probability that every checked stabilizer yields the trivial
            outcome.
        """
        for phase_mask, base_sign in structure.zero_constraints:
            phase = (effect_z & phase_mask).bit_count() & 1
            if base_sign ^ phase:
                return 0.0
        for phase_mask, base_sign in structure.kernel_constraints:
            phase = (effect_z & phase_mask).bit_count() & 1
            if base_sign ^ phase:
                return 0.0

        eta = 0.0
        for coefficient, phase_mask, base_sign in structure.logical_terms:
            phase = (effect_z & phase_mask).bit_count() & 1
            eta += (-1 if base_sign ^ phase else 1) * coefficient

        return 2.0 ** (-structure.rank) * eta

    @staticmethod
    def _try_solve(
            row_basis: Mapping[int, int],
            source_basis: Mapping[int, int],
            target: int,
    ) -> None | int:
        """Return one source vector whose image equals a target row.

        :param row_basis: A row-echelon basis for the image of the reduced
            matrix ``S``, keyed by pivot bit.
        :param source_basis: A map with the same pivots as ``row_basis``.
            Each value is a bitmask over columns of ``S`` whose image is the
            corresponding row-basis vector.
        :param target: The integer row encoding to solve for.
        :return source: A bitmask over columns of ``S`` satisfying
            ``S source = target``, or ``None`` if ``target`` is outside the
            image of ``S``.
        """
        solution = 0
        row = target
        while row:
            pivot = row.bit_length() - 1
            if pivot not in row_basis:
                return None
            row ^= row_basis[pivot]
            solution ^= source_basis[pivot]
        return solution

    @staticmethod
    def _chi(
            source: int,
            columns: Sequence[tuple[int, int, int]],
    ) -> int:
        """Evaluate the phase function from the trivial-syndrome algorithm.

        :param source: A bitmask selecting columns from ``columns``, and
            therefore selecting a product of retained columns
            ``F^dag Z_i F = i^phi_i X(x_i) Z(z_i)``.
        :param columns: The retained nonzero columns of the reduced matrix ``S``.
            Each entry is ``(x_mask, z_mask, j_power)`` for one transformed
            generator.
        :return chi: The sign exponent modulo 2, such that the selected product
            contributes sign ``(-1)^chi`` to the trivial-syndrome trace.
        """
        phi_sum = 0
        commutation_sum = 0
        x_product = 0
        z_product = 0
        for column_index, (x_mask, z_mask, j_power) in enumerate(columns):
            if not source & (1 << column_index):
                continue
            phi_sum += j_power
            # If the current product is X(a)Z(b) and this column is X(c)Z(d),
            # rewriting X(a)Z(b)X(c)Z(d) as X(a+c)Z(b+d) moves X(c) past Z(b).
            # Each overlapping qubit contributes ZX = -XZ, so this adds b dot c.
            commutation_sum ^= (z_product & x_mask).bit_count() & 1
            x_product ^= x_mask
            z_product ^= z_mask
        y_count = (x_product & z_product).bit_count()
        return (((phi_sum - y_count) // 2) + commutation_sum) & 1

    def _rho(self, x_mask: int, z_mask: int) -> int:
        """Pack Pauli masks into the row encoding used by ``rho``.

        The row keeps all X support and only the Z support on unencoded logical
        qubits.

        :param x_mask: Bitmask encoding X support in unencoded coordinates.
        :param z_mask: Bitmask encoding Z support in unencoded coordinates.
        :return row: Integer bitmask with X support in the low ``n`` bits and
            logical-qubit Z support in the next ``n`` bits.
        """
        logical_z_support = z_mask & self.unencoded_logical_mask
        return x_mask | (logical_z_support << self.qubit_count)

    def _get_logical_rho(self, logical_pauli_mask: int) -> int:
        """Construct ``rho(I^r tensor L)`` from a compact logical Pauli mask.

        :param logical_pauli_mask: Integer encoding a phaseless logical Pauli.
            The low k bits encode logical X support and the high k bits encode
            logical Z support.
        :return row: The integer row encoding of
            ``rho(I^r tensor L)``, with all X support followed by the logical
            Z support.
        """
        logical_mask = (1 << self.logical_qubit_count) - 1
        logical_x = (logical_pauli_mask & logical_mask) << self.stabilizer_rank
        logical_z = (logical_pauli_mask >> self.logical_qubit_count) << self.stabilizer_rank
        return self._rho(logical_x, logical_z)


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
