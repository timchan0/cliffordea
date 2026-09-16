"""Acceptance probabilities for stabilizer codes suffering Clifford errors."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import stim

from cliffordea.accept.types import PauliMask


LogicalCoefficients = Mapping[int, float]
"""Sparse Pauli-basis coefficients for a logical density operator.

Each key encodes a phaseless logical Pauli on ``k`` logical qubits. The low
``k`` bits encode X support and the high ``k`` bits encode Z support.
"""

PhaseConstraint = tuple[int, int]
"""A physical X-support phase mask followed by its base sign bit."""

LogicalAcceptanceTerm = tuple[float, int, int]
"""A logical coefficient, physical phase mask, and base sign bit."""

_SIGN_TO_J_POWER: dict[complex, Literal[0, 1, 2, 3]] = {
    (1 + 0j): 0,
    (0 + 1j): 1,
    (-1 + 0j): 2,
    (0 - 1j): 3,
}


@dataclass(frozen=True, slots=True)
class AcceptanceStructure:
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


def pauli_masks_and_j_power(
    pauli_string: stim.PauliString,
) -> tuple[int, int, int]:
    """Convert a Pauli string to integer X/Z masks and an i-power.

    :param pauli_string: A signed Pauli string ``P``.
    :return: Integer bitmask for the X component.
    :return: Integer bitmask for the Z component.
    :return: Power ``p`` such that ``P = 1j**p X(x) Z(z)``.
    """
    xs, zs = pauli_string.to_numpy(bit_packed=True)
    x_mask = int.from_bytes(xs.tobytes(), "little")
    z_mask = int.from_bytes(zs.tobytes(), "little")
    j_power = (
        _SIGN_TO_J_POWER[pauli_string.sign]
        + (x_mask & z_mask).bit_count()
    ) % 4
    return x_mask, z_mask, j_power


def trivial_syndrome_probability(
    encoder: stim.Tableau,
    error: stim.Tableau,
    *,
    logical_qubit_count: int,
    logical_coefficients: LogicalCoefficients,
) -> float:
    """Return the trivial-syndrome probability after a Clifford error.

    The encoder maps the first ``n - k`` input axes to stabilizer generators
    and the final ``k`` axes to the chosen logical Pauli frame. Logical Pauli
    masks place their X support in the low ``k`` bits and Z support in the high
    ``k`` bits.

    :param encoder: Encoding Clifford ``C`` for the stabilizer code.
    :param error: Physical Clifford error ``E`` in encoded coordinates.
    :param logical_qubit_count: Number ``k`` of encoded logical qubits.
    :param logical_coefficients: Nonzero coefficients in the logical Pauli
        expansion of the input density operator.
    :return: Probability that every stabilizer measurement is trivial.
    """
    unencoded_error = encoder.inverse() * error * encoder
    return unencoded_trivial_syndrome_probability(
        unencoded_error,
        stabilizer_rank=len(encoder) - logical_qubit_count,
        logical_coefficients=logical_coefficients,
    )


def unencoded_trivial_syndrome_probability(
    unencoded_error: stim.Tableau,
    *,
    stabilizer_rank: int,
    logical_coefficients: LogicalCoefficients,
    checked_stabilizer_count: int | None = None,
) -> float:
    """Return acceptance for an error already conjugated by the encoder.

    :param unencoded_error: Clifford ``C^dag E C`` in unencoded coordinates.
    :param stabilizer_rank: Number of independent stabilizer generators.
    :param logical_coefficients: Nonzero coefficients in the logical Pauli
        expansion of the input density operator.
    :param checked_stabilizer_count: Leading stabilizer generators included in
        the acceptance calculation, or all generators when omitted.
    :return: Probability that every stabilizer measurement is trivial.
    """
    qubit_count = len(unencoded_error)
    logical_qubit_count = qubit_count - stabilizer_rank
    unencoded_logical_mask = (
        ((1 << qubit_count) - 1) ^ ((1 << stabilizer_rank) - 1)
    )
    if checked_stabilizer_count is None:
        checked_stabilizer_count = stabilizer_rank
    columns: list[tuple[int, int, int]] = []
    for stabilizer_index in range(checked_stabilizer_count):
        transformed_generator = unencoded_error.inverse_z_output(
            stabilizer_index,
        )
        column = pauli_masks_and_j_power(transformed_generator)
        x_mask, z_mask, _ = column
        if _rho(
            x_mask,
            z_mask,
            qubit_count=qubit_count,
            unencoded_logical_mask=unencoded_logical_mask,
        ) == 0:
            if _chi(1, (column,)):
                return 0.0
            continue
        columns.append(column)
    structure = build_acceptance_structure(
        columns=columns,
        phase_masks=(0,) * len(columns),
        logical_coefficients=logical_coefficients,
        qubit_count=qubit_count,
        stabilizer_rank=stabilizer_rank,
        logical_qubit_count=logical_qubit_count,
        unencoded_logical_mask=unencoded_logical_mask,
    )
    return evaluate_acceptance_structure(structure, 0)


def build_acceptance_structure(
    *,
    columns: Sequence[tuple[int, int, int]],
    phase_masks: Sequence[PauliMask],
    logical_coefficients: LogicalCoefficients,
    qubit_count: int,
    stabilizer_rank: int,
    logical_qubit_count: int,
    unencoded_logical_mask: int,
) -> AcceptanceStructure:
    """Precompute row and sign data for an acceptance calculation.

    :param columns: Transformed stabilizer generators represented by X/Z masks
        and powers of ``1j``.
    :param phase_masks: Physical X supports whose overlap with a deferred Z
        factor flips each corresponding column sign.
    :param logical_coefficients: Nonzero logical Pauli-basis coefficients.
    :param qubit_count: Number of physical qubits in unencoded coordinates.
    :param stabilizer_rank: Number of independent stabilizer generators.
    :param logical_qubit_count: Number of encoded logical qubits.
    :param unencoded_logical_mask: Mask selecting the final logical axes in
        unencoded coordinates.
    :return: Reusable phase constraints and logical contributions.
    """
    zero_constraints: list[PhaseConstraint] = []
    retained_columns: list[tuple[int, int, int]] = []
    retained_phase_masks: list[PauliMask] = []
    rows: list[int] = []
    for column, phase_mask in zip(columns, phase_masks, strict=True):
        x_mask, z_mask, _ = column
        row = _rho(
            x_mask,
            z_mask,
            qubit_count=qubit_count,
            unencoded_logical_mask=unencoded_logical_mask,
        )
        if row == 0:
            zero_constraints.append((phase_mask, _chi(1, (column,))))
            continue
        retained_columns.append(column)
        retained_phase_masks.append(phase_mask)
        rows.append(row)

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
            source ^= source_basis[pivot]
        if row == 0:
            kernel_basis.append(source)

    for kernel_vector in kernel_basis:
        kernel_constraints.append((
            _xor_selected_masks(kernel_vector, retained_phase_masks),
            _chi(kernel_vector, retained_columns),
        ))

    logical_terms: list[LogicalAcceptanceTerm] = []
    for logical_pauli_mask, coefficient in logical_coefficients.items():
        if coefficient == 0:
            continue
        target = _get_logical_rho(
            logical_pauli_mask,
            qubit_count=qubit_count,
            stabilizer_rank=stabilizer_rank,
            logical_qubit_count=logical_qubit_count,
            unencoded_logical_mask=unencoded_logical_mask,
        )
        solution = _try_solve(row_basis, source_basis, target)
        if solution is not None:
            logical_terms.append((
                coefficient,
                _xor_selected_masks(solution, retained_phase_masks),
                _chi(solution, retained_columns),
            ))

    return AcceptanceStructure(
        zero_constraints=tuple(zero_constraints),
        kernel_constraints=tuple(kernel_constraints),
        logical_terms=tuple(logical_terms),
        rank=len(row_basis),
    )


def evaluate_acceptance_structure(
    structure: AcceptanceStructure,
    effect_z: PauliMask,
) -> float:
    """Evaluate a precomputed acceptance structure for one Z support.

    :param structure: Precomputed structural constraints and logical terms.
    :param effect_z: Deferred physical Z support controlling column signs.
    :return: Probability that every checked stabilizer is trivial.
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


def _try_solve(
    row_basis: Mapping[int, int],
    source_basis: Mapping[int, int],
    target: int,
) -> None | int:
    """Return one source vector whose image equals a target row.

    :param row_basis: Row-echelon image basis keyed by pivot bit.
    :param source_basis: Source vector corresponding to each image basis row.
    :param target: Integer row encoding to solve for.
    :return: Source vector satisfying the equation, or ``None``.
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


def _chi(
    source: int,
    columns: Sequence[tuple[int, int, int]],
) -> int:
    """Evaluate the phase function from the acceptance algorithm.

    :param source: Bitmask selecting transformed-generator columns.
    :param columns: Selected generators represented by X/Z masks and i-powers.
    :return: Sign exponent modulo two.
    """
    phi_sum = 0
    commutation_sum = 0
    x_product = 0
    z_product = 0
    for column_index, (x_mask, z_mask, j_power) in enumerate(columns):
        if not source & (1 << column_index):
            continue
        phi_sum += j_power
        commutation_sum ^= (z_product & x_mask).bit_count() & 1
        x_product ^= x_mask
        z_product ^= z_mask
    y_count = (x_product & z_product).bit_count()
    return (((phi_sum - y_count) // 2) + commutation_sum) & 1


def _rho(
    x_mask: int,
    z_mask: int,
    *,
    qubit_count: int,
    unencoded_logical_mask: int,
) -> int:
    """Pack a Pauli into the row encoding used by the acceptance algorithm.

    :param x_mask: X support in unencoded coordinates.
    :param z_mask: Z support in unencoded coordinates.
    :param qubit_count: Number of unencoded physical qubits.
    :param unencoded_logical_mask: Mask selecting the final logical axes.
    :return: All X support followed by logical-qubit Z support.
    """
    logical_z_support = z_mask & unencoded_logical_mask
    return x_mask | (logical_z_support << qubit_count)


def _get_logical_rho(
    logical_pauli_mask: int,
    *,
    qubit_count: int,
    stabilizer_rank: int,
    logical_qubit_count: int,
    unencoded_logical_mask: int,
) -> int:
    """Construct ``rho(I^r tensor L)`` from a compact logical Pauli mask.

    :param logical_pauli_mask: Logical X support below logical Z support.
    :param qubit_count: Number of unencoded physical qubits.
    :param stabilizer_rank: Number of leading stabilizer ancillas.
    :param logical_qubit_count: Number of encoded logical qubits.
    :param unencoded_logical_mask: Mask selecting the final logical axes.
    :return: Row encoding of the requested logical Pauli.
    """
    logical_mask = (1 << logical_qubit_count) - 1
    logical_x = (
        logical_pauli_mask & logical_mask
    ) << stabilizer_rank
    logical_z = (
        logical_pauli_mask >> logical_qubit_count
    ) << stabilizer_rank
    return _rho(
        logical_x,
        logical_z,
        qubit_count=qubit_count,
        unencoded_logical_mask=unencoded_logical_mask,
    )
