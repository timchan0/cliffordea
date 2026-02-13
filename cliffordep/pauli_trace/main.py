"""
Trace of a product of Pauli projector factors (I + P)/2.

Public API:
    ProjectorProductTracer: Class for computing the trace.
"""
from typing import Sequence, Literal
from fractions import Fraction

import numpy as np

from cliffordep.pauli_trace._character_summers import BruteForceCharacterSummer, PolynomialCharacterSummer

# --------------------------
# Validation & basic helpers
# --------------------------

def _validate_inputs(
    px: Sequence[np.ndarray[tuple[int], np.dtype[np.bool_]]],
    pz: Sequence[np.ndarray[tuple[int], np.dtype[np.bool_]]],
    signs: Sequence[int],
):
    """Validate and convert inputs to numpy arrays (dtype=uint8)."""
    try:
        px_array: np.ndarray[tuple[int, int], np.dtype[np.uint8]] = np.array(px, dtype=np.uint8)
        pz_array: np.ndarray[tuple[int, int], np.dtype[np.uint8]] = np.array(pz, dtype=np.uint8)
    except Exception as exc:
        raise ValueError(
            f"Could not interpret px/pz as m x qubit_count binary arrays: {exc}"
        ) from exc
    signs_array: np.ndarray[tuple[int], np.dtype[np.uint8]] = np.array(signs, dtype=np.uint8)
    return px_array, pz_array, signs_array


# --------------------------
# GF(2) linear algebra
# --------------------------


def _gf2_kernel_basis(
        matrix: np.ndarray[tuple[int, int], np.dtype[np.uint8]]
) -> np.ndarray[tuple[int, int], np.dtype[np.uint8]]:
    """
    Compute a GF(2) basis for the kernel of a matrix.
    
    :param matrix: A p x m matrix.
    :return basis: An m x k matrix B whose columns span the kernel:
        matrix @ x = 0  <=>  x = B @ y for some y in GF(2)^k.
    """
    row_count, col_count = matrix.shape
    reduced = matrix.copy()
    row = 0
    pivot_cols = []
    row_for_col = {}
    for col in range(col_count):
        if row >= row_count:
            break
        sel = None
        for r in range(row, row_count):
            if reduced[r, col] == 1:
                sel = r
                break
        if sel is None:
            continue
        if sel != row:
            reduced[[sel, row]] = reduced[[row, sel]]
        pivot_cols.append(col)
        row_for_col[col] = row
        # eliminate other rows to help back-substitution
        for r in range(row_count):
            if r != row and reduced[r, col] == 1:
                reduced[r, :] ^= reduced[row, :]
        row += 1
    free_cols = [c for c in range(col_count) if c not in pivot_cols]
    basis: list[np.ndarray[tuple[int], np.dtype[np.uint8]]] = []
    for f in free_cols:
        vec = np.zeros(col_count, dtype=np.uint8)
        vec[f] = 1
        # back-substitute pivot variables
        for col in reversed(pivot_cols):
            r = row_for_col[col]
            s = 0
            for j in range(col + 1, col_count):
                if reduced[r, j] == 1 and vec[j] == 1:
                    s ^= 1
            vec[col] = s
        basis.append(vec)
    if len(basis) == 0:
        return np.zeros((col_count, 0), dtype=np.uint8)
    return np.stack(basis, axis=1)


# --------------------------
# Pairing matrix (packed)
# --------------------------

def _pack_bits_to_uint64(bit_array: np.ndarray[tuple[int, int], np.dtype[np.uint8]]):
    """
    Pack binary 2D array (m x qubit_count) into shape (m, word_count) of uint64 words.
    Words pack little-endian bit order within each 64-bit word.
    """
    m, qubit_count = bit_array.shape
    word_count = (qubit_count + 63) // 64
    out = np.zeros((m, word_count), dtype=np.uint64)
    for bit in range(qubit_count):
        word = bit // 64
        offset = bit % 64
        out[:, word] |= (bit_array[:, bit].astype(np.uint64) << np.uint64(offset))
    return out


def _popcount_uint64_word(word: np.uint64):
    """Return popcount of a single uint64 word."""
    return int(word).bit_count()


def _popcount_and_mod2_word(a_word: np.uint64, b_word: np.uint64):
    """Return popcount(a_word & b_word) mod 2."""
    return _popcount_uint64_word(a_word & b_word) & 1


def _get_pairing_matrix_from_packed(
        px_packed: np.ndarray[tuple[int, int], np.dtype[np.uint64]],
        pz_packed: np.ndarray[tuple[int, int], np.dtype[np.uint64]],
):
    """
    Compute pairing matrix from packed px/pz arrays (uint64 arrays).
    
    Implemented using wordwise popcounts.
    
    :param px_packed: m x w array of X components,
        where m is the Pauli count and w is the word count.
    :param pz_packed: m x w array of Z components.
    :return pairing_matrix: m x m matrix P where P_{ij} = z_i·x_j mod 2.
    """
    pauli_count, word_count = px_packed.shape
    pairing_matrix = np.zeros((pauli_count, pauli_count), dtype=np.uint8)
    for i in range(pauli_count):
        zi = pz_packed[i]
        for j in range(pauli_count):
            xj = px_packed[j]
            parity = 0
            for w in range(word_count):
                parity ^= _popcount_and_mod2_word(zi[w], xj[w])
            pairing_matrix[i, j] = parity & 1
    return pairing_matrix


def _get_pairing_matrix(
        px_array: np.ndarray[tuple[int, int], np.dtype[np.uint8]],
        pz_array: np.ndarray[tuple[int, int], np.dtype[np.uint8]],
):
    """
    Compute pairing matrix using numpy matrix products (simple and vectorized).

    For large qubit_count, prefer the packed-word version.
    
    :param px_array: m x n array of X components,
        where m is the Pauli count and n is the qubit count.
    :param pz_array: m x n array of Z components.
    :return pairing_matrix: m x m matrix P where P_{ij} = z_i·x_j mod 2.
    """
    pairing_matrix: np.ndarray[tuple[int, int], np.dtype[np.uint8]] = (
        pz_array @ px_array.T) & 1 # type: ignore
    return pairing_matrix


# --------------------------
# Cross-term accumulation
# --------------------------

def _get_cross_term_accumulation(
        pairing_matrix: np.ndarray[tuple[int, int], np.dtype[np.uint8]],
        kernel_basis: np.ndarray[tuple[int, int], np.dtype[np.uint8]],
) -> np.ndarray[tuple[int, int], np.dtype[np.uint8]]:
    """Compute the cross-term accumulation matrix C.
    
    :param pairing_matrix: m x m pairing matrix P where P_{ij} = z_i·x_j mod 2.
    :param kernel_basis: m x k basis matrix B for the kernel of the Pauli matrix V.
    :return cross_C: The k x k matrix C over GF(2)
        where C_{ab} = sum_{i<j} P_{ij} B_{ia} B_{jb}  (mod 2).
        Equivalently, C = B^T strict_upper_triangle(P) B.
    """
    return (kernel_basis.T @ np.triu(pairing_matrix, k=1) @ kernel_basis) % 2


# --------------------------
# High-level API
# --------------------------

class ProjectorProductTracer:

    def __init__(
        self,
        mode: Literal['deterministic', 'brute'] = 'deterministic',
        use_packed: bool = True,
        assume_nonnegative: bool = False,
    ) -> None:
        """
        :param mode: Mode to count solutions:
            * 'deterministic' uses polynomial-time GF(2) elimination,
            * 'brute' uses exponential-time brute force.
        :param use_packed: When constructing the pairing matrix,
            use 64-bit-word packing for large n (faster).
        :param assume_nonnegative: Whether to assume the trace is nonnegative.
            If True, the quadratic character sum is computed via
            one GF(2) Gaussian elimination on an augmented matrix.
            This can be faster, but will give the wrong answer if the result is actually negative.

            Note: nonnegativity is not automatic for an arbitrary product of projectors when the
            projectors do not commute (the product need not be Hermitian/PSD). This flag is intended
            for use-cases where the returned trace is known to be nonnegative (e.g.
            when all Pauli projectors mutually commute so the product is itself a projector, or
            when the expression is derived from a bona fide probability).
        """
        self.USE_PACKED = use_packed
        self.MODE = mode
        self.ASSUME_NONNEGATIVE = assume_nonnegative
        self.CHARACTER_SUMMER = PolynomialCharacterSummer(assume_nonnegative) \
            if mode == 'deterministic' else BruteForceCharacterSummer()

    def trace(
        self,
        px: Sequence[np.ndarray[tuple[int], np.dtype[np.bool_]]],
        pz: Sequence[np.ndarray[tuple[int], np.dtype[np.bool_]]],
        signs: Sequence[int],
    ):
        """
        Compute exactly T = tr[ prod_i (I + P_i)/2 ] for Paulis P_i given in binary symplectic form.

        Let m be the number of Paulis and n be the number of qubits.
        
        :param px: Length-m sequence of n-bit vectors (X component).
        :param pz: Length-m sequence of n-bit vectors (Z component).
        :param signs: Length-m sequence of bits (0 => +1, 1 => -1).
        :return trace: The trace T.
        """
        px_array, pz_array, signs_array = _validate_inputs(px, pz, signs)
        pauli_count, qubit_count = px_array.shape
        if self.USE_PACKED and qubit_count > 64:
            px_packed = _pack_bits_to_uint64(px_array)
            pz_packed = _pack_bits_to_uint64(pz_array)
            pairing_matrix = _get_pairing_matrix_from_packed(px_packed, pz_packed)
        else:
            pairing_matrix = _get_pairing_matrix(px_array, pz_array)
        # Solve linear constraint: V^T x = 0
        paulis_in_bsf: np.ndarray[tuple[int, int], np.dtype[np.uint8]] = np.concatenate(
            [px_array, pz_array], axis=1).astype(np.uint8)  # m x 2n matrix V
        kernel_basis = _gf2_kernel_basis(paulis_in_bsf.T)  # m x k matrix B
        _, nullity = kernel_basis.shape
        if nullity == 0:
            numerator = 2 ** qubit_count
        else:
            linear: np.ndarray[tuple[int], np.dtype[np.uint8]] = (
                kernel_basis.T @ signs_array) & 1 # type: ignore
            cross_C = _get_cross_term_accumulation(pairing_matrix, kernel_basis)
            character_sum = self.CHARACTER_SUMMER.sum(linear, cross_C)
            numerator = (2 ** qubit_count) * character_sum
        denominator = 2 ** pauli_count
        return Fraction(numerator, denominator)

# --------------------------
# Utility: brute force verifier (small sizes)
# --------------------------

def brute_force_projector_product_trace(
    px: Sequence[np.ndarray[tuple[int], np.dtype[np.bool_]]],
    pz: Sequence[np.ndarray[tuple[int], np.dtype[np.bool_]]],
    signs: Sequence[int],
):
    """
    Brute force expansion over all subsets (for testing small instances).

    :param px: See `ProjectorProductTracer.trace`.
    :param pz: See `ProjectorProductTracer.trace`.
    :param signs: See `ProjectorProductTracer.trace`.
    :param qubit_count: See `ProjectorProductTracer.trace`.

    :return trace: The trace.
    :return size_of_0_set: Number of subsets giving +1 overall sign.
    :return size_of_1_set: Number of subsets giving -1 overall sign.
    """
    px_array, pz_array, signs_array = _validate_inputs(px, pz, signs)
    pauli_count, qubit_count = px_array.shape
    pairing_matrix = _get_pairing_matrix(px_array, pz_array)
    paulis_in_bsf: np.ndarray[tuple[int, int], np.dtype[np.uint8]] = np.concatenate(
        [px_array, pz_array], axis=1).astype(np.uint8)  # m x 2n matrix V
    size_of_0_set = 0
    size_of_1_set = 0
    for mask in range(1 << pauli_count):
        # pauli_selector is a vector x of length m indicating which Paulis are included
        pauli_selector = np.array([
            (mask >> i) & 1 for i in range(pauli_count)], dtype=np.uint8)
        if np.any((paulis_in_bsf.T @ pauli_selector) % 2):
            continue
        external_sign_product = int((signs_array @ pauli_selector) & 1)
        internal_sign_product = 0
        for i in range(pauli_count):
            if not pauli_selector[i]:
                continue
            for j in range(i + 1, pauli_count):
                if pauli_selector[j] and pairing_matrix[i, j]:
                    internal_sign_product ^= 1
        total_sign = (external_sign_product ^ internal_sign_product) & 1
        if total_sign:
            size_of_1_set += 1
        else:
            size_of_0_set += 1
    T = Fraction((2 ** qubit_count) * (size_of_0_set - size_of_1_set), 2 ** pauli_count)
    return T, size_of_0_set, size_of_1_set