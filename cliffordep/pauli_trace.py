"""
Trace of a product of Pauli projector factors (I + P)/2.

Public API:
    ProjectorProductTracer: Class for computing the trace.
"""
import abc
import itertools
from typing import Sequence, Literal
from fractions import Fraction

import numpy as np

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


def _get_a_symmetric(cross_C: np.ndarray[tuple[int, int], np.dtype[np.uint8]]):
    """Get the hollow symmetric k x k matrix A = C + C^T from cross-term accumulation matrix C."""
    out: np.ndarray[tuple[int, int], np.dtype[np.uint8]] = cross_C ^ cross_C.T
    return out


# --------------------------
# Congruence reduction & deterministic counting
# --------------------------

class _CharacterSummer(abc.ABC):
    
    @abc.abstractmethod
    def sum(
        self,
        vector: np.ndarray[tuple[int], np.dtype[np.uint8]],
        matrix: np.ndarray[tuple[int, int], np.dtype[np.uint8]],
    ) -> int:
        """Compute a quadratic character sum over GF(2).

        Does not modify the input arrays.

        :param vector: Vector b in GF(2)^k defining the linear part.
        :param matrix: Matrix M in GF(2)^{k x k} defining the quadratic part.
        :return character_sum: The quantity
            S = sum_{y in GF(2)^k} (-1)^{ y^T M y + b^T y },
            which is an integer in {0, ±2^t}_{t = 0}^k.
        """


class BruteForceCharacterSummer(_CharacterSummer):

    def sum(self, vector, matrix) -> int:
        nullity, = vector.shape
        root_count = self._count_roots_quadratic(constant=0, vector=vector, matrix=matrix)
        return 2*root_count - 2**nullity
    
    @staticmethod
    def _count_roots_quadratic(
        constant: Literal[0, 1],
        vector: np.ndarray[tuple[int], np.dtype[np.uint8]],
        matrix: np.ndarray[tuple[int, int], np.dtype[np.uint8]],
    ):
        """
        Brute-force count roots y in GF(2)^k of a quadratic Boolean function.

        Does not modify the input arrays.
        
        :param constant: Scalar in GF(2) defining the constant term.
        :param vector: Vector in GF(2)^k defining the linear part.
        :param matrix: Matrix in GF(2)^{k x k} defining the quadratic part.

        :return root_count: Number of y in GF(2)^k satisfying
            y^T matrix y + vector^T y + constant = 0 (mod 2).
        """
        k, _ = matrix.shape
        root_count = 0
        for bits in itertools.product((0, 1), repeat=k):
            y = np.array(bits, dtype=np.uint8)
            function_val = constant
            function_val ^= int(vector @ y) & 1
            function_val ^= int(y.T @ matrix @ y) & 1
            root_count += (function_val ^ 1)
        return root_count
    

class PolynomialCharacterSummer(_CharacterSummer):

    def __init__(self, assume_nonnegative: bool) -> None:
        self.ASSUMER = Nonnegative if assume_nonnegative else NoAssumptions

    def sum(self, vector, matrix) -> int:
        # Over GF(2), y_i^2 = y_i, so any diagonal terms in matrix can be moved into the linear term.
        # This makes the quadratic part alternating (zero diagonal).
        hollow_symmetric = _get_a_symmetric(matrix)
        linear: np.ndarray[tuple[int], np.dtype[np.uint8]] = vector ^ np.diagonal(matrix)
        s_qb = self.ASSUMER.sum(linear, hollow_symmetric)
        return s_qb
    

class _Assummer(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def sum(
        linear: np.ndarray[tuple[int], np.dtype[np.uint8]],
        hollow_symmetric: np.ndarray[tuple[int, int], np.dtype[np.uint8]],
    ) -> int:
        """Compute a quadratic character sum over GF(2).

        Does not modify the input arrays.

        :param linear: Vector in GF(2)^k defining the linear part.
        :param hollow_symmetric: Hollow symmetric matrix A [i.e. A = A^T and diag(A) = 0]
            in GF(2)^{k x k} defining the quadratic part .
        :return character_sum: Specifically,
            S = sum_{y in GF(2)^k} (-1)^{ y^T A y + linear^T y }.
            For alternating A, the sum is an integer in {0, ±2^t}:
                if A x = linear is inconsistent, S = 0;
                else, S = ±2^{k - rank(A)/2} where the sign is determined by the Arf invariant.
        """


class NoAssumptions(_Assummer):

    @staticmethod
    def sum(linear, hollow_symmetric):
        a = hollow_symmetric.copy()
        b = linear.copy()
        k, _ = a.shape
        if k == 0:
            return 1

        # Eliminate into direct sum of 2x2 blocks [[0,1],[1,0]] and free variables.
        pairs = 0
        arf = 0
        n = k
        while True:
            pivot_i = None
            pivot_j = None
            for i in range(n):
                for j in range(i + 1, n):
                    if a[i, j] == 1:
                        pivot_i = i
                        pivot_j = j
                        break
                if pivot_i is not None:
                    break
            if pivot_i is None:
                break

            # Move pivot pair to positions (0, 1).
            if pivot_i != 0:
                a[[0, pivot_i], :] = a[[pivot_i, 0], :]
                a[:, [0, pivot_i]] = a[:, [pivot_i, 0]]
                b[0], b[pivot_i] = b[pivot_i], b[0]
                if pivot_j == 0:
                    pivot_j = pivot_i
            if pivot_j != 1:
                a[[1, pivot_j], :] = a[[pivot_j, 1], :]
                a[:, [1, pivot_j]] = a[:, [pivot_j, 1]]
                b[1], b[pivot_j] = b[pivot_j], b[1]

            # Clear couplings from variables 2..n-1 into pivot pair, using congruence ops.
            # Operation: y_q <- y_q + y_p  (invertible over GF(2))
            # Corresponds to: col_q ^= col_p; row_q ^= row_p; b_q ^= b_p
            for t in range(2, n):
                if a[t, 0] == 1:
                    a[:, t] ^= a[:, 1]
                    a[t, :] ^= a[1, :]
                    b[t] ^= b[1]
                if a[t, 1] == 1:
                    a[:, t] ^= a[:, 0]
                    a[t, :] ^= a[0, :]
                    b[t] ^= b[0]

            # Now (0, 1) is an isolated J-block.
            arf ^= int(b[0] & b[1])
            pairs += 1

            # Drop the first two variables and continue.
            if n == 2:
                n = 0
                a = a[:0, :0]
                b = b[:0]
                break
            a = a[2:n, 2:n]
            b = b[2:n]
            n = int(a.shape[0])

        # Remaining variables are uncoupled; sum is zero unless all their linear terms vanish.
        if np.any(b):
            return 0

        # Each J-block contributes ±2 depending on the two linear coefficients.
        # Each free variable contributes 2.
        magnitude_power = k - pairs
        magnitude = 2 ** magnitude_power
        return -magnitude if (arf & 1) else magnitude


class Nonnegative(_Assummer):
    """Class for computing a *guaranteed nonnegative* quadratic character sum via one GF(2) elimination.

    If you need the sign (the Arf invariant), use `CharacterSummer`.
    """

    @staticmethod
    def sum(linear, hollow_symmetric):
        k, _ = hollow_symmetric.shape
        if k == 0:
            return 1

        # Gaussian elimination over GF(2) on the augmented system [A | b].
        augmented = np.empty((k, k + 1), dtype=np.uint8)
        augmented[:, :k] = hollow_symmetric
        augmented[:, k] = linear
        pivot_row = 0
        rank_a = 0
        for col in range(k):
            if pivot_row >= k:
                break
            pivots = np.flatnonzero(augmented[pivot_row:, col])
            if pivots.size == 0:
                continue
            r = pivot_row + int(pivots[0])
            if r != pivot_row:
                augmented[[pivot_row, r], :] = augmented[[r, pivot_row], :]

            below = np.flatnonzero(augmented[pivot_row + 1:, col]) + (pivot_row + 1)
            if below.size:
                augmented[below, col:] ^= augmented[pivot_row, col:]

            pivot_row += 1
            rank_a += 1

        # Inconsistency check: any zero row in A-part with RHS = 1.
        zero_rows = ~np.any(augmented[:, :k], axis=1)
        if np.any(augmented[zero_rows, k]):
            return 0
        
        return 2 ** (k - (rank_a // 2))

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