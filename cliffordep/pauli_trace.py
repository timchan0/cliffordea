"""
Trace of a product of Pauli projector factors (I + P)/2.

Public API:
    trace_of_projector_product_symplectic
"""

import itertools
from typing import Sequence, Literal
from fractions import Fraction

import numpy as np

# --------------------------
# Validation & basic helpers
# --------------------------

def _validate_symplectic_inputs(
    px: Sequence[np.ndarray[tuple[int], np.dtype[np.bool_]]],
    pz: Sequence[np.ndarray[tuple[int], np.dtype[np.bool_]]],
    signs: Sequence[int],
    qubit_count: int,
):
    """Validate and convert inputs to numpy arrays (dtype=uint8)."""
    pauli_count = len(px)
    try:
        px_array = np.array(px, dtype=np.uint8).reshape((pauli_count, qubit_count))
        pz_array = np.array(pz, dtype=np.uint8).reshape((pauli_count, qubit_count))
    except Exception as exc:
        raise ValueError(
            f"Could not interpret px/pz as m x qubit_count binary arrays: {exc}"
        ) from exc
    signs_array: np.ndarray[tuple[int], np.dtype[np.uint8]] = np.array(signs, dtype=np.uint8)
    return pauli_count, px_array, pz_array, signs_array


# --------------------------
# GF(2) linear algebra
# --------------------------


def _gf2_kernel_basis(
        matrix: np.ndarray[tuple[int, int], np.dtype[np.uint8]]
) -> np.ndarray[tuple[int, int], np.dtype[np.uint8]]:
    """
    Compute a GF(2) basis for the kernel of a matrix.
    
    :param matrix: A p x q matrix.
    :return basis: A q x k matrix B whose columns span the kernel:
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


def _deterministic_count_from_congruence(
    a_symmetric: np.ndarray[tuple[int, int], np.dtype[np.uint8]],
    linear: np.ndarray[tuple[int], np.dtype[np.uint8]],
    constant: Literal[0, 1],
    target: Literal[0, 1],
) -> int:
    """
    Deterministic counting without enumeration:
    count y in GF(2)^k satisfying y^T a_symmetric y + linear^T y + constant = target (mod 2).
    """
    a = a_symmetric.copy()
    b = linear.copy()
    k, _ = a.shape

    if k == 0:
        return constant ^ target ^ 1

    # Over GF(2), y_i^2 = y_i, so any diagonal terms can be moved into the linear term.
    # This makes the quadratic part alternating (zero diagonal).
    b ^= np.diag(a)
    np.fill_diagonal(a, 0)

    # Compute character sum S = sum_y (-1)^{q(y)+b^T y + constant}.
    # Then counts are:
    #   N(target) = (2^k + (-1)^(target + constant) * S_qb) / 2
    # where S_qb = sum_y (-1)^{q(y)+b^T y}.
    s_qb = _character_sum_quadratic_gf2_alternating(a, b)
    if s_qb == 0:
        # Balanced: exactly half solutions.
        return 2 ** (k - 1)

    s_total = -s_qb if constant else s_qb
    if target:
        s_total = -s_total
    return (2 ** k + s_total) // 2


def _character_sum_quadratic_gf2_alternating(
        a_alternating: np.ndarray[tuple[int, int], np.dtype[np.uint8]],
        linear: np.ndarray[tuple[int], np.dtype[np.uint8]],
) -> int:
    """Compute S = sum_{y in GF(2)^k} (-1)^{ y^T A y + linear^T y }.

    :param a_alternating: Symmetric matrix A over GF(2) with zero diagonal
    (an alternating quadratic form in the representation used by this module).

    Returns an integer in {0, ±2^t}.
    """
    a = a_alternating.copy()
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


def _count_solutions_quadratic(
    cross_C: np.ndarray[tuple[int, int], np.dtype[np.uint8]],
    linear: np.ndarray[tuple[int], np.dtype[np.uint8]],
    constant: Literal[0, 1],
    target: Literal[0, 1],
    enum_threshold: int = 22,
    mode: Literal['auto', 'deterministic', 'brute'] = 'auto',
):
    """
    Count solutions y in GF(2)^k of a quadratic Boolean equation.
    
    :param cross_C: A cross-term accumulation matrix
        in GF(2)^{k x k} defining the quadratic part.
    :param linear: Vector in GF(2)^k defining the linear part.
    :param constant: Constant term in the quadratic Boolean function.
    :param target: Target value of the quadratic Boolean function.
    :param mode: Mode to count solutions:
      1. 'auto' uses brute force for k <= enum_threshold, deterministic otherwise,
      2. 'deterministic' uses always deterministic,
      3. 'brute' uses always brute force.

    :return solution_count: Number of y in GF(2)^k satisfying
        y^T a_symmetric y + linear^T y + constant = target (mod 2).
    """
    k, _ = cross_C.shape
    if mode == 'brute' or (mode == 'auto' and k <= enum_threshold):
        out = 0
        for bits in itertools.product((0, 1), repeat=k):
            y = np.array(bits, dtype=np.uint8)
            function_val = constant
            function_val ^= int(linear @ y) & 1
            function_val ^= int(y.T @ cross_C @ y) & 1
            out += (function_val ^ target ^ 1)
        return out
    else:
        a_symmetric = _get_a_symmetric(cross_C)
        linear_total: np.ndarray[tuple[int], np.dtype[np.uint8]] = linear ^ np.diagonal(cross_C)
        return _deterministic_count_from_congruence(a_symmetric, linear_total, constant, target)


# --------------------------
# High-level API
# --------------------------

def trace_of_projector_product_symplectic(
    px: Sequence[np.ndarray[tuple[int], np.dtype[np.bool_]]],
    pz: Sequence[np.ndarray[tuple[int], np.dtype[np.bool_]]],
    signs: Sequence[int],
    qubit_count: int,
    enum_threshold: int = 22,
    mode: Literal['auto', 'deterministic', 'brute'] = 'auto',
    use_packed: bool = True,
):
    """
    Compute exactly T = tr[ prod_i (I + P_i)/2 ] for Paulis P_i given in symplectic form.
    
    :param px: Length-m sequence of n-bit vectors (X component).
    :param pz: Length-m sequence of n-bit vectors (Z component).
    :param signs: Length-m sequence of bits (0 => +1, 1 => -1).
    :param qubit_count: Number n of qubits.
    :param enum_threshold: Threshold when mode='auto' decides to brute force.
    :param mode: Mode to count solutions:
        1. 'auto' uses brute force for k <= enum_threshold, deterministic otherwise,
        2. 'deterministic' uses always deterministic,
        3. 'brute' uses always brute force.
    :param use_packed: When constructing the pairing matrix,
        use 64-bit-word packing for large n (faster).
    :return trace: The trace T.
    """
    pauli_count, px_array, pz_array, signs_array = _validate_symplectic_inputs(
        px, pz, signs, qubit_count)
    if use_packed and qubit_count > 64:
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
        size_of_0_set = 1
    else:
        constant = 0
        linear: np.ndarray[tuple[int], np.dtype[np.uint8]] = (
            kernel_basis.T @ signs_array) & 1 # type: ignore
        cross_C = _get_cross_term_accumulation(pairing_matrix, kernel_basis)
        size_of_0_set = _count_solutions_quadratic(
            cross_C=cross_C,
            linear=linear,
            constant=constant,
            target=0,
            enum_threshold=enum_threshold,
            mode=mode,
        )
    numerator = (2 ** qubit_count) * (2*size_of_0_set - 2**nullity)
    denominator = 2 ** pauli_count
    return Fraction(numerator, denominator)


# --------------------------
# Utility: brute force verifier (small sizes)
# --------------------------

def brute_force_trace_symplectic(
    px: Sequence[np.ndarray[tuple[int], np.dtype[np.bool_]]],
    pz: Sequence[np.ndarray[tuple[int], np.dtype[np.bool_]]],
    signs: Sequence[int],
    qubit_count: int,
):
    """
    Brute force expansion over all subsets (for testing small instances).

    :param px: See `trace_of_projector_product_symplectic`.
    :param pz: See `trace_of_projector_product_symplectic`.
    :param signs: See `trace_of_projector_product_symplectic`.
    :param qubit_count: See `trace_of_projector_product_symplectic`.

    :return trace: The trace.
    :return size_of_0_set: Number of subsets giving +1 overall sign.
    :return size_of_1_set: Number of subsets giving -1 overall sign.
    """
    pauli_count, px_array, pz_array, signs_array = _validate_symplectic_inputs(
        px, pz, signs, qubit_count)
    paulis_in_bsf = np.concatenate([px_array, pz_array], axis=1).astype(np.uint8)
    pairing_matrix = _get_pairing_matrix(px_array, pz_array)
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