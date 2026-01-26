"""
Trace of a product of Pauli projector factors (I + P)/2.

Public API:
    trace_of_projector_product_symplectic
"""

import itertools
from typing import Sequence
from fractions import Fraction

import numpy as np


# --------------------------
# Validation & basic helpers
# --------------------------

def _validate_symplectic_inputs(px: Sequence[np.ndarray],
                                pz: Sequence[np.ndarray],
                                signs: Sequence[int],
                                qubit_count: int):
    """Validate and convert inputs to numpy arrays (dtype=uint8)."""
    # if len(px) != len(pz):
    #     raise ValueError(
    #         f"px and pz must have same length; got {len(px)} and {len(pz)}"
    #     )
    pauli_count = len(px)
    try:
        px_array = np.array(px, dtype=np.uint8).reshape((pauli_count, qubit_count))
        pz_array = np.array(pz, dtype=np.uint8).reshape((pauli_count, qubit_count))
    except Exception as exc:
        raise ValueError(
            f"Could not interpret px/pz as m x qubit_count binary arrays: {exc}"
        ) from exc
    # if not np.all((px_array == 0) | (px_array == 1)):
    #     raise ValueError("px must be binary (0/1)")
    # if not np.all((pz_array == 0) | (pz_array == 1)):
    #     raise ValueError("pz must be binary (0/1)")
    # if len(signs) != pauli_count:
    #     raise ValueError(f"signs must have length {pauli_count}, got {len(signs)}")
    signs_array = np.array(signs, dtype=np.uint8)
    # if not np.all((signs_array == 0) | (signs_array == 1)):
    #     raise ValueError("signs must be bits 0 or 1")
    return pauli_count, px_array, pz_array, signs_array


# --------------------------
# GF(2) linear algebra
# --------------------------


def gf2_nullspace_basis(matrix: np.ndarray) -> np.ndarray:
    """
    Compute a GF(2) basis for the nullspace of `matrix` (shape p x q).
    Returns array B of shape (q, k) whose columns span the nullspace:
        matrix @ x = 0  <=>  x = B @ y for some y in GF(2)^k.
    """
    matrix = matrix.copy().astype(np.uint8)
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
    basis = []
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
# Symplectic omega (packed)
# --------------------------

def pack_bits_to_uint64(bit_array: np.ndarray) -> np.ndarray:
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


def _popcount_uint64_word(word: np.uint64) -> int:
    """Return popcount of a single uint64 word."""
    return int(int(word).bit_count())


def _popcount_and_mod2_word(a_word: np.uint64, b_word: np.uint64) -> int:
    """Return popcount(a_word & b_word) mod 2."""
    return _popcount_uint64_word(a_word & b_word) & 1


def symplectic_omega_from_packed(px_packed: np.ndarray, pz_packed: np.ndarray):
    """
    Compute omega parity matrix from packed px/pz arrays (uint64 arrays).
    omega[i,j] = z_i·x_j mod 2.
    Implemented using wordwise popcounts.
    """
    pauli_count, word_count = px_packed.shape
    omega = np.zeros((pauli_count, pauli_count), dtype=np.uint8)
    for i in range(pauli_count):
        xi = px_packed[i]
        zi = pz_packed[i]
        for j in range(pauli_count):
            xj = px_packed[j]
            zj = pz_packed[j]
            parity = 0
            for w in range(word_count):
                parity ^= _popcount_and_mod2_word(zi[w], xj[w])
            omega[i, j] = parity & 1
    return omega


def symplectic_omega(px_array: np.ndarray, pz_array: np.ndarray) -> np.ndarray:
    """
    Compute omega using numpy matrix products (simple and vectorized).
    For large qubit_count, prefer the packed-word version.
    omega[i,j] = z_i·x_j mod 2.
    """
    omega = (pz_array @ px_array.T) & 1
    return omega


# --------------------------
# Congruence reduction & deterministic counting
# --------------------------


def deterministic_count_from_congruence(a_symmetric: np.ndarray, linear: np.ndarray, constant: int, target: int) -> int:
    """
    Deterministic counting without enumeration:
    count y in GF(2)^k satisfying y^T a_symmetric y + linear^T y + constant = target (mod 2).
    """
    a = a_symmetric.copy().astype(np.uint8)
    b = linear.copy().astype(np.uint8)
    k = int(a.shape[0])

    if k == 0:
        return 1 if ((constant ^ target) & 1) == 0 else 0

    # Over GF(2), y_i^2 = y_i, so any diagonal terms can be moved into the linear term.
    # This makes the quadratic part alternating (zero diagonal).
    diag = np.diag(a).copy()
    if np.any(diag):
        b ^= diag
        np.fill_diagonal(a, 0)

    # Compute character sum S = sum_y (-1)^{q(y)+b^T y + constant}.
    # Then counts are:
    #   N(target) = (2^k + (-1)^target * (-1)^constant * S_qb) / 2
    # where S_qb = sum_y (-1)^{q(y)+b^T y}.
    s_qb = _character_sum_quadratic_gf2_alternating(a, b)
    if s_qb == 0:
        # Balanced: exactly half solutions.
        return 2 ** (k - 1)

    s_total = s_qb if (constant & 1) == 0 else -s_qb
    if (target & 1) == 1:
        s_total = -s_total
    return (2 ** k + s_total) // 2


def _character_sum_quadratic_gf2_alternating(a_alternating: np.ndarray, linear: np.ndarray) -> int:
    """Compute S = sum_{y in GF(2)^k} (-1)^{ y^T A y + linear^T y }.

    Assumes A is symmetric over GF(2) with zero diagonal (an alternating quadratic form
    in the representation used by this module).

    Returns an integer in {0, ±2^t}.
    """
    a = a_alternating.copy().astype(np.uint8)
    b = linear.copy().astype(np.uint8)
    k = int(a.shape[0])
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

        # Move pivot pair to positions (0,1).
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

        # Now (0,1) is an isolated J-block.
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


def count_solutions_quadratic(a_symmetric: np.ndarray, linear: np.ndarray, constant: int,
                              target: int, enum_threshold: int = 22, mode: str = 'auto') -> int:
    """
    Hybrid counting wrapper.
    mode:
      - 'auto' : brute force for k <= enum_threshold, deterministic otherwise
      - 'deterministic' : always deterministic
      - 'brute' : always brute force
    """
    k = a_symmetric.shape[0]
    if mode == 'brute':
        cnt = 0
        for bits in itertools.product([0, 1], repeat=k):
            y = np.array(bits, dtype=np.uint8)
            val = 0
            for a in range(k):
                for b in range(a + 1, k):
                    if a_symmetric[a, b] and y[a] and y[b]:
                        val ^= 1
            val ^= (int(linear @ y) & 1)
            val ^= constant & 1
            if val == (target & 1):
                cnt += 1
        return cnt
    elif mode == 'auto':
        if k <= enum_threshold:
            return count_solutions_quadratic(a_symmetric, linear, constant, target, enum_threshold, mode='brute')
        else:
            return deterministic_count_from_congruence(a_symmetric, linear, constant, target)
    elif mode == 'deterministic':
        return deterministic_count_from_congruence(a_symmetric, linear, constant, target)
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected 'auto', 'deterministic', or 'brute'.")


# --------------------------
# High-level API
# --------------------------

def trace_of_projector_product_symplectic(
    px: Sequence[np.ndarray],
    pz: Sequence[np.ndarray],
    signs: Sequence[int],
    qubit_count: int,
    enum_threshold: int = 22,
    mode: str = 'auto',
    use_packed_omega: bool = True
):
    """
    Compute exactly T = tr( prod_i (I + P_i)/2 ) for Paulis P_i given in symplectic form.
    Args:
      px, pz: length-m lists/arrays of qubit_count-bit vectors (X- and Z- components).
      signs : length-m list/array of bits (0 => +1, 1 => -1).
      qubit_count: number of qubits.
      enum_threshold: threshold when mode='auto' decides to brute force.
      mode  : 'auto'|'deterministic'|'brute'.
      use_packed_omega: use 64-bit-word packing for large qubit_count (faster).
    Returns:
      integer trace T.
    """
    pauli_count, px_array, pz_array, signs_array = _validate_symplectic_inputs(px, pz, signs, qubit_count)
    if use_packed_omega and qubit_count > 64:
        px_packed = pack_bits_to_uint64(px_array)
        pz_packed = pack_bits_to_uint64(pz_array)
        omega = symplectic_omega_from_packed(px_packed, pz_packed)
    else:
        omega = symplectic_omega(px_array, pz_array)
    # Solve linear constraint: V^T x = 0
    V = np.concatenate([px_array, pz_array], axis=1).astype(np.uint8)
    nullspace_basis = gf2_nullspace_basis(V.T)  # m x k
    k = nullspace_basis.shape[1]
    B = nullspace_basis.astype(np.uint8)
    if k == 0:
        N_plus = 1
        N_minus = 0
        numerator = (2 ** qubit_count) * (N_plus - N_minus)
        denominator = 2 ** pauli_count
        return Fraction(numerator, denominator)
    linear_y = (B.T @ signs_array) % 2
    # Build cross-term accumulation
    cross_C = np.zeros((k, k), dtype=np.uint8)
    for i in range(pauli_count):
        bi = B[i, :]
        if not bi.any():
            continue
        for j in range(i + 1, pauli_count):
            if omega[i, j] == 0:
                continue
            bj = B[j, :]
            if not bj.any():
                continue
            cross_C ^= np.outer(bi, bj).astype(np.uint8)
    # a_symmetric: symmetric k x k with zero diagonal (cross-term coefficients)
    a_symmetric = np.zeros((k, k), dtype=np.uint8)
    diag_C = np.zeros(k, dtype=np.uint8)
    for a in range(k):
        diag_C[a] = int(cross_C[a, a] & 1)
        for b in range(a + 1, k):
            val = (cross_C[a, b] ^ cross_C[b, a]) & 1
            a_symmetric[a, b] = val
            a_symmetric[b, a] = val
    linear_total = (linear_y ^ diag_C) & 1
    constant = 0
    N_plus = count_solutions_quadratic(a_symmetric, linear_total, constant, target=0,
                                       enum_threshold=enum_threshold, mode=mode)
    N_minus = count_solutions_quadratic(a_symmetric, linear_total, constant, target=1,
                                        enum_threshold=enum_threshold, mode=mode)
    numerator = (2 ** qubit_count) * (N_plus - N_minus)
    denominator = 2 ** pauli_count
    return Fraction(numerator, denominator)


# --------------------------
# Utility: brute force verifier (small sizes)
# --------------------------

def brute_force_trace_symplectic(
    px: Sequence[np.ndarray],
    pz: Sequence[np.ndarray],
    signs: Sequence[int],
    qubit_count: int
):
    """
    Brute force expansion over all subsets (for testing small instances).
    Returns (T, N_plus, N_minus).
    """
    pauli_count, px_array, pz_array, signs_array = _validate_symplectic_inputs(px, pz, signs, qubit_count)
    V = np.concatenate([px_array, pz_array], axis=1).astype(np.uint8)
    omega = symplectic_omega(px_array, pz_array)
    N_plus = 0
    N_minus = 0
    for mask in range(1 << pauli_count):
        x = np.array([(mask >> idx) & 1 for idx in range(pauli_count)], dtype=np.uint8)
        if np.any((V.T @ x) % 2):
            continue
        s = int((signs_array @ x) & 1)
        q = 0
        for i in range(pauli_count):
            if x[i] == 0:
                continue
            for j in range(i + 1, pauli_count):
                if x[j] == 1 and omega[i, j] == 1:
                    q ^= 1
        sigma = (s ^ q) & 1
        if sigma == 0:
            N_plus += 1
        else:
            N_minus += 1
    T = Fraction((2 ** qubit_count) * (N_plus - N_minus), 2 ** pauli_count)
    return T, N_plus, N_minus