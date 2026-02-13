import numpy as np


import abc


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