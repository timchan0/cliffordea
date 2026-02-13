# --------------------------
# Congruence reduction & deterministic counting
# --------------------------

import itertools
from typing import Literal
import numpy as np


import abc

from cliffordep.pauli_trace._assumers import NoAssumptions, Nonnegative


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


def _get_a_symmetric(cross_C: np.ndarray[tuple[int, int], np.dtype[np.uint8]]):
    """Get the hollow symmetric k x k matrix A = C + C^T from cross-term accumulation matrix C."""
    out: np.ndarray[tuple[int, int], np.dtype[np.uint8]] = cross_C ^ cross_C.T
    return out


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