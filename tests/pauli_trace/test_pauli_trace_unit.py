import numpy as np

from cliffordep.pauli_trace import (
    _get_cross_term_accumulation,
    _get_a_symmetric,
)


class TestGetCrossTermAccumulation:


    def test_against_slow_implementation(self):
        """Simple example: manual double-sum vs function."""
        # pairing: only (0,1) and (0,2) are 1
        pairing = np.array([
            [0, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
        ], dtype=np.uint8)
        # kernel basis: 3 rows, 2 basis vectors
        kernel = np.array([
            [1, 0],
            [0, 1],
            [1, 1],
        ], dtype=np.uint8)

        expected = self._slow_implementation(pairing, kernel)

        out = _get_cross_term_accumulation(pairing, kernel)
        assert out.dtype == np.uint8
        assert np.array_equal(out, expected)

    
    @staticmethod
    def _slow_implementation(
        pairing_matrix: np.ndarray[tuple[int, int], np.dtype[np.uint8]],
        kernel_basis: np.ndarray[tuple[int, int], np.dtype[np.uint8]],
    ):
        """Slow version of _get_cross_term_accumulation for testing."""
        pauli_count, nullity = kernel_basis.shape
        out = np.zeros((nullity, nullity), dtype=np.uint8)
        for i in range(pauli_count):
            for j in range(i + 1, pauli_count):
                if pairing_matrix[i, j]:
                    out ^= np.outer(kernel_basis[i], kernel_basis[j]).astype(np.uint8)
        return out


    def test_ignores_zero_rows(self):
        """Rows with all-zero basis vectors should be ignored."""
        pairing = np.zeros((3, 3), dtype=np.uint8)
        pairing[0, 2] = 1
        kernel = np.array([
            [1],
            [0],
            [1],
        ], dtype=np.uint8)

        # Only (0,2) contributes: outer([1],[1]) = [[1]]
        expected = np.array([[1]], dtype=np.uint8)
        out = _get_cross_term_accumulation(pairing, kernel)
        assert np.array_equal(out, expected)


class TestGetASymmetric:

    
    def test_basic(self):
        """Simple example for getting A from C."""
        cross_C = np.array([
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ], dtype=np.uint8)

        expected = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=np.uint8)

        out = _get_a_symmetric(cross_C)
        assert out.dtype == np.uint8
        assert np.array_equal(out, expected)

    
    def test_against_slow_implementation(self):
        """Compare fast and slow implementations on random inputs."""
        for nullity in [1, 2, 3, 5, 10]:
            cross_C = np.random.randint(0, 2, size=(nullity, nullity), dtype=np.uint8)

            expected = self._slow_implementation(cross_C)
            out = _get_a_symmetric(cross_C)
            assert out.dtype == np.uint8
            assert np.array_equal(out, expected)
    
    @staticmethod
    def _slow_implementation(cross_C: np.ndarray[tuple[int, int], np.dtype[np.uint8]]):
        """Slow version of _get_a_symmetric for testing."""
        nullity, _ = cross_C.shape
        a_symmetric = np.zeros((nullity, nullity), dtype=np.uint8)
        for a in range(nullity):
            for b in range(a + 1, nullity):
                val = (cross_C[a, b] ^ cross_C[b, a]) & 1
                a_symmetric[a, b] = val
                a_symmetric[b, a] = val
        return a_symmetric