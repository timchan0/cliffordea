import numpy as np

from cliffordep.pauli_trace import _get_cross_term_accumulation


def test_get_cross_term_accumulation_basic():
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

    expected = _get_cross_term_accumulation_slow(pairing, kernel)

    out = _get_cross_term_accumulation(pairing, kernel)
    assert out.dtype == np.uint8
    assert np.array_equal(out, expected)


def _get_cross_term_accumulation_slow(
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


def test_get_cross_term_accumulation_ignores_zero_rows():
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