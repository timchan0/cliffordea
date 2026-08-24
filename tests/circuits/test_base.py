import pytest
import stim

from cliffordep import circuits


def test_find_data_indices():
    """Test DATA_INDICES agrees with hardcoded values."""

    all_circuits: dict[
        tuple[int, int],
        circuits.Distance3DoubleCheck | circuits.Distance5DoubleCheck,
    ] = {
        (1, 0): circuits.D3A1(),
        (2, 0): circuits.D3A2(),
        (3, 0): circuits.D3A3(),
        (4, 0): circuits.D3A4(),
        (5, 0): circuits.D3A5(),
        (6, 0): circuits.D3A6(),
        (7, 0): circuits.D3A7(),
        (6, 2): circuits.D3A6F2(),
        (6, 3): circuits.D3A6F3(),
        (19, 0): circuits.D5A19(),
    }
    _DATA_INDICES = {
        (1, 0): (0, 1, 3, 4, 5, 6, 7),
        (2, 0): (0, 1, 3, 4, 5, 7, 8),
        (3, 0): (0, 2, 4, 5, 6, 8, 9),
        (4, 0): (0, 2, 4, 6, 7, 9, 10),
        (5, 0): (0, 3, 5, 7, 8, 10, 11),
        (6, 0): (0, 3, 5, 7, 8, 10, 11),
        (7, 0): (0, 3, 5, 7, 8, 10, 11),
        (6, 2): (0, 3, 7, 9, 10, 12, 13),
        (6, 3): (0, 3, 7, 9, 11, 13, 14),
        (19, 0): tuple(sorted((3, 5, 0, 9, 14, 22, 32, 29, 34, 31, 24, 26, 20, 18, 13, 7, 11, 16, 36)))
    }
    for ancilla_flag_counts, circuit in all_circuits.items():
        assert circuit.DATA_INDICES == _DATA_INDICES[ancilla_flag_counts]


def test_find_stabilizer_generators():
    """Test STABILIZER_GENERATORS agrees with hardcoded values."""

    all_circuits: dict[
        tuple[int, int],
        circuits.Distance5DoubleCheck,
    ] = {
        (19, 0): circuits.D5A19(),
    }
    _STABILIZER_GENERATORS = {
        (19, 0): {tuple(sorted(indices)) for indices in [
            (0, 9, 5, 3), (14, 32, 29, 22), (11, 16, 24, 18, 13, 7),
            (22, 29, 34, 31, 24, 16), (3, 5, 11, 7), (13, 18, 26, 20),
            (9, 14, 22, 16, 11, 5), (24, 31, 26, 18), (29, 32, 36, 34),
        ]},
    }
    for ancilla_flag_counts, circuit in all_circuits.items():
        assert set(circuit._STABILIZER_GENERATOR_INDICES) == _STABILIZER_GENERATORS[ancilla_flag_counts]


@pytest.mark.parametrize(
    "ancilla_flag_counts, circuit",
    [
        ((19, 0), circuits.D5A19()),
        ((19, 3), circuits.D5A19F3()),
        ((19, 15), circuits.D5A19F15()),
        ((19, 18), circuits.D5A19F18()),
    ])
def test_find_logical_s_gate(ancilla_flag_counts, circuit: circuits.Distance5DoubleCheck):
    """Test LOGICAL_S agrees with hardcoded values."""
    _LOGICAL_S = {
        (19, 0): stim.Circuit(
"""
S 0 5 7 14 16 18 20 29 31 36
S_DAG 9 11 13 22 24 26 34 32 3
"""
        ),
        (19, 3): stim.Circuit(
"""
S 16 18 20 22 32 34 6 8 0 39
S_DAG 3 10 12 14 26 24 28 35 37
"""
        ),
        (19, 15): stim.Circuit(
"""
S 20 22 24 26 41 43 7 9 0 51
S_DAG 4 14 32 16 18 36 34 48 46
"""
        ),
        (19, 18): stim.Circuit(
"""
S 21 23 25 27 44 46 7 9 0 54
S_DAG 4 15 35 17 19 39 37 51 49
"""
        ),
    }
    assert circuit.LOGICAL_S.to_tableau() == _LOGICAL_S[ancilla_flag_counts].to_tableau()