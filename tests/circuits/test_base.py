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