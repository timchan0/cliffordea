from collections.abc import Iterable

from cliffordea import circuits
from cliffordea.circuits.main import DoubleCheck


Qubit = tuple[float, float]
Support = frozenset[Qubit]


_BELL_PAIR_SUPPORTS: tuple[Support, ...] = tuple(map(frozenset, (
    ((3, 4), (4, 6)),
    ((4, 2), (5, 1)),
    ((4, 4), (5, 5)),
    ((5, 3), (6, 3)),
    ((6, 1), (7, 2)),
    ((7, 0), (8, 0)),
)))


def _span(generators: Iterable[Support]) -> set[Support]:
    """Generate every support obtainable by multiplying CSS generators.

    Supports multiply by symmetric difference because applying the same Pauli
    twice cancels on each qubit.

    :param generators: Supports whose products generate the group.
    :return: Supports of every element in the generated group.
    """
    result: set[Support] = {frozenset()}
    for generator in generators:
        result |= {element ^ generator for element in tuple(result)}
    return result


def _layout_supports(
        layout: DoubleCheck,
) -> tuple[Support, dict[str, tuple[Support, ...]]]:
    """Express a colour-code layout and its checks using qubit coordinates.

    Coordinates put the distance-3 and distance-5 layouts in a common frame,
    allowing their data qubits and stabilizer supports to be compared directly.

    :param layout: Distance-3 or distance-5 colour-code circuit layout.
    :return data_support: Support containing every data-qubit coordinate.
    :return generators: X- and Z-type stabilizer-generator supports.
    """
    coordinates = layout.INNER_CIRCUIT.get_final_qubit_coordinates()
    data_coordinates: tuple[Qubit, ...] = tuple(
        (coordinates[qubit][0], coordinates[qubit][1])
        for qubit in layout.DATA_INDICES
    )
    generators: dict[str, tuple[Support, ...]] = {
        basis: tuple(frozenset(
            data_coordinates[index]
            for index in range(len(generator))
            if generator[index]
        ) for generator in layout.STABILIZER_GENERATORS_RESTRICTED[basis])
        for basis in ('X', 'Z')
    }
    return frozenset(data_coordinates), generators


def test_inherits_syndrome_and_logicals():
    """Check that Bell-pair growth preserves the old syndrome and logicals.

    The prepared Bell pairs fix seven of the nine distance-5 checks in each
    basis. Restricted to the old patch, these checks recover the complete
    distance-3 syndrome; the same Bell pairs extend the old logical operators
    across all new data qubits. The other two checks are genuinely new and
    acquire their values from the following stabilizer round.
    """
    d3_data, d3_generators = _layout_supports(
        circuits.DoubleCheck(ancilla_count=6)
    )
    d5_data, d5_generators = _layout_supports(
        circuits.DoubleCheck(distance=5, ancilla_count=19)
    )

    for basis in ('X', 'Z'):
        prepared_span = _span((*d3_generators[basis], *_BELL_PAIR_SUPPORTS))
        inherited = tuple(
            generator for generator in d5_generators[basis]
            if generator in prepared_span
        )
        fresh = tuple(
            generator for generator in d5_generators[basis]
            if generator not in prepared_span
        )

        assert len(inherited) == 7
        assert len(fresh) == 2
        assert _span(generator & d3_data for generator in inherited) == _span(
            d3_generators[basis]
        )

        # The all-data logical pulls back to the old logical modulo Bell pairs.
        assert d3_data ^ d5_data in _span(_BELL_PAIR_SUPPORTS)
