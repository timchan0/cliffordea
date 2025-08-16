import pytest

from cliffordep.pauli_string_tools import CliffordString


@pytest.fixture
def terms() -> dict[str, complex]:
    return {"X": 2, "Y": -1j}


@pytest.fixture
def string_1(terms: dict[str, complex]) -> CliffordString:
    return CliffordString(terms)


@pytest.fixture
def string_2():
    return CliffordString({"_": 1j, "Z": -1j}, denominator_squared=1.5)


@pytest.fixture
def string_3() -> CliffordString:
    return CliffordString({"XX": 2, "YY": -1j})