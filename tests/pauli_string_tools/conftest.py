import pytest

from cliffordep.pauli_string_tools import PauliSum


@pytest.fixture
def terms() -> dict[str, complex]:
    return {"X": 2, "Y": -1j}


@pytest.fixture
def string_1(terms: dict[str, complex]) -> PauliSum:
    return PauliSum(terms)


@pytest.fixture
def string_2():
    return PauliSum({"_": 1j, "Z": -1j}, denominator_squared=1.5)


@pytest.fixture
def string_3() -> PauliSum:
    return PauliSum({"XX": 2, "YY": -1j})


@pytest.fixture
def string_4() -> PauliSum:
    return PauliSum({"XX": 4, "XY": -2j, "YX": -2j, "YY": -1}, denominator_squared=1.5)