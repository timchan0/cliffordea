import pytest

from cliffordep.pauli_string_tools import CliffordString, FrozenCliffordString

@pytest.fixture
def frozen_string_1(string_1: CliffordString):
    return string_1.frozen_copy()

@pytest.fixture
def frozen_string_2(string_2: CliffordString):
    return string_2.frozen_copy()

@pytest.fixture
def frozen_string_3(string_3: CliffordString):
    return string_3.frozen_copy()


def test_not_equals(frozen_string_1: FrozenCliffordString, frozen_string_2: FrozenCliffordString):
    assert frozen_string_1 != frozen_string_2