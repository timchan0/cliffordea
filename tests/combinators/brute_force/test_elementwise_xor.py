import numpy as np
import pytest

from cliffordep.combinators import SlowFaultSourceCombinator

def test_1():
    assert SlowFaultSourceCombinator._any_defects([np.array([True, False, True])])

def test_2():
    syndromes = (
        np.array([True, False, False]),
        np.array([True, True, False]),
    )
    assert SlowFaultSourceCombinator._any_defects(syndromes)

def test_length_1():
    syndromes = (
        np.array([True]),
        np.array([False]),
        np.array([False]),
    )
    assert SlowFaultSourceCombinator._any_defects(syndromes)

def test_all_false():
    syndromes = (
        np.array([False, False, False]),
        np.array([False, False, False]),
        np.array([False, False, False]),
    )
    assert not SlowFaultSourceCombinator._any_defects(syndromes)

def test_all_true():
    syndromes = (
        np.array([True, True, True]),
        np.array([True, True, True]),
        np.array([True, True, True]),
    )
    assert SlowFaultSourceCombinator._any_defects(syndromes)

def test_mixed():
    syndromes = (
        np.array([True, False, True]),
        np.array([False, True, True]),
        np.array([True, True, False]),
    )
    assert not SlowFaultSourceCombinator._any_defects(syndromes)

def test_different_lengths():
    syndromes = (
        np.array([True, False]),
        np.array([False, True, True]),
    )
    with pytest.raises(ValueError):
        SlowFaultSourceCombinator._any_defects(syndromes)