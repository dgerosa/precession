import pytest
import numpy as np
import precession as pre


def test_silly():
    """
    Ensure test procude works.
    """

    assert True


def isarray(var):
    """Check if a variable is an array
    """

    if isinstance(var, np.ndarray):
        return 1
    elif isinstance(var, (list, tuple)):
        return 0 #2
    else:
        return 0


def test_mass1():
    """
    Test primary mass from mass ratio.
    """

    q = .5
    m1 = pre.mass1(q)

    assert isarray(m1)
    assert m1 == .5


def test_mass2():
    """
    Test secondary mass from mass ratio.
    """

    q = .5
    m2 = pre.mass2(q)

    assert isarray(m2)
    assert m2 == .5
