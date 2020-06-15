import pytest
import numpy as np
import precession as pre


def test_silly():
    """
    Test the test procedure.
    """

    assert True


def isfloat(var):
    """
    Check a variable is a scalar float.

    Parameters
    ----------
    var, any:
        Variable to check the type of.

    Returns
    -------
    bool:
        Truth of variable being a scalar float.
    """

    if isinstance(var, float):
        return 1
    else:
        return 0


def isarray(var):
    """
    Check if a variable is a numpy array.

    Parameters
    ----------
    var, any:
        Variable to check the type of.

    Returns
    -------
    bool:
        Truth of variable being a numpy array.
    """

    if isinstance(var, np.ndarray):
        return 1
    elif isinstance(var, (list, tuple)):
        return 0 #2
    else:
        return 0


def test_mass1():
    """
    Test computation of primary mass.
    """

    # Test scalar input returns scalar output.
    q = 1.0
    m1 = pre.mass1(q)
    check_scalar = isfloat(q)
    assert check_scalar

    # Test function returns correct scalar value.
    if check_scalar:
        check_val = m1 == 0.5
        assert check_val

    # Test vector input returns vector output.
    q = [1.0, 0.0]
    m1 = pre.mass1(q)
    check_vector = isarray(m1)
    assert check_vector

    # Test function returns correct vector values.
    if check_vector:
        check_vals = m1 == np.array([0.5, 1.0])
        assert check_vals
