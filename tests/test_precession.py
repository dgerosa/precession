import pytest
import numpy as np
import precession as pre


def test_silly():
    """
    Test the test procedure.
    """

    assert True


def assert_scalar(val, func, *args):
    """
    Assert that the output of an array is a scalar with correct check value.

    Parameters
    ----------
    val, float:
        The check value for the function being tested.

    func, function:
        The function to be tested.

    *args:
        The parameters passed to func.
    """

    output = func(*args)
    # Test scalar input returns scalar output
    check_scalar = isinstance(output, float)
    assert check_scalar, 'Scalar input does not return scalar output'
    # Test function returns correct scalar value
    if check_scalar:
        check_val = val == output
        assert check_val, 'Incorrect scalar value returned'


def assert_vector(vals, func, *args):
    """
    Assert that the output of an array is a vector with correct check values.

    Parameters
    ----------
    vals, array:
        The check values for the function being tested.

    func, function:
        The function to be tested.

    *args:
        The parameters passed to func.
    """

    output = func(*args)
    # Test vector input returns vector output
    check_vector = isinstance(output, np.ndarray)
    assert check_vector, 'Vector input does not return vector output'
    # Test function returns correct vector values
    if check_vector:
        check_vals = (vals == output).all()
        assert check_vals, 'Incorrect vector values returned'


def test_mass1():
    """
    Test computation of primary mass
    """

    # Test scalar input
    q = 1.0
    m1 = 0.5
    assert_scalar(m1, pre.mass1, q)
    # Test vector input
    q = [1.0, 0.0]
    m1 = [0.5, 1.0]
    assert_vector(m1, pre.mass1, q)


#def test_mass1():
#    """
#    Test computation of primary mass.
#    """
#
#    # Test scalar input returns scalar output
#    q = 1.0
#    m1 = pre.mass1(q)
#    check_scalar = isfloat(m1)
#    assert check_scalar, 'Scalar input does not return scalar output'
#
#    # Test function returns correct scalar value
#    if check_scalar:
#        check_val = m1 == 0.5
#        assert check_val, 'Incorrect scalar value returned'
#
#    # Test vector input returns vector output
#    q = [1.0, 0.0]
#    m1 = pre.mass1(q)
#    check_vector = isarray(m1)
#    assert check_vector, 'Vector input does not return vector output'
#
#    # Test function returns correct vector values
#    if check_vector:
#        check_vals = (m1 == [0.5, 1.0]).all()
#        assert check_vals, 'Incorrect vector values returned'
