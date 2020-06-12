"""
precession
"""

import numpy as np
import scipy as sp
# Using precession_v1 functions for now
import precession as pre
import sys, os, time
import warnings
import itertools

__all__ = [] # Why is this necessary?


#getnone=itertools.repeat(None)
def flen(x):
    if hasattr(x, "__len__"):
        return len(x)
    else:
        return 1

def toarray(*args):
    return np.squeeze(np.array([*args]))

def mass1(q):
    """
    Mass of the heavier black hole in units of the total mass.

    Parameters
    ----------
    q: float
        Mass ratio: 0 <= q <= 1.

    Returns
    -------
    m1: float
        Mass of the primary black hole.
    """

    q = toarray(q)
    m1 = 1/(1+q)

    return m1


def mass2(q):
    """
    Mass of the lighter black hole in units of the total mass.

    Parameters
    ----------
    q: float
        Mass ratio: 0 <= q <= 1.

    Returns
    -------
    m2: float
        Mass of the secondary black hole.

    """

    q = toarray(q)
    m2 = q/(1+q)

    return m2


def spin1(q,chi1):
    """
    Spin angular momentum of the heavier black hole.

    Parameters
    ----------
    q: float
        Mass ratio: 0 <= q <= 1.
    chi1: float or numpy array
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.

    Returns
    -------
    S1: float
        Spin of the primary black hole.
    """

    chi1 = (chi1)
    S1 = chi1*(mass1(q))**2

    return S1


def spin2(q,chi2):
    """
    Spin angular momentum of the lighter black hole.

    Parameters
    ----------
    q: float
        Mass ratio: 0 <= q <= 1.
    chi2: float or numpy array
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.

    Returns
    -------
    S2: float
        Spin of the secondary black hole.
    """

    chi2 = toarray(chi2)
    S2 = chi2*(mass2(q))**2

    return S2


def spinmags(q,chi1,chi2):
    """
    Spins of the black holes in units of the total mass.

    Parameters
    ----------
    q: float
        Mass ratio: 0 <= q <= 1.
    chi1: float or numpy array
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.
    chi2: float
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.

    Returns
    -------
    S1: float
        Spin of the primary black hole.
    S2: float
        Spin of the secondary black hole.
    """

    S1 = spin1(q,chi1)
    S2 = spin2(q,chi2)

    return np.array([S1,S2])


def angularmomentum(r,q):
    """
    Newtonian angular momentum of the binary.

    Parameters
    ----------
    r: float
        Binary separation.
    q: float
        Mass ratio: 0 <= q <= 1.

    Returns
    -------
    Smin: float
        Binary angular momentum
    """

    r = toarray(r)
    L = mass1(q)*mass2(q)*r**0.5

    return L


def Jlimits_LS1S2(r,q,chi1,chi2):
    """
    Limits on the magnitude of the total angular momentum due to the vector relation J=L+S1+S2

    Parameters
    ----------
    r: float
        Binary separation.
    q: float
        Mass ratio: 0 <= q <= 1.
    chi1: float
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.
    chi2: float
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.

    Returns
    -------
    Jmin:
        Minimum value of the total angular momentum.
    Jmax:
        Maximum value of the total angular momentum.
    """


    S1,S2 = spinmags(q,chi1,chi2)
    L = angularmomentum(r,q)
    Jmin = np.maximum.reduce([np.zeros(flen(L)), L-S1-S2, np.abs(S1-S2)-L])
    Jmax = L+S1+S1

    return np.array([Jmin,Jmax])


def Jdiscriminant_coefficients(r,xi,q,chi1,chi2):
    """
    Coefficients of the quintic equation in J that defines the spin-orbit resonances.

    Parameters
    ----------
    r: float
        Binary separation.
    xi: float
        Effective spin
    q: float
        Mass ratio: 0 <= q <= 1.
    chi1: float
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.
    chi2: float
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.

    Returns
    -------
    sigma10:
        Coefficient of J^10.
    sigma8:
        Coefficient of J^8.
    sigma6:
        Coefficient of J^6.
    sigma4:
        Coefficient of J^4.
    sigma2:
        Coefficient of J^2.
    sigma0:
        Coefficient of J^0.
    """


    q,xi=toarray(q)
    L=angularmomentum(r,q)
    S1,S2= spinmags(q,chi1,chi2)

    sigma0 = \
    ( L )**( 2 ) * ( ( -1 + q ) )**( 2 ) * ( ( 1 + q ) )**( 2 ) * ( ( ( \
    L )**( 2 ) * ( ( 1 + q ) )**( 2 ) + ( ( -1 + ( q )**( 2 ) ) * ( S1 + \
    -1 * S2 ) * ( S1 + S2 ) + 2 * L * q * xi ) ) )**( 2 ) * ( ( L )**( 6 \
    ) * ( q )**( 2 ) * ( ( 1 + q ) )**( 4 ) + ( 4 * ( L )**( 5 ) * ( q \
    )**( 2 ) * ( ( 1 + q ) )**( 4 ) * xi + ( -4 * L * q * ( ( 1 + q ) \
    )**( 3 ) * ( q * ( S1 )**( 2 ) + -1 * ( S2 )**( 2 ) ) * ( q * ( -5 + \
    4 * q ) * ( S1 )**( 2 ) + ( -4 + 5 * q ) * ( S2 )**( 2 ) ) * xi + ( \
    16 * L * ( q )**( 3 ) * ( 1 + q ) * ( q * ( S1 )**( 2 ) + ( S2 )**( 2 \
    ) ) * ( xi )**( 3 ) + ( 4 * ( ( 1 + q ) )**( 2 ) * ( ( -1 * q * ( S1 \
    )**( 2 ) + ( S2 )**( 2 ) ) )**( 2 ) * ( -1 * ( -1 + q ) * ( ( 1 + q ) \
    )**( 2 ) * ( q * ( S1 )**( 2 ) + -1 * ( S2 )**( 2 ) ) + ( q )**( 2 ) \
    * ( xi )**( 2 ) ) + ( 4 * ( L )**( 3 ) * q * ( ( 1 + q ) )**( 2 ) * \
    xi * ( ( 1 + q ) * ( q * ( 1 + ( 6 * q + -4 * ( q )**( 2 ) ) ) * ( S1 \
    )**( 2 ) + ( -4 + q * ( 6 + q ) ) * ( S2 )**( 2 ) ) + 4 * ( q )**( 2 \
    ) * ( xi )**( 2 ) ) + ( 2 * ( L )**( 4 ) * ( ( ( 1 + q ) )**( 4 ) * ( \
    ( q )**( 2 ) * ( 1 + -2 * ( -1 + q ) * q ) * ( S1 )**( 2 ) + ( -2 + q \
    * ( 2 + q ) ) * ( S2 )**( 2 ) ) + 2 * ( q )**( 2 ) * ( ( 1 + q ) )**( \
    2 ) * ( 1 + q * ( 4 + q ) ) * ( xi )**( 2 ) ) + ( L )**( 2 ) * ( ( ( \
    1 + q ) )**( 4 ) * ( ( q )**( 2 ) * ( 1 + -8 * ( -1 + q ) * q ) * ( \
    S1 )**( 4 ) + ( -2 * q * ( 10 + q * ( -19 + 10 * q ) ) * ( S1 )**( 2 \
    ) * ( S2 )**( 2 ) + ( -8 + q * ( 8 + q ) ) * ( S2 )**( 4 ) ) ) + ( -8 \
    * ( q )**( 2 ) * ( ( 1 + q ) )**( 2 ) * ( ( -4 + q ) * q * ( S1 )**( \
    2 ) + ( 1 + -4 * q ) * ( S2 )**( 2 ) ) * ( xi )**( 2 ) + 16 * ( q \
    )**( 4 ) * ( xi )**( 4 ) ) ) ) ) ) ) ) ) )

    sigma2 = \
    -4 * ( L )**( 2 ) * ( ( -1 + q ) )**( 2 ) * ( ( 1 + q ) )**( 4 ) * ( \
    ( L )**( 8 ) * ( q )**( 2 ) * ( ( 1 + q ) )**( 4 ) * ( 1 + q * ( 3 + \
    q ) ) + ( ( L )**( 7 ) * ( q )**( 2 ) * ( ( 1 + q ) )**( 4 ) * ( 3 + \
    q * ( 14 + 3 * q ) ) * xi + ( ( L )**( 5 ) * q * ( ( 1 + q ) )**( 2 ) \
    * xi * ( ( 1 + q ) * ( q * ( 2 + q * ( 18 + -1 * q * ( 3 + q ) * ( \
    -19 + 12 * q ) ) ) * ( S1 )**( 2 ) + ( -12 + q * ( -17 + q * ( 57 + 2 \
    * q * ( 9 + q ) ) ) ) * ( S2 )**( 2 ) ) + 4 * ( q )**( 2 ) * ( 3 + q \
    * ( 17 + 3 * q ) ) * ( xi )**( 2 ) ) + ( -1 * ( L )**( 6 ) * ( ( 1 + \
    q ) )**( 2 ) * ( ( ( 1 + q ) )**( 2 ) * ( ( q )**( 2 ) * ( -1 + q * ( \
    -7 + ( -1 + q ) * q * ( 9 + 2 * q ) ) ) * ( S1 )**( 2 ) + -1 * ( -2 + \
    q * ( -7 + q * ( 9 + q * ( 7 + q ) ) ) ) * ( S2 )**( 2 ) ) + -2 * ( q \
    )**( 2 ) * ( 1 + q * ( 14 + q * ( 32 + q * ( 14 + q ) ) ) ) * ( xi \
    )**( 2 ) ) + ( -1 * ( -1 + q ) * ( ( 1 + q ) )**( 2 ) * ( q * ( S1 \
    )**( 2 ) + -1 * ( S2 )**( 2 ) ) * ( ( -1 + q ) * ( ( 1 + q ) )**( 2 ) \
    * ( q * ( S1 )**( 2 ) + -1 * ( S2 )**( 2 ) ) * ( q * ( -5 + 2 * q ) * \
    ( S1 )**( 4 ) + ( 2 * ( 1 + ( q + ( q )**( 2 ) ) ) * ( S1 )**( 2 ) * \
    ( S2 )**( 2 ) + ( 2 + -5 * q ) * ( S2 )**( 4 ) ) ) + -2 * ( q )**( 2 \
    ) * ( ( -2 + q ) * q * ( S1 )**( 4 ) + ( ( 1 + ( q )**( 2 ) ) * ( S1 \
    )**( 2 ) * ( S2 )**( 2 ) + ( 1 + -2 * q ) * ( S2 )**( 4 ) ) ) * ( xi \
    )**( 2 ) ) + ( ( L )**( 4 ) * ( ( q )**( 2 ) * ( S1 )**( 4 ) + ( 6 * \
    ( q )**( 3 ) * ( S1 )**( 4 ) + ( 25 * ( q )**( 4 ) * ( S1 )**( 4 ) + \
    ( 55 * ( q )**( 5 ) * ( S1 )**( 4 ) + ( 49 * ( q )**( 6 ) * ( S1 )**( \
    4 ) + ( -8 * ( q )**( 7 ) * ( S1 )**( 4 ) + ( -45 * ( q )**( 8 ) * ( \
    S1 )**( 4 ) + ( -29 * ( q )**( 9 ) * ( S1 )**( 4 ) + ( -6 * ( q )**( \
    10 ) * ( S1 )**( 4 ) + ( -2 * ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( -6 * \
    q * ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( -32 * ( q )**( 2 ) * ( S1 )**( \
    2 ) * ( S2 )**( 2 ) + ( -58 * ( q )**( 3 ) * ( S1 )**( 2 ) * ( S2 \
    )**( 2 ) + ( 10 * ( q )**( 4 ) * ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( 80 \
    * ( q )**( 5 ) * ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( 10 * ( q )**( 6 ) \
    * ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( -58 * ( q )**( 7 ) * ( S1 )**( 2 \
    ) * ( S2 )**( 2 ) + ( -32 * ( q )**( 8 ) * ( S1 )**( 2 ) * ( S2 )**( \
    2 ) + ( -6 * ( q )**( 9 ) * ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( -2 * ( \
    q )**( 10 ) * ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( -6 * ( S2 )**( 4 ) + \
    ( -29 * q * ( S2 )**( 4 ) + ( -45 * ( q )**( 2 ) * ( S2 )**( 4 ) + ( \
    -8 * ( q )**( 3 ) * ( S2 )**( 4 ) + ( 49 * ( q )**( 4 ) * ( S2 )**( 4 \
    ) + ( 55 * ( q )**( 5 ) * ( S2 )**( 4 ) + ( 25 * ( q )**( 6 ) * ( S2 \
    )**( 4 ) + ( 6 * ( q )**( 7 ) * ( S2 )**( 4 ) + ( ( q )**( 8 ) * ( S2 \
    )**( 4 ) + ( -2 * ( q )**( 2 ) * ( ( 1 + q ) )**( 2 ) * ( ( -1 + q * \
    ( -4 + q * ( -40 + 3 * q * ( -5 + 3 * q ) ) ) ) * ( S1 )**( 2 ) + -1 \
    * ( -9 + q * ( 15 + q * ( 40 + q * ( 4 + q ) ) ) ) * ( S2 )**( 2 ) ) \
    * ( xi )**( 2 ) + 8 * ( q )**( 4 ) * ( 3 + q ) * ( 1 + 3 * q ) * ( xi \
    )**( 4 ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) \
    + ( -1 * ( L )**( 2 ) * ( -1 * ( q )**( 2 ) * ( S1 )**( 6 ) + ( -7 * \
    ( q )**( 3 ) * ( S1 )**( 6 ) + ( -9 * ( q )**( 4 ) * ( S1 )**( 6 ) + \
    ( 3 * ( q )**( 5 ) * ( S1 )**( 6 ) + ( 3 * ( q )**( 6 ) * ( S1 )**( 6 \
    ) + ( -9 * ( q )**( 7 ) * ( S1 )**( 6 ) + ( ( q )**( 8 ) * ( S1 )**( \
    6 ) + ( 13 * ( q )**( 9 ) * ( S1 )**( 6 ) + ( 6 * ( q )**( 10 ) * ( \
    S1 )**( 6 ) + ( 15 * q * ( S1 )**( 4 ) * ( S2 )**( 2 ) + ( 11 * ( q \
    )**( 2 ) * ( S1 )**( 4 ) * ( S2 )**( 2 ) + ( -35 * ( q )**( 3 ) * ( \
    S1 )**( 4 ) * ( S2 )**( 2 ) + ( 17 * ( q )**( 4 ) * ( S1 )**( 4 ) * ( \
    S2 )**( 2 ) + ( 105 * ( q )**( 5 ) * ( S1 )**( 4 ) * ( S2 )**( 2 ) + \
    ( 9 * ( q )**( 6 ) * ( S1 )**( 4 ) * ( S2 )**( 2 ) + ( -93 * ( q )**( \
    7 ) * ( S1 )**( 4 ) * ( S2 )**( 2 ) + ( -41 * ( q )**( 8 ) * ( S1 \
    )**( 4 ) * ( S2 )**( 2 ) + ( 8 * ( q )**( 9 ) * ( S1 )**( 4 ) * ( S2 \
    )**( 2 ) + ( 4 * ( q )**( 10 ) * ( S1 )**( 4 ) * ( S2 )**( 2 ) + ( 4 \
    * ( S1 )**( 2 ) * ( S2 )**( 4 ) + ( 8 * q * ( S1 )**( 2 ) * ( S2 )**( \
    4 ) + ( -41 * ( q )**( 2 ) * ( S1 )**( 2 ) * ( S2 )**( 4 ) + ( -93 * \
    ( q )**( 3 ) * ( S1 )**( 2 ) * ( S2 )**( 4 ) + ( 9 * ( q )**( 4 ) * ( \
    S1 )**( 2 ) * ( S2 )**( 4 ) + ( 105 * ( q )**( 5 ) * ( S1 )**( 2 ) * \
    ( S2 )**( 4 ) + ( 17 * ( q )**( 6 ) * ( S1 )**( 2 ) * ( S2 )**( 4 ) + \
    ( -35 * ( q )**( 7 ) * ( S1 )**( 2 ) * ( S2 )**( 4 ) + ( 11 * ( q \
    )**( 8 ) * ( S1 )**( 2 ) * ( S2 )**( 4 ) + ( 15 * ( q )**( 9 ) * ( S1 \
    )**( 2 ) * ( S2 )**( 4 ) + ( 6 * ( S2 )**( 6 ) + ( 13 * q * ( S2 )**( \
    6 ) + ( ( q )**( 2 ) * ( S2 )**( 6 ) + ( -9 * ( q )**( 3 ) * ( S2 \
    )**( 6 ) + ( 3 * ( q )**( 4 ) * ( S2 )**( 6 ) + ( 3 * ( q )**( 5 ) * \
    ( S2 )**( 6 ) + ( -9 * ( q )**( 6 ) * ( S2 )**( 6 ) + ( -7 * ( q )**( \
    7 ) * ( S2 )**( 6 ) + ( -1 * ( q )**( 8 ) * ( S2 )**( 6 ) + ( 2 * ( q \
    )**( 2 ) * ( ( 1 + q ) )**( 2 ) * ( q * ( -12 + q * ( 26 + ( -30 * q \
    + 9 * ( q )**( 2 ) ) ) ) * ( S1 )**( 4 ) + ( 2 * ( 1 + q * ( -5 + q * \
    ( 15 + ( -5 + q ) * q ) ) ) * ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( 9 + \
    -2 * q * ( 15 + q * ( -13 + 6 * q ) ) ) * ( S2 )**( 4 ) ) ) * ( xi \
    )**( 2 ) + -8 * ( q )**( 4 ) * ( ( 1 + q * ( -1 + 3 * q ) ) * ( S1 \
    )**( 2 ) + ( 3 + ( -1 + q ) * q ) * ( S2 )**( 2 ) ) * ( xi )**( 4 ) ) \
    ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) \
    ) ) ) + ( ( L )**( 3 ) * q * xi * ( -8 * ( q )**( 8 ) * ( S1 )**( 2 ) \
    * ( 3 * ( S1 )**( 2 ) + ( S2 )**( 2 ) ) + ( -8 * ( S2 )**( 2 ) * ( ( \
    S1 )**( 2 ) + 3 * ( S2 )**( 2 ) ) + ( q * ( 3 * ( S1 )**( 4 ) + ( -8 \
    * ( S1 )**( 2 ) * ( S2 )**( 2 ) + -51 * ( S2 )**( 4 ) ) ) + ( ( q \
    )**( 7 ) * ( -51 * ( S1 )**( 4 ) + ( -8 * ( S1 )**( 2 ) * ( S2 )**( 2 \
    ) + 3 * ( S2 )**( 4 ) ) ) + ( 2 * ( q )**( 2 ) * ( 7 * ( S1 )**( 4 ) \
    + ( 2 * ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( -1 * ( S2 )**( 4 ) + 4 * ( \
    ( S1 )**( 2 ) + ( S2 )**( 2 ) ) * ( xi )**( 2 ) ) ) ) + ( ( q )**( 6 \
    ) * ( -2 * ( S1 )**( 4 ) + ( 4 * ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( 14 \
    * ( S2 )**( 4 ) + 8 * ( ( S1 )**( 2 ) + ( S2 )**( 2 ) ) * ( xi )**( 2 \
    ) ) ) ) + ( ( q )**( 5 ) * ( 65 * ( S1 )**( 4 ) + ( -40 * ( S1 )**( 2 \
    ) * ( S2 )**( 2 ) + ( 31 * ( S2 )**( 4 ) + 4 * ( 19 * ( S1 )**( 2 ) + \
    3 * ( S2 )**( 2 ) ) * ( xi )**( 2 ) ) ) ) + ( ( q )**( 3 ) * ( 31 * ( \
    S1 )**( 4 ) + ( -40 * ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( 65 * ( S2 \
    )**( 4 ) + 4 * ( 3 * ( S1 )**( 2 ) + 19 * ( S2 )**( 2 ) ) * ( xi )**( \
    2 ) ) ) ) + 4 * ( q )**( 4 ) * ( 15 * ( S1 )**( 4 ) + ( -22 * ( S1 \
    )**( 2 ) * ( S2 )**( 2 ) + ( 15 * ( S2 )**( 4 ) + ( 18 * ( ( S1 )**( \
    2 ) + ( S2 )**( 2 ) ) * ( xi )**( 2 ) + 4 * ( xi )**( 4 ) ) ) ) ) ) ) \
    ) ) ) ) ) ) + -1 * L * q * ( 1 + q ) * xi * ( 8 * ( S1 )**( 2 ) * ( \
    S2 )**( 4 ) + ( 12 * ( S2 )**( 6 ) + ( 4 * ( q )**( 7 ) * ( 3 * ( S1 \
    )**( 6 ) + 2 * ( S1 )**( 4 ) * ( S2 )**( 2 ) ) + ( ( q )**( 6 ) * ( \
    -17 * ( S1 )**( 6 ) + ( -6 * ( S1 )**( 4 ) * ( S2 )**( 2 ) + 3 * ( S1 \
    )**( 2 ) * ( S2 )**( 4 ) ) ) + ( q * ( 3 * ( S1 )**( 4 ) * ( S2 )**( \
    2 ) + ( -6 * ( S1 )**( 2 ) * ( S2 )**( 4 ) + -17 * ( S2 )**( 6 ) ) ) \
    + ( -1 * ( q )**( 2 ) * ( 20 * ( S1 )**( 6 ) + ( -3 * ( S1 )**( 4 ) * \
    ( S2 )**( 2 ) + ( 22 * ( S1 )**( 2 ) * ( S2 )**( 4 ) + ( 21 * ( S2 \
    )**( 6 ) + 4 * ( S2 )**( 2 ) * ( 2 * ( S1 )**( 2 ) + 3 * ( S2 )**( 2 \
    ) ) * ( xi )**( 2 ) ) ) ) ) + ( ( q )**( 4 ) * ( 37 * ( S1 )**( 6 ) + \
    ( 3 * ( S1 )**( 4 ) * ( S2 )**( 2 ) + ( 11 * ( S1 )**( 2 ) * ( S2 \
    )**( 4 ) + ( 9 * ( S2 )**( 6 ) + 4 * ( 5 * ( S1 )**( 4 ) + ( 3 * ( S1 \
    )**( 2 ) * ( S2 )**( 2 ) + -3 * ( S2 )**( 4 ) ) ) * ( xi )**( 2 ) ) ) \
    ) ) + ( ( q )**( 3 ) * ( 9 * ( S1 )**( 6 ) + ( 11 * ( S1 )**( 4 ) * ( \
    S2 )**( 2 ) + ( 3 * ( S1 )**( 2 ) * ( S2 )**( 4 ) + ( 37 * ( S2 )**( \
    6 ) + 4 * ( -3 * ( S1 )**( 4 ) + ( 3 * ( S1 )**( 2 ) * ( S2 )**( 2 ) \
    + 5 * ( S2 )**( 4 ) ) ) * ( xi )**( 2 ) ) ) ) ) + -1 * ( q )**( 5 ) * \
    ( 21 * ( S1 )**( 6 ) + ( 20 * ( S2 )**( 6 ) + ( 2 * ( S1 )**( 4 ) * ( \
    11 * ( S2 )**( 2 ) + 6 * ( xi )**( 2 ) ) + ( S1 )**( 2 ) * ( -3 * ( \
    S2 )**( 4 ) + 8 * ( S2 )**( 2 ) * ( xi )**( 2 ) ) ) ) ) ) ) ) ) ) ) ) \
    ) ) ) ) ) ) ) ) )

    sigma4 = \
    2 * ( L )**( 2 ) * ( ( -1 + q ) )**( 2 ) * ( ( 1 + q ) )**( 4 ) * ( \
    ( L )**( 6 ) * ( q )**( 2 ) * ( ( 1 + q ) )**( 4 ) * ( 3 + q * ( 14 + \
    3 * q ) ) + ( -2 * ( -1 + q ) * ( ( 1 + q ) )**( 4 ) * ( q * ( S1 \
    )**( 2 ) + -1 * ( S2 )**( 2 ) ) * ( ( q )**( 2 ) * ( 10 + ( -8 + q ) \
    * q ) * ( S1 )**( 4 ) + ( -2 * q * ( 4 + q * ( -5 + 4 * q ) ) * ( S1 \
    )**( 2 ) * ( S2 )**( 2 ) + ( 1 + 2 * q * ( -4 + 5 * q ) ) * ( S2 )**( \
    4 ) ) ) + ( 6 * ( L )**( 5 ) * ( q )**( 2 ) * ( ( 1 + q ) )**( 4 ) * \
    ( 1 + q * ( 8 + q ) ) * xi + ( 2 * ( q )**( 2 ) * ( ( 1 + q ) )**( 2 \
    ) * ( ( q )**( 2 ) * ( 6 + ( -6 + q ) * q ) * ( S1 )**( 4 ) + ( -2 * \
    q * ( 3 + q * ( -5 + 3 * q ) ) * ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( 1 \
    + 6 * ( -1 + q ) * q ) * ( S2 )**( 4 ) ) ) * ( xi )**( 2 ) + ( 2 * ( \
    L )**( 3 ) * q * ( ( 1 + q ) )**( 2 ) * xi * ( ( 1 + q ) * ( q * ( 3 \
    + q * ( 31 + -2 * q * ( 6 + q ) * ( -3 + 2 * q ) ) ) * ( S1 )**( 2 ) \
    + ( -4 + q * ( -18 + q * ( 9 + q ) * ( 4 + 3 * q ) ) ) * ( S2 )**( 2 \
    ) ) + 4 * ( q )**( 2 ) * ( 1 + q * ( 11 + q ) ) * ( xi )**( 2 ) ) + ( \
    -2 * ( L )**( 4 ) * ( ( 1 + q ) )**( 2 ) * ( ( ( 1 + q ) )**( 2 ) * ( \
    ( S2 )**( 2 ) + q * ( q * ( -2 + q * ( -13 + q * ( -9 + q * ( 11 + q \
    ) ) ) ) * ( S1 )**( 2 ) + -1 * ( 11 + 2 * q ) * ( -1 + ( q + ( q )**( \
    2 ) ) ) * ( S2 )**( 2 ) ) ) + -1 * ( q )**( 2 ) * ( 1 + q * ( 26 + q \
    * ( 72 + q * ( 26 + q ) ) ) ) * ( xi )**( 2 ) ) + ( -2 * L * q * ( 1 \
    + q ) * xi * ( ( ( 1 + q ) )**( 2 ) * ( ( q )**( 2 ) * ( -30 + q * ( \
    39 + q * ( -19 + 4 * q ) ) ) * ( S1 )**( 4 ) + ( 3 * ( 1 + q ) * ( q \
    + ( q )**( 3 ) ) * ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( 4 + q * ( -19 + \
    ( 39 * q + -30 * ( q )**( 2 ) ) ) ) * ( S2 )**( 4 ) ) ) + -4 * ( q \
    )**( 2 ) * ( q * ( 3 + ( -1 + q ) * q ) * ( S1 )**( 2 ) + ( 1 + q * ( \
    -1 + 3 * q ) ) * ( S2 )**( 2 ) ) * ( xi )**( 2 ) ) + ( L )**( 2 ) * ( \
    3 * ( q )**( 2 ) * ( S1 )**( 4 ) + ( 36 * ( q )**( 3 ) * ( S1 )**( 4 \
    ) + ( 101 * ( q )**( 4 ) * ( S1 )**( 4 ) + ( 100 * ( q )**( 5 ) * ( \
    S1 )**( 4 ) + ( ( q )**( 6 ) * ( S1 )**( 4 ) + ( -68 * ( q )**( 7 ) * \
    ( S1 )**( 4 ) + ( -53 * ( q )**( 8 ) * ( S1 )**( 4 ) + ( -20 * ( q \
    )**( 9 ) * ( S1 )**( 4 ) + ( -4 * ( q )**( 10 ) * ( S1 )**( 4 ) + ( \
    -30 * q * ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( -74 * ( q )**( 2 ) * ( S1 \
    )**( 2 ) * ( S2 )**( 2 ) + ( -40 * ( q )**( 3 ) * ( S1 )**( 2 ) * ( \
    S2 )**( 2 ) + ( 26 * ( q )**( 4 ) * ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( \
    44 * ( q )**( 5 ) * ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( 26 * ( q )**( 6 \
    ) * ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( -40 * ( q )**( 7 ) * ( S1 )**( \
    2 ) * ( S2 )**( 2 ) + ( -74 * ( q )**( 8 ) * ( S1 )**( 2 ) * ( S2 \
    )**( 2 ) + ( -30 * ( q )**( 9 ) * ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( \
    -4 * ( S2 )**( 4 ) + ( -20 * q * ( S2 )**( 4 ) + ( -53 * ( q )**( 2 ) \
    * ( S2 )**( 4 ) + ( -68 * ( q )**( 3 ) * ( S2 )**( 4 ) + ( ( q )**( 4 \
    ) * ( S2 )**( 4 ) + ( 100 * ( q )**( 5 ) * ( S2 )**( 4 ) + ( 101 * ( \
    q )**( 6 ) * ( S2 )**( 4 ) + ( 36 * ( q )**( 7 ) * ( S2 )**( 4 ) + ( \
    3 * ( q )**( 8 ) * ( S2 )**( 4 ) + ( -4 * ( q )**( 2 ) * ( ( 1 + q ) \
    )**( 2 ) * ( q * ( -12 + q * ( -8 + ( -8 + q ) * q ) ) * ( S1 )**( 2 \
    ) + ( 1 + -4 * q * ( 2 + q * ( 2 + 3 * q ) ) ) * ( S2 )**( 2 ) ) * ( \
    xi )**( 2 ) + 8 * ( q )**( 4 ) * ( 1 + q * ( 4 + q ) ) * ( xi )**( 4 \
    ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) \
    )

    sigma6 = \
    -4 * ( L )**( 2 ) * ( ( -1 + q ) )**( 2 ) * q * ( ( 1 + q ) )**( 6 ) \
    * ( ( L )**( 4 ) * q * ( ( 1 + q ) )**( 2 ) * ( 1 + q * ( 8 + q ) ) + \
    ( ( ( 1 + q ) )**( 2 ) * ( ( q )**( 2 ) * ( 10 + 3 * ( -4 + q ) * q ) \
    * ( S1 )**( 4 ) + ( -2 * q * ( 6 + q * ( -11 + 6 * q ) ) * ( S1 )**( \
    2 ) * ( S2 )**( 2 ) + ( 3 + 2 * q * ( -6 + 5 * q ) ) * ( S2 )**( 4 ) \
    ) ) + ( ( L )**( 3 ) * q * ( ( 1 + q ) )**( 2 ) * ( 1 + q * ( 18 + q \
    ) ) * xi + ( -1 * L * q * ( 1 + q ) * ( q * ( -20 + q * ( 3 + q ) ) * \
    ( S1 )**( 2 ) + ( 1 + ( 3 + -20 * q ) * q ) * ( S2 )**( 2 ) ) * xi + \
    ( -2 * ( q )**( 2 ) * ( ( -2 + q ) * q * ( S1 )**( 2 ) + ( 1 + -2 * q \
    ) * ( S2 )**( 2 ) ) * ( xi )**( 2 ) + ( 4 * L * ( q )**( 3 ) * ( xi \
    )**( 3 ) + ( L )**( 2 ) * ( ( ( 1 + q ) )**( 2 ) * ( -1 * q * ( -1 + \
    q * ( -13 + ( q + 5 * ( q )**( 2 ) ) ) ) * ( S1 )**( 2 ) + ( -5 + q * \
    ( -1 + q * ( 13 + q ) ) ) * ( S2 )**( 2 ) ) + 4 * ( q )**( 2 ) * ( 2 \
    + q * ( 7 + 2 * q ) ) * ( xi )**( 2 ) ) ) ) ) ) ) )

    sigma8 = \
    ( L )**( 2 ) * ( ( -1 + q ) )**( 2 ) * ( q )**( 2 ) * ( ( 1 + q ) \
    )**( 6 ) * ( ( ( 1 + q ) )**( 2 ) * ( ( L )**( 2 ) * ( 1 + q * ( 18 + \
    q ) ) + ( 4 * ( 5 + -3 * q ) * q * ( S1 )**( 2 ) + 4 * ( -3 + 5 * q ) \
    * ( S2 )**( 2 ) ) ) + ( 20 * L * q * ( ( 1 + q ) )**( 2 ) * xi + 4 * \
    ( q )**( 2 ) * ( xi )**( 2 ) ) )

    sigma10 = \
    -4 * ( L )**( 2 ) * ( ( -1 + q ) )**( 2 ) * ( q )**( 3 ) * ( ( 1 + q \
    ) )**( 8 )

    return np.array([sigma10, sigma8, sigma6, sigma4, sigma2, sigma0])


def wraproots(coefficientfunction, *args,**kwargs):
    """
    Find roots of a polynomial given coefficients. Wrapper of numpy.roots.

    Parameters
    ----------
    coefficientfunction: callable
        Function returnin the polynomial coefficients ordered from highest to lowest degree.
    *args, **kwargs:
        Parameters of `coefficientfunction`.

    Returns
    -------
    sols: array
        Roots of the polynomial, ordered according to their real part. Complex roots are masked with nans.
    """

    coeffs= coefficientfunction(*args,**kwargs)

    if np.ndim(coeffs)==1:
        sols = np.sort_complex(np.roots(coeffs))
    else:
        sols = np.array([np.sort_complex(np.roots(x)) for x in coeffs.T])

    sols = np.real(np.where(np.isreal(sols),sols,np.nan))

    return sols


def Jresonances(r,xi,q,chi1,chi2):
    """
    Total angular momentum of the two spin-orbit resonances.

    Parameters
    ----------
    r: float
        Binary separation.
    xi: float
        Effective spin
    q: float
        Mass ratio: 0 <= q <= 1.
    chi1: float
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.
    chi2: float
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.

    Returns
    -------
    Jmin: float
        Spin-orbit resonance that minimizes J (DeltaPhi=pi)
    Jmax: float
        Spin-orbit resonance that minimizes J (DeltaPhi=pi)
    """

    # The good solutions are the last two. That's because the discriminant quintic asymptotes to -infinity and the physical region is when it's positive


    J2roots= wraproots(Jdiscriminant_coefficients,r,xi,q,chi1,chi2)

    if np.ndim(J2roots)==1:
        Jmin,Jmax = J2roots[~np.isnan(J2roots)][-2:]**0.5
    else:
        Jmin,Jmax = np.array([x[~np.isnan(x)][-2:]**0.5 for x in J2roots]).T

    return np.array([Jmin,Jmax])


def Jlimits(r=None,xi=None,q=None,chi1=None,chi2=None):
    """
    Limits on the magnitude of the total angular momentum. The contraints considered depend on the inputs provided.
        - If r, q, chi1, and chi2 are provided, enforce J=L+S1+S2.
        - If r, xi, q, chi1, and chi2 are provided, the limits are given by the two spin-orbit resonances.

    Parameters
    ----------
    r: float, optional
        Binary separation.
    xi: float, optional
        Effective spin
    q: float
        Mass ratio: 0 <= q <= 1.
    chi1: float, optional
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.
    chi2: float, optional
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.
    conincident: boolean, optional
        If True, assume that the input is a spin-orbit resonance and return repeated roots

    Returns
    -------
    Jmin:
        Minimum value of the total angular momentum.
    Jmax:
        Maximum value of the total angular momentum.
    """

    if r is not None and xi is None and q is not None and chi1 is not None and chi2 is not None:
        Jmin,Jmax = Jlimits_LS1S2(r,q,chi1,chi2)


    elif r is not None and xi is not None and q is not None and chi1 is not None and chi2 is not None:
        #TODO: Assert that the xi values are compatible with q,chi1,chi2 (either explicitely or with a generic 'limits_check' function)
        Jmin,Jmax = Jresonances(r,xi,q,chi1,chi2)

    else:
        raise TypeError

    return np.array([Jmin,Jmax])


def xilimits_definition(q,chi1,chi2):
    """
    Limits on the effective spin based only on the definition xi = (1+q)S1.L + (1+1/q)S2.L

    Parameters
    ----------
    q: float
        Mass ratio: 0 <= q <= 1.
    chi1: float
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.
    chi2: float
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.

    Returns
    -------
    ximin:
        Minimum value of the effective spin.
    ximax:
        Maximum value of the effective spin.
    """

    q=toarray(q)
    S1,S2 = spinmags(q,chi1,chi2)
    xilim = (1+q)*S1 + (1+1/q)*S2

    return np.array([-xilim,xilim])


def xidiscriminant_coefficients(J,r,q,chi1,chi2):
    """
    Coefficients of the 6-degree equation in xi that defines the spin-orbit resonances.

    Parameters
    ----------
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    q: float
        Mass ratio: 0 <= q <= 1.
    chi1: float
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.
    chi2: float
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.

    Returns
    -------
    sigma6:
        Coefficient of xi^6.
    sigma5:
        Coefficient of xi^5.
    sigma4:
        Coefficient of xi^4.
    sigma3:
        Coefficient of xi^3.
    sigma2:
        Coefficient of xi^2.
    sigma1:
        Coefficient of xi^1.
    sigma0:
        Coefficient of xi^0.
    """


    J,q=toarray(J,q)
    L=angularmomentum(r,q)
    S1,S2= spinmags(q,chi1,chi2)

    sigma0 = \
    -1 * ( L )**( 2 ) * ( ( 1 + q ) )**( 6 ) * ( ( -1 + ( q )**( 2 ) ) \
    )**( 2 ) * ( 4 * ( J )**( 10 ) * ( q )**( 3 ) + ( -1 * ( J )**( 8 ) * \
    ( q )**( 2 ) * ( ( L )**( 2 ) * ( 1 + q * ( 18 + q ) ) + ( 4 * ( 5 + \
    -3 * q ) * q * ( S1 )**( 2 ) + 4 * ( -3 + 5 * q ) * ( S2 )**( 2 ) ) ) \
    + ( 4 * ( J )**( 6 ) * q * ( q * ( ( L )**( 4 ) * ( 1 + q * ( 8 + q ) \
    ) + ( -1 * ( L )**( 2 ) * ( -1 + q * ( -13 + ( q + 5 * ( q )**( 2 ) ) \
    ) ) * ( S1 )**( 2 ) + q * ( 10 + 3 * ( -4 + q ) * q ) * ( S1 )**( 4 ) \
    ) ) + ( ( ( L )**( 2 ) * ( -5 + q * ( -1 + q * ( 13 + q ) ) ) + -2 * \
    q * ( 6 + q * ( -11 + 6 * q ) ) * ( S1 )**( 2 ) ) * ( S2 )**( 2 ) + ( \
    3 + 2 * q * ( -6 + 5 * q ) ) * ( S2 )**( 4 ) ) ) + ( -1 * ( ( q )**( \
    2 ) * ( ( ( L )**( 2 ) + ( S1 )**( 2 ) ) )**( 2 ) * ( ( L )**( 2 ) + \
    -4 * ( -1 + q ) * q * ( S1 )**( 2 ) ) + ( 2 * ( ( L )**( 4 ) * ( -2 + \
    q * ( 2 + q ) ) + ( ( L )**( 2 ) * q * ( -10 + ( 19 + -10 * q ) * q ) \
    * ( S1 )**( 2 ) + 6 * ( -1 + q ) * ( q )**( 2 ) * ( S1 )**( 4 ) ) ) * \
    ( S2 )**( 2 ) + ( ( ( L )**( 2 ) * ( -8 + q * ( 8 + q ) ) + -12 * ( \
    -1 + q ) * q * ( S1 )**( 2 ) ) * ( S2 )**( 4 ) + 4 * ( -1 + q ) * ( \
    S2 )**( 6 ) ) ) ) * ( ( ( L )**( 2 ) * ( 1 + q ) + ( -1 + q ) * ( S1 \
    + -1 * S2 ) * ( S1 + S2 ) ) )**( 2 ) + ( 2 * ( J )**( 4 ) * ( -1 * ( \
    L )**( 6 ) * ( q )**( 2 ) * ( 3 + q * ( 14 + 3 * q ) ) + ( 2 * ( L \
    )**( 4 ) * ( ( q )**( 2 ) * ( -2 + q * ( -13 + q * ( -9 + q * ( 11 + \
    q ) ) ) ) * ( S1 )**( 2 ) + ( 1 + -1 * q * ( 11 + 2 * q ) * ( -1 + ( \
    q + ( q )**( 2 ) ) ) ) * ( S2 )**( 2 ) ) + ( ( L )**( 2 ) * ( ( q \
    )**( 2 ) * ( -3 + q * ( -24 + q * ( 13 + 4 * q * ( 1 + q ) ) ) ) * ( \
    S1 )**( 4 ) + ( 2 * q * ( 15 + q * ( -23 + q * ( 22 + q * ( -23 + 15 \
    * q ) ) ) ) * ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( 4 + q * ( 4 + q * ( \
    13 + -3 * q * ( 8 + q ) ) ) ) * ( S2 )**( 4 ) ) ) + 2 * ( -1 + q ) * \
    ( ( q )**( 3 ) * ( 10 + ( -8 + q ) * q ) * ( S1 )**( 6 ) + ( -9 * ( q \
    )**( 2 ) * ( 2 + ( -2 + q ) * q ) * ( S1 )**( 4 ) * ( S2 )**( 2 ) + ( \
    9 * q * ( 1 + 2 * ( -1 + q ) * q ) * ( S1 )**( 2 ) * ( S2 )**( 4 ) + \
    ( -1 + 2 * ( 4 + -5 * q ) * q ) * ( S2 )**( 6 ) ) ) ) ) ) ) + 4 * ( J \
    )**( 2 ) * ( ( L )**( 8 ) * ( q )**( 2 ) * ( 1 + q * ( 3 + q ) ) + ( \
    ( L )**( 6 ) * ( ( q )**( 2 ) * ( 1 + q * ( 7 + -1 * ( -1 + q ) * q * \
    ( 9 + 2 * q ) ) ) * ( S1 )**( 2 ) + ( -2 + q * ( -7 + q * ( 9 + q * ( \
    7 + q ) ) ) ) * ( S2 )**( 2 ) ) + ( -1 * ( ( -1 + q ) )**( 2 ) * ( ( \
    -1 * q * ( S1 )**( 2 ) + ( S2 )**( 2 ) ) )**( 2 ) * ( q * ( -5 + 2 * \
    q ) * ( S1 )**( 4 ) + ( 2 * ( 1 + ( q + ( q )**( 2 ) ) ) * ( S1 )**( \
    2 ) * ( S2 )**( 2 ) + ( 2 + -5 * q ) * ( S2 )**( 4 ) ) ) + ( ( L )**( \
    4 ) * ( ( q )**( 2 ) * ( 1 + q * ( 2 + -1 * ( -1 + q ) * q * ( 11 + 6 \
    * q ) ) ) * ( S1 )**( 4 ) + ( 2 * ( -1 + ( q + ( -14 * ( q )**( 2 ) + \
    ( 25 * ( q )**( 3 ) + ( -14 * ( q )**( 4 ) + ( ( q )**( 5 ) + -1 * ( \
    q )**( 6 ) ) ) ) ) ) ) * ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( -6 + q * ( \
    -5 + q * ( 11 + q * ( 2 + q ) ) ) ) * ( S2 )**( 4 ) ) ) + ( L )**( 2 \
    ) * ( -1 + q ) * ( -1 * ( q )**( 2 ) * ( 1 + q * ( 4 + q * ( -5 + 6 * \
    q ) ) ) * ( S1 )**( 6 ) + ( q * ( 15 + q * ( -34 + q * ( 37 + -4 * ( \
    -1 + q ) * q ) ) ) * ( S1 )**( 4 ) * ( S2 )**( 2 ) + ( ( 4 + -1 * q * \
    ( 4 + q * ( 37 + q * ( -34 + 15 * q ) ) ) ) * ( S1 )**( 2 ) * ( S2 \
    )**( 4 ) + ( 6 + ( -1 + q ) * q * ( 5 + q ) ) * ( S2 )**( 6 ) ) ) ) ) \
    ) ) ) ) ) ) ) )

    sigma1 = \
    -4 * ( L )**( 3 ) * ( ( -1 + q ) )**( 2 ) * q * ( ( 1 + q ) )**( 7 ) \
    * ( -5 * ( J )**( 8 ) * ( q )**( 2 ) * ( 1 + q ) + ( ( J )**( 6 ) * q \
    * ( ( L )**( 2 ) * ( 1 + q ) * ( 1 + q * ( 18 + q ) ) + ( -1 * q * ( \
    -20 + q * ( 3 + q ) ) * ( S1 )**( 2 ) + ( -1 + q * ( -3 + 20 * q ) ) \
    * ( S2 )**( 2 ) ) ) + ( ( J )**( 4 ) * ( q * ( -3 * ( L )**( 4 ) * ( \
    1 + q ) * ( 1 + q * ( 8 + q ) ) + ( ( L )**( 2 ) * ( -3 + q * ( -31 + \
    2 * q * ( 6 + q ) * ( -3 + 2 * q ) ) ) * ( S1 )**( 2 ) + q * ( -30 + \
    q * ( 39 + q * ( -19 + 4 * q ) ) ) * ( S1 )**( 4 ) ) ) + ( ( ( L )**( \
    2 ) * ( 4 + -1 * q * ( -18 + q * ( 9 + q ) * ( 4 + 3 * q ) ) ) + 3 * \
    q * ( 1 + q ) * ( 1 + ( q )**( 2 ) ) * ( S1 )**( 2 ) ) * ( S2 )**( 2 \
    ) + ( 4 + q * ( -19 + ( 39 * q + -30 * ( q )**( 2 ) ) ) ) * ( S2 )**( \
    4 ) ) ) + ( -1 * ( ( L )**( 2 ) * ( 1 + q ) + ( -1 + q ) * ( S1 + -1 \
    * S2 ) * ( S1 + S2 ) ) * ( ( L )**( 6 ) * q * ( 1 + q * ( 3 + q ) ) + \
    ( ( L )**( 4 ) * ( ( q )**( 2 ) * ( 9 + ( 7 + -8 * q ) * q ) * ( S1 \
    )**( 2 ) + ( -8 + q * ( 7 + 9 * q ) ) * ( S2 )**( 2 ) ) + ( -1 * ( -1 \
    + q ) * ( q * ( S1 )**( 2 ) + -1 * ( S2 )**( 2 ) ) * ( q * ( -5 + 8 * \
    q ) * ( S1 )**( 4 ) + ( 2 * ( -2 + ( q + -2 * ( q )**( 2 ) ) ) * ( S1 \
    )**( 2 ) * ( S2 )**( 2 ) + ( 8 + -5 * q ) * ( S2 )**( 4 ) ) ) + ( L \
    )**( 2 ) * ( q * ( -1 + ( q + ( 19 * ( q )**( 2 ) + -16 * ( q )**( 3 \
    ) ) ) ) * ( S1 )**( 4 ) + ( 2 * ( 2 + q * ( -15 + q * ( 23 + q * ( \
    -15 + 2 * q ) ) ) ) * ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( -16 + q * ( \
    19 + ( q + -1 * ( q )**( 2 ) ) ) ) * ( S2 )**( 4 ) ) ) ) ) ) + ( J \
    )**( 2 ) * ( ( L )**( 6 ) * q * ( 1 + q ) * ( 3 + q * ( 14 + 3 * q ) \
    ) + ( ( L )**( 4 ) * ( q * ( 2 + q * ( 18 + -1 * q * ( 3 + q ) * ( \
    -19 + 12 * q ) ) ) * ( S1 )**( 2 ) + ( -12 + q * ( -17 + q * ( 57 + 2 \
    * q * ( 9 + q ) ) ) ) * ( S2 )**( 2 ) ) + ( ( L )**( 2 ) * ( q * ( 3 \
    + q * ( 5 + q * ( 7 + 3 * ( 7 + -8 * q ) * q ) ) ) * ( S1 )**( 4 ) + \
    ( -4 * ( 1 + q ) * ( 2 + q * ( -6 + q * ( 11 + 2 * ( -3 + q ) * q ) ) \
    ) * ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( -24 + q * ( 21 + q * ( 7 + q * \
    ( 5 + 3 * q ) ) ) ) * ( S2 )**( 4 ) ) ) + -1 * ( -1 + q ) * ( ( q \
    )**( 2 ) * ( 20 + q * ( -29 + 12 * q ) ) * ( S1 )**( 6 ) + ( q * ( -3 \
    + 2 * ( q )**( 2 ) * ( -7 + 4 * q ) ) * ( S1 )**( 4 ) * ( S2 )**( 2 ) \
    + ( ( -8 + ( 14 * q + 3 * ( q )**( 3 ) ) ) * ( S1 )**( 2 ) * ( S2 \
    )**( 4 ) + ( -12 + ( 29 + -20 * q ) * q ) * ( S2 )**( 6 ) ) ) ) ) ) ) \
    ) ) ) )

    sigma2 = \
    -4 * ( L )**( 2 ) * ( ( -1 + q ) )**( 2 ) * ( q )**( 2 ) * ( ( 1 + q \
    ) )**( 6 ) * ( -1 * ( J )**( 8 ) * ( q )**( 2 ) + ( -1 * ( L )**( 8 ) \
    * ( 1 + q * ( 1 + q ) * ( 10 + q * ( 9 + q ) ) ) + ( -1 * ( ( -1 + q \
    ) )**( 2 ) * ( ( S1 + -1 * S2 ) )**( 2 ) * ( ( S1 + S2 ) )**( 2 ) * ( \
    ( -1 * q * ( S1 )**( 2 ) + ( S2 )**( 2 ) ) )**( 2 ) + ( 2 * ( J )**( \
    6 ) * q * ( 2 * ( L )**( 2 ) * ( 2 + q * ( 7 + 2 * q ) ) + ( -1 * ( \
    -2 + q ) * q * ( S1 )**( 2 ) + ( -1 + 2 * q ) * ( S2 )**( 2 ) ) ) + ( \
    2 * ( L )**( 6 ) * ( ( 1 + 2 * ( q )**( 2 ) * ( -11 + q * ( -7 + 5 * \
    q ) ) ) * ( S1 )**( 2 ) + ( 10 + q * ( -14 + ( -22 * q + ( q )**( 3 ) \
    ) ) ) * ( S2 )**( 2 ) ) + ( ( L )**( 4 ) * ( ( -1 + 6 * q * ( 3 + ( q \
    )**( 2 ) * ( -12 + 7 * q ) ) ) * ( S1 )**( 4 ) + ( -2 * ( 9 + q * ( \
    -33 + q * ( 35 + ( -33 * q + 9 * ( q )**( 2 ) ) ) ) ) * ( S1 )**( 2 ) \
    * ( S2 )**( 2 ) + -1 * ( -42 + ( 72 * q + ( -18 * ( q )**( 3 ) + ( q \
    )**( 4 ) ) ) ) * ( S2 )**( 4 ) ) ) + ( 2 * ( L )**( 2 ) * ( -1 + q ) \
    * ( 2 * q * ( 2 + q * ( -8 + 5 * q ) ) * ( S1 )**( 6 ) + ( ( -1 + ( q \
    + ( 15 * ( q )**( 2 ) + -9 * ( q )**( 3 ) ) ) ) * ( S1 )**( 4 ) * ( \
    S2 )**( 2 ) + ( ( 9 + q * ( -15 + ( -1 + q ) * q ) ) * ( S1 )**( 2 ) \
    * ( S2 )**( 4 ) + -2 * ( 5 + 2 * ( -4 + q ) * q ) * ( S2 )**( 6 ) ) ) \
    ) + ( ( J )**( 4 ) * ( -1 * ( L )**( 4 ) * ( 1 + q * ( 26 + q * ( 72 \
    + q * ( 26 + q ) ) ) ) + ( -1 * ( q )**( 2 ) * ( 6 + ( -6 + q ) * q ) \
    * ( S1 )**( 4 ) + ( 2 * q * ( 3 + q * ( -5 + 3 * q ) ) * ( S1 )**( 2 \
    ) * ( S2 )**( 2 ) + ( ( -1 + -6 * ( -1 + q ) * q ) * ( S2 )**( 4 ) + \
    2 * ( L )**( 2 ) * ( q * ( -12 + q * ( -8 + ( -8 + q ) * q ) ) * ( S1 \
    )**( 2 ) + ( 1 + -4 * q * ( 2 + q * ( 2 + 3 * q ) ) ) * ( S2 )**( 2 ) \
    ) ) ) ) ) + 2 * ( J )**( 2 ) * ( ( L )**( 6 ) * ( 1 + q * ( 14 + q * \
    ( 32 + q * ( 14 + q ) ) ) ) + ( ( L )**( 4 ) * ( ( 1 + q * ( 4 + q * \
    ( 40 + 3 * ( 5 + -3 * q ) * q ) ) ) * ( S1 )**( 2 ) + ( -9 + q * ( 15 \
    + q * ( 40 + q * ( 4 + q ) ) ) ) * ( S2 )**( 2 ) ) + ( ( -1 + q ) * ( \
    q * ( S1 )**( 2 ) + -1 * ( S2 )**( 2 ) ) * ( ( -2 + q ) * q * ( S1 \
    )**( 4 ) + ( ( 1 + ( q )**( 2 ) ) * ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( \
    1 + -2 * q ) * ( S2 )**( 4 ) ) ) + ( L )**( 2 ) * ( q * ( 12 + q * ( \
    -26 + ( 30 * q + -9 * ( q )**( 2 ) ) ) ) * ( S1 )**( 4 ) + ( -2 * ( 1 \
    + q * ( -5 + q * ( 15 + ( -5 + q ) * q ) ) ) * ( S1 )**( 2 ) * ( S2 \
    )**( 2 ) + ( -9 + 2 * q * ( 15 + q * ( -13 + 6 * q ) ) ) * ( S2 )**( \
    4 ) ) ) ) ) ) ) ) ) ) ) ) ) )

    sigma3 = \
    -16 * ( L )**( 3 ) * ( ( -1 + q ) )**( 2 ) * ( q )**( 3 ) * ( ( 1 + q \
    ) )**( 5 ) * ( ( J )**( 6 ) * q * ( 1 + q ) + ( -1 * ( L )**( 6 ) * ( \
    1 + q ) * ( 2 + q * ( 7 + 2 * q ) ) + ( -1 * ( J )**( 4 ) * ( ( L \
    )**( 2 ) * ( 1 + q ) * ( 1 + q * ( 11 + q ) ) + ( q * ( 3 + ( -1 + q \
    ) * q ) * ( S1 )**( 2 ) + ( 1 + q * ( -1 + 3 * q ) ) * ( S2 )**( 2 ) \
    ) ) + ( ( L )**( 4 ) * ( ( 3 + q * ( -5 + q * ( -19 + 2 * q ) ) ) * ( \
    S1 )**( 2 ) + ( 2 + q * ( -19 + q * ( -5 + 3 * q ) ) ) * ( S2 )**( 2 \
    ) ) + ( -1 * ( -1 + q ) * ( S1 + -1 * S2 ) * ( S1 + S2 ) * ( q * ( -1 \
    + 2 * q ) * ( S1 )**( 4 ) + ( -1 * ( 1 + ( q )**( 2 ) ) * ( S1 )**( 2 \
    ) * ( S2 )**( 2 ) + -1 * ( -2 + q ) * ( S2 )**( 4 ) ) ) + ( ( L )**( \
    2 ) * ( ( -1 + q * ( 11 + q * ( -15 + 2 * q ) ) ) * ( S1 )**( 4 ) + ( \
    ( 2 + ( q + ( ( q )**( 2 ) + 2 * ( q )**( 3 ) ) ) ) * ( S1 )**( 2 ) * \
    ( S2 )**( 2 ) + -1 * ( -2 + q * ( 15 + ( -11 + q ) * q ) ) * ( S2 \
    )**( 4 ) ) ) + ( J )**( 2 ) * ( ( L )**( 4 ) * ( 1 + q ) * ( 3 + q * \
    ( 17 + 3 * q ) ) + ( q * ( 3 + q * ( -5 + 3 * q ) ) * ( S1 )**( 4 ) + \
    ( ( -2 + q ) * ( 1 + q ) * ( -1 + 2 * q ) * ( S1 )**( 2 ) * ( S2 )**( \
    2 ) + ( ( 3 + q * ( -5 + 3 * q ) ) * ( S2 )**( 4 ) + ( L )**( 2 ) * ( \
    ( 2 + ( q + ( 17 * ( q )**( 2 ) + 2 * ( q )**( 3 ) ) ) ) * ( S1 )**( \
    2 ) + ( 2 + q * ( 17 + ( q + 2 * ( q )**( 2 ) ) ) ) * ( S2 )**( 2 ) ) \
    ) ) ) ) ) ) ) ) ) )

    sigma4 = \
    16 * ( L )**( 4 ) * ( ( -1 + q ) )**( 2 ) * ( q )**( 4 ) * ( ( 1 + q \
    ) )**( 4 ) * ( ( J )**( 4 ) * ( 1 + q * ( 4 + q ) ) + ( 2 * ( L )**( \
    4 ) * ( 3 + q * ( 7 + 3 * q ) ) + ( ( 1 + 6 * ( -1 + q ) * q ) * ( S1 \
    )**( 4 ) + ( -2 * ( 3 + q * ( -5 + 3 * q ) ) * ( S1 )**( 2 ) * ( S2 \
    )**( 2 ) + ( ( 6 + ( -6 + q ) * q ) * ( S2 )**( 4 ) + ( 2 * ( L )**( \
    2 ) * ( ( -3 + ( 6 * q + 4 * ( q )**( 2 ) ) ) * ( S1 )**( 2 ) + ( 4 + \
    -3 * ( -2 + q ) * q ) * ( S2 )**( 2 ) ) + -2 * ( J )**( 2 ) * ( ( L \
    )**( 2 ) * ( 3 + q ) * ( 1 + 3 * q ) + ( ( 1 + q * ( -1 + 3 * q ) ) * \
    ( S1 )**( 2 ) + ( 3 + ( -1 + q ) * q ) * ( S2 )**( 2 ) ) ) ) ) ) ) ) )

    sigma5 = \
    -64 * ( L )**( 5 ) * ( ( -1 + q ) )**( 2 ) * ( q )**( 5 ) * ( ( 1 + \
    q ) )**( 3 ) * ( ( J )**( 2 ) * ( 1 + q ) + ( -2 * ( L )**( 2 ) * ( 1 \
    + q ) + ( ( 1 + -2 * q ) * ( S1 )**( 2 ) + ( -2 + q ) * ( S2 )**( 2 ) \
    ) ) )

    sigma6 = \
    64 * ( L )**( 6 ) * ( q )**( 6 ) * ( ( -1 + ( q )**( 2 ) ) )**( 2 )

    return np.array([sigma6, sigma5, sigma4, sigma3, sigma2, sigma1, sigma0])


def xiresonances(J,r,q,chi1,chi2):
    """
    Total angular momentum of the two spin-orbit resonances.

    Parameters
    ----------
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    q: float
        Mass ratio: 0 <= q <= 1.
    chi1: float
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.
    chi2: float
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.

    Returns
    -------
    ximin: float
        Spin-orbit resonance that minimizes xi (either DeltaPhi=0 or DeltaPhi=pi)
    ximax: float
        Spin-orbit resonance that minimizes xi (always DeltaPhi=pi)
    """

    #Altough there are 6 solutions in general, we know that only two can lie between Smin and Smax.

    Smin,Smax = Slimits_LJS1S2(J,r,q,chi1,chi2)
    xiroots= wraproots(xidiscriminant_coefficients,J,r,q,chi1,chi2)

    def _compute(Smin,Smax,J,r,xiroots,q,chi1,chi2):
        Sroots = np.array([Slimits_plusminus(J,r,x,q,chi1,chi2,coincident=True)[0] for x in xiroots])
        with np.errstate(invalid='ignore'):
            xires = xiroots[np.logical_and(Sroots>Smin, Sroots<Smax)]
        return xires

    if np.ndim(xiroots)==1:
        ximin,ximax =_compute(Smin,Smax,J,r,xiroots,q,chi1,chi2)
    else:
        ximin,ximax =np.array(list(map(_compute, Smin,Smax,J,r,xiroots,q,chi1,chi2))).T

    return np.array([ximin,ximax])


def xilimits(J=None,r=None,q=None,chi1=None,chi2=None):
    """
    Limits on the projected effective spin. The contraints considered depend on the inputs provided.
        - If q, chi1, and chi2 are provided, enforce xi = (1+q)S1.L + (1+1/q)S2.L.
        - If J, r, q, chi1, and chi2 are provided, the limits are given by the two spin-orbit resonances.

    Parameters
    ----------
    J: float, optional
        Magnitude of the total angular momentum.
    r: float, optional
        Binary separation.
    q: float
        Mass ratio: 0 <= q <= 1.
    chi1: float, optional
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.
    chi2: float, optional
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.
    conincident: boolean, optional
        If True, assume that the input is a spin-orbit resonance and return repeated roots

    Returns
    -------
    ximin:
        Minimum value of the effective spin.
    ximax:
        Maximum value of the effective spin.
    """

    if J is None and r is None and q is not None and chi1 is not None and chi2 is not None:
        ximin,ximax = xilimits_definition(q,chi1,chi2)


    elif J is not None and r is not None and q is not None and chi1 is not None and chi2 is not None:
        #TODO: Assert that the xi values are compatible with q,chi1,chi2 (either explicitely or with a generic 'limits_check' function)
        ximin,ximax = xiresonances(J,r,q,chi1,chi2)

    else:
        raise TypeError

    return np.array([ximin,ximax])


def Slimits_S1S2(q,chi1,chi2):
    """
    Limits on the total spin magnitude due to the vector relation S=S1+S2

    Parameters
    ----------
    q: float
        Mass ratio: 0 <= q <= 1.
    chi1: float
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.
    chi2: float
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.

    Returns
    -------
    Smin:
        Minimum value of the total spin.
    Smax:
        Maximum value of the total spin.
    """

    S1,S2= spinmags(q,chi1,chi2)
    Smin = np.abs(S1-S2)
    Smax = S1+S2

    return np.array([Smin,Smax])


def Slimits_LJ(J,r,q):
    """
    Limits on the total spin magnitude due to the vector relation S=J-L

    Parameters
    ----------
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    q: float
        Mass ratio: 0 <= q <= 1.

    Returns
    -------
    Smin:
        Minimum value of the total spin.
    Smax:
        Maximum value of the total spin.
    """

    L= angularmomentum(r,q)
    Smin = np.abs(J-L)
    Smax = J+L

    return np.array([Smin,Smax])


def Slimits_LJS1S2(J,r,q,chi1,chi2):
    """
    Limits on the total spin magnitude due to the vector relations S=S1+S2 and S=J-L.

    Parameters
    ----------
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    q: float
        Mass ratio: 0 <= q <= 1.

    Returns
    -------
    Smin:
        Minimum value of the total spin.
    Smax:
        Maximum value of the total spin.
    """

    SminS1S2,SmaxS1S2 = Slimits_S1S2(q,chi1,chi2)
    SminLJ, SmaxLJ = Slimits_LJ(J,r,q)
    Smin = np.maximum(SminS1S2,SminLJ)
    Smax = np.minimum(SmaxS1S2,SmaxLJ)

    return np.array([Smin,Smax])


def Scubic_coefficients(J,r,xi,q,chi1,chi2):
    """
    Coefficients of the cubic equation in S^2 that identifies the effective potentials.

    Parameters
    ----------
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    xi: float
        Effective spin
    q: float
        Mass ratio: 0 <= q <= 1.
    chi1: float
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.
    chi2: float
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.

    Returns
    -------
    sigma6:
        Coefficient of S^6.
    sigma4:
        Coefficient of S^4.
    sigma2:
        Coefficient of S^2.
    sigma0:
        Coefficient of S^0.
    """

    J,xi,q=toarray(J,xi,q)
    L=angularmomentum(r,q)
    S1,S2= spinmags(q,chi1,chi2)

    sigma6 = q * ( ( 1 + q ) )**( 2 )

    sigma4 = \
    ( ( 1 + q ) )**( 2 ) * ( -2 * ( J )**( 2 ) * q + ( ( L )**( 2 ) * ( 1 \
    + ( q )**( 2 ) ) + ( ( -1 + q ) * ( q * ( S1 )**( 2 ) + -1 * ( S2 \
    )**( 2 ) ) + 2 * L * q * xi ) ) )

    sigma2 = \
    ( ( J )**( 4 ) * q * ( ( 1 + q ) )**( 2 ) + ( L * ( L * ( ( 1 + q ) \
    )**( 2 ) * ( ( L )**( 2 ) * q + 2 * ( -1 + q ) * ( ( S1 )**( 2 ) + -1 \
    * q * ( S2 )**( 2 ) ) ) + ( 2 * q * ( 1 + q ) * ( ( L )**( 2 ) * ( 1 \
    + q ) + ( -1 + q ) * ( S1 + -1 * S2 ) * ( S1 + S2 ) ) * xi + 4 * L * \
    ( q )**( 2 ) * ( xi )**( 2 ) ) ) + -2 * ( J )**( 2 ) * ( ( 1 + q ) \
    )**( 2 ) * ( ( S2 )**( 2 ) + q * ( ( -1 + q ) * ( S1 )**( 2 ) + ( -1 \
    * ( S2 )**( 2 ) + L * ( L + xi ) ) ) ) ) )

    sigma0 = \
    ( -1 + ( q )**( 2 ) ) * ( ( J )**( 4 ) * ( 1 + q ) * ( q * ( S1 )**( \
    2 ) + -1 * ( S2 )**( 2 ) ) + ( ( L )**( 2 ) * ( ( -1 + ( q )**( 2 ) ) \
    * ( ( ( S1 )**( 2 ) + -1 * ( S2 )**( 2 ) ) )**( 2 ) + ( ( L )**( 2 ) \
    * ( 1 + q ) * ( q * ( S1 )**( 2 ) + -1 * ( S2 )**( 2 ) ) + 2 * L * q \
    * ( S1 + -1 * S2 ) * ( S1 + S2 ) * xi ) ) + 2 * ( J )**( 2 ) * L * ( \
    L * ( 1 + q ) * ( -1 * q * ( S1 )**( 2 ) + ( S2 )**( 2 ) ) + q * ( -1 \
    * ( S1 )**( 2 ) + ( S2 )**( 2 ) ) * xi ) ) )

    return np.array([sigma6, sigma4, sigma2, sigma0])


def S2roots(J,r,xi,q,chi1,chi2,coincident=False):
    """
    Coefficients of the cubic equation in S^2 that identifies the effective potentials.

    Parameters
    ----------
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    xi: float
        Effective spin
    q: float
        Mass ratio: 0 <= q <= 1.
    chi1: float
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.
    chi2: float
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.
    coincident: boolean, optional (default: False)
        If True, assume that the input is a spin-orbit resonance and return repeated roots

    Returns
    -------
    S32: float
        Spurious root.
    Sminus2:
        Lowest physical root (or unphysical).
    Splus2:
        Lowest physical root (or unphysical).

    """

    sigma6,sigma4,sigma2,sigma0= Scubic_coefficients(J,r,xi,q,chi1,chi2)

    sigmap = (sigma4**2/(3*sigma6**2) - sigma2/sigma6)/3
    sigmaq = ((2*sigma4**3)/(27*sigma6**3) - (sigma4*sigma2)/(3*sigma6**2) + sigma0/sigma6) /2
    #delta = sigmaq**2+sigmap**3

    if not coincident:
        # Mask values if there is only one solution and not three
        with np.errstate(invalid='ignore'):
            Sminus2,Splus2,S32= 2*sigmap**(1/2) * np.sin(np.arcsin(sigmaq*sigmap**(-3/2))/3 + (2*np.pi/3)*np.outer([0,1,2],np.ones(flen(sigmap)))) - sigma4/(3*sigma6)
    elif coincident:
        S32 = -2*sigmaq**(1/3) - sigma4/(3*sigma6)
        Sminus2=Splus2  = sigmaq**(1/3) - sigma4/(3*sigma6)

    #print(np.roots([sigma6,sigma4,sigma2,sigma0])) # You can test this against numpy.roots
    return np.array([Sminus2,Splus2,S32])


def Slimits_plusminus(J,r,xi,q,chi1,chi2,coincident=False):
    """
    Limits on the total spin magnitude compatible with both J and xi.

    Parameters
    ----------
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    xi: float
        Effective spin
    q: float
        Mass ratio: 0 <= q <= 1.
    chi1: float
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.
    chi2: float
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.
    conincident: boolean, optional (default: False)
        If True, assume that the input is a spin-orbit resonance and return repeated roots

    Returns
    -------
    Sminus:
        Minimum value of the total spin.
    Splus:
        Maximum value of the total spin.
    """

    Sminus2,Splus2,_= S2roots(J,r,xi,q,chi1,chi2,coincident=coincident)
    with np.errstate(invalid='ignore'):
        Sminus=Sminus2**0.5
        Splus=Splus2**0.5

    return np.array([Sminus,Splus])


def Slimits(J=None,r=None,xi=None,q=None,chi1=None,chi2=None,coincident=False):
    """
    Limits on the total spin magnitude. The contraints considered depend on the inputs provided.
        - If q, chi1, and chi2 are provided, enforce S=S1+S2.
        - If J, r, and q are provided, enforce S=J-L.
        - If J, r, q, chi1, and chi2 are provided, enforce S=S1+S2 and S=J-L.
        - If J, r, xi, q, chi1, and chi2 are provided, compute solve the cubic equation of the effective potentials (Sminus and Splus).

    Parameters
    ----------
    J: float, optional
        Magnitude of the total angular momentum.
    r: float, optional
        Binary separation.
    xi: float, optional
        Effective spin
    q: float
        Mass ratio: 0 <= q <= 1.
    chi1: float, optional
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.
    chi2: float, optional
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.
    conincident: boolean, optional (default: False)
        If True, assume that the input is a spin-orbit resonance and return repeated roots

    Returns
    -------
    Sminus:
        Minimum value of the total spin.
    Splus:
        Maximum value of the total spin.
    """

    if J is None and r is None and xi is None and q is not None and chi1 is not None and chi2 is not None:
        Smin,Smax = Slimits_S1S2(q,chi1,chi2)

    elif J is not None and r is not None and xi is None and q is not None and chi1 is None and chi2 is None:
        Smin,Smax = Slimits_LJ(J,r,q)

    elif J is not None and r is not None and xi is None and q is not None and chi1 is not None and chi2 is not None:
        Smin,Smax = Slimits_LJS1S2(J,r,q,chi1,chi2)

    elif J is not None and r is not None and xi is not None and q is not None and chi1 is not None and chi2 is not None:
        #TODO: Assert that Slimits_LJS1S2 is also respected (either explicitely or with a generic 'limits_check' function)
        Smin,Smax = Slimits_plusminus(J,r,xi,q,chi1,chi2,coincident=coincident)

    else:
        raise TypeError

    return np.array([Smin,Smax])


def limits_check(S=None, J=None,r=None,xi=None,q=None,chi1=None,chi2=None):
    """
    Check if a the inputs are consistent with the geometrical constraints.

    Parameters
    ----------

    Returns
    -------
    Boolean flag.
    """

    def _limits_check(testvalue,interval):
        """Check if a value is within a given interval"""
        return np.logical_and(testvalue>interval[0],testvalue<interval[1])

    raise NotImplementedError


def effectivepotential_Sphi(S,varphi,J,r,q,chi1,chi2):
    """
    Effective spin as a function of total spin magnitude S, nutation angle varphi and total angularm momentum J.

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    varphi: float
        Generalized nutation coordinate (Eq 9 in arxiv:1506.03492).
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    q: float
        Mass ratio: 0 <= q <= 1.
    chi1: float
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.
    chi2: float
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.

    Returns
    -------
    xi: float
        Effective spin
    """

    S,varphi,J,q=toarray(S,varphi,J,q)
    S1,S2 = spinmags(q,chi1,chi2)
    L = angularmomentum(r,q)

    xi = \
    1/4 * ( L )**( -1 ) * ( q )**( -1 ) * ( S )**( -2 ) * ( ( ( J )**( 2 \
    ) + ( -1 * ( L )**( 2 ) + -1 * ( S )**( 2 ) ) ) * ( ( ( 1 + q ) )**( \
    2 ) * ( S )**( 2 ) + ( -1 + ( q )**( 2 ) ) * ( ( S1 )**( 2 ) + -1 * ( \
    S2 )**( 2 ) ) ) + -1 * ( 1 + -1 * ( q )**( 2 ) ) * ( ( ( J )**( 2 ) + \
    -1 * ( ( L + -1 * S ) )**( 2 ) ) )**( 1/2 ) * ( ( -1 * ( J )**( 2 ) + \
    ( ( L + S ) )**( 2 ) ) )**( 1/2 ) * ( ( ( S )**( 2 ) + -1 * ( ( S1 + \
    -1 * S2 ) )**( 2 ) ) )**( 1/2 ) * ( ( -1 * ( S )**( 2 ) + ( ( S1 + S2 \
    ) )**( 2 ) ) )**( 1/2 ) * numpy.cos( varphi ) )

    return xi


def effectivepotential_plus(S,J,r,q,chi1,chi2):
    """
    Upper effective potential.

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    q: float
        Mass ratio: 0 <= q <= 1.
    chi1: float
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.
    chi2: float
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.

    Returns
    -------
    xi: float
        Effective spin
    """

    varphi = np.pi*np.ones(flen(S))
    xi = effectivepotential_Sphi(S,varphi,J,r,q,chi1,chi2)

    return xi


def effectivepotential_minus(S,J,r,q,chi1,chi2):
    """
    Lower effective potential.

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    q: float
        Mass ratio: 0 <= q <= 1.
    chi1: float
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.
    chi2: float
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.

    Returns
    -------
    xi: float
        Effective spin
    """

    varphi = np.zeros(flen(S))
    xi = effectivepotential_Sphi(S,varphi,J,r,q,chi1,chi2)

    return xi


def spinangle_costheta1(S,J,r,xi,q,chi1,chi2):
    """
    Cosine of the angle theta1 between the orbital angular momentum and the spin of the primary black hole.

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    xi: float
        Effective spin.
    q: float
        Mass ratio: 0 <= q <= 1.
    chi1: float
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.
    chi2: float
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.

    Returns
    -------
    costheta1: float
        Cosine of the angle between orbital angular momentum and primary spin.
    """

    S,J,q=toarray(S,J,q)
    S1,S2 = spinmags(q,chi1,chi2)
    L = angularmomentum(r,q)

    costheta1= ( ((J**2-L**2-S**2)/L) - (2.*q*xi)/(1.+q) )/(2.*(1.-q)*S1)

    return costheta1

def spinangle_theta1(S,J,r,xi,q,chi1,chi2):
    """
    Angle theta1 between the orbital angular momentum and the spin of the primary black hole.

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    xi: float
        Effective spin.
    q: float
        Mass ratio: 0 <= q <= 1.
    chi1: float
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.
    chi2: float
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.

    Returns
    -------
    theta1: float
        Angle between orbital angular momentum and primary spin.
    """

    costheta1=spinangle_costheta1(S,J,r,xi,q,chi1,chi2)
    theta1 = np.arccos(costheta1)

    return theta1


def spinangle_costheta2(S,J,r,xi,q,chi1,chi2):
    """
    Cosine of the angle theta2 between the orbital angular momentum and the spin of the secondary black hole.

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    xi: float
        Effective spin.
    q: float
        Mass ratio: 0 <= q <= 1.
    chi1: float
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.
    chi2: float
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.

    Returns
    -------
    costheta2: float
        Cosine of the angle between orbital angular momentum and secondary spin.
    """

    S,J,q=toarray(S,J,q)
    S1,S2 = spinmags(q,chi1,chi2)
    L = angularmomentum(r,q)

    costheta2= ( ((J**2-L**2-S**2)*(-q/L)) + (2*q*xi)/(1+q) )/(2*(1-q)*S2)

    return costheta2

def spinangle_theta2(S,J,r,xi,q,chi1,chi2):
    """
    Angle theta2 between the orbital angular momentum and the spin of the secondary black hole.

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    xi: float
        Effective spin.
    q: float
        Mass ratio: 0 <= q <= 1.
    chi1: float
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.
    chi2: float
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.

    Returns
    -------
    theta2: float
        Angle between orbital angular momentum and secondary spin.
    """

    costheta2=spinangle_costheta2(S,J,r,xi,q,chi1,chi2)
    theta2 = np.arccos(costheta2)

    return theta2


def spinangle_costheta12(S,q,chi1,chi2):
    """
    Cosine of the angle theta12 between the two spins.

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    q: float
        Mass ratio: 0 <= q <= 1.
    chi1: float
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.
    chi2: float
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.

    Returns
    -------
    costheta12: float
        Cosine of the angle between the two spins.
    """

    S=toarray(S)
    S1,S2 = spinmags(q,chi1,chi2)
    costheta12=(S**2-S1**2-S2**2)/(2*S1*S2)

    return costheta12


def spinangle_theta12(S,q,chi1,chi2):
    """
    Angle theta12 between the two spins.

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    q: float
        Mass ratio: 0 <= q <= 1.
    chi1: float
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.
    chi2: float
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.

    Returns
    -------
    theta12: float
        Angle between the two spins.
    """

    costheta12=spinangle_costheta12(S,q,chi1,chi2)
    theta12 = np.arccos(costheta12)

    return theta2


def spinangle_cosdeltaphi(S,J,r,xi,q,chi1,chi2):
    """
    Cosine of the angle deltaphi between the projections of the two spins onto the orbital plane.

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    xi: float
        Effective spin.
    q: float
        Mass ratio: 0 <= q <= 1.
    chi1: float
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.
    chi2: float
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.

    Returns
    -------
    cosdeltaphi: float
        Cosine of the angle between the projections of the two spins onto the orbital plane.
    """

    q=toarray(q)
    S1,S2 = spinmags(q,chi1,chi2)
    costheta1=spinangle_costheta1(S,J,r,xi,q,chi1,chi2)
    costheta2=spinangle_costheta2(S,J,r,xi,q,chi1,chi2)
    costheta12=spinangle_costheta12(S,q,chi1,chi2)
    cosdeltaphi= (costheta12 - costheta1*costheta2)/((1-costheta1**2)*(1-costheta2**2))**0.5

    return cosdeltaphi


def spinangle_deltaphi(S,J,r,xi,q,chi1,chi2,sign=+1):
    """
    Angle deltaphi between the projections of the two spins onto the orbital plane. By default this is returned in [0,pi]. Setting sign=-1 returns the other half of the  precession cycle [-pi,0].

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    q: float
        Mass ratio: 0 <= q <= 1.
    xi: float
        Effective spin.
    chi1: float
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.
    chi2: float
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.
    sign: optional (default: +1)
        If positive returns values in [0,pi], if negative returns values in [-pi,0].

    Returns
    -------
    deltaphi: float
        Cosine of the angle between the projections of the two spins onto the orbital plane.
    """

    cosdeltaphi=spinangle_cosdeltaphi(S,J,r,xi,q,chi1,chi2)
    deltaphi = np.sign(sign)*np.arccos(cosdeltaphi)

    return deltaphi






def newlen(var):
    """Redefine len function
    """

    try:
        n = len(var)
    except:
        n = 1
    return n


def isarray(var):
    """Check if a variable is an array
    """

    if isinstance(var, (list, tuple, np.ndarray)):
        return True
    else:
        return False


class Binary:
    """Test class of a binary black hole

    Parameters
    ----------
    q: float
        Mass ratio m2/m1, 0 <= q <= 1.

    chi1: float
        Kerr parameter of the heavier black hole, 0 <= chi1 <= 1.

    chi2: float
        Kerr parameter of the lighter black hole, 0 <= chi2 <= 1.
    """

    # How should we represent a single binary?
    # The fixed parameters without initialization are
    # M, q, chi1, chi2 (or m1, m2, S1, S2)
    # To initialize a binary we also require
    # r (or L) and xi, J, S (or theta1, theta2 and DeltaPhi)
    # Different parameters are required for different evolutions:
    # finite/infinite precession-averaged or orbit-averaged
    # For the precession-averaged we can include finite and infinite in one
    # and allow for r = np.inf with a flag to check for this
    # Should we ignore the total mass M?
    # Or include it everywhere as an optional parameter?


    def __init__(self, q, chi1, chi2):

        assert q >= 0. and q <= 1.
        assert chi1 >= 0. and chi1 <= 1.
        assert chi2 >= 0. and chi2 <= 1.

        self.q = q
        self.chi1 = chi1
        self.chi2 = chi2
        self.M = 1.
        self.m1 = self.M / (1. + self.q)
        self.m2 = self.m1 * self.q
        self.S1 = self.chi1 * self.m1**2
        self.S2 = self.chi2 * self.m2**2

    def config_angles(self, ri, theta1, theta2, deltaphi):
        """
        """

        xii, Ji, Si = pre.from_the_angles(theta1, theta2, deltaphi, self.q,
                                          self.S1, self.S2, ri)
        self.ri = ri
        self.xii = xii
        self.Ji = Ji
        self.Si = Si

        return xii, Ji, Si

    def orb_evolve(self, r):
        """
        """

        assert r[0] == self.ri
        J, xi, S = pre.orbit_averaged(self.Ji, self.xii, self.Si, r, self.q,
                                      self.S1, self.S2)
        self.r = r
        self.xi = xi
        self.J = J
        self.S = S

        return xi, J, S


if __name__ == '__main__':

    #r=[10,10]
    #xi=[0.35,-0.6]
    #q=[0.8,0.2]
    #chi1=[1,1]
    #chi2=[1,1]
    #J=[1,0.23]

    #print(Jresonances(r[0],xi[0],q[0],chi1[0],chi2[0]))
    #print(Jresonances(r[1],xi[1],q[1],chi1[1],chi2[1]))
    #print(Jresonances(r,xi,q,chi1,chi2))
    #print(Jlimits(r=r,xi=xi,q=q,chi1=chi1,chi2=chi2))
    #print(Jlimits(r=r,q=q,chi1=chi1,chi2=chi2))


    #print(xiresonances(J[0],r[0],q[0],chi1[0],chi2[0]))
    #print(xiresonances(J[1],r[1],q[1],chi1[1],chi2[1]))
    #print(xiresonances(J,r,q,chi1,chi2))

    #print(S2roots(J[0],r[0],xi[0],q[0],chi1[0],chi2[0]))
    #print(Slimits_plusminus(J,r,xi,q,chi1,chi2))

    #print(xilimits(q=q,chi1=chi1,chi2=chi2))

    #print(xilimits(J=J,r=r,q=q,chi1=chi1,chi2=chi2))
    #S=[0.4,0.6668]

    #print(effectivepotential_plus(S,J,r,q,chi1,chi2))
    #print(effectivepotential_minus(S,J,r,q,chi1,chi2))

    #print(Slimits_cycle(J,r,xi,q,chi1,chi2))


    #M,m1,m2,S1,S2=pre.get_fixed(q[0],chi1[0],chi2[0])
    #print(pre.J_allowed(xi[0],q[0],S1[0],S2[0],r[0]))

    #print(Jresonances(r,xi,q,chi1,chi2))

    #S2roots(J,r,xi,q,chi1,chi2)

    #print(Jlimits(r,q,chi1,chi2))
    #print(S2roots(J,r,xi,q,chi1,chi2))


    #print(Slimits_check([0.24,4,6],q,chi1,chi2,which='S1S2'))

    S=[0.2,0.35]
    q=[0.8,0.8]
    chi1=[1,1]
    chi2=[0.8,0.8]
    r=[20,20]
    J=[1.29,1.29]
    xi=[0.3,0.3]
    print(spinangle_deltaphi(S,J,r,xi,q,chi1,chi2)/np.pi)
