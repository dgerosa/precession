"""
precession
"""

import numpy as np
import scipy as sp
# Using precession_v1 functions for now
import precession as pre
import sys, os

__all__ = [] # Why is this necessary?


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

    q = np.array(q)
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

    q = np.array(q)
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

    chi1 = np.array(chi1)
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

    chi2 = np.array(chi2)
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

    r = np.array(r)
    L = mass1(q)*mass2(q)*r**(3/2)

    return L


def xilimits(q,chi1,chi2):
    """
    Limits on the effective spin xi.

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

    q=np.array(q)
    S1,S2 = spinmags(q,chi1,chi2)
    xilim = (1+q)*S1 + (1+1/q)*S2

    return np.array([-xilim,xilim])


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
    Jmin = np.maximum.reduce([0, L-S1-S2, np.abs(S1-S2)-L])
    Jmax = L+S1+S1

    return np.array([Jmin,Jmax])


def deltacoeffs(r,xi,q,chi1,chi2):
    """Not finished
    Polynomial coefficients of the discriminant as a function of J"""

    L=angularmomentum(r,q)
    xi=np.array(xi)
    q=np.array(q)
    S1,S2= spinmags(q,chi1,chi2)

    delta0 = \
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

    delta2 = \
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

    delta4 = \
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

    delta6 = \
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

    delta8 = \
    ( L )**( 2 ) * ( ( -1 + q ) )**( 2 ) * ( q )**( 2 ) * ( ( 1 + q ) \
    )**( 6 ) * ( ( ( 1 + q ) )**( 2 ) * ( ( L )**( 2 ) * ( 1 + q * ( 18 + \
    q ) ) + ( 4 * ( 5 + -3 * q ) * q * ( S1 )**( 2 ) + 4 * ( -3 + 5 * q ) \
    * ( S2 )**( 2 ) ) ) + ( 20 * L * q * ( ( 1 + q ) )**( 2 ) * xi + 4 * \
    ( q )**( 2 ) * ( xi )**( 2 ) ) )

    delta10 = \
    -4 * ( L )**( 2 ) * ( ( -1 + q ) )**( 2 ) * ( q )**( 3 ) * ( ( 1 + q \
    ) )**( 8 )

    return delta10, delta8, delta6, delta4, delta2, delta0

def deltaroots(r,xi,q,chi1,chi2):
    coeffs= deltacoeffs(r,xi,q,chi1,chi2)
    J2 = np.sort_complex(np.roots(coeffs)) # Complex numbers, sort according to real part

    return J2

def Jresonances(r,xi,q,chi1,chi2):
    # The good solutions are the last two. That's because the discriminant quintic asymptotes to -infinity and the physical region is when it's positive

    J2roots= deltaroots(r,xi,q,chi1,chi2)
    Jresonances = np.real(J2roots[np.isreal(J2roots)][-2:]**0.5)

    return Jresonances


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


def Slimits_LJS1S2(J,r,q):
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

    SminS1S2,SmaxS1S2 = Slimits_S1S2(q,chi1,chi2)
    SminLJ, SmaxLJ = Slimits_LJ(J,r,q)
    Smin = np.maximum(SminS1S1,SminLJ)
    Smax = np.minimum(SmaxS1S1,SmaxLJ)

    return np.array([Smin,Smax])


def sigmacoeffs(J,r,xi,q,chi1,chi2):
    """Not finished"""

    J=np.array(J)
    L=angularmomentum(r,q)
    xi=np.array(xi)
    q=np.array(q)
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

    return sigma6, sigma4, sigma2, sigma0

def S2roots(J,r,xi,q,chi1,chi2):
    """Not finished"""

    a,b,c,d= sigmacoeffs(J,r,xi,q,chi1,chi2)
    pt = (b**2/(3*a**2) - c/a)/3
    qt = ((2*b**3)/(27*a**3) - (b*c)/(3*a**2) + d/a) /2
    asint = np.arcsin(qt*pt**(-3/2))/3
    S32,Sminus2,Splus2 =  2*pt**(1/2) * np.sin(asint+ (2*np.pi/3)*np.array([2,0,1])) - b/(3*a)

    return S32,Sminus2,Splus2




def limits_check(function=None, S=None,J=None,r=None,q=None,chi1=None,chi2=None):
    """
    THIS FUNCTION DOESN'T WORK (YET)

    Check if a given variable satisfies the relevant geometrical constraints. The behaviour is set by `function`. For instance, to check if some values of J are compatible with the provides values of r, q, chi1, and chi2 use function=`Jlimits`. Not all the parameters are necessary. For instance, to check the limits in J one does not need to provide values of S.

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chi1: float or numpy array
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.
    chi2: float or numpy array
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.

    Returns
    -------
    Boolean flag.
    """

    def _limits_check(testvalue,interval):
        """Check if a value is within a given interval"""
        return np.logical_and(testvalue>interval[0],testvalue<interval[1])

    if function=='Jlimits':
        return _limits_check(J,Jlimits(r,q,chi1,chi2))

    elif function=='Slimits_S1S2':
        return _limits_check(S,Slimits_S1S2(q,chi1,chi2))

    elif function=='Slimits_JL':
        return _limits_check(S,Slimits_LJ(J,r,q))

    elif function=='Slimits_JL':
        return _limits_check(S,Slimits_LJ(J,r,q))

    else:
        raise ValueError




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

    r=10
    xi=0
    q=0.5
    chi1=0.5
    chi2=0.5
    J=6.9

    print(Jresonances(r,xi,q,chi1,chi2))

    #print(Jlimits(r,q,chi1,chi2))
    #print(S2roots(J,r,xi,q,chi1,chi2))





    #print(Slimits_check([0.24,4,6],q,chi1,chi2,which='S1S2'))
    #
    # # Example
    # #--------
    # q = .7
    # chi1 = .8
    # chi2 = .9
    # b = Binary(q, chi1, chi2)
    #
    # ri = 1000.
    # t1i = np.pi/2.
    # t2i = np.pi/3.
    # dpi = np.pi/4.
    # xii, Ji, Si = b.config_angles(ri, t1i, t2i, dpi)
    #
    # rf = 10.
    # nr = 1000
    # r = np.linspace(ri, rf, nr)
    # xi, J, S = b.orb_evolve(r)
    # #--------
