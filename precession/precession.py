"""
precession
"""

import numpy as np
import scipy, scipy.special, scipy.integrate
import sys, os, time
import warnings
import itertools


#getnone=itertools.repeat(None)
def flen(x):
    if hasattr(x, "__len__"):
        return len(x)
    else:
        return 1


def toarray(*args):
    return np.squeeze(np.array([*args]))


#def isarray(var):
#    """
#    Check if a variable is an array
#    """
#
#    if isinstance(var, np.ndarray):
#        return 1
#    elif isinstance(var, (list, tuple)):
#        return 0 #2
#    else:
#        return 0


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


def masses(q):
    """
    """

    m1 = mass1(q)
    m2 = mass2(q)

    return np.array([m1, m2])


def massratio(m1, m2):
    """
    Mass ratio, 0 < q = m2/m1 < 1.

    Parameters
    ----------
    m1: float
        Mass of the primary black hole.

    m2: float
        Mass of the secondary black hole.

    Returns
    -------
    q: float
        Mass ratio, 0 < q < 1.

    """

    m1 = toarray(m1)
    m2 = toarray(m2)
    q = m2 / m1

    return q


def symmetricmassratio(q):
    """
    Symmetric mass ratio eta = m1*m2/(m1+m2)^2 = q/(1+q)^2

    Parameters
    ----------
    q: float
        Mass ratio: 0 <= q <= 1.

    Returns
    -------
    eta: float
        Symmetric mass ratio.

    """

    q = toarray(q)
    eta = q/(1+q)**2

    return eta


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

    chi1 = toarray(chi1)
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


def orbitalseparation(L, q):
    """
    """

    L = toarray(L)
    m1, m2 = masses(q)
    r = (L / (m1 * m2))**2

    return r


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

    q,xi=toarray(q,xi)
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
    Roots of the cubic equation in S^2 that identifies the effective potentials.

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
    Sminus2:
        Lowest physical root (or unphysical).
    Splus2:
        Highest physical root (or unphysical).
    S32: float
        Spurious root.
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
    return toarray([Sminus2,Splus2,S32])


def Scubic_coefficients_NEW(kappa,u,xi,q,chi1,chi2):
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

    kappa,u,xi,q=toarray(kappa,u,xi,q)
    S1,S2= spinmags(q,chi1,chi2)

    sigma6 = q * ( ( 1 + q ) )**( 2 ) * ( u )**( 2 )

    sigma4 = \
    1/4 * ( ( 1 + q ) )**( 2 ) * ( 1 + ( 4 * ( S2 )**( 2 ) * ( u )**( 2 \
    ) + ( ( q )**( 2 ) * ( 1 + 4 * ( S1 )**( 2 ) * ( u )**( 2 ) ) + -2 * \
    q * ( 1 + 2 * u * ( ( ( S1 )**( 2 ) + ( S2 )**( 2 ) ) * u + ( -1 * xi \
    + 2 * kappa ) ) ) ) ) )

    sigma2 = \
    ( -1 * q * ( -1 * xi + ( kappa + q * kappa ) ) * ( q * xi + -1 * ( 1 \
    + q ) * kappa ) + ( -1/2 * ( -1 + ( q )**( 2 ) ) * ( S2 )**( 2 ) * ( \
    -1 + ( ( q )**( 2 ) + ( 2 * q * u * xi + -4 * ( 1 + q ) * u * kappa ) \
    ) ) + -1/2 * ( -1 + ( q )**( 2 ) ) * ( S1 )**( 2 ) * ( -1 + q * ( q + \
    ( -2 * u * xi + 4 * ( 1 + q ) * u * kappa ) ) ) ) )

    sigma0 = \
    ( 1/4 * ( ( -1 + ( q )**( 2 ) ) )**( 2 ) * ( ( ( S1 )**( 2 ) + -1 * ( \
    S2 )**( 2 ) ) )**( 2 ) + ( -1 * q * ( -1 + ( q )**( 2 ) ) * ( S1 + -1 \
    * S2 ) * ( S1 + S2 ) * xi * kappa + ( -1 + q ) * ( ( 1 + q ) )**( 2 ) \
    * ( q * ( S1 )**( 2 ) + -1 * ( S2 )**( 2 ) ) * ( kappa )**( 2 ) ) )

    return np.array([sigma6, sigma4, sigma2, sigma0])


## TODO: generalize sigma6=0 to vector sigma6
def S2roots_NEW(kappa,u,xi,q,chi1,chi2,coincident=False):
    """
    Roots of the cubic equation in S^2 that identifies the effective potentials.

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
    Sminus2:
        Lowest physical root (or unphysical).
    Splus2:
        Highest physical root (or unphysical).
    S32: float
        Spurious root.
    """

    sigma6,sigma4,sigma2,sigma0= Scubic_coefficients_NEW(kappa,u,xi,q,chi1,chi2)

    #sigma6bool = sigma6 == 0.0

    if sigma6 == 0.0:
        Sminus2 = (-sigma2 - (sigma2**2 - 4*sigma4*sigma0)**0.5) / (2*sigma4)
        Splus2 = (-sigma2 + (sigma2**2 - 4*sigma4*sigma0)**0.5) / (2*sigma4)
        S32 = -np.inf

    else:
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
    return toarray([Sminus2,Splus2,S32])


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


# TODO: Check inter-compatibility of Slimits, Jlimits, xilimits
# Tags for each limit check that fails?
def limits_check(S=None, J=None, r=None, xi=None, q=None, chi1=None, chi2=None):
    """
    Check if the inputs are consistent with the geometrical constraints.

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
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

    Returns
    -------
    check: bool
        True if the given parameters are compatible with each other, false if not.
    """
    # q, ch1, chi2
    # 0, 1

    # J: r, xi, q, chi1, chi2
    # r, q, chi1, chi2 -> Jlimits_LS1S2
    # r, xi, q, chi1, chi2 -> Jresonances

    # xi: J, r, q, chi1, chi2
    # q, chi1, chi2 -> xilimits_definition
    # J, r, q, chi1, chi2 -> xiresonances

    # S: J, r, xi, q, chi1, chi2
    # q, chi1, chi2 -> Slimits_S1S2
    # J, r, q -> Slimits_LJ
    # J, r, q, chi1, chi2 -> Slimits_LJS1S2
    # J, r, xi, q, chi1, chi2 -> Slimits_plusminus

    def _limits_check(testvalue, interval):
        """Check if a value is within a given interval"""
        return np.logical_and(testvalue>=interval[0], testvalue<=interval[1])

    Slim = Slimits(J, r, xi, q, chi1, chi2)
    Sbool = _limits_check(S, Slim)

    Jlim = Jlimits(r, xi, q, chi1, chi2)
    Jbool = _limits_check(J, Jlim)

    xilim = xilimits(J, r, q, chi1, chi2)
    xibool = _limits_check(xi, xilim)

    check = all((Sbool, Jbool, xibool))

    if r is not None:
        rbool = _limits_check(r, [10.0, np.inf])
        check = all((check, rbool))

    if q is not None:
        qbool = _limits_check(q, [0.0, 1.0])
        check = all((check, qbool))

    if chi1 is not None:
        chi1bool = _limits_check(chi1, [0.0, 1.0])
        check = all((check, chi1bool))

    if chi2 is not None:
        chi2bool = _limits_check(chi2, [0.0, 1.0])
        check = all((check, chi2bool))

    return check


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
    ) )**( 2 ) ) )**( 1/2 ) * np.cos( varphi ) )

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


def eval_costheta1(S,J,r,xi,q,chi1,chi2):
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


def eval_theta1(S,J,r,xi,q,chi1,chi2):
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

    costheta1=eval_costheta1(S,J,r,xi,q,chi1,chi2)
    theta1 = np.arccos(costheta1)

    return theta1


def eval_costheta2(S,J,r,xi,q,chi1,chi2):
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


def eval_theta2(S,J,r,xi,q,chi1,chi2):
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

    costheta2=eval_costheta2(S,J,r,xi,q,chi1,chi2)
    theta2 = np.arccos(costheta2)

    return theta2


def eval_costheta12(S,q,chi1,chi2):
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


def eval_theta12(S,q,chi1,chi2):
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

    costheta12=eval_costheta12(S,q,chi1,chi2)
    theta12 = np.arccos(costheta12)

    return theta12


def eval_cosdeltaphi(S,J,r,xi,q,chi1,chi2):
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
    costheta1=eval_costheta1(S,J,r,xi,q,chi1,chi2)
    costheta2=eval_costheta2(S,J,r,xi,q,chi1,chi2)
    costheta12=eval_costheta12(S,q,chi1,chi2)
    cosdeltaphi= (costheta12 - costheta1*costheta2)/((1-costheta1**2)*(1-costheta2**2))**0.5

    return cosdeltaphi


def eval_deltaphi(S,J,r,xi,q,chi1,chi2,sign=+1):
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
        Angle between the projections of the two spins onto the orbital plane.
    """

    cosdeltaphi=eval_cosdeltaphi(S,J,r,xi,q,chi1,chi2)
    deltaphi = np.sign(sign)*np.arccos(cosdeltaphi)

    return deltaphi


def eval_costhetaL(S,J,r,q,chi1,chi2):
    """
    Cosine of the angle thetaL betwen orbital angular momentum and total angular momentum.

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
    costhetaL: float
        Cosine of the angle betwen orbital angular momentum and total angular momentum.
    """

    S,J=toarray(S,J)
    S1,S2 = spinmags(q,chi1,chi2)
    L = angularmomentum(r,q)
    costhetaL=(J**2+L**2-S**2)/(2*J*L)

    return costhetaL


def eval_thetaL(S,J,r,q,chi1,chi2):
    """
    Angle thetaL betwen orbital angular momentum and total angular momentum.

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
    thetaL: float
        Angle betwen orbital angular momentum and total angular momentum.
    """

    costhetaL=eval_costhetaL(S,J,r,q,chi1,chi2)
    thetaL=np.arccos(costhetaL)

    return thetaL


def eval_xi(theta1,theta2,q,chi1,chi2):
    """
    Effective spin from the spin angles.

    Parameters
    ----------
    theta1: float
        Angle between orbital angular momentum and primary spin.
    theta1: float
        Angle between orbital angular momentum and primary spin.
    q: float
        Mass ratio: 0 <= q <= 1.
    chi1: float
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.
    chi2: float
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.

    Returns
    -------
    xi: float
        Effective spin.
    """

    theta1,theta2,q=toarray(theta1,theta2,q)
    S1,S2 = spinmags(q,chi1,chi2)
    xi=(1+q)*(q*S1*np.cos(theta1)+S2*np.cos(theta2))/q

    return xi


def eval_J(theta1,theta2,deltaphi,r,q,chi1,chi2):
    """
    Magnitude of the total angular momentum from the spin angles.

    Parameters
    ----------
    theta1: float
        Angle between orbital angular momentum and primary spin.
    theta1: float
        Angle between orbital angular momentum and primary spin.
    deltaphi: float
        Angle between the projections of the two spins onto the orbital plane.
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
    J: float
        Magnitude of the total angular momentum.
    """

    theta1,theta2,deltaphi,q=toarray(theta1,theta2,deltaphi,q)
    S1,S2 = spinmags(q,chi1,chi2)
    L = angularmomentum(r,q)
    S=eval_S(theta1,theta2,deltaphi,q,chi1,chi2)
    J=(L**2+S**2+2*L*(S1*np.cos(theta1)+S2*np.cos(theta2)))**0.5

    return J


def eval_S(theta1,theta2,deltaphi,q,chi1,chi2):
    """
    Magnitude of the total spin from the spin angles.

    Parameters
    ----------
    theta1: float
        Angle between orbital angular momentum and primary spin.
    theta1: float
        Angle between orbital angular momentum and primary spin.
    deltaphi: float
        Angle between the projections of the two spins onto the orbital plane.
    q: float
        Mass ratio: 0 <= q <= 1.
    chi1: float
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.
    chi2: float
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.

    Returns
    -------
    S: float
        Magnitude of the total spin.
    """

    theta1,theta2,deltaphi=toarray(theta1,theta2,deltaphi)
    S1,S2 = spinmags(q,chi1,chi2)

    S=(S1**2+S2**2+2*S1*S2*(np.sin(theta1)*np.sin(theta2)*np.cos(deltaphi)+np.cos(theta1)*np.cos(theta2)))**0.5

    return S


def conserved_to_angles(S,J,r,xi,q,chi1,chi2,sign=+1):
    """
    Convert conserved quantities (S,J,xi) into angles (theta1,theta2,deltaphi).
    Setting sign=+1 (default) returns deltaphi in [0, pi], setting sign=-1 returns deltaphi in [-pi,0].

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
    theta1: float
        Angle between orbital angular momentum and primary spin.
    theta2: float
        Angle between orbital angular momentum and secondary spin.
    deltaphi: float
        Angle between the projections of the two spins onto the orbital plane.
    """

    theta1=eval_theta1(S,J,r,xi,q,chi1,chi2)
    theta2=eval_theta2(S,J,r,xi,q,chi1,chi2)
    deltaphi=eval_deltaphi(S,J,r,xi,q,chi1,chi2,sign=sign)

    return np.array([theta1,theta2,deltaphi])


def angles_to_conserved(theta1,theta2,deltaphi,r,q,chi1,chi2):
    """
    Convert angles (theta1,theta2,deltaphi) into conserved quantities (S,J,xi).

    Parameters
    ----------
    theta1: float
        Angle between orbital angular momentum and primary spin.
    theta1: float
        Angle between orbital angular momentum and primary spin.
    deltaphi: float
        Angle between the projections of the two spins onto the orbital plane.
    r: float
        Binary separation.
    q: float
        Mass ratio: 0 <= q <= 1.
    chi1: float
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.
    chi2: float
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.

    Returns
    ----------
    S: float
        Magnitude of the total spin.
    J: float
        Magnitude of the total angular momentum.
    xi: float
        Effective spin.
    """

    S=eval_S(theta1,theta2,deltaphi,q,chi1,chi2)
    J=eval_J(theta1,theta2,deltaphi,r,q,chi1,chi2)
    xi=eval_xi(theta1,theta2,q,chi1,chi2)

    return np.array([S,J,xi])


## TODO:
def eval_varphi(S, J, r, xi, q, chi1, chi2, sign=1):
    """
    """

    L = angularmomentum(r, q)
    S1, S2 = spinmags(q, chi1, chi2)

    t1 = (1+q) / (4*q * S**2 * L)
    t2 = J**2 - L**2 - S**2
    t3 = S**2 * (1+q) - (S1**2 - S2**2) * (1-q)
    t4 = (1-q) * ((L+S)**2 - J**2)**0.5
    t5 = (J**2 - (L-S)**2)**0.5
    t6 = ((S1+S2)**2 - S**2)**0.5
    t7 = (S**2 - (S1-S2)**2)**0.5

    cosvarphi= ((t2*t3) - (xi/t1)) / (t4*t5*t6*t7)
    varphi = np.arccos(cosvarphi) * sign

    return varphi


## TODO:
def vectors_to_conserved(S1vec, S2vec, Lvec, q):
    """
    """

    S1vec, S2vec, Lvec, q = toarray(S1vec, S2vec, Lvec, q)
    vecs = [S1vec, S2vec, Lvec]
    for i in range(len(vecs)):
        if len(vecs[i].shape) == 1:
            vecs[i] = np.array([vecs[i]])
    S1vec, S2vec, Lvec = vecs
    S = np.linalg.norm(S1vec+S2vec, axis=-1)
    J = np.linalg.norm(S1vec+S2vec+Lvec, axis=-1)
    L = np.linalg.norm(Lvec, axis=-1)
    m1, m2 = masses(q)
    #xi = np.einsum('ij, ij->i', S1vec/m1 + S2vec/m2, Lvec/L)
    xi = np.array([np.dot(s0, l) for s0, l in zip(S1vec/m1+S2vec/m2, Lvec/L)])

    return toarray(S, J, xi)


## TODO:
def vectors_to_angles(S1vec, S2vec, Lvec):
    """
    """

    S1vec, S2vec, Lvec = toarray(S1vec, S2vec, Lvec)
    vecs = [S1vec, S2vec, Lvec]
    for i in range(len(vecs)):
        if len(vecs[i].shape) == 1:
            vecs[i] = np.array([vecs[i]])
    S1vec, S2vec, Lvec = vecs
    S1vec = S1vec / np.linalg.norm(S1vec, axis=-1)
    S2vec = S2vec / np.linalg.norm(S2vec, axis=-1)
    Lvec = Lvec / np.linalg.norm(Lvec, axis=-1)
    #theta1 = np.arccos(np.einsum('ij, ij->i', S1vec, Lvec))
    #theta2 = np.arccos(np.einsum('ij, ij->i', S2vec, Lvec))
    theta1 = np.arccos(np.array([np.dot(s1, l) for s1, l in zip(S1vec, Lvec)]))
    theta2 = np.arccos(np.array([np.dot(s2, l) for s2, l in zip(S2vec, Lvec)]))
    S1cL = np.cross(S1vec, Lvec)
    S1cL = S1cL / np.linalg.norm(S1cL, axis=-1)
    S2cL = np.cross(S2vec, Lvec)
    S2cL = S2cL / np.linalg.norm(S2cL, axis=-1)
    #deltaphi = np.arccos(np.einsum('ij, ij->i', S1cL, S2cL))
    deltaphi = np.arccos(np.array([np.dot(s1cl, s2cl) for s1cl, s2cl in zip(S1cL, S2cL)]))

    return toarray(theta1, theta2, deltaphi)


## TODO:
def conserved_to_Jframe(S, J, r, xi, q, chi1, chi2):
    """
    Convert the conserved quantities to angular momentum vectors in the frame
    aligned with the total angular momentum.

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

    Returns
    -------

    """

    S, J = toarray(S, J)
    L = angularmomentum(r, q)
    S1, S2 = spinmags(q, chi1, chi2)
    #if q == 1:
    #else:
    varphi = eval_varphi(S, J, r, xi, q, chi1, chi2)
    thetaL = eval_thetaL(S, J, r, q, chi1, chi2)

    Jx = 0.0
    Jy = 0.0
    Jz = J
    Jvec = np.array([Jx, Jy, Jz])

    Lx = L * np.sin(thetaL)
    Ly = 0.0
    Lz = L * np.cos(thetaL)
    Lvec = np.array([Lx, Ly, Lz])

    Svec = Jvec - Lvec

    A1 = (J**2 - (L-S)**2)**0.5
    A2 = ((L+S)**2 - J**2)**0.5
    A3 = (S**2 - (S1-S2)**2)**0.5
    A4 = ((S1+S2)**2 - S**2)**0.5

    #Lx_new = A1 * A2 / (2*J)
    #Ly_new = 0.0
    #Lz_new = (J**2 + L**2 - S**2) / (2*J)
    #Lvec_new = np.array([Lx_new, Ly_new, Lz_new])

    S1x = (-(S**2+S1**2-S2**2)*A1*A2 + (J**2-L**2+S**2)*A3*A4*np.cos(varphi)) / (4*J*S**2)
    S1y = A3 * A4 * np.sin(varphi) / (2*S)
    S1z = ((S**2+S1**2-S2**2)*(J**2-L**2+S**2) + A1*A2*A3*A4*np.cos(varphi)) / (4*J*S**2)
    S1vec = np.array([S1x, S1y, S1z])

    S2x = -((S**2+S2**2-S1**2)*A1*A2 + (J**2-L**2+S**2)*A3*A4*np.cos(varphi)) / (4*J*S**2)
    S2y = -A3*A4*np.sin(varphi) / (2*S)
    S2z = ((S**2+S2**2-S1**2)*(J**2-L**2+S**2) - A1*A2*A3*A4*np.cos(varphi)) / (4*J*S**2)
    S2vec = np.array([S2x, S2y, S2z])

    return np.array([Svec, S1vec, S2vec, Jvec, Lvec])


## TODO:
def angles_to_Jframe(theta1, theta2, deltaphi, r, q, chi1, chi2):
    """
    """

    S, J, xi = angles_to_conserved(theta1, theta2, deltaphi, r, q, chi1, chi2)

    return conserved_to_Jframe(S, J, r, xi, q, chi1, chi2)


## TODO:
def angles_to_Lframe(theta1, theta2, deltaphi, r, q, chi1, chi2):
    """
    """

    L = angularmomentum(r, q)
    S1, S2 = spinmags(q, chi1, chi2)

    Lx = 0.0
    Ly = 0.0
    Lz = L
    Lvec = np.array([Lx, Ly, Lz])

    S1x = np.sin(theta1)
    S1y = 0.0
    S1z = np.cos(theta1)
    S1vec = np.array([S1x, S1y, S1z]) * S1

    S2x = np.sin(theta2) * np.cos(deltaphi)
    S2y = np.sin(theta2) * np.sin(deltaphi)
    S2z = np.cos(theta2)
    S2vec = np.array([S2x, S2y, S2z]) * S2

    Svec = S1vec + S2vec
    Jvec = Lvec + Svec

    return toarray(Svec, S1vec, S2vec, Jvec, Lvec)


## TODO:
def conserved_to_Lframe(S, J, r, xi, q, chi1, chi2):
    """
    """

    theta1, theta2, deltaphi = conserved_to_angles(S, J, r, xi, q, chi1, chi2)

    return angles_to_Lframe(theta1, theta2, deltaphi, r, q, chi1, chi2)


def morphology(J,r,xi,q,chi1,chi2,simpler=False):
    """
    Evaluate the spin morphology and return "L0" for librating about DeltaPhi=0, "Lpi" for librating about DeltaPhi=pi, "C-" for circulating from DeltaPhi=pi to DeltaPhi=0, and "C+" for circulating from DeltaPhi=0 to DeltaPhi=pi. If simpler=True, do not distinguish between the two circulating morphologies and return "C" for both.

    Parameters
    ----------
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
    simpler: optional (default: False)
        If True does not distinguish between positive and negative circulation.

    Returns
    -------
    morph: string
        Spin morphology.
    """

    Smin,Smax = Slimits_plusminus(J,r,xi,q,chi1,chi2)
    # Pairs of booleans based on the values of deltaphi at S- and S+
    status = np.array([eval_cosdeltaphi(Smin,J,r,xi,q,chi1,chi2) >0.5 ,eval_cosdeltaphi(Smax,J,r,xi,q,chi1,chi2) >0.5]).T

    # Map to labels
    if simpler:
        dictlabel = {(False,False):"Lpi", (True,True):"L0", (False, True):"C", (True, False):"C"}
    else:
        dictlabel = {(False,False):"Lpi", (True,True):"L0", (False, True):"C-", (True, False):"C+"}

    # Subsitute pairs with labels
    morphs = np.zeros(flen(J))
    for k, v in dictlabel.items():
        morphs=np.where((status == k).all(axis=1),v,morphs)

    return np.squeeze(morphs)


def Speriod_prefactor(r,xi,q):
    """
    Numerical prefactor to the precession period.

    Parameters
    ----------
    r: float
        Binary separation.
    xi: float
        Effective spin.
    q: float
        Mass ratio: 0 <= q <= 1.

    Returns
    -------
    mathcalA: string
        Numerical prefactor to the precession period.
    """

    r,xi=toarray(r,xi)
    eta=symmetricmassratio(q)
    mathcalA = (3/2)*(1/(r**3*eta**0.5))*(1-(xi/r**0.5))

    return mathcalA



# J, r, xi, q, chi1, chi2 or Sminus2, Splus2, S32, a?
def dS2dtsquared(S,J,r,q,chi1,chi2):
    """
    Squared time derivative of the squared total spin, on the precession timescale.

    Parameters
    ----------
    S: float
        Magnitude of the total spin.

    J: float
        Magnitude of the total angular momentum.

    r: float
        Orbital separation.

    q: float
        Mass ratio.

    chi1: float
        Dimensionless spin of the primary.

    chi2: float
        Dimensionless spin of the secondary.

    Returns
    -------
    float
        Squared time derivative of the squared total spin.
    """

    mathcalA = Speriod_prefactor(r,xi,q)
    Sminus2,Splus2,S32 = S2roots(J,r,xi,q,chi1,chi2)

    return - mathcalA**2 * (S**2-Splus2) * (S**2-Sminus2) * (S**2-S32)


def dS2dt(S,J,r,q,chi1,chi2):
    """
    Time derivative of the squared total spin, on the precession timescale.

    Parameters
    ----------
    S: float
        Magnitude of the total spin.

    J: float
        Magnitude of the total angular momentum.

    r: float
        Orbital separation.

    q: float
        Mass ratio.

    chi1: float
        Dimensionless spin of the primary.

    chi2: float
        Dimensionless spin of the secondary.

    Returns
    -------
    float
        Time derivative of the squared total spin.
    """

    return dS2dtsquared(S,J,r,q,chi1,chi2)**0.5


def dSdt(S,J,r,q,chi1,chi2):
    """
    Ttime derivative of the total spin, on the precession timescale.

    Parameters
    ----------
    S: float
        Magnitude of the total spin.

    J: float
        Magnitude of the total angular momentum.

    r: float
        Orbital separation.

    q: float
        Mass ratio.

    chi1: float
        Dimensionless spin of the primary.

    chi2: float
        Dimensionless spin of the secondary.

    Returns
    -------
    float
        Time derivative of the total spin.
    """

    return dS2dt(S,J,r,q,chi1,chi2) / (2*S)


def elliptic_parameter(Sminus2,Splus2,S32):
    """
    Parameter m entering elliptic functiosn for the evolution of S.

    Parameters
    ----------
    Sminus2, Splus2, S32: floats
        Roots of d(S^2)/dt=0 with S32<=Sminus2<=Splus2.

    Returns
    -------
    m: string
        Parameter of the ellptic functions.
    """

    m = (Splus2-Sminus2)/(Splus2-S32)

    return m



def Speriod(J,r,xi,q,chi1,chi2):
    """
    Period of S as it oscillates from S- to S+ and back to S-.

    Parameters
    ----------
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
    tau: string
        Nutation period.
    """

    mathcalA=Speriod_prefactor(r,xi,q)
    Sminus2,Splus2,S32 = S2roots(J,r,xi,q,chi1,chi2)
    m = elliptic_parameter(Sminus2,Splus2,S32)
    tau = 4*scipy.special.ellipk(m) / (mathcalA* (Splus2-S32)**0.5)

    return tau


def Soft(t,J,r,xi,q,chi1,chi2):
    """
    Evolution of S on the precessional timescale (without radiation reaction).

    Parameters
    ----------
    t: float, array
        Time
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
    S: float
        Magnitude of the total spin.
    """

    t=toarray(t)
    mathcalA=Speriod_prefactor(r,xi,q)
    Sminus2,Splus2,S32 = S2roots(J,r,xi,q,chi1,chi2)
    m = elliptic_parameter(Sminus2,Splus2,S32)
    sn,cn,dn,pn = scipy.special.ellipj(t.T*mathcalA*(Splus2-S32)**0.5/2,m)
    S2 = Sminus2 + (Splus2-Sminus2)*((Sminus2-S32)/(Splus2-S32)) *(sn/dn)**2
    S=S2.T**0.5

    return S


def S2av_mfactor(m):
    """
    Factor depending on the elliptic parameter in the precession averaged squared total spin.

    Parameters
    ----------
    m: float
        Elliptic parameter.

    Returns
    -------
    mfactor: float
        Value of the factor for the given m, (1 - E(m)/K(m)) / m.
        For m=0 the factor is 1/2.
    """

    m=toarray(m)

    with warnings.catch_warnings(): #Filter out warning for m=0
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        mfactor = (1- scipy.special.ellipe(m)/scipy.special.ellipk(m))/m

    # If m=0 return the limit 1/2
    return np.where(m==0, 1/2, mfactor)


def S2av(J, r, xi, q, chi1, chi2):
    """
    Analytic precession averaged expression for the squared total spin.

    Parameters
    ----------
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
    """

    Sminus2, Splus2, S32 = S2roots(J, r, xi, q, chi1, chi2)
    m = elliptic_parameter(Sminus2, Splus2, S32)
    S2 = Splus2 - (Splus2-Sminus2)*S2av_mfactor(m)

    return S2


def S2av_NEW(kappa, u, xi, q, chi1, chi2):
    """
    Analytic precession averaged expression for the squared total spin.

    Parameters
    ----------
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
    """

    Sminus2, Splus2, S32 = S2roots_NEW(kappa, u, xi, q, chi1, chi2)
    m = elliptic_parameter(Sminus2, Splus2, S32)
    S2 = Splus2 - (Splus2-Sminus2)*S2av_mfactor(m)

    return S2


def eval_kappa(J, r, q):
    """
    Change of dependant variable to regularize the infinite orbital separation
    limit of the precession-averaged evolution equation.

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
    kappa: float
        New dependant variable, (J^2-L^2)/(2L).
    """

    J = toarray(J)
    L = angularmomentum(r, q)
    kappa = (J**2 - L**2) / (2*L)

    return kappa


def eval_u(r, q):
    """
    Change of independant variable to regularize the infinite orbital separation
    limit of the precession-averaged evolution equation.

    Parameters
    ----------
    r: float
        Binary separation.

    q: float
        Mass ratio: 0 <= q <= 1.

    Returns
    -------
    u: float
        New independant variable, 1/(2L).
    """

    L = angularmomentum(r, q)
    u = 1 / (2*L)

    return u


def eval_kappainf(theta1inf, theta2inf, q, chi1, chi2):
    """
    Infinite orbital separation limit of the parameter kappa.

    Parameters
    ----------
    theta1inf: float
        Asymptotic value of the angle theta1 between S1 and L.

    theta2inf: float
        Asymptotic value of the angle theta2 between S2 and L.

    q: float
        Mass ratio: 0 <= q <= 1.

    chi1: float
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.

    chi2: float
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.

    Returns
    -------
    kappainf: float
        Asymptotic value of kappa.
    """

    S1, S2 = spinmags(q, chi1, chi2)
    kappainf = S1*np.cos(theta1inf) + S2*np.cos(theta2inf)

    return kappainf


def eval_costheta1inf(kappainf, xi, q, chi1, chi2):
    """
    Infinite orbital separation limit of the cosine of the angle between the
    orbital angular momentum and the primary spin.

    Parameters
    ----------
    kappainf: float
        Asymptotic value of kappa.

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
    costheta1inf: float
        Asymptotic value of the cosine of the angle between the orbital angular
        momentum and the primary spin.
    """

    S1, S2 = spinmags(q, chi1, chi2)
    costheta1inf = (-xi + kappainf*(1+1/q)) / (S1*(1/q-q))

    return costheta1inf


def eval_theta1inf(kappainf, xi, q, chi1, chi2):
    """
    Infinite orbital separation limit of the angle between the orbital angular
    momentum and the primary spin.

    Parameters
    ----------
    kappainf: float
        Asymptotic value of kappa.

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
    theta1inf: float
        Asymptotic value of the angle between the orbital angular momentum and
        the primary spin.
    """

    costheta1inf = eval_costheta1inf(kappainf, xi, q, chi1, chi2)
    theta1inf = np.arccos(costheta1inf)

    return theta1inf


def eval_costheta2inf(kappainf, xi, q, chi1, chi2):
    """
    Infinite orbital separation limit of the cosine of the angle between the
    orbital angular momentum and the secondary spin.

    Parameters
    ----------
    kappainf: float
        Asymptotic value of kappa.

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
    costheta2inf: float
        Asymptotic value of the cosine of the angle between the orbital angular
        momentum and the secondary spin.
    """

    S1, S2 = spinmags(q, chi1, chi2)
    costheta2inf = (xi - kappainf*(1+q)) / (S2*(1/q-q))

    return costheta2inf


def eval_theta2inf(kappainf, xi, q, chi1, chi2):
    """
    Infinite orbital separation limit of the angle between the orbital angular
    momentum and the secondary spin.

    Parameters
    ----------
    kappainf: float
        Asymptotic value of kappa.

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
    theta2inf: float
        Asymptotic value of the angle between the orbital angular momentum and
        the secondary spin.
    """

    costheta2inf = eval_costheta2inf(kappainf, xi, q, chi1, chi2)
    theta2inf = np.arccos(costheta2inf)

    return theta2inf


## TODO: not needed with S2roots_NEW modification for sigma6=0
def S2rootsinf(theta1inf, theta2inf, q, chi1, chi2):
    """
    Infinite orbital separation limit of the roots of the cubic equation in S^2.

    Parameters
    ----------
    theta1inf: float
        Asymptotic value of the angle between the orbital angular momentum and
        the primary spin.

    theta2inf: float
        Asymptotic value of the angle between the orbital angular momentum and
        the secondary spin.

    q: float
        Mass ratio: 0 <= q <= 1.

    chi1: float
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.

    chi2: float
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.

    Returns
    -------
    Sminus2inf: float
        Asymptotic value of the root Sminus2.

    Splus2inf: float
        Asymptotic value of the root Splus2.

    S32inf: float
        Asymptotic value of the root S32, -inf.
    """

    S1, S2 = spinmags(q, chi1, chi2)
    costheta1inf = np.cos(theta1inf)
    costheta2inf = np.cos(theta2inf)
    sintheta1inf = np.sin(theta1inf)
    sintheta2inf = np.sin(theta2inf)
    Sminus2inf = S1**2 + S2**2 + 2*S1*S2*(costheta1inf*costheta2inf - sintheta1inf*sintheta2inf)
    Splus2inf = S1**2 + S2**2 + 2*S1*S2*(costheta1inf*costheta2inf + sintheta1inf*sintheta2inf)
    S32inf = -np.inf

    return toarray([Sminus2inf, Splus2inf, S32inf])


def S2rootsinf_NEW(kappainf, xi, q, chi1, chi2):
    """
    """

    uinf = 0.0
    Sminus2inf, Splus2inf, S32inf = S2roots_NEW(kappainf, uinf, xi, q, chi1, chi2)

    return toarray([Sminus2inf, Splus2inf, S32inf])


def S2avinf(theta1inf, theta2inf, q, chi1, chi2):
    """
    Infinite orbital separation limit of the precession averaged values of S^2.

    Parameters
    ----------
    theta1inf: float
        Asymptotic value of the angle between the orbital angular momentum and
        the primary spin.

    theta2inf: float
        Asymptotic value of the angle between the orbital angular momentum and
        the secondary spin.

    q: float
        Mass ratio: 0 <= q <= 1.

    chi1: float
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.

    chi2: float
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.

    Returns
    -------
    S2avinf: flaot
        Asymptotic value of S2av.
    """

    S1, S2 = spinmags(q, chi1, chi2)
    S2inf = S1**2 + S2**2 + 2*S1*S2*np.cos(theta1inf)*np.cos(theta2inf)

    return S2inf


def S2avinf_NEW(kappainf, xi, q, chi1, chi2):
    """
    """

    #S1, S2 = spinmags(q, chi1, chi2)
    #eta = symmetricmassratio(q)
    #S2inf = S1**2 + S2**2 + (2.0*q/(1.0-q)**2)*(kappainf*(xi-kappainf)-xi**2*eta)

    uinf = 0.0
    Sminus2, Splus2, S32 = S2roots_NEW(kappainf, uinf, xi, q, chi1, chi2)
    S2inf = (Sminus2+Splus2) / 2.0

    return S2inf


# TODO: write the integrator. First understand how the S2 roots behave at r->infinity. Write another function for solving the quadratic instead of the cubic?
#def kappaofu():
#    scipy.integrate.odeint(S2av, kappa_initial, u_outputs, args=(xi,q,chi1,chi2))
def kappaofu(kappa0, u, xi, q, chi1, chi2):
    """
    """

    kappa = scipy.integrate.odeint(S2av_NEW, kappa0, u, args=(xi,q,chi1,chi2))

    return toarray(kappa)


## TODO: A function to precession-average a generic quantity
def precession_average(J, r, xi, q, chi1, chi2, func, *args, **kwargs):
    """
    Average a function over a precession cycle.

    Parameters
    ----------
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

    func: function
        Function to precession-average, with call func(S**2, *args, **kwargs).

    *args: tuple
        Extra arguments to pass to func.

    **kwargs: tuple
        Extra keyword arguments to pass to func.

    Returns
    -------
    func_av: float
        Precession averaged value of func.
    """

    #TODO: add kwargs, not only args

    Sminus2, Splus2, S32 = S2roots(J, r, xi, q, chi1, chi2)
    a = Speriod_prefactor(r, xi ,q)

    def _integrand(S2):
        return func(S2, *args, **kwargs) / np.abs(dS2dt(S2, Sminus2, Splus2, S32, a))

    tau = Speriod(J, r, xi, q, chi1, chi2)
    func_av = (2/tau) * scipy.integrate.quad(_integrand, Sminus2, Splus2)[0]

    return func_av


def r_updown(q, chi1, chi2):
    """
    The critical separations r_ud+/- marking the region of the up-down precessional instability.

    Parameters
    ----------
    q: float
        Mass ratio m2/m1, 0 <= q <= 1.

    chi1:
        Dimensionless spin of the primary black hole, 0 <= chi1 <= 1.

    chi2:
        Dimensionless spin of the secondary black hole, 0 <= chi2 <= 1.

    Returns
    -------
    r_udp: float
        Outer critical separation marking the instability onset.

    r_udm: float
        Inner critical separation marking the end of the unstable region.

    """

    q, chi1, chi2 = toarray(q, chi1, chi2)
    r_plus = (chi1**.5+(q*chi2)**.5)**4./(1.-q)**2.
    r_minus = (chi1**.5-(q*chi2)**.5)**4./(1.-q)**2.

    return np.array([r_plus, r_minus])


def omega2_aligned(r, q, chi1, chi2, alpha1, alpha2):
    """
    Squared oscillation frequency of a given perturbed aligned-spin binary.

    Parameters
    ----------
    r: float
        Orbital separation.

    q: float
        Mass ratio m2/m1, 0 <= q <= 1.

    chi1: float
        Dimensionless spin of the primary black hole, 0 <= chi1 <= 1.

    chi2: float
        Dimensionless spin of the secondary black hole, 0 <= chi2 <= 1.

    alpha1: int
        Alignment of the primary black hole, +1 for up or -1 for down.

    alpha2: int
        Alignment of the secondary black hole, +1 for up or -1 for down.

    Returns
    -------
    omega2: float
        Squared oscillation frequency of the given aligned binary.
    """

    L = angularmomentum(r, q)
    S1, S2 = spinmags(q, chi1, chi2)
    # Slightly rewritten from Eq. 18 in arXiv:2003.02281 to regularized for q=1
    a = (3*q**5/(2*(1+q)**11*L**7))**2
    b = L**2*(1-q)**2 - 2*L*(q*alpha1*S1-alpha2*S2)*(1-q) + (q*alpha1*S1+alpha2*S2)**2
    c = (L - (q*alpha1*S1+alpha2*S2)/(1+q))**2
    omega2 = a*b*c

    return omega2


def omega2_upup(r, q, chi1, chi2):
    """
    Squared oscillation frequency of a perturbed up-up binary.

    Parameters
    ----------
    r: float
        Orbital separation.

    q: float
        Mass ratio m2/m1, 0 <= q <= 1.

    chi1: float
        Dimensionless spin of the primary black hole, 0 <= chi1 <= 1.

    chi2: float
        Dimensionless spin of the secondary black hole, 0 <= chi2 <= 1.

    Returns
    -------
    omega2: float
        Squared oscillation frequency of the up-up binary, omega2 > 0 (for r > M).
    """

    omega2 = omega2_aligned(r, q, chi1, chi2, 1, 1)

    return omega2


def omega2_downdown(r, q, chi1, chi2):
    """
    Squared oscillation frequency of a perturbed down-down binary.

    Parameters
    ----------
    r: float
        Orbital separation.

    q: float
        Mass ratio m2/m1, 0 <= q <= 1.

    chi1: float
        Dimensionless spin of the primary black hole, 0 <= chi1 <= 1.

    chi2: float
        Dimensionless spin of the secondary black hole, 0 <= chi2 <= 1.

    Returns
    -------
    omega2: float
        Squared oscillation frequency of the down-down binary, omega2 > 0 (for r > M).
    """

    omega2 = omega2_aligned(r, q, chi1, chi2, -1, -1)

    return omega2


def omega2_downup(r, q, chi1, chi2):
    """
    Squared oscillation frequency of a perturbed down-up binary.

    Parameters
    ----------
    r: float
        Orbital separation.

    q: float
        Mass ratio m2/m1, 0 <= q <= 1.

    chi1: float
        Dimensionless spin of the primary black hole, 0 <= chi1 <= 1.

    chi2: float
        Dimensionless spin of the secondary black hole, 0 <= chi2 <= 1.

    Returns
    -------
    omega2: float
        Squared oscillation frequency of the down-up binary, omega2 > 0 (for r > M).
    """

    omega2 = omega2_aligned(r, q, chi1, chi2, -1, 1)

    return omega2


def omega2_updown(r, q, chi1, chi2):
    """
    Squared oscillation frequency of a perturbed up-down binary.

    Parameters
    ----------
    r: float
        Orbital separation.

    q: float
        Mass ratio m2/m1, 0 <= q <= 1.

    chi1: float
        Dimensionless spin of the primary black hole, 0 <= chi1 <= 1.

    chi2: float
        Dimensionless spin of the secondary black hole, 0 <= chi2 <= 1.

    Returns
    -------
    omega2: float
        Squared oscillation frequency of the up-down binary, omega2 < 0 when r_ud+ > r > r_ud-.
    """

    omega2 = omega2_aligned(r, q, chi1, chi2, 1, -1)

    return omega2


# TODO: nutation
def r_wide(q, chi1, chi2):
    """
    The critical separation r_wide below which the binary component with
    smaller dimensionless spin may undergo wide nutations.

    Parameters
    ----------
    q: float
        Mass ratio m2/m1, 0 <= q <= 1.

    chi1:
        Dimensionless spin of the primary black hole, 0 <= chi1 <= 1.

    chi2:
        Dimensionless spin of the secondary black hole, 0 <= chi2 <= 1.

    Returns
    -------
    r_wide: float
        Critical orbital separation for wide nutation.
        If chi1 < chi2 (chi1 > chi2) the primary (secondary) spin may undergo
        wide nutations.

    """

    q, chi1, chi2 = toarray(q, chi1, chi2)
    r_wide = ((q*chi2 - chi1) / (1-q))**2

    return r_wide



if __name__ == '__main__':

    r=[10,10]
    xi=[0.35,-0.675]
    q=[0.8,0.2]
    chi1=[1,1]
    chi2=[1,1]
    J=[1,0.23]

    print(S2roots(J,r,xi,q,chi1,chi2))
    #print( dSdtprefactor(r,xi,q) )
    kappa=eval_kappa(J,r,q)
    u=eval_u(r,q)
    print(S2roots_NEW(kappa,u,xi,q,chi1,chi2))


    #print(Jresonances(r[0],xi[0],q[0],chi1[0],chi2[0]))
    #print(Jresonances(r[1],xi[1],q[1],chi1[1],chi2[1]))
    #print(Jresonances(r,xi,q,chi1,chi2))
    #print(Jlimits(r=r,xi=xi,q=q,chi1=chi1,chi2=chi2))
    #print(Jlimits(r=r,q=q,chi1=chi1,chi2=chi2))


    #print(xiresonances(J[0],r[0],q[0],chi1[0],chi2[0]))
    #print(xiresonances(J[1],r[1],q[1],chi1[1],chi2[1]))
    #print(xiresonances(J,r,q,chi1,chi2))

    #
    # t0=time.time()
    # [S2roots(J[0],r[0],xi[0],q[0],chi1[0],chi2[0]) for i in range(100)]
    # #print(Slimits_plusminus(J,r,xi,q,chi1,chi2))
    # print(time.time()-t0)
    #
    # @np.vectorize
    # def ell(x):
    #   if x==0:
    #     return 1/2
    #   else:
    #       return (1- scipy.special.ellipe(x)/scipy.special.ellipk(x))/x
    #
    # # Should be equivalent to
    # def ell(x):
    #     return np.where(x==0, 1/2, (1- scipy.special.ellipe(x)/scipy.special.ellipk(x))/x)
    #
    # t0=time.time()
    # [ell(0.5) for i in range(100)]
    # print(time.time()-t0)

    #print(xilimits(q=q,chi1=chi1,chi2=chi2))

    #print(xilimits(J=J,r=r,q=q,chi1=chi1,chi2=chi2))
    #S=[0.4,0.6668]

    #print(effectivepotential_plus(S,J,r,q,chi1,chi2))
    #print(effectivepotential_minus(S,J,r,q,chi1,chi2))

    #print(Slimits_cycle(J,r,xi,q,chi1,chi2))


    #M,m1,m2,S1,S2=pre.get_fixed(q[0],chi1[0],chi2[0])
    #print(pre.J_allowed(xi[0],q[0],S1[0],S2[0],r[0]))

    #print(Jresonances(r,xi,q,chi1,chi2))


    #print(Jlimits(r,q,chi1,chi2))
    #print(S2roots(J,r,xi,q,chi1,chi2))



    #print(Slimits_check([0.24,4,6],q,chi1,chi2,which='S1S2'))

    # q=[0.7,0.7]
    # chi1=[0.7,0.7]
    # chi2=[0.9,0.9]
    # r=[30,30]
    # J=[1.48,1.48]
    # xi=[0.25,0.18]
    #print("stillworks",S2roots(J,r,xi,q,chi1,chi2))


    #print(morphology(J,r,xi,q,chi1,chi2))
    #print(morphology(J[0],r[0],xi[0],q[0],chi1[0],chi2[0]))

    # theta1=[0.567,1]
    # theta2=[1,1]
    # deltaphi=[1,2]
    #S,J,xi = angles_to_conserved(theta1,theta2,deltaphi,r,q,chi1,chi2)
    #print(S,J,xi)
    #theta1,theta2,deltaphi=conserved_to_angles(S,J,r,xi,q,chi1,chi2)
    #print(theta1,theta2,deltaphi)
    #print(eval_costheta1(0.4,J[0],r[0],xi[0],q[0],chi1[0],chi2[0]))

    #print(eval_thetaL([0.5,0.6],J,r,q,chi1,chi2))

    # tau = Speriod(J[0],r[0],xi[0],q[0],chi1[0],chi2[0])
    # Smin,Smax = Slimits_plusminus(J[0],r[0],xi[0],q[0],chi1[0],chi2[0])
    # t= np.linspace(0,tau,200)
    # S= Soft([t,t],J,r,xi,q,chi1,chi2)
    #
    # #print(t)
    # print(np.shape([t,t]))
    # print(np.shape(S))
    # #S= Soft(t,J[0],r[0],xi[0],q[0],chi1[0],chi2[0])

    #print(S[1:5])

    #S= Soft(t[4],J[0],r[0],xi[0],q[0],chi1[0],chi2[0])

    #print(S)

    #import pylab as plt
    #plt.plot(t/1e5,S)
    #plt.axhline(Smin)
    #plt.axhline(Smax)
    #plt.show()
