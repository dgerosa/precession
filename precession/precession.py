"""
precession
"""

import numpy as np
import scipy, scipy.special, scipy.integrate
import sys, os, time
import warnings
import itertools
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


#### Utilities ####
def flen(x):
    # TODO: write docstrings
    #https://stackoverflow.com/a/26533085
    return getattr(x, '__len__', lambda:1)()

def toarray(*args):
    """
    Convert a series of variables to numpy arrays if necessary.

    Call
    ----
    x,y,z,... = toarray(x,y,z,...)

    Parameters
    ----------
    x,y,z,...: generic
        Any number of input quantities.

    Returns
    -------
    x,y,z,...: array
        Corresponding number of output quantities.
    """
    if flen(args) == 1 :
        return np.squeeze(args)
#    elif all(flen(x)==flen(args[0]) for x in args):
#        return np.squeeze(np.array([*args]))
    else:
        return [np.squeeze(x) for x in args]


def norm_nested(x):
    """
    Norm of 2D array of shape (x,3) along last axis.

    Call
    ----
    x = normalize_nested(x)

    Parameters
    ----------
    x : array
        Input array.

    Returns
    -------
    x : array
        Normalized array.
    """

    return np.linalg.norm(x, axis=1)


def normalize_nested(x):
    """
    Normalize 2D array (x,3) along last axis.

    Call
    ----
    x = normalize_nested(x)

    Parameters
    ----------
    x : array
        Input array.

    Returns
    -------
    x : array
        Normalized array.
    """

    return x/norm_nested(x)[:,None]


def dot_nested(x,y):
    """
    Dot product between 2D arrays along last axis.

    Call
    ----
    z = dot_nested(x,y)

    Parameters
    ----------
    x : array
        Input array.
    y : array
        Input array.

    Returns
    -------
    z : array
        Dot product array.
    """

    return np.einsum('ij,ij->i',x,y)


def sample_unitsphere(N=1):
    """
    Sample points uniformly on a sphere of unit radius. Returns array of shape (N,3).

    Call
    ----
    vec = sample_unitsphere(N = 1)

    Parameters
    ----------
    N: integer, optional (default: 1)
    	Number of samples.

    Returns
    -------
    vec: array
    	Vector in Cartesian coomponents.
    """

    vec = np.random.randn(3, N)
    vec /= np.linalg.norm(vec, axis=0)
    return vec.T


def wraproots(coefficientfunction, *args,**kwargs):
    """
    Find roots of a polynomial given coefficients, ordered according to their real part. Complex roots are masked with nans. This is a wrapper to numpy.roots.

    Call
    ----
    sols = precession.wraproots(coefficientfunction, *args,**kwargs)

    Parameters
    ----------
    coefficientfunction: callable
        Function returning  the polynomial coefficients ordered from highest to lowest degree.
    *args, **kwargs:
        Parameters of `coefficientfunction`.

    Returns
    -------
    sols: array
        Roots of the polynomial.
    """

    coeffs= coefficientfunction(*args,**kwargs)

    sols = np.array([np.sort_complex(np.roots(x)) for x in coeffs.T])
    #This avoid a for loop, but its not reliable because enforces the same dtype to all outputs
    #sols = np.sort_complex(np.apply_along_axis(np.roots,0,coeffs).T)

    sols = np.real(np.where(np.isreal(sols),sols,np.nan))

    return sols

#### Definitions ####

def eval_m1(q):
    """
    Mass of the heavier black hole in units of the total mass.

    Call
    ----
    m1 = eval_m1(q)

    Parameters
    ----------
    q: float
    	Mass ratio: 0<=q<=1.

    Returns
    -------
    m1: float
    	Mass of the primary (heavier) black hole.
    """
    q = np.atleast_1d(q)
    m1 = 1/(1+q)

    return m1

def eval_m2(q):
    """
    Mass of the lighter black hole in units of the total mass.

    Call
    ----
    m2 = eval_m2(q)

    Parameters
    ----------
    q: float
    	Mass ratio: 0<=q<=1.

    Returns
    -------
    m2: float
    	Mass of the secondary (lighter) black hole.
    """

    q = np.atleast_1d(q)
    m2 = q/(1+q)

    return m2


def masses(q):
    """
    Masses of the two black holes in units of the total mass.

    Call
    ----
    m1,m2 = masses(q)

    Parameters
    ----------
    q: float
    	Mass ratio: 0<=q<=1.

    Returns
    -------
    m1: float
    	Mass of the primary (heavier) black hole.
    m2: float
    	Mass of the secondary (lighter) black hole.
    """


    m1 = eval_m1(q)
    m2 = eval_m2(q)

    return np.stack([m1,m2])


def eval_q(m1, m2):
    """
    Mass ratio, 0 < q = m2/m1 < 1.

    Call
    ----
    q = eval_q(m1,m2)

    Parameters
    ----------
    m1: float
    	Mass of the primary (heavier) black hole.
    m2: float
    	Mass of the secondary (lighter) black hole.

    Returns
    -------
    q: float
    	Mass ratio: 0<=q<=1.
    """

    m1 = np.atleast_1d(m1)
    m2 = np.atleast_1d(m2)
    q = m2 / m1
    assert (q<1).all(), "The convention used in this code is q=m2/m1<1."

    return q

def eval_eta(q):
    """
    Symmetric mass ratio eta = m1*m2/(m1+m2)^2 = q/(1+q)^2.

    Call
    ----
    eta = eval_eta(q)

    Parameters
    ----------
    q: float
    	Mass ratio: 0<=q<=1.

    Returns
    -------
    eta: float
    	Symmetric mass ratio 0<=eta<=1/4.
    """

    q = np.atleast_1d(q)
    eta = q/(1+q)**2

    return eta

def eval_S1(q,chi1):
    """
    Spin angular momentum of the heavier black hole.

    Call
    ----
    S1 = eval_S1(q,chi1)

    Parameters
    ----------
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.

    Returns
    -------
    S1: float
    	Magnitude of the primary spin.
    """

    chi1 = np.atleast_1d(chi1)
    S1 = chi1*(eval_m1(q))**2

    return S1

def eval_S2(q,chi2):
    """
    Spin angular momentum of the lighter black hole.

    Call
    ----
    S2 = eval_S2(q,chi2)

    Parameters
    ----------
    q: float
    	Mass ratio: 0<=q<=1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    S2: float
    	Magnitude of the secondary spin.
    """

    chi2 = np.atleast_1d(chi2)
    S2 = chi2*(eval_m2(q))**2

    return S2


def spinmags(q,chi1,chi2):
    """
    Spins of the black holes in units of the total mass.

    Call
    ----
    S1,S2 = spinmags(q,chi1,chi2)

    Parameters
    ----------
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    S1: float
    	Magnitude of the primary spin.
    S2: float
    	Magnitude of the secondary spin.
    """

    S1 = eval_S1(q,chi1)
    S2 = eval_S2(q,chi2)

    return np.stack([S1,S2])


def eval_L(r,q):
    """
    Newtonian angular momentum of the binary.

    Call
    ----
    L = eval_L(r,q)

    Parameters
    ----------
    r: float
    	Binary separation.
    q: float
    	Mass ratio: 0<=q<=1.

    Returns
    -------
    L: float
    	Magnitude of the Newtonian orbital angular momentum.
    """

    r = np.atleast_1d(r)
    L = eval_m1(q)*eval_m2(q)*r**0.5

    return L


def eval_v(r):
    """
    Newtonian orbital velocity of the binary.

    Call
    ----
    v = eval_v(r)

    Parameters
    ----------
    r: float
    	Binary separation.

    Returns
    -------
    v: float
    	Newtonian orbital velocity.
    """

    r = np.atleast_1d(r)
    v= 1/r**0.5

    return v


def eval_r(L=None, u=None, q=None):
    """
    Orbital separation of the binary. Valid inputs are either (L,q) or (u,q).

    Call
    ----
    r = eval_r(L=None,u=None,q=None)

    Parameters
    ----------
    L: float, optional (default: None)
    	Magnitude of the Newtonian orbital angular momentum.
    u: float, optional (default: None)
    	Compactified separation 1/(2L).
    q: float, optional (default: None)
    	Mass ratio: 0<=q<=1.

    Returns
    -------
    r: float
    	Binary separation.
    """

    if L is not None and u is None and q is not None:

        L = np.atleast_1d(L)
        m1, m2 = masses(q)
        r = (L / (m1 * m2))**2

    elif L is None and u is not None and q is not None:

        u = np.atleast_1d(u)
        r= (2*eval_m1(q)*eval_m2(q)*u)**(-2)

    else:
        raise TypeError("Provide either (L,q) or (u,q).")

    return r


#### Limits ####

def Jlimits_LS1S2(r,q,chi1,chi2):
    """
    Limits on the magnitude of the total angular momentum due to the vector relation J=L+S1+S2.

    Call
    ----
    Jmin,Jmax = Jlimits_LS1S2(r,q,chi1,chi2)

    Parameters
    ----------
    r: float
    	Binary separation.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    Jmin: float
    	Minimum value of the total angular momentum J.
    Jmax: float
    	Maximum value of the total angular momentum J.
    """

    S1,S2 = spinmags(q,chi1,chi2)
    L = eval_L(r,q)
    Jmin = np.maximum.reduce([np.zeros(flen(L)), L-S1-S2, np.abs(S1-S2)-L])
    Jmax = L+S1+S2

    return np.stack([Jmin,Jmax])


def kappadiscriminant_coefficients(u,xi,q,chi1,chi2):
    """
    Coefficients of the quintic equation in kappa that defines the spin-orbit resonances.

    Call
    ----
    coeff5,coeff4,coeff3,coeff2,coeff1,coeff0 = kappadiscriminant_coefficients(u,xi,q,chi1,chi2)

    Parameters
    ----------
    u: float
    	Compactified separation 1/(2L).
    xi: float
    	Effective spin.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    coeff5: float
    	Coefficient to the x^5 term in polynomial.
    coeff4: float
    	Coefficient to the x^4 term in polynomial.
    coeff3: float
    	Coefficient to the x^3 term in polynomial.
    coeff2: float
    	Coefficient to the x^2 term in polynomial.
    coeff1: float
    	Coefficient to the x^1 term in polynomial.
    coeff0: float
    	Coefficient to the x^0 term in polynomial.
    """

    u=np.atleast_1d(u)
    q=np.atleast_1d(q)
    xi=np.atleast_1d(xi)
    S1,S2= spinmags(q,chi1,chi2)

    coeff0 = \
    ( 16 * ( ( -1 + ( q )**( 2 ) ) )**( 2 ) * ( ( ( -1 + ( q )**( 2 ) ) \
    )**( 2 ) * ( S1 )**( 2 ) + -1 * ( q )**( 2 ) * ( xi )**( 2 ) ) * ( ( \
    ( -1 + ( q )**( 2 ) ) )**( 2 ) * ( S2 )**( 2 ) + -1 * ( q )**( 2 ) * \
    ( xi )**( 2 ) ) + ( -32 * q * ( ( 1 + q ) )**( 2 ) * u * xi * ( -5 * \
    ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( ( S2 )**( 4 ) + ( ( q )**( 8 ) * ( \
    ( S1 )**( 4 ) + -5 * ( S1 )**( 2 ) * ( S2 )**( 2 ) ) + ( q * ( ( S1 \
    )**( 4 ) + -1 * ( S2 )**( 4 ) ) + ( ( q )**( 7 ) * ( -1 * ( S1 )**( 4 \
    ) + ( S2 )**( 4 ) ) + ( -1 * ( q )**( 3 ) * ( ( S1 )**( 2 ) + -1 * ( \
    S2 )**( 2 ) ) * ( 3 * ( S1 )**( 2 ) + ( 3 * ( S2 )**( 2 ) + 2 * ( xi \
    )**( 2 ) ) ) + ( ( q )**( 5 ) * ( ( S1 )**( 2 ) + -1 * ( S2 )**( 2 ) \
    ) * ( 3 * ( S1 )**( 2 ) + ( 3 * ( S2 )**( 2 ) + 2 * ( xi )**( 2 ) ) ) \
    + ( ( q )**( 2 ) * ( -1 * ( S1 )**( 4 ) + ( -3 * ( S2 )**( 4 ) + ( 3 \
    * ( S2 )**( 2 ) * ( xi )**( 2 ) + 5 * ( S1 )**( 2 ) * ( 4 * ( S2 )**( \
    2 ) + ( xi )**( 2 ) ) ) ) ) + ( ( q )**( 6 ) * ( -3 * ( S1 )**( 4 ) + \
    ( -1 * ( S2 )**( 4 ) + ( 5 * ( S2 )**( 2 ) * ( xi )**( 2 ) + ( S1 \
    )**( 2 ) * ( 20 * ( S2 )**( 2 ) + 3 * ( xi )**( 2 ) ) ) ) ) + ( q \
    )**( 4 ) * ( 3 * ( S1 )**( 4 ) + ( 3 * ( S2 )**( 4 ) + ( -8 * ( S2 \
    )**( 2 ) * ( xi )**( 2 ) + ( -4 * ( xi )**( 4 ) + -2 * ( S1 )**( 2 ) \
    * ( 15 * ( S2 )**( 2 ) + 4 * ( xi )**( 2 ) ) ) ) ) ) ) ) ) ) ) ) ) ) \
    ) + ( ( u )**( 2 ) * ( -32 * q * ( 1 + q ) * u * xi * ( 4 * ( q )**( \
    9 ) * ( S1 )**( 2 ) * ( 3 * ( S1 )**( 4 ) + ( -8 * ( S1 )**( 2 ) * ( \
    S2 )**( 2 ) + ( S2 )**( 4 ) ) ) + ( 4 * ( S2 )**( 2 ) * ( ( S1 )**( 4 \
    ) + ( -8 * ( S1 )**( 2 ) * ( S2 )**( 2 ) + 3 * ( S2 )**( 4 ) ) ) + ( \
    -1 * q * ( ( S1 )**( 6 ) + ( 21 * ( S1 )**( 4 ) * ( S2 )**( 2 ) + ( \
    43 * ( S1 )**( 2 ) * ( S2 )**( 4 ) + -49 * ( S2 )**( 6 ) ) ) ) + ( ( \
    q )**( 8 ) * ( 49 * ( S1 )**( 6 ) + ( -43 * ( S1 )**( 4 ) * ( S2 )**( \
    2 ) + ( -21 * ( S1 )**( 2 ) * ( S2 )**( 4 ) + -1 * ( S2 )**( 6 ) ) ) \
    ) + ( ( q )**( 7 ) * ( 37 * ( S1 )**( 6 ) + ( 23 * ( S2 )**( 6 ) + ( \
    -4 * ( S2 )**( 4 ) * ( xi )**( 2 ) + ( 5 * ( S1 )**( 4 ) * ( 9 * ( S2 \
    )**( 2 ) + 4 * ( xi )**( 2 ) ) + ( S1 )**( 2 ) * ( -41 * ( S2 )**( 4 \
    ) + 16 * ( S2 )**( 2 ) * ( xi )**( 2 ) ) ) ) ) ) + ( ( q )**( 2 ) * ( \
    23 * ( S1 )**( 6 ) + ( 37 * ( S2 )**( 6 ) + ( 20 * ( S2 )**( 4 ) * ( \
    xi )**( 2 ) + ( -1 * ( S1 )**( 4 ) * ( 41 * ( S2 )**( 2 ) + 4 * ( xi \
    )**( 2 ) ) + ( S1 )**( 2 ) * ( 45 * ( S2 )**( 4 ) + 16 * ( S2 )**( 2 \
    ) * ( xi )**( 2 ) ) ) ) ) ) + ( ( q )**( 6 ) * ( -75 * ( S1 )**( 6 ) \
    + ( 63 * ( S2 )**( 6 ) + ( 48 * ( S2 )**( 4 ) * ( xi )**( 2 ) + ( ( \
    S1 )**( 4 ) * ( 53 * ( S2 )**( 2 ) + -40 * ( xi )**( 2 ) ) + ( S1 \
    )**( 2 ) * ( 23 * ( S2 )**( 4 ) + 24 * ( S2 )**( 2 ) * ( xi )**( 2 ) \
    ) ) ) ) ) + ( ( q )**( 3 ) * ( 63 * ( S1 )**( 6 ) + ( -5 * ( S2 )**( \
    4 ) * ( 15 * ( S2 )**( 2 ) + 8 * ( xi )**( 2 ) ) + ( ( S1 )**( 4 ) * \
    ( 23 * ( S2 )**( 2 ) + 48 * ( xi )**( 2 ) ) + ( S1 )**( 2 ) * ( 53 * \
    ( S2 )**( 4 ) + 24 * ( S2 )**( 2 ) * ( xi )**( 2 ) ) ) ) ) + ( ( q \
    )**( 5 ) * ( -111 * ( S1 )**( 6 ) + ( 3 * ( S2 )**( 6 ) + ( 28 * ( S2 \
    )**( 4 ) * ( xi )**( 2 ) + ( 16 * ( S2 )**( 2 ) * ( xi )**( 4 ) + ( \
    -3 * ( S1 )**( 4 ) * ( 5 * ( S2 )**( 2 ) + 28 * ( xi )**( 2 ) ) + ( \
    S1 )**( 2 ) * ( 27 * ( S2 )**( 4 ) + ( -8 * ( S2 )**( 2 ) * ( xi )**( \
    2 ) + -32 * ( xi )**( 4 ) ) ) ) ) ) ) ) + ( q )**( 4 ) * ( 3 * ( S1 \
    )**( 6 ) + ( ( S1 )**( 4 ) * ( 27 * ( S2 )**( 2 ) + 28 * ( xi )**( 2 \
    ) ) + ( ( S1 )**( 2 ) * ( -15 * ( S2 )**( 4 ) + ( -8 * ( S2 )**( 2 ) \
    * ( xi )**( 2 ) + 16 * ( xi )**( 4 ) ) ) + -1 * ( S2 )**( 2 ) * ( 111 \
    * ( S2 )**( 4 ) + ( 84 * ( S2 )**( 2 ) * ( xi )**( 2 ) + 32 * ( xi \
    )**( 4 ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + -16 * ( ( q )**( 12 ) * ( S1 \
    )**( 2 ) * ( ( S1 )**( 4 ) + ( -10 * ( S1 )**( 2 ) * ( S2 )**( 2 ) + \
    ( S2 )**( 4 ) ) ) + ( ( S2 )**( 2 ) * ( ( S1 )**( 4 ) + ( -10 * ( S1 \
    )**( 2 ) * ( S2 )**( 2 ) + ( S2 )**( 4 ) ) ) + ( 2 * ( q )**( 11 ) * \
    ( ( S1 )**( 6 ) + ( -22 * ( S1 )**( 4 ) * ( S2 )**( 2 ) + -11 * ( S1 \
    )**( 2 ) * ( S2 )**( 4 ) ) ) + ( q * ( -22 * ( S1 )**( 4 ) * ( S2 \
    )**( 2 ) + ( -44 * ( S1 )**( 2 ) * ( S2 )**( 4 ) + 2 * ( S2 )**( 6 ) \
    ) ) + ( 2 * ( q )**( 9 ) * ( -4 * ( S1 )**( 6 ) + ( ( S2 )**( 6 ) + ( \
    19 * ( S2 )**( 4 ) * ( xi )**( 2 ) + ( 2 * ( S1 )**( 2 ) * ( S2 )**( \
    2 ) * ( 11 * ( S2 )**( 2 ) + ( xi )**( 2 ) ) + ( S1 )**( 4 ) * ( 77 * \
    ( S2 )**( 2 ) + 43 * ( xi )**( 2 ) ) ) ) ) ) + ( 2 * ( q )**( 3 ) * ( \
    ( S1 )**( 6 ) + ( -4 * ( S2 )**( 6 ) + ( 43 * ( S2 )**( 4 ) * ( xi \
    )**( 2 ) + ( ( S1 )**( 4 ) * ( 22 * ( S2 )**( 2 ) + 19 * ( xi )**( 2 \
    ) ) + ( S1 )**( 2 ) * ( 77 * ( S2 )**( 4 ) + 2 * ( S2 )**( 2 ) * ( xi \
    )**( 2 ) ) ) ) ) ) + ( ( q )**( 2 ) * ( ( S1 )**( 6 ) + ( -3 * ( S2 \
    )**( 6 ) + ( 23 * ( S2 )**( 4 ) * ( xi )**( 2 ) + ( -1 * ( S1 )**( 4 \
    ) * ( 61 * ( S2 )**( 2 ) + ( xi )**( 2 ) ) + -1 * ( S1 )**( 2 ) * ( \
    17 * ( S2 )**( 4 ) + 22 * ( S2 )**( 2 ) * ( xi )**( 2 ) ) ) ) ) ) + ( \
    -1 * ( q )**( 10 ) * ( 3 * ( S1 )**( 6 ) + ( -1 * ( S2 )**( 6 ) + ( ( \
    S2 )**( 4 ) * ( xi )**( 2 ) + ( ( S1 )**( 4 ) * ( 17 * ( S2 )**( 2 ) \
    + -23 * ( xi )**( 2 ) ) + ( S1 )**( 2 ) * ( 61 * ( S2 )**( 4 ) + 22 * \
    ( S2 )**( 2 ) * ( xi )**( 2 ) ) ) ) ) ) + ( 2 * ( q )**( 7 ) * ( 6 * \
    ( S1 )**( 6 ) + ( -4 * ( S2 )**( 6 ) + ( 5 * ( S2 )**( 4 ) * ( xi \
    )**( 2 ) + ( 4 * ( S2 )**( 2 ) * ( xi )**( 4 ) + ( -1 * ( S1 )**( 4 ) \
    * ( 88 * ( S2 )**( 2 ) + 67 * ( xi )**( 2 ) ) + ( S1 )**( 2 ) * ( 22 \
    * ( S2 )**( 4 ) + ( -2 * ( S2 )**( 2 ) * ( xi )**( 2 ) + -36 * ( xi \
    )**( 4 ) ) ) ) ) ) ) ) + ( -2 * ( q )**( 5 ) * ( 4 * ( S1 )**( 6 ) + \
    ( -6 * ( S2 )**( 6 ) + ( 67 * ( S2 )**( 4 ) * ( xi )**( 2 ) + ( 36 * \
    ( S2 )**( 2 ) * ( xi )**( 4 ) + ( -1 * ( S1 )**( 4 ) * ( 22 * ( S2 \
    )**( 2 ) + 5 * ( xi )**( 2 ) ) + 2 * ( S1 )**( 2 ) * ( 44 * ( S2 )**( \
    4 ) + ( ( S2 )**( 2 ) * ( xi )**( 2 ) + -2 * ( xi )**( 4 ) ) ) ) ) ) \
    ) ) + ( ( q )**( 8 ) * ( 2 * ( S1 )**( 6 ) + ( -3 * ( S2 )**( 6 ) + ( \
    104 * ( S2 )**( 4 ) * ( xi )**( 2 ) + ( 32 * ( S2 )**( 2 ) * ( xi \
    )**( 4 ) + ( ( S1 )**( 4 ) * ( 169 * ( S2 )**( 2 ) + 56 * ( xi )**( 2 \
    ) ) + 8 * ( S1 )**( 2 ) * ( 28 * ( S2 )**( 4 ) + ( 12 * ( S2 )**( 2 ) \
    * ( xi )**( 2 ) + -1 * ( xi )**( 4 ) ) ) ) ) ) ) ) + ( ( q )**( 4 ) * \
    ( -3 * ( S1 )**( 6 ) + ( 8 * ( S1 )**( 4 ) * ( 28 * ( S2 )**( 2 ) + \
    13 * ( xi )**( 2 ) ) + ( 2 * ( S2 )**( 2 ) * ( ( S2 )**( 4 ) + ( 28 * \
    ( S2 )**( 2 ) * ( xi )**( 2 ) + -4 * ( xi )**( 4 ) ) ) + ( S1 )**( 2 \
    ) * ( 169 * ( S2 )**( 4 ) + ( 96 * ( S2 )**( 2 ) * ( xi )**( 2 ) + 32 \
    * ( xi )**( 4 ) ) ) ) ) ) + 2 * ( q )**( 6 ) * ( ( S1 )**( 6 ) + ( ( \
    S2 )**( 6 ) + ( -91 * ( S2 )**( 4 ) * ( xi )**( 2 ) + ( -44 * ( S2 \
    )**( 2 ) * ( xi )**( 4 ) + ( -8 * ( xi )**( 6 ) + ( -1 * ( S1 )**( 4 \
    ) * ( 153 * ( S2 )**( 2 ) + 91 * ( xi )**( 2 ) ) + -1 * ( S1 )**( 2 ) \
    * ( 153 * ( S2 )**( 4 ) + ( 74 * ( S2 )**( 2 ) * ( xi )**( 2 ) + 44 * \
    ( xi )**( 4 ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + ( u )**( 4 \
    ) * ( -256 * ( ( 1 + q ) )**( 2 ) * ( ( -1 + ( q )**( 2 ) ) )**( 2 ) \
    * ( ( ( S1 )**( 2 ) + -1 * ( S2 )**( 2 ) ) )**( 2 ) * ( ( -1 * q * ( \
    S1 )**( 2 ) + ( S2 )**( 2 ) ) )**( 2 ) * ( u )**( 2 ) * ( ( q )**( 4 \
    ) * ( S1 )**( 2 ) + ( ( S2 )**( 2 ) + ( ( q )**( 3 ) * ( ( S1 )**( 2 \
    ) + -1 * ( S2 )**( 2 ) ) + ( q * ( -1 * ( S1 )**( 2 ) + ( S2 )**( 2 ) \
    ) + -1 * ( q )**( 2 ) * ( ( S1 )**( 2 ) + ( ( S2 )**( 2 ) + ( xi )**( \
    2 ) ) ) ) ) ) ) + ( -128 * ( -1 + q ) * q * ( ( 1 + q ) )**( 3 ) * ( \
    ( S1 )**( 2 ) + -1 * ( S2 )**( 2 ) ) * u * xi * ( -4 * ( S1 )**( 2 ) \
    * ( S2 )**( 4 ) + ( 8 * ( S2 )**( 6 ) + ( ( q )**( 6 ) * ( 8 * ( S1 \
    )**( 6 ) + -4 * ( S1 )**( 4 ) * ( S2 )**( 2 ) ) + ( -1 * q * ( S2 \
    )**( 2 ) * ( ( S1 )**( 4 ) + ( 10 * ( S1 )**( 2 ) * ( S2 )**( 2 ) + \
    -3 * ( S2 )**( 4 ) ) ) + ( ( q )**( 5 ) * ( 3 * ( S1 )**( 6 ) + ( -10 \
    * ( S1 )**( 4 ) * ( S2 )**( 2 ) + -1 * ( S1 )**( 2 ) * ( S2 )**( 4 ) \
    ) ) + ( ( q )**( 3 ) * ( -3 * ( S1 )**( 6 ) + ( 11 * ( S1 )**( 2 ) * \
    ( S2 )**( 4 ) + ( -3 * ( S2 )**( 6 ) + ( 4 * ( S2 )**( 4 ) * ( xi \
    )**( 2 ) + ( S1 )**( 4 ) * ( 11 * ( S2 )**( 2 ) + 4 * ( xi )**( 2 ) ) \
    ) ) ) ) + ( ( q )**( 2 ) * ( 5 * ( S1 )**( 6 ) + ( 5 * ( S1 )**( 4 ) \
    * ( S2 )**( 2 ) + ( -13 * ( S2 )**( 6 ) + ( -8 * ( S2 )**( 4 ) * ( xi \
    )**( 2 ) + -1 * ( S1 )**( 2 ) * ( ( S2 )**( 4 ) + -4 * ( S2 )**( 2 ) \
    * ( xi )**( 2 ) ) ) ) ) ) + ( q )**( 4 ) * ( -13 * ( S1 )**( 6 ) + ( \
    5 * ( S2 )**( 6 ) + ( -1 * ( S1 )**( 4 ) * ( ( S2 )**( 2 ) + 8 * ( xi \
    )**( 2 ) ) + ( S1 )**( 2 ) * ( 5 * ( S2 )**( 4 ) + 4 * ( S2 )**( 2 ) \
    * ( xi )**( 2 ) ) ) ) ) ) ) ) ) ) ) ) + -16 * ( ( 1 + q ) )**( 2 ) * \
    ( 8 * ( q )**( 10 ) * ( S1 )**( 4 ) * ( ( S1 )**( 4 ) + ( -4 * ( S1 \
    )**( 2 ) * ( S2 )**( 2 ) + ( S2 )**( 4 ) ) ) + ( 8 * ( S2 )**( 4 ) * \
    ( ( S1 )**( 4 ) + ( -4 * ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( S2 )**( 4 \
    ) ) ) + ( 4 * ( q )**( 9 ) * ( 9 * ( S1 )**( 8 ) + ( -13 * ( S1 )**( \
    6 ) * ( S2 )**( 2 ) + ( 7 * ( S1 )**( 4 ) * ( S2 )**( 4 ) + 5 * ( S1 \
    )**( 2 ) * ( S2 )**( 6 ) ) ) ) + ( 4 * q * ( 5 * ( S1 )**( 6 ) * ( S2 \
    )**( 2 ) + ( 7 * ( S1 )**( 4 ) * ( S2 )**( 4 ) + ( -13 * ( S1 )**( 2 \
    ) * ( S2 )**( 6 ) + 9 * ( S2 )**( 8 ) ) ) ) + ( 2 * ( q )**( 3 ) * ( \
    9 * ( S1 )**( 8 ) + ( -27 * ( S2 )**( 8 ) + ( -28 * ( S2 )**( 6 ) * ( \
    xi )**( 2 ) + ( -16 * ( S1 )**( 6 ) * ( 7 * ( S2 )**( 2 ) + ( xi )**( \
    2 ) ) + ( 2 * ( S1 )**( 4 ) * ( 53 * ( S2 )**( 4 ) + -6 * ( S2 )**( 2 \
    ) * ( xi )**( 2 ) ) + -8 * ( S1 )**( 2 ) * ( 5 * ( S2 )**( 6 ) + -3 * \
    ( S2 )**( 4 ) * ( xi )**( 2 ) ) ) ) ) ) ) + ( -2 * ( q )**( 7 ) * ( \
    27 * ( S1 )**( 8 ) + ( -9 * ( S2 )**( 8 ) + ( 16 * ( S2 )**( 6 ) * ( \
    xi )**( 2 ) + ( 4 * ( S1 )**( 6 ) * ( 10 * ( S2 )**( 2 ) + 7 * ( xi \
    )**( 2 ) ) + ( -2 * ( S1 )**( 4 ) * ( 53 * ( S2 )**( 4 ) + 12 * ( S2 \
    )**( 2 ) * ( xi )**( 2 ) ) + 4 * ( S1 )**( 2 ) * ( 28 * ( S2 )**( 6 ) \
    + 3 * ( S2 )**( 4 ) * ( xi )**( 2 ) ) ) ) ) ) ) + ( ( q )**( 8 ) * ( \
    31 * ( S1 )**( 8 ) + ( -1 * ( S2 )**( 8 ) + ( ( S1 )**( 6 ) * ( -52 * \
    ( S2 )**( 2 ) + 88 * ( xi )**( 2 ) ) + ( 2 * ( S1 )**( 4 ) * ( 69 * ( \
    S2 )**( 4 ) + -32 * ( S2 )**( 2 ) * ( xi )**( 2 ) ) + ( S1 )**( 2 ) * \
    ( -68 * ( S2 )**( 6 ) + 8 * ( S2 )**( 4 ) * ( xi )**( 2 ) ) ) ) ) ) + \
    ( -1 * ( q )**( 2 ) * ( ( S1 )**( 8 ) + ( 68 * ( S1 )**( 6 ) * ( S2 \
    )**( 2 ) + ( -31 * ( S2 )**( 8 ) + ( -88 * ( S2 )**( 6 ) * ( xi )**( \
    2 ) + ( -2 * ( S1 )**( 4 ) * ( 69 * ( S2 )**( 4 ) + 4 * ( S2 )**( 2 ) \
    * ( xi )**( 2 ) ) + ( S1 )**( 2 ) * ( 52 * ( S2 )**( 6 ) + 64 * ( S2 \
    )**( 4 ) * ( xi )**( 2 ) ) ) ) ) ) ) + ( 8 * ( q )**( 5 ) * ( 11 * ( \
    S2 )**( 6 ) * ( xi )**( 2 ) + ( 12 * ( S2 )**( 4 ) * ( xi )**( 4 ) + \
    ( ( S1 )**( 6 ) * ( 42 * ( S2 )**( 2 ) + 11 * ( xi )**( 2 ) ) + ( -3 \
    * ( S1 )**( 4 ) * ( 20 * ( S2 )**( 4 ) + ( ( S2 )**( 2 ) * ( xi )**( \
    2 ) + -4 * ( xi )**( 4 ) ) ) + ( S1 )**( 2 ) * ( 42 * ( S2 )**( 6 ) + \
    ( -3 * ( S2 )**( 4 ) * ( xi )**( 2 ) + -20 * ( S2 )**( 2 ) * ( xi \
    )**( 4 ) ) ) ) ) ) ) + ( ( q )**( 6 ) * ( -87 * ( S1 )**( 8 ) + ( 49 \
    * ( S2 )**( 8 ) + ( 112 * ( S2 )**( 6 ) * ( xi )**( 2 ) + ( -16 * ( \
    S2 )**( 4 ) * ( xi )**( 4 ) + ( 4 * ( S1 )**( 6 ) * ( 33 * ( S2 )**( \
    2 ) + -50 * ( xi )**( 2 ) ) + ( -2 * ( S1 )**( 4 ) * ( 73 * ( S2 )**( \
    4 ) + ( -104 * ( S2 )**( 2 ) * ( xi )**( 2 ) + 48 * ( xi )**( 4 ) ) ) \
    + 4 * ( S1 )**( 2 ) * ( 5 * ( S2 )**( 6 ) + ( -38 * ( S2 )**( 4 ) * ( \
    xi )**( 2 ) + 24 * ( S2 )**( 2 ) * ( xi )**( 4 ) ) ) ) ) ) ) ) ) + ( \
    q )**( 4 ) * ( 49 * ( S1 )**( 8 ) + ( 4 * ( S1 )**( 6 ) * ( 5 * ( S2 \
    )**( 2 ) + 28 * ( xi )**( 2 ) ) + ( -2 * ( S1 )**( 4 ) * ( 73 * ( S2 \
    )**( 4 ) + ( 76 * ( S2 )**( 2 ) * ( xi )**( 2 ) + 8 * ( xi )**( 4 ) ) \
    ) + ( -1 * ( S2 )**( 4 ) * ( 87 * ( S2 )**( 4 ) + ( 200 * ( S2 )**( 2 \
    ) * ( xi )**( 2 ) + 96 * ( xi )**( 4 ) ) ) + 4 * ( S1 )**( 2 ) * ( 33 \
    * ( S2 )**( 6 ) + ( 52 * ( S2 )**( 4 ) * ( xi )**( 2 ) + 24 * ( S2 \
    )**( 2 ) * ( xi )**( 4 ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) )

    coeff1 = \
    ( 32 * ( ( -1 + q ) )**( 2 ) * q * ( ( 1 + q ) )**( 4 ) * xi * ( ( q \
    )**( 4 ) * ( S1 )**( 2 ) + ( ( S2 )**( 2 ) + ( q * ( ( S1 )**( 2 ) + \
    -1 * ( S2 )**( 2 ) ) + ( ( q )**( 3 ) * ( -1 * ( S1 )**( 2 ) + ( S2 \
    )**( 2 ) ) + -1 * ( q )**( 2 ) * ( ( S1 )**( 2 ) + ( ( S2 )**( 2 ) + \
    ( xi )**( 2 ) ) ) ) ) ) ) + ( 32 * ( ( 1 + q ) )**( 2 ) * u * ( -12 * \
    q * ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( -12 * ( q )**( 9 ) * ( S1 )**( \
    2 ) * ( S2 )**( 2 ) + ( ( q )**( 10 ) * ( S1 )**( 2 ) * ( ( S1 )**( 2 \
    ) + ( S2 )**( 2 ) ) + ( ( S2 )**( 2 ) * ( ( S1 )**( 2 ) + ( S2 )**( 2 \
    ) ) + ( ( q )**( 6 ) * ( 6 * ( S1 )**( 4 ) + ( -4 * ( S2 )**( 4 ) + ( \
    9 * ( S2 )**( 2 ) * ( xi )**( 2 ) + ( -8 * ( xi )**( 4 ) + ( S1 )**( \
    2 ) * ( 2 * ( S2 )**( 2 ) + -15 * ( xi )**( 2 ) ) ) ) ) ) + ( -12 * ( \
    q )**( 5 ) * ( 3 * ( S2 )**( 2 ) * ( xi )**( 2 ) + ( 2 * ( xi )**( 4 \
    ) + 3 * ( S1 )**( 2 ) * ( 2 * ( S2 )**( 2 ) + ( xi )**( 2 ) ) ) ) + ( \
    ( q )**( 2 ) * ( ( S1 )**( 4 ) + ( -4 * ( S2 )**( 4 ) + ( 7 * ( S2 \
    )**( 2 ) * ( xi )**( 2 ) + -1 * ( S1 )**( 2 ) * ( 3 * ( S2 )**( 2 ) + \
    ( xi )**( 2 ) ) ) ) ) + ( 6 * ( q )**( 3 ) * ( 3 * ( S2 )**( 2 ) * ( \
    xi )**( 2 ) + ( S1 )**( 2 ) * ( 8 * ( S2 )**( 2 ) + 3 * ( xi )**( 2 ) \
    ) ) + ( 6 * ( q )**( 7 ) * ( 3 * ( S2 )**( 2 ) * ( xi )**( 2 ) + ( S1 \
    )**( 2 ) * ( 8 * ( S2 )**( 2 ) + 3 * ( xi )**( 2 ) ) ) + ( ( q )**( 8 \
    ) * ( -4 * ( S1 )**( 4 ) + ( ( S2 )**( 4 ) + ( -1 * ( S2 )**( 2 ) * ( \
    xi )**( 2 ) + ( S1 )**( 2 ) * ( -3 * ( S2 )**( 2 ) + 7 * ( xi )**( 2 \
    ) ) ) ) ) + ( q )**( 4 ) * ( -4 * ( S1 )**( 4 ) + ( 6 * ( S2 )**( 4 ) \
    + ( -15 * ( S2 )**( 2 ) * ( xi )**( 2 ) + ( -8 * ( xi )**( 4 ) + ( S1 \
    )**( 2 ) * ( 2 * ( S2 )**( 2 ) + 9 * ( xi )**( 2 ) ) ) ) ) ) ) ) ) ) \
    ) ) ) ) ) ) + ( ( u )**( 4 ) * ( 256 * ( -1 + q ) * ( ( 1 + q ) )**( \
    4 ) * ( q * ( S1 )**( 2 ) + -1 * ( S2 )**( 2 ) ) * u * ( 2 * ( q )**( \
    6 ) * ( S1 )**( 4 ) * ( ( S1 )**( 2 ) + ( S2 )**( 2 ) ) + ( 2 * ( S2 \
    )**( 4 ) * ( ( S1 )**( 2 ) + ( S2 )**( 2 ) ) + ( ( q )**( 5 ) * ( -3 \
    * ( S1 )**( 6 ) + ( 2 * ( S1 )**( 4 ) * ( S2 )**( 2 ) + -7 * ( S1 \
    )**( 2 ) * ( S2 )**( 4 ) ) ) + ( q * ( -7 * ( S1 )**( 4 ) * ( S2 )**( \
    2 ) + ( 2 * ( S1 )**( 2 ) * ( S2 )**( 4 ) + -3 * ( S2 )**( 6 ) ) ) + \
    ( ( q )**( 3 ) * ( 3 * ( S1 )**( 6 ) + ( 5 * ( S1 )**( 2 ) * ( S2 \
    )**( 4 ) + ( 3 * ( S2 )**( 6 ) + ( 4 * ( S2 )**( 4 ) * ( xi )**( 2 ) \
    + ( S1 )**( 4 ) * ( 5 * ( S2 )**( 2 ) + 4 * ( xi )**( 2 ) ) ) ) ) ) + \
    ( ( q )**( 2 ) * ( 5 * ( S1 )**( 6 ) + ( -7 * ( S1 )**( 4 ) * ( S2 \
    )**( 2 ) + ( -7 * ( S2 )**( 6 ) + ( -2 * ( S2 )**( 4 ) * ( xi )**( 2 \
    ) + ( S1 )**( 2 ) * ( 5 * ( S2 )**( 4 ) + -2 * ( S2 )**( 2 ) * ( xi \
    )**( 2 ) ) ) ) ) ) + -1 * ( q )**( 4 ) * ( 7 * ( S1 )**( 6 ) + ( -5 * \
    ( S2 )**( 6 ) + ( ( S1 )**( 4 ) * ( -5 * ( S2 )**( 2 ) + 2 * ( xi \
    )**( 2 ) ) + ( S1 )**( 2 ) * ( 7 * ( S2 )**( 4 ) + 2 * ( S2 )**( 2 ) \
    * ( xi )**( 2 ) ) ) ) ) ) ) ) ) ) ) + 128 * q * ( ( 1 + q ) )**( 3 ) \
    * xi * ( 8 * ( S1 )**( 2 ) * ( S2 )**( 4 ) + ( 12 * ( S2 )**( 6 ) + ( \
    4 * ( q )**( 7 ) * ( 3 * ( S1 )**( 6 ) + 2 * ( S1 )**( 4 ) * ( S2 \
    )**( 2 ) ) + ( ( q )**( 6 ) * ( -17 * ( S1 )**( 6 ) + ( -6 * ( S1 \
    )**( 4 ) * ( S2 )**( 2 ) + 3 * ( S1 )**( 2 ) * ( S2 )**( 4 ) ) ) + ( \
    q * ( 3 * ( S1 )**( 4 ) * ( S2 )**( 2 ) + ( -6 * ( S1 )**( 2 ) * ( S2 \
    )**( 4 ) + -17 * ( S2 )**( 6 ) ) ) + ( ( q )**( 3 ) * ( 9 * ( S1 )**( \
    6 ) + ( 37 * ( S2 )**( 6 ) + ( 20 * ( S2 )**( 4 ) * ( xi )**( 2 ) + ( \
    ( S1 )**( 4 ) * ( 11 * ( S2 )**( 2 ) + -12 * ( xi )**( 2 ) ) + 3 * ( \
    S1 )**( 2 ) * ( ( S2 )**( 4 ) + 4 * ( S2 )**( 2 ) * ( xi )**( 2 ) ) ) \
    ) ) ) + ( -1 * ( q )**( 5 ) * ( 21 * ( S1 )**( 6 ) + ( 20 * ( S2 )**( \
    6 ) + ( 2 * ( S1 )**( 4 ) * ( 11 * ( S2 )**( 2 ) + 6 * ( xi )**( 2 ) \
    ) + ( S1 )**( 2 ) * ( -3 * ( S2 )**( 4 ) + 8 * ( S2 )**( 2 ) * ( xi \
    )**( 2 ) ) ) ) ) + ( -1 * ( q )**( 2 ) * ( 20 * ( S1 )**( 6 ) + ( -3 \
    * ( S1 )**( 4 ) * ( S2 )**( 2 ) + ( 3 * ( S2 )**( 4 ) * ( 7 * ( S2 \
    )**( 2 ) + 4 * ( xi )**( 2 ) ) + ( S1 )**( 2 ) * ( 22 * ( S2 )**( 4 ) \
    + 8 * ( S2 )**( 2 ) * ( xi )**( 2 ) ) ) ) ) + ( q )**( 4 ) * ( 37 * ( \
    S1 )**( 6 ) + ( 9 * ( S2 )**( 6 ) + ( -12 * ( S2 )**( 4 ) * ( xi )**( \
    2 ) + ( ( S1 )**( 4 ) * ( 3 * ( S2 )**( 2 ) + 20 * ( xi )**( 2 ) ) + \
    ( S1 )**( 2 ) * ( 11 * ( S2 )**( 4 ) + 12 * ( S2 )**( 2 ) * ( xi )**( \
    2 ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) + ( u )**( 2 ) * ( 32 * q * ( ( 1 + q \
    ) )**( 2 ) * xi * ( 8 * ( q )**( 8 ) * ( S1 )**( 2 ) * ( 2 * ( S1 \
    )**( 2 ) + ( S2 )**( 2 ) ) + ( 8 * ( S2 )**( 2 ) * ( ( S1 )**( 2 ) + \
    2 * ( S2 )**( 2 ) ) + ( ( q )**( 7 ) * ( 65 * ( S1 )**( 4 ) + ( 2 * ( \
    S1 )**( 2 ) * ( S2 )**( 2 ) + -3 * ( S2 )**( 4 ) ) ) + ( q * ( -3 * ( \
    S1 )**( 4 ) + ( 2 * ( S1 )**( 2 ) * ( S2 )**( 2 ) + 65 * ( S2 )**( 4 \
    ) ) ) + ( 2 * ( q )**( 6 ) * ( 7 * ( S1 )**( 4 ) + ( -14 * ( S1 )**( \
    2 ) * ( S2 )**( 2 ) + ( 23 * ( S2 )**( 4 ) + -4 * ( S2 )**( 2 ) * ( \
    xi )**( 2 ) ) ) ) + ( -4 * ( q )**( 4 ) * ( 19 * ( S1 )**( 4 ) + ( 19 \
    * ( S2 )**( 4 ) + ( 14 * ( S2 )**( 2 ) * ( xi )**( 2 ) + ( 4 * ( xi \
    )**( 4 ) + -2 * ( S1 )**( 2 ) * ( 5 * ( S2 )**( 2 ) + -7 * ( xi )**( \
    2 ) ) ) ) ) ) + ( 2 * ( q )**( 2 ) * ( 23 * ( S1 )**( 4 ) + ( 7 * ( \
    S2 )**( 4 ) + -2 * ( S1 )**( 2 ) * ( 7 * ( S2 )**( 2 ) + 2 * ( xi \
    )**( 2 ) ) ) ) + ( ( q )**( 5 ) * ( -133 * ( S1 )**( 4 ) + ( 71 * ( \
    S2 )**( 4 ) + ( 12 * ( S2 )**( 2 ) * ( xi )**( 2 ) + -2 * ( S1 )**( 2 \
    ) * ( ( S2 )**( 2 ) + 38 * ( xi )**( 2 ) ) ) ) ) + ( q )**( 3 ) * ( \
    71 * ( S1 )**( 4 ) + ( -2 * ( S1 )**( 2 ) * ( ( S2 )**( 2 ) + -6 * ( \
    xi )**( 2 ) ) + -19 * ( 7 * ( S2 )**( 4 ) + 4 * ( S2 )**( 2 ) * ( xi \
    )**( 2 ) ) ) ) ) ) ) ) ) ) ) ) + 64 * ( ( 1 + q ) )**( 2 ) * u * ( 4 \
    * ( q )**( 10 ) * ( S1 )**( 4 ) * ( ( S1 )**( 2 ) + ( S2 )**( 2 ) ) + \
    ( 4 * ( S2 )**( 4 ) * ( ( S1 )**( 2 ) + ( S2 )**( 2 ) ) + ( ( q )**( \
    9 ) * ( 23 * ( S1 )**( 6 ) + ( 26 * ( S1 )**( 4 ) * ( S2 )**( 2 ) + \
    15 * ( S1 )**( 2 ) * ( S2 )**( 4 ) ) ) + ( q * ( 15 * ( S1 )**( 4 ) * \
    ( S2 )**( 2 ) + ( 26 * ( S1 )**( 2 ) * ( S2 )**( 4 ) + 23 * ( S2 )**( \
    6 ) ) ) + ( ( q )**( 8 ) * ( 25 * ( S1 )**( 6 ) + ( -1 * ( S2 )**( 6 \
    ) + ( ( S1 )**( 4 ) * ( -23 * ( S2 )**( 2 ) + 20 * ( xi )**( 2 ) ) + \
    ( S1 )**( 2 ) * ( -25 * ( S2 )**( 4 ) + 4 * ( S2 )**( 2 ) * ( xi )**( \
    2 ) ) ) ) ) + ( ( q )**( 3 ) * ( 13 * ( S1 )**( 6 ) + ( -33 * ( S2 \
    )**( 6 ) + ( -32 * ( S2 )**( 4 ) * ( xi )**( 2 ) + ( -1 * ( S1 )**( 4 \
    ) * ( 107 * ( S2 )**( 2 ) + 24 * ( xi )**( 2 ) ) + -3 * ( S1 )**( 2 ) \
    * ( 43 * ( S2 )**( 4 ) + 8 * ( S2 )**( 2 ) * ( xi )**( 2 ) ) ) ) ) ) \
    + ( -1 * ( q )**( 7 ) * ( 33 * ( S1 )**( 6 ) + ( -13 * ( S2 )**( 6 ) \
    + ( 24 * ( S2 )**( 4 ) * ( xi )**( 2 ) + ( ( S1 )**( 4 ) * ( 129 * ( \
    S2 )**( 2 ) + 32 * ( xi )**( 2 ) ) + ( S1 )**( 2 ) * ( 107 * ( S2 \
    )**( 4 ) + 24 * ( S2 )**( 2 ) * ( xi )**( 2 ) ) ) ) ) ) + ( -1 * ( q \
    )**( 2 ) * ( ( S1 )**( 6 ) + ( 25 * ( S1 )**( 4 ) * ( S2 )**( 2 ) + ( \
    ( S1 )**( 2 ) * ( 23 * ( S2 )**( 4 ) + -4 * ( S2 )**( 2 ) * ( xi )**( \
    2 ) ) + -5 * ( 5 * ( S2 )**( 6 ) + 4 * ( S2 )**( 4 ) * ( xi )**( 2 ) \
    ) ) ) ) + ( ( q )**( 6 ) * ( -63 * ( S1 )**( 6 ) + ( 35 * ( S2 )**( 6 \
    ) + ( 16 * ( S2 )**( 4 ) * ( xi )**( 2 ) + ( -8 * ( S2 )**( 2 ) * ( \
    xi )**( 4 ) + ( ( S1 )**( 4 ) * ( 9 * ( S2 )**( 2 ) + -60 * ( xi )**( \
    2 ) ) + ( S1 )**( 2 ) * ( 35 * ( S2 )**( 4 ) + ( 20 * ( S2 )**( 2 ) * \
    ( xi )**( 2 ) + -24 * ( xi )**( 4 ) ) ) ) ) ) ) ) + ( ( q )**( 5 ) * \
    ( -3 * ( S1 )**( 6 ) + ( -3 * ( S2 )**( 6 ) + ( 32 * ( S2 )**( 4 ) * \
    ( xi )**( 2 ) + ( 8 * ( S2 )**( 2 ) * ( xi )**( 4 ) + ( ( S1 )**( 4 ) \
    * ( 195 * ( S2 )**( 2 ) + 32 * ( xi )**( 2 ) ) + ( S1 )**( 2 ) * ( \
    195 * ( S2 )**( 4 ) + ( 96 * ( S2 )**( 2 ) * ( xi )**( 2 ) + 8 * ( xi \
    )**( 4 ) ) ) ) ) ) ) ) + ( q )**( 4 ) * ( 35 * ( S1 )**( 6 ) + ( ( S1 \
    )**( 4 ) * ( 35 * ( S2 )**( 2 ) + 16 * ( xi )**( 2 ) ) + ( ( S1 )**( \
    2 ) * ( 9 * ( S2 )**( 4 ) + ( 20 * ( S2 )**( 2 ) * ( xi )**( 2 ) + -8 \
    * ( xi )**( 4 ) ) ) + -3 * ( 21 * ( S2 )**( 6 ) + ( 20 * ( S2 )**( 4 \
    ) * ( xi )**( 2 ) + 8 * ( S2 )**( 2 ) * ( xi )**( 4 ) ) ) ) ) ) ) ) ) \
    ) ) ) ) ) ) ) ) ) ) )

    coeff2 = \
    ( -16 * ( ( -1 + q ) )**( 2 ) * ( ( 1 + q ) )**( 4 ) * ( ( q )**( 6 \
    ) * ( S1 )**( 2 ) + ( ( S2 )**( 2 ) + ( -4 * ( q )**( 3 ) * ( xi )**( \
    2 ) + ( ( q )**( 2 ) * ( ( S1 )**( 2 ) + ( -2 * ( S2 )**( 2 ) + -1 * \
    ( xi )**( 2 ) ) ) + ( q )**( 4 ) * ( -2 * ( S1 )**( 2 ) + ( ( S2 )**( \
    2 ) + -1 * ( xi )**( 2 ) ) ) ) ) ) ) + ( -32 * q * ( ( 1 + q ) )**( 4 \
    ) * u * xi * ( 4 * ( q )**( 6 ) * ( S1 )**( 2 ) + ( 4 * ( S2 )**( 2 ) \
    + ( ( q )**( 5 ) * ( 19 * ( S1 )**( 2 ) + -3 * ( S2 )**( 2 ) ) + ( q \
    * ( -3 * ( S1 )**( 2 ) + 19 * ( S2 )**( 2 ) ) + ( ( q )**( 2 ) * ( 26 \
    * ( S1 )**( 2 ) + ( -30 * ( S2 )**( 2 ) + -4 * ( xi )**( 2 ) ) ) + ( \
    ( q )**( 4 ) * ( -30 * ( S1 )**( 2 ) + ( 26 * ( S2 )**( 2 ) + -4 * ( \
    xi )**( 2 ) ) ) + -16 * ( q )**( 3 ) * ( ( S1 )**( 2 ) + ( ( S2 )**( \
    2 ) + 2 * ( xi )**( 2 ) ) ) ) ) ) ) ) ) + ( -256 * ( ( 1 + q ) )**( 4 \
    ) * ( u )**( 4 ) * ( ( q )**( 8 ) * ( S1 )**( 6 ) + ( ( S2 )**( 6 ) + \
    ( -1 * ( q )**( 7 ) * ( 7 * ( S1 )**( 6 ) + 9 * ( S1 )**( 4 ) * ( S2 \
    )**( 2 ) ) + ( -1 * q * ( 9 * ( S1 )**( 2 ) * ( S2 )**( 4 ) + 7 * ( \
    S2 )**( 6 ) ) + ( ( q )**( 2 ) * ( 18 * ( S1 )**( 4 ) * ( S2 )**( 2 ) \
    + ( 9 * ( S1 )**( 2 ) * ( S2 )**( 4 ) + ( ( S2 )**( 6 ) + -1 * ( S2 \
    )**( 4 ) * ( xi )**( 2 ) ) ) ) + ( ( q )**( 6 ) * ( ( S1 )**( 6 ) + ( \
    18 * ( S1 )**( 2 ) * ( S2 )**( 4 ) + ( S1 )**( 4 ) * ( 9 * ( S2 )**( \
    2 ) + -1 * ( xi )**( 2 ) ) ) ) + ( ( q )**( 5 ) * ( 17 * ( S1 )**( 6 \
    ) + ( -10 * ( S2 )**( 6 ) + ( 6 * ( S1 )**( 2 ) * ( S2 )**( 2 ) * ( \
    xi )**( 2 ) + ( S1 )**( 4 ) * ( 9 * ( S2 )**( 2 ) + 6 * ( xi )**( 2 ) \
    ) ) ) ) + ( ( q )**( 3 ) * ( -10 * ( S1 )**( 6 ) + ( 17 * ( S2 )**( 6 \
    ) + ( 6 * ( S2 )**( 4 ) * ( xi )**( 2 ) + ( S1 )**( 2 ) * ( 9 * ( S2 \
    )**( 4 ) + 6 * ( S2 )**( 2 ) * ( xi )**( 2 ) ) ) ) ) + -1 * ( q )**( \
    4 ) * ( 2 * ( S1 )**( 6 ) + ( 3 * ( S1 )**( 4 ) * ( 9 * ( S2 )**( 2 ) \
    + 2 * ( xi )**( 2 ) ) + ( 2 * ( S2 )**( 4 ) * ( ( S2 )**( 2 ) + 3 * ( \
    xi )**( 2 ) ) + ( S1 )**( 2 ) * ( 27 * ( S2 )**( 4 ) + 10 * ( S2 )**( \
    2 ) * ( xi )**( 2 ) ) ) ) ) ) ) ) ) ) ) ) ) + ( u )**( 2 ) * ( -32 * \
    ( ( 1 + q ) )**( 2 ) * ( 4 * ( q )**( 10 ) * ( S1 )**( 4 ) + ( 4 * ( \
    S2 )**( 4 ) + ( ( q )**( 9 ) * ( 38 * ( S1 )**( 4 ) + 30 * ( S1 )**( \
    2 ) * ( S2 )**( 2 ) ) + ( q * ( 30 * ( S1 )**( 2 ) * ( S2 )**( 2 ) + \
    38 * ( S2 )**( 4 ) ) + ( ( q )**( 2 ) * ( -3 * ( S1 )**( 4 ) + ( 2 * \
    ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( 53 * ( S2 )**( 4 ) + 4 * ( S2 )**( \
    2 ) * ( xi )**( 2 ) ) ) ) + ( ( q )**( 8 ) * ( 53 * ( S1 )**( 4 ) + ( \
    -3 * ( S2 )**( 4 ) + 2 * ( S1 )**( 2 ) * ( ( S2 )**( 2 ) + 2 * ( xi \
    )**( 2 ) ) ) ) + ( -4 * ( q )**( 7 ) * ( 13 * ( S1 )**( 4 ) + ( -6 * \
    ( S2 )**( 2 ) * ( ( S2 )**( 2 ) + -2 * ( xi )**( 2 ) ) + ( S1 )**( 2 \
    ) * ( 29 * ( S2 )**( 2 ) + 9 * ( xi )**( 2 ) ) ) ) + ( 4 * ( q )**( 3 \
    ) * ( 6 * ( S1 )**( 4 ) + ( -13 * ( S2 )**( 4 ) + ( -9 * ( S2 )**( 2 \
    ) * ( xi )**( 2 ) + -1 * ( S1 )**( 2 ) * ( 29 * ( S2 )**( 2 ) + 12 * \
    ( xi )**( 2 ) ) ) ) ) + ( -1 * ( q )**( 6 ) * ( 121 * ( S1 )**( 4 ) + \
    ( -67 * ( S2 )**( 4 ) + ( 104 * ( S2 )**( 2 ) * ( xi )**( 2 ) + ( 8 * \
    ( xi )**( 4 ) + 2 * ( S1 )**( 2 ) * ( ( S2 )**( 2 ) + 46 * ( xi )**( \
    2 ) ) ) ) ) ) + ( ( q )**( 4 ) * ( 67 * ( S1 )**( 4 ) + ( -121 * ( S2 \
    )**( 4 ) + ( -92 * ( S2 )**( 2 ) * ( xi )**( 2 ) + ( -8 * ( xi )**( 4 \
    ) + -2 * ( S1 )**( 2 ) * ( ( S2 )**( 2 ) + 52 * ( xi )**( 2 ) ) ) ) ) \
    ) + -2 * ( q )**( 5 ) * ( 5 * ( S1 )**( 4 ) + ( 5 * ( S2 )**( 4 ) + ( \
    54 * ( S2 )**( 2 ) * ( xi )**( 2 ) + ( 16 * ( xi )**( 4 ) + ( S1 )**( \
    2 ) * ( -86 * ( S2 )**( 2 ) + 54 * ( xi )**( 2 ) ) ) ) ) ) ) ) ) ) ) \
    ) ) ) ) ) + -128 * q * ( ( 1 + q ) )**( 3 ) * u * xi * ( 4 * ( q )**( \
    7 ) * ( S1 )**( 4 ) + ( 4 * ( S2 )**( 4 ) + ( ( q )**( 6 ) * ( -11 * \
    ( S1 )**( 4 ) + 3 * ( S1 )**( 2 ) * ( S2 )**( 2 ) ) + ( q * ( 3 * ( \
    S1 )**( 2 ) * ( S2 )**( 2 ) + -11 * ( S2 )**( 4 ) ) + ( ( q )**( 2 ) \
    * ( -30 * ( S1 )**( 4 ) + ( 9 * ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( 5 * \
    ( S2 )**( 4 ) + -4 * ( S2 )**( 2 ) * ( xi )**( 2 ) ) ) ) + ( ( q )**( \
    5 ) * ( 5 * ( S1 )**( 4 ) + ( -30 * ( S2 )**( 4 ) + ( S1 )**( 2 ) * ( \
    9 * ( S2 )**( 2 ) + -4 * ( xi )**( 2 ) ) ) ) + ( ( q )**( 3 ) * ( -21 \
    * ( S1 )**( 4 ) + ( 29 * ( S2 )**( 4 ) + ( 4 * ( S2 )**( 2 ) * ( xi \
    )**( 2 ) + 12 * ( S1 )**( 2 ) * ( ( S2 )**( 2 ) + -1 * ( xi )**( 2 ) \
    ) ) ) ) + ( q )**( 4 ) * ( 29 * ( S1 )**( 4 ) + ( 4 * ( S1 )**( 2 ) * \
    ( 3 * ( S2 )**( 2 ) + ( xi )**( 2 ) ) + -3 * ( 7 * ( S2 )**( 4 ) + 4 \
    * ( S2 )**( 2 ) * ( xi )**( 2 ) ) ) ) ) ) ) ) ) ) ) ) ) ) )

    coeff3 = \
    ( -32 * ( ( -1 + q ) )**( 2 ) * ( q )**( 2 ) * ( ( 1 + q ) )**( 6 ) * \
    xi + ( 64 * q * ( ( 1 + q ) )**( 4 ) * u * ( 5 * ( q )**( 6 ) * ( S1 \
    )**( 2 ) + ( 5 * ( S2 )**( 2 ) + ( -1 * q * ( ( S1 )**( 2 ) + ( S2 \
    )**( 2 ) ) + ( -1 * ( q )**( 5 ) * ( ( S1 )**( 2 ) + ( S2 )**( 2 ) ) \
    + ( 2 * ( q )**( 3 ) * ( ( S1 )**( 2 ) + ( ( S2 )**( 2 ) + -12 * ( xi \
    )**( 2 ) ) ) + ( ( q )**( 2 ) * ( 5 * ( S1 )**( 2 ) + ( -10 * ( S2 \
    )**( 2 ) + -8 * ( xi )**( 2 ) ) ) + ( q )**( 4 ) * ( -10 * ( S1 )**( \
    2 ) + ( 5 * ( S2 )**( 2 ) + -8 * ( xi )**( 2 ) ) ) ) ) ) ) ) ) + ( u \
    )**( 2 ) * ( 128 * ( q )**( 2 ) * ( ( 1 + q ) )**( 4 ) * xi * ( ( q \
    )**( 4 ) * ( S1 )**( 2 ) + ( ( S2 )**( 2 ) + ( 4 * ( q )**( 3 ) * ( ( \
    S1 )**( 2 ) + -5 * ( S2 )**( 2 ) ) + ( 4 * q * ( -5 * ( S1 )**( 2 ) + \
    ( S2 )**( 2 ) ) + -1 * ( q )**( 2 ) * ( 17 * ( S1 )**( 2 ) + ( 17 * ( \
    S2 )**( 2 ) + 4 * ( xi )**( 2 ) ) ) ) ) ) ) + -256 * q * ( ( 1 + q ) \
    )**( 4 ) * u * ( 3 * ( q )**( 6 ) * ( S1 )**( 4 ) + ( 3 * ( S2 )**( 4 \
    ) + ( -6 * ( q )**( 5 ) * ( ( S1 )**( 4 ) + 2 * ( S1 )**( 2 ) * ( S2 \
    )**( 2 ) ) + ( -6 * q * ( 2 * ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( S2 \
    )**( 4 ) ) + ( ( q )**( 2 ) * ( 10 * ( S1 )**( 4 ) + ( -2 * ( S1 )**( \
    2 ) * ( S2 )**( 2 ) + ( -11 * ( S2 )**( 4 ) + -2 * ( S2 )**( 2 ) * ( \
    xi )**( 2 ) ) ) ) + ( -1 * ( q )**( 4 ) * ( 11 * ( S1 )**( 4 ) + ( \
    -10 * ( S2 )**( 4 ) + 2 * ( S1 )**( 2 ) * ( ( S2 )**( 2 ) + ( xi )**( \
    2 ) ) ) ) + 4 * ( q )**( 3 ) * ( 2 * ( S1 )**( 4 ) + ( ( S2 )**( 2 ) \
    * ( 2 * ( S2 )**( 2 ) + ( xi )**( 2 ) ) + ( S1 )**( 2 ) * ( 5 * ( S2 \
    )**( 2 ) + ( xi )**( 2 ) ) ) ) ) ) ) ) ) ) ) ) )

    coeff4 = \
    ( 16 * ( ( -1 + q ) )**( 2 ) * ( q )**( 2 ) * ( ( 1 + q ) )**( 6 ) + \
    ( 640 * ( q )**( 3 ) * ( ( 1 + q ) )**( 6 ) * u * xi + -256 * ( q \
    )**( 2 ) * ( ( 1 + q ) )**( 4 ) * ( u )**( 2 ) * ( 3 * ( q )**( 4 ) * \
    ( S1 )**( 2 ) + ( 3 * ( S2 )**( 2 ) + ( ( q )**( 3 ) * ( ( S1 )**( 2 \
    ) + -5 * ( S2 )**( 2 ) ) + ( q * ( -5 * ( S1 )**( 2 ) + ( S2 )**( 2 ) \
    ) + -1 * ( q )**( 2 ) * ( 7 * ( S1 )**( 2 ) + ( 7 * ( S2 )**( 2 ) + ( \
    xi )**( 2 ) ) ) ) ) ) ) ) )
    coeff5 = \
    -256 * ( q )**( 3 ) * ( ( 1 + q ) )**( 6 ) * u

    return np.stack([coeff5, coeff4, coeff3, coeff2, coeff1, coeff0])


def Jresonances(r,xi,q,chi1,chi2):
    """
    Total angular momentum of the two spin-orbit resonances. The resonances minimizes and maximizes J for a given value of xi. The minimum corresponds to DeltaPhi=pi and the maximum corresponds to DeltaPhi=0.

    Call
    ----
    Jmin,Jmax = Jresonances(r,xi,q,chi1,chi2)

    Parameters
    ----------
    r: float
    	Binary separation.
    xi: float
    	Effective spin.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    Jmin: float
    	Minimum value of the total angular momentum J.
    Jmax: float
    	Maximum value of the total angular momentum J.
    """

    # There are in principle five solutions, but only two are physical.

    r=np.atleast_1d(r)
    xi=np.atleast_1d(xi)
    q=np.atleast_1d(q)
    chi1=np.atleast_1d(chi1)
    chi2=np.atleast_1d(chi2)

    u = eval_u(r, q)
    kapparoots = wraproots(kappadiscriminant_coefficients,u,xi,q,chi1,chi2)
    def _compute(kapparoots,r,xi,q,chi1,chi2):
        kapparoots = kapparoots[np.isfinite(kapparoots)]
        Jroots = eval_J(kappa=kapparoots,r=np.tile(r,kapparoots.shape),q=np.tile(q,kapparoots.shape))
        Sroots = Satresonance(Jroots,np.tile(r,Jroots.shape),np.tile(xi,Jroots.shape),np.tile(q,Jroots.shape),np.tile(chi1,Jroots.shape),np.tile(chi2,Jroots.shape))
        Smin,Smax = Slimits_LJS1S2(Jroots,np.tile(r,Jroots.shape),np.tile(q,Jroots.shape),np.tile(chi1,Jroots.shape),np.tile(chi2,Jroots.shape))
        Jres = Jroots[np.logical_and(Sroots>Smin,Sroots<Smax)]
        assert len(Jres)<=2, "I found more than two resonances, this should not be possible."
        # If you didn't find enough solutions, append nans
        Jres=np.concatenate([Jres,np.repeat(np.nan,2-len(Jres))])
        return Jres

    Jmin,Jmax =np.array(list(map(_compute, kapparoots,r,xi,q,chi1,chi2))).T

    return np.stack([Jmin,Jmax])


def Jlimits(r=None,xi=None,q=None,chi1=None,chi2=None):
    """
    Limits on the magnitude of the total angular momentum. The contraints considered depend on the inputs provided.
    - If r, q, chi1, and chi2 are provided, enforce J=L+S1+S2.
    - If r, xi, q, chi1, and chi2 are provided, the limits are given by the two spin-orbit resonances.

    Call
    ----
    Jmin,Jmax = Jlimits(r=None,xi=None,q=None,chi1=None,chi2=None)

    Parameters
    ----------
    r: float, optional (default: None)
    	Binary separation.
    xi: float, optional (default: None)
    	Effective spin.
    q: float, optional (default: None)
    	Mass ratio: 0<=q<=1.
    chi1: float, optional (default: None)
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float, optional (default: None)
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    Jmin: float
    	Minimum value of the total angular momentum J.
    Jmax: float
    	Maximum value of the total angular momentum J.
    """

    if r is not None and xi is None and q is not None and chi1 is not None and chi2 is not None:
        Jmin,Jmax = Jlimits_LS1S2(r,q,chi1,chi2)

    elif r is not None and xi is not None and q is not None and chi1 is not None and chi2 is not None:
        Jmin,Jmax = Jresonances(r,xi,q,chi1,chi2)
        # Check precondition
        Jmin_cond,Jmax_cond = Jlimits_LS1S2(r,q,chi1,chi2)
        assert (Jmin>Jmin_cond).all() and (Jmax<Jmax_cond).all(), "Input values are incompatible."

    else:
        raise TypeError("Provide either (r,q,chi1,chi2) or (r,xi,q,chi1,chi2).")

    return np.stack([Jmin,Jmax])


def xilimits_definition(q,chi1,chi2):
    """
    Limits on the effective spin based only on the definition xi = (1+q)S1.L + (1+1/q)S2.L.

    Call
    ----
    ximin,ximax = xilimits_definition(q,chi1,chi2)

    Parameters
    ----------
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    ximin: float
    	Minimum value of the effective spin xi.
    ximax: float
    	Maximum value of the effective spin xi.
    """

    q=np.atleast_1d(q)
    S1,S2 = spinmags(q,chi1,chi2)
    xilim = (1+q)*S1 + (1+1/q)*S2

    return np.stack([-xilim,xilim])


def xidiscriminant_coefficients(kappa,u,q,chi1,chi2):
    """
    Coefficients of the sixth-degree equation in xi that defines the spin-orbit resonances.

    Call
    ----
    coeff6,coeff5,coeff4,coeff3,coeff2,coeff1,coeff0 = xidiscriminant_coefficients(kappa,u,q,chi1,chi2)

    Parameters
    ----------
    kappa: float
    	Regularized angular momentum (J^2-L^2)/(2L).
    u: float
    	Compactified separation 1/(2L).
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    coeff6: float
    	Coefficient to the x^6 term in polynomial.
    coeff5: float
    	Coefficient to the x^5 term in polynomial.
    coeff4: float
    	Coefficient to the x^4 term in polynomial.
    coeff3: float
    	Coefficient to the x^3 term in polynomial.
    coeff2: float
    	Coefficient to the x^2 term in polynomial.
    coeff1: float
    	Coefficient to the x^1 term in polynomial.
    coeff0: float
    	Coefficient to the x^0 term in polynomial.
    """

    kappa=np.atleast_1d(kappa)
    u=np.atleast_1d(u)
    q=np.atleast_1d(q)
    S1,S2= spinmags(q,chi1,chi2)

    coeff0 = \
    ( 16 * ( ( -1 + q ) )**( 2 ) * ( ( 1 + q ) )**( 6 ) * ( ( ( -1 + q ) \
    )**( 2 ) * ( S1 )**( 2 ) + -1 * ( kappa )**( 2 ) ) * ( ( ( -1 + q ) \
    )**( 2 ) * ( S2 )**( 2 ) + -1 * ( q )**( 2 ) * ( kappa )**( 2 ) ) + ( \
    32 * ( ( 1 + q ) )**( 6 ) * u * kappa * ( ( q )**( 6 ) * ( S1 )**( 2 \
    ) * ( ( S1 )**( 2 ) + ( S2 )**( 2 ) ) + ( ( S2 )**( 2 ) * ( ( S1 )**( \
    2 ) + ( S2 )**( 2 ) ) + ( -2 * q * ( S2 )**( 2 ) * ( 8 * ( S1 )**( 2 \
    ) + ( 2 * ( S2 )**( 2 ) + -5 * ( kappa )**( 2 ) ) ) + ( -2 * ( q )**( \
    5 ) * ( S1 )**( 2 ) * ( 2 * ( S1 )**( 2 ) + ( 8 * ( S2 )**( 2 ) + -5 \
    * ( kappa )**( 2 ) ) ) + ( -2 * ( q )**( 3 ) * ( 2 * ( S1 )**( 4 ) + \
    ( 2 * ( S2 )**( 4 ) + ( -7 * ( S2 )**( 2 ) * ( kappa )**( 2 ) + ( 4 * \
    ( kappa )**( 4 ) + ( S1 )**( 2 ) * ( 40 * ( S2 )**( 2 ) + -7 * ( \
    kappa )**( 2 ) ) ) ) ) ) + ( ( q )**( 4 ) * ( 6 * ( S1 )**( 4 ) + ( ( \
    S2 )**( 4 ) + ( -2 * ( S2 )**( 2 ) * ( kappa )**( 2 ) + 11 * ( S1 \
    )**( 2 ) * ( 5 * ( S2 )**( 2 ) + -2 * ( kappa )**( 2 ) ) ) ) ) + ( q \
    )**( 2 ) * ( ( S1 )**( 4 ) + ( 6 * ( S2 )**( 4 ) + ( -22 * ( S2 )**( \
    2 ) * ( kappa )**( 2 ) + ( S1 )**( 2 ) * ( 55 * ( S2 )**( 2 ) + -2 * \
    ( kappa )**( 2 ) ) ) ) ) ) ) ) ) ) ) + ( ( u )**( 4 ) * ( -256 * ( ( \
    -1 + q ) )**( 3 ) * ( ( 1 + q ) )**( 6 ) * ( ( ( S1 )**( 2 ) + -1 * ( \
    S2 )**( 2 ) ) )**( 2 ) * ( ( q * ( S1 )**( 2 ) + -1 * ( S2 )**( 2 ) ) \
    )**( 3 ) * ( u )**( 2 ) + ( 256 * ( ( -1 + q ) )**( 2 ) * ( ( 1 + q ) \
    )**( 6 ) * ( ( -1 * q * ( S1 )**( 2 ) + ( S2 )**( 2 ) ) )**( 2 ) * ( \
    2 * ( q )**( 2 ) * ( S1 )**( 2 ) * ( ( S1 )**( 2 ) + ( S2 )**( 2 ) ) \
    + ( 2 * ( S2 )**( 2 ) * ( ( S1 )**( 2 ) + ( S2 )**( 2 ) ) + q * ( -5 \
    * ( S1 )**( 4 ) + ( 2 * ( S1 )**( 2 ) * ( S2 )**( 2 ) + -5 * ( S2 \
    )**( 4 ) ) ) ) ) * u * kappa + -16 * ( -1 + q ) * ( ( 1 + q ) )**( 6 \
    ) * ( -8 * ( S2 )**( 4 ) * ( ( S1 )**( 4 ) + ( -4 * ( S1 )**( 2 ) * ( \
    S2 )**( 2 ) + ( ( S2 )**( 4 ) + 2 * ( S2 )**( 2 ) * ( kappa )**( 2 ) \
    ) ) ) + ( 8 * ( q )**( 5 ) * ( S1 )**( 4 ) * ( ( S1 )**( 4 ) + ( ( S2 \
    )**( 4 ) + ( S1 )**( 2 ) * ( -4 * ( S2 )**( 2 ) + 2 * ( kappa )**( 2 \
    ) ) ) ) + ( 4 * ( q )**( 4 ) * ( 3 * ( S1 )**( 8 ) + ( 5 * ( S1 )**( \
    2 ) * ( S2 )**( 6 ) + ( ( S1 )**( 6 ) * ( 11 * ( S2 )**( 2 ) + -32 * \
    ( kappa )**( 2 ) ) + ( S1 )**( 4 ) * ( ( S2 )**( 4 ) + -36 * ( S2 \
    )**( 2 ) * ( kappa )**( 2 ) ) ) ) ) + ( -4 * q * ( 5 * ( S1 )**( 6 ) \
    * ( S2 )**( 2 ) + ( ( S1 )**( 4 ) * ( S2 )**( 4 ) + ( 3 * ( S2 )**( 8 \
    ) + ( -32 * ( S2 )**( 6 ) * ( kappa )**( 2 ) + ( S1 )**( 2 ) * ( 11 * \
    ( S2 )**( 6 ) + -36 * ( S2 )**( 4 ) * ( kappa )**( 2 ) ) ) ) ) ) + ( \
    ( q )**( 2 ) * ( ( S1 )**( 8 ) + ( 128 * ( S1 )**( 6 ) * ( S2 )**( 2 \
    ) + ( 21 * ( S2 )**( 8 ) + ( -160 * ( S2 )**( 6 ) * ( kappa )**( 2 ) \
    + ( -2 * ( S1 )**( 4 ) * ( 55 * ( S2 )**( 4 ) + 144 * ( S2 )**( 2 ) * \
    ( kappa )**( 2 ) ) + 24 * ( S1 )**( 2 ) * ( 5 * ( S2 )**( 6 ) + -12 * \
    ( S2 )**( 4 ) * ( kappa )**( 2 ) ) ) ) ) ) ) + -1 * ( q )**( 3 ) * ( \
    21 * ( S1 )**( 8 ) + ( ( S2 )**( 8 ) + ( 40 * ( S1 )**( 6 ) * ( 3 * ( \
    S2 )**( 2 ) + -4 * ( kappa )**( 2 ) ) + ( -2 * ( S1 )**( 4 ) * ( 55 * \
    ( S2 )**( 4 ) + 144 * ( S2 )**( 2 ) * ( kappa )**( 2 ) ) + 32 * ( S1 \
    )**( 2 ) * ( 4 * ( S2 )**( 6 ) + -9 * ( S2 )**( 4 ) * ( kappa )**( 2 \
    ) ) ) ) ) ) ) ) ) ) ) ) ) + ( u )**( 2 ) * ( 64 * ( ( 1 + q ) )**( 6 \
    ) * u * kappa * ( 4 * ( q )**( 6 ) * ( S1 )**( 4 ) * ( ( S1 )**( 2 ) \
    + ( S2 )**( 2 ) ) + ( 4 * ( S2 )**( 4 ) * ( ( S1 )**( 2 ) + ( S2 )**( \
    2 ) ) + ( q * ( 15 * ( S1 )**( 4 ) * ( S2 )**( 2 ) + ( 10 * ( S1 )**( \
    2 ) * ( S2 )**( 4 ) + ( 7 * ( S2 )**( 6 ) + -12 * ( S2 )**( 4 ) * ( \
    kappa )**( 2 ) ) ) ) + ( ( q )**( 5 ) * ( 7 * ( S1 )**( 6 ) + ( 15 * \
    ( S1 )**( 2 ) * ( S2 )**( 4 ) + 2 * ( S1 )**( 4 ) * ( 5 * ( S2 )**( 2 \
    ) + -6 * ( kappa )**( 2 ) ) ) ) + ( -1 * ( q )**( 4 ) * ( 27 * ( S1 \
    )**( 6 ) + ( ( S2 )**( 6 ) + ( ( S1 )**( 4 ) * ( 87 * ( S2 )**( 2 ) + \
    -48 * ( kappa )**( 2 ) ) + ( S1 )**( 2 ) * ( 85 * ( S2 )**( 4 ) + -48 \
    * ( S2 )**( 2 ) * ( kappa )**( 2 ) ) ) ) ) + ( -1 * ( q )**( 2 ) * ( \
    ( S1 )**( 6 ) + ( 85 * ( S1 )**( 4 ) * ( S2 )**( 2 ) + ( 27 * ( S2 \
    )**( 6 ) + ( -48 * ( S2 )**( 4 ) * ( kappa )**( 2 ) + ( S1 )**( 2 ) * \
    ( 87 * ( S2 )**( 4 ) + -48 * ( S2 )**( 2 ) * ( kappa )**( 2 ) ) ) ) ) \
    ) + ( q )**( 3 ) * ( 17 * ( S1 )**( 6 ) + ( 17 * ( S2 )**( 6 ) + ( \
    -40 * ( S2 )**( 4 ) * ( kappa )**( 2 ) + ( ( S1 )**( 4 ) * ( 143 * ( \
    S2 )**( 2 ) + -40 * ( kappa )**( 2 ) ) + 11 * ( S1 )**( 2 ) * ( 13 * \
    ( S2 )**( 4 ) + -8 * ( S2 )**( 2 ) * ( kappa )**( 2 ) ) ) ) ) ) ) ) ) \
    ) ) ) + -16 * ( ( 1 + q ) )**( 6 ) * ( ( S2 )**( 2 ) * ( ( S1 )**( 4 \
    ) + ( -10 * ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( ( S2 )**( 4 ) + 8 * ( \
    S2 )**( 2 ) * ( kappa )**( 2 ) ) ) ) + ( ( q )**( 6 ) * ( S1 )**( 2 ) \
    * ( ( S1 )**( 4 ) + ( ( S2 )**( 4 ) + ( S1 )**( 2 ) * ( -10 * ( S2 \
    )**( 2 ) + 8 * ( kappa )**( 2 ) ) ) ) + ( -4 * ( q )**( 5 ) * ( S1 \
    )**( 2 ) * ( ( S1 )**( 4 ) + ( 7 * ( S2 )**( 4 ) + ( -15 * ( S2 )**( \
    2 ) * ( kappa )**( 2 ) + -1 * ( S1 )**( 2 ) * ( 4 * ( S2 )**( 2 ) + \
    11 * ( kappa )**( 2 ) ) ) ) ) + ( -4 * q * ( S2 )**( 2 ) * ( 7 * ( S1 \
    )**( 4 ) + ( ( S2 )**( 4 ) + ( -11 * ( S2 )**( 2 ) * ( kappa )**( 2 ) \
    + -1 * ( S1 )**( 2 ) * ( 4 * ( S2 )**( 2 ) + 15 * ( kappa )**( 2 ) ) \
    ) ) ) + ( ( q )**( 2 ) * ( ( S1 )**( 6 ) + ( 6 * ( S2 )**( 6 ) + ( \
    -118 * ( S2 )**( 4 ) * ( kappa )**( 2 ) + ( 48 * ( S2 )**( 2 ) * ( \
    kappa )**( 4 ) + ( ( S1 )**( 4 ) * ( 92 * ( S2 )**( 2 ) + -6 * ( \
    kappa )**( 2 ) ) + ( S1 )**( 2 ) * ( 37 * ( S2 )**( 4 ) + -236 * ( S2 \
    )**( 2 ) * ( kappa )**( 2 ) ) ) ) ) ) ) + ( ( q )**( 4 ) * ( 6 * ( S1 \
    )**( 6 ) + ( ( S2 )**( 6 ) + ( -6 * ( S2 )**( 4 ) * ( kappa )**( 2 ) \
    + ( ( S1 )**( 4 ) * ( 37 * ( S2 )**( 2 ) + -118 * ( kappa )**( 2 ) ) \
    + 4 * ( S1 )**( 2 ) * ( 23 * ( S2 )**( 4 ) + ( -59 * ( S2 )**( 2 ) * \
    ( kappa )**( 2 ) + 12 * ( kappa )**( 4 ) ) ) ) ) ) ) + -4 * ( q )**( \
    3 ) * ( ( S1 )**( 6 ) + ( ( S2 )**( 6 ) + ( -18 * ( S2 )**( 4 ) * ( \
    kappa )**( 2 ) + ( 20 * ( S2 )**( 2 ) * ( kappa )**( 4 ) + ( 9 * ( S1 \
    )**( 4 ) * ( 3 * ( S2 )**( 2 ) + -2 * ( kappa )**( 2 ) ) + ( S1 )**( \
    2 ) * ( 27 * ( S2 )**( 4 ) + ( -88 * ( S2 )**( 2 ) * ( kappa )**( 2 ) \
    + 20 * ( kappa )**( 4 ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) )


    coeff1 = \
    ( ( u )**( 4 ) * ( -128 * ( ( -1 + q ) )**( 2 ) * q * ( ( 1 + q ) \
    )**( 5 ) * ( ( S1 )**( 2 ) + -1 * ( S2 )**( 2 ) ) * ( 4 * ( S2 )**( 4 \
    ) * ( ( S1 )**( 2 ) + -2 * ( S2 )**( 2 ) ) + ( ( q )**( 3 ) * ( 8 * ( \
    S1 )**( 6 ) + -4 * ( S1 )**( 4 ) * ( S2 )**( 2 ) ) + ( -1 * ( q )**( \
    2 ) * ( S1 )**( 2 ) * ( 5 * ( S1 )**( 4 ) + ( 6 * ( S1 )**( 2 ) * ( \
    S2 )**( 2 ) + ( S2 )**( 4 ) ) ) + q * ( S2 )**( 2 ) * ( ( S1 )**( 4 ) \
    + ( 6 * ( S1 )**( 2 ) * ( S2 )**( 2 ) + 5 * ( S2 )**( 4 ) ) ) ) ) ) * \
    u + 128 * ( -1 + q ) * q * ( ( 1 + q ) )**( 5 ) * ( -4 * ( S2 )**( 4 \
    ) * ( 2 * ( S1 )**( 2 ) + 3 * ( S2 )**( 2 ) ) + ( 4 * ( q )**( 4 ) * \
    ( 3 * ( S1 )**( 6 ) + 2 * ( S1 )**( 4 ) * ( S2 )**( 2 ) ) + ( ( q \
    )**( 3 ) * ( -29 * ( S1 )**( 6 ) + ( -14 * ( S1 )**( 4 ) * ( S2 )**( \
    2 ) + 3 * ( S1 )**( 2 ) * ( S2 )**( 4 ) ) ) + ( 20 * ( q )**( 2 ) * ( \
    ( S1 )**( 6 ) + -1 * ( S2 )**( 6 ) ) + q * ( -3 * ( S1 )**( 4 ) * ( \
    S2 )**( 2 ) + ( 14 * ( S1 )**( 2 ) * ( S2 )**( 4 ) + 29 * ( S2 )**( 6 \
    ) ) ) ) ) ) ) * kappa ) + ( 32 * ( ( -1 + q ) )**( 2 ) * q * ( ( 1 + \
    q ) )**( 5 ) * kappa * ( ( q )**( 3 ) * ( S1 )**( 2 ) + ( ( S2 )**( 2 \
    ) + ( q * ( ( S1 )**( 2 ) + ( -2 * ( S2 )**( 2 ) + -1 * ( kappa )**( \
    2 ) ) ) + ( q )**( 2 ) * ( -2 * ( S1 )**( 2 ) + ( ( S2 )**( 2 ) + -1 \
    * ( kappa )**( 2 ) ) ) ) ) ) + ( -32 * q * ( ( 1 + q ) )**( 5 ) * u * \
    ( ( q )**( 5 ) * ( S1 )**( 2 ) * ( ( S1 )**( 2 ) + ( -5 * ( S2 )**( 2 \
    ) + 4 * ( kappa )**( 2 ) ) ) + ( ( S2 )**( 2 ) * ( -5 * ( S1 )**( 2 ) \
    + ( ( S2 )**( 2 ) + 4 * ( kappa )**( 2 ) ) ) + ( -1 * ( q )**( 2 ) * \
    ( 4 * ( S1 )**( 4 ) + ( -6 * ( S2 )**( 4 ) + ( 45 * ( S2 )**( 2 ) * ( \
    kappa )**( 2 ) + ( 20 * ( kappa )**( 4 ) + ( S1 )**( 2 ) * ( 10 * ( \
    S2 )**( 2 ) + -29 * ( kappa )**( 2 ) ) ) ) ) ) + ( q * ( ( S1 )**( 4 \
    ) + ( -4 * ( S2 )**( 4 ) + ( 15 * ( S2 )**( 2 ) * ( kappa )**( 2 ) + \
    3 * ( S1 )**( 2 ) * ( 5 * ( S2 )**( 2 ) + -1 * ( kappa )**( 2 ) ) ) ) \
    ) + ( ( q )**( 4 ) * ( -4 * ( S1 )**( 4 ) + ( ( S2 )**( 4 ) + ( -3 * \
    ( S2 )**( 2 ) * ( kappa )**( 2 ) + 15 * ( S1 )**( 2 ) * ( ( S2 )**( 2 \
    ) + ( kappa )**( 2 ) ) ) ) ) + ( q )**( 3 ) * ( 6 * ( S1 )**( 4 ) + ( \
    -4 * ( S2 )**( 4 ) + ( 29 * ( S2 )**( 2 ) * ( kappa )**( 2 ) + ( -20 \
    * ( kappa )**( 4 ) + -5 * ( S1 )**( 2 ) * ( 2 * ( S2 )**( 2 ) + 9 * ( \
    kappa )**( 2 ) ) ) ) ) ) ) ) ) ) ) + ( u )**( 2 ) * ( 32 * q * ( ( 1 \
    + q ) )**( 5 ) * kappa * ( 8 * ( q )**( 5 ) * ( S1 )**( 2 ) * ( 2 * ( \
    S1 )**( 2 ) + ( S2 )**( 2 ) ) + ( 8 * ( S2 )**( 2 ) * ( ( S1 )**( 2 ) \
    + 2 * ( S2 )**( 2 ) ) + ( q * ( -3 * ( S1 )**( 4 ) + ( -22 * ( S1 \
    )**( 2 ) * ( S2 )**( 2 ) + ( 17 * ( S2 )**( 4 ) + 4 * ( S2 )**( 2 ) * \
    ( kappa )**( 2 ) ) ) ) + ( ( q )**( 2 ) * ( 55 * ( S1 )**( 4 ) + ( \
    -85 * ( S2 )**( 4 ) + ( 12 * ( S2 )**( 2 ) * ( kappa )**( 2 ) + 2 * ( \
    S1 )**( 2 ) * ( 7 * ( S2 )**( 2 ) + -40 * ( kappa )**( 2 ) ) ) ) ) + \
    ( ( q )**( 4 ) * ( 17 * ( S1 )**( 4 ) + ( -3 * ( S2 )**( 4 ) + ( S1 \
    )**( 2 ) * ( -22 * ( S2 )**( 2 ) + 4 * ( kappa )**( 2 ) ) ) ) + ( q \
    )**( 3 ) * ( -85 * ( S1 )**( 4 ) + ( 55 * ( S2 )**( 4 ) + ( -80 * ( \
    S2 )**( 2 ) * ( kappa )**( 2 ) + 2 * ( S1 )**( 2 ) * ( 7 * ( S2 )**( \
    2 ) + 6 * ( kappa )**( 2 ) ) ) ) ) ) ) ) ) ) + -32 * q * ( ( 1 + q ) \
    )**( 5 ) * u * ( 4 * ( S2 )**( 2 ) * ( ( S1 )**( 4 ) + ( -8 * ( S1 \
    )**( 2 ) * ( S2 )**( 2 ) + ( 3 * ( S2 )**( 4 ) + 4 * ( S2 )**( 2 ) * \
    ( kappa )**( 2 ) ) ) ) + ( 4 * ( q )**( 5 ) * ( S1 )**( 2 ) * ( 3 * ( \
    S1 )**( 4 ) + ( ( S2 )**( 4 ) + ( S1 )**( 2 ) * ( -8 * ( S2 )**( 2 ) \
    + 4 * ( kappa )**( 2 ) ) ) ) + ( ( q )**( 4 ) * ( ( S1 )**( 6 ) + ( \
    -1 * ( S2 )**( 6 ) + ( ( S1 )**( 4 ) * ( 85 * ( S2 )**( 2 ) + -76 * ( \
    kappa )**( 2 ) ) + ( S1 )**( 2 ) * ( -37 * ( S2 )**( 4 ) + 12 * ( S2 \
    )**( 2 ) * ( kappa )**( 2 ) ) ) ) ) + ( ( q )**( 3 ) * ( -39 * ( S1 \
    )**( 6 ) + ( 3 * ( S2 )**( 4 ) * ( 9 * ( S2 )**( 2 ) + -40 * ( kappa \
    )**( 2 ) ) + ( ( S1 )**( 4 ) * ( -103 * ( S2 )**( 2 ) + 156 * ( kappa \
    )**( 2 ) ) + ( S1 )**( 2 ) * ( 83 * ( S2 )**( 4 ) + 12 * ( S2 )**( 2 \
    ) * ( kappa )**( 2 ) ) ) ) ) + ( q * ( -1 * ( S1 )**( 6 ) + ( -37 * ( \
    S1 )**( 4 ) * ( S2 )**( 2 ) + ( ( S2 )**( 6 ) + ( -76 * ( S2 )**( 4 ) \
    * ( kappa )**( 2 ) + ( S1 )**( 2 ) * ( 85 * ( S2 )**( 4 ) + 12 * ( S2 \
    )**( 2 ) * ( kappa )**( 2 ) ) ) ) ) ) + ( q )**( 2 ) * ( 27 * ( S1 \
    )**( 6 ) + ( ( S1 )**( 4 ) * ( 83 * ( S2 )**( 2 ) + -120 * ( kappa \
    )**( 2 ) ) + ( ( S1 )**( 2 ) * ( -103 * ( S2 )**( 4 ) + 12 * ( S2 \
    )**( 2 ) * ( kappa )**( 2 ) ) + -39 * ( ( S2 )**( 6 ) + -4 * ( S2 \
    )**( 4 ) * ( kappa )**( 2 ) ) ) ) ) ) ) ) ) ) ) ) ) )

    coeff2 = \
    ( 32 * ( q )**( 2 ) * ( ( 1 + q ) )**( 4 ) * u * kappa * ( ( ( -1 + \
    q ) )**( 2 ) * ( -1 + ( 18 * q + 7 * ( q )**( 2 ) ) ) * ( S1 )**( 2 ) \
    + ( -1 * ( ( -1 + q ) )**( 2 ) * ( -7 + ( -18 * q + ( q )**( 2 ) ) ) \
    * ( S2 )**( 2 ) + -16 * q * ( 1 + ( 3 * q + ( q )**( 2 ) ) ) * ( \
    kappa )**( 2 ) ) ) + ( -16 * ( ( -1 + q ) )**( 2 ) * ( q )**( 2 ) * ( \
    ( 1 + q ) )**( 4 ) * ( ( ( -1 + q ) )**( 2 ) * ( S1 )**( 2 ) + ( ( ( \
    -1 + q ) )**( 2 ) * ( S2 )**( 2 ) + -1 * ( 1 + ( 4 * q + ( q )**( 2 ) \
    ) ) * ( kappa )**( 2 ) ) ) + ( ( u )**( 2 ) * ( -16 * ( q )**( 2 ) * \
    ( ( 1 + q ) )**( 4 ) * ( ( ( -1 + q ) )**( 2 ) * ( -1 + ( 40 * q + 23 \
    * ( q )**( 2 ) ) ) * ( S1 )**( 4 ) + ( -1 * ( ( -1 + q ) )**( 2 ) * ( \
    -23 + ( -40 * q + ( q )**( 2 ) ) ) * ( S2 )**( 4 ) + ( -8 * ( -1 + ( \
    11 * q + ( 2 * ( q )**( 2 ) + 12 * ( q )**( 3 ) ) ) ) * ( S2 )**( 2 ) \
    * ( kappa )**( 2 ) + ( -16 * ( q )**( 2 ) * ( kappa )**( 4 ) + -2 * ( \
    S1 )**( 2 ) * ( ( ( -1 + q ) )**( 2 ) * ( 11 + ( -24 * q + 11 * ( q \
    )**( 2 ) ) ) * ( S2 )**( 2 ) + 4 * q * ( 12 + ( 2 * q + ( 11 * ( q \
    )**( 2 ) + -1 * ( q )**( 3 ) ) ) ) * ( kappa )**( 2 ) ) ) ) ) ) + 256 \
    * ( q )**( 2 ) * ( ( 1 + q ) )**( 4 ) * u * kappa * ( ( q )**( 4 ) * \
    ( S1 )**( 2 ) * ( 5 * ( S1 )**( 2 ) + ( S2 )**( 2 ) ) + ( ( S2 )**( 2 \
    ) * ( ( S1 )**( 2 ) + 5 * ( S2 )**( 2 ) ) + ( -2 * q * ( 3 * ( S1 \
    )**( 4 ) + ( 4 * ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( 9 * ( S2 )**( 4 ) \
    + -1 * ( S2 )**( 2 ) * ( kappa )**( 2 ) ) ) ) + ( -2 * ( q )**( 3 ) * \
    ( 9 * ( S1 )**( 4 ) + ( 3 * ( S2 )**( 4 ) + ( S1 )**( 2 ) * ( 4 * ( \
    S2 )**( 2 ) + -1 * ( kappa )**( 2 ) ) ) ) + 4 * ( q )**( 2 ) * ( 4 * \
    ( S1 )**( 4 ) + ( 4 * ( S2 )**( 4 ) + ( -1 * ( S2 )**( 2 ) * ( kappa \
    )**( 2 ) + ( S1 )**( 2 ) * ( 5 * ( S2 )**( 2 ) + -1 * ( kappa )**( 2 \
    ) ) ) ) ) ) ) ) ) ) + ( u )**( 4 ) * ( 256 * ( q )**( 2 ) * ( ( 1 + q \
    ) )**( 4 ) * ( ( ( S1 )**( 2 ) + -1 * ( S2 )**( 2 ) ) )**( 2 ) * ( ( \
    ( q )**( 2 ) * ( S1 )**( 2 ) + ( ( S2 )**( 2 ) + -1 * q * ( ( S1 )**( \
    2 ) + ( S2 )**( 2 ) ) ) ) )**( 2 ) * ( u )**( 2 ) + ( -512 * ( -1 + q \
    ) * ( q )**( 2 ) * ( ( 1 + q ) )**( 4 ) * ( ( q )**( 3 ) * ( S1 )**( \
    4 ) * ( ( S1 )**( 2 ) + ( S2 )**( 2 ) ) + ( -1 * ( S2 )**( 4 ) * ( ( \
    S1 )**( 2 ) + ( S2 )**( 2 ) ) + ( -1 * ( q )**( 2 ) * ( 2 * ( S1 )**( \
    6 ) + ( ( S1 )**( 4 ) * ( S2 )**( 2 ) + 3 * ( S1 )**( 2 ) * ( S2 )**( \
    4 ) ) ) + q * ( 3 * ( S1 )**( 4 ) * ( S2 )**( 2 ) + ( ( S1 )**( 2 ) * \
    ( S2 )**( 4 ) + 2 * ( S2 )**( 6 ) ) ) ) ) ) * u * kappa + -128 * ( q \
    )**( 2 ) * ( ( 1 + q ) )**( 4 ) * ( ( S2 )**( 2 ) * ( ( S1 )**( 4 ) + \
    ( -8 * ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( 11 * ( S2 )**( 4 ) + -2 * ( \
    S2 )**( 2 ) * ( kappa )**( 2 ) ) ) ) + ( ( q )**( 4 ) * ( S1 )**( 2 ) \
    * ( 11 * ( S1 )**( 4 ) + ( ( S2 )**( 4 ) + -2 * ( S1 )**( 2 ) * ( 4 * \
    ( S2 )**( 2 ) + ( kappa )**( 2 ) ) ) ) + ( 2 * ( q )**( 2 ) * ( 11 * \
    ( S1 )**( 6 ) + ( 11 * ( S2 )**( 6 ) + ( -6 * ( S2 )**( 4 ) * ( kappa \
    )**( 2 ) + ( -1 * ( S1 )**( 4 ) * ( 5 * ( S2 )**( 2 ) + 6 * ( kappa \
    )**( 2 ) ) + -5 * ( S1 )**( 2 ) * ( ( S2 )**( 4 ) + 2 * ( S2 )**( 2 ) \
    * ( kappa )**( 2 ) ) ) ) ) ) + ( q * ( -4 * ( S1 )**( 6 ) + ( -5 * ( \
    S1 )**( 4 ) * ( S2 )**( 2 ) + ( -29 * ( S2 )**( 6 ) + ( 12 * ( S2 \
    )**( 4 ) * ( kappa )**( 2 ) + 2 * ( S1 )**( 2 ) * ( 11 * ( S2 )**( 4 \
    ) + 6 * ( S2 )**( 2 ) * ( kappa )**( 2 ) ) ) ) ) ) + ( q )**( 3 ) * ( \
    -29 * ( S1 )**( 6 ) + ( -4 * ( S2 )**( 6 ) + ( 2 * ( S1 )**( 4 ) * ( \
    11 * ( S2 )**( 2 ) + 6 * ( kappa )**( 2 ) ) + ( S1 )**( 2 ) * ( -5 * \
    ( S2 )**( 4 ) + 12 * ( S2 )**( 2 ) * ( kappa )**( 2 ) ) ) ) ) ) ) ) ) \
    ) ) ) ) )

    coeff3 = \
    ( -32 * ( ( -1 + q ) )**( 2 ) * ( q )**( 3 ) * ( ( 1 + q ) )**( 4 ) \
    * kappa + ( ( u )**( 4 ) * ( 512 * ( q )**( 3 ) * ( ( 1 + q ) )**( 3 \
    ) * ( ( S1 )**( 2 ) + -1 * ( S2 )**( 2 ) ) * ( ( S2 )**( 2 ) * ( ( S1 \
    )**( 2 ) + -2 * ( S2 )**( 2 ) ) + ( ( q )**( 3 ) * ( 2 * ( S1 )**( 4 \
    ) + -1 * ( S1 )**( 2 ) * ( S2 )**( 2 ) ) + ( -1 * ( q )**( 2 ) * ( 3 \
    * ( S1 )**( 4 ) + ( -1 * ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( S2 )**( 4 \
    ) ) ) + q * ( ( S1 )**( 4 ) + ( -1 * ( S1 )**( 2 ) * ( S2 )**( 2 ) + \
    3 * ( S2 )**( 4 ) ) ) ) ) ) * u + -512 * ( q )**( 3 ) * ( ( 1 + q ) \
    )**( 3 ) * ( 2 * ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( 3 * ( S2 )**( 4 ) \
    + ( ( q )**( 3 ) * ( 3 * ( S1 )**( 4 ) + 2 * ( S1 )**( 2 ) * ( S2 \
    )**( 2 ) ) + ( q * ( 3 * ( S1 )**( 4 ) + ( -3 * ( S1 )**( 2 ) * ( S2 \
    )**( 2 ) + -5 * ( S2 )**( 4 ) ) ) + ( q )**( 2 ) * ( -5 * ( S1 )**( 4 \
    ) + ( -3 * ( S1 )**( 2 ) * ( S2 )**( 2 ) + 3 * ( S2 )**( 4 ) ) ) ) ) \
    ) ) * kappa ) + ( 32 * ( q )**( 3 ) * ( ( 1 + q ) )**( 3 ) * u * ( -1 \
    * ( ( -1 + q ) )**( 2 ) * ( 5 + 3 * q ) * ( S1 )**( 2 ) + ( -1 * ( ( \
    -1 + q ) )**( 2 ) * ( 3 + 5 * q ) * ( S2 )**( 2 ) + 4 * ( 1 + ( 9 * q \
    + ( 9 * ( q )**( 2 ) + ( q )**( 3 ) ) ) ) * ( kappa )**( 2 ) ) ) + ( \
    u )**( 2 ) * ( 128 * ( q )**( 3 ) * ( ( 1 + q ) )**( 3 ) * kappa * ( \
    ( -2 + ( 5 * q + -19 * ( q )**( 2 ) ) ) * ( S1 )**( 2 ) + -1 * q * ( \
    ( 19 + ( -5 * q + 2 * ( q )**( 2 ) ) ) * ( S2 )**( 2 ) + 4 * ( 1 + q \
    ) * ( kappa )**( 2 ) ) ) + 128 * ( q )**( 3 ) * ( ( 1 + q ) )**( 3 ) \
    * u * ( ( 1 + ( -14 * q + ( 20 * ( q )**( 2 ) + -5 * ( q )**( 3 ) ) ) \
    ) * ( S1 )**( 4 ) + ( ( -5 + ( 20 * q + ( -14 * ( q )**( 2 ) + ( q \
    )**( 3 ) ) ) ) * ( S2 )**( 4 ) + ( 4 * ( 1 + ( -1 * q + 3 * ( q )**( \
    2 ) ) ) * ( S2 )**( 2 ) * ( kappa )**( 2 ) + ( S1 )**( 2 ) * ( 2 * ( \
    -2 + ( q + ( ( q )**( 2 ) + -2 * ( q )**( 3 ) ) ) ) * ( S2 )**( 2 ) + \
    4 * q * ( 3 + ( -1 * q + ( q )**( 2 ) ) ) * ( kappa )**( 2 ) ) ) ) ) \
    ) ) ) )

    coeff4 = \
    ( 16 * ( q )**( 4 ) * ( ( -1 + ( q )**( 2 ) ) )**( 2 ) + ( 256 * ( q \
    )**( 4 ) * ( ( 1 + q ) )**( 2 ) * ( ( 1 + ( -6 * q + 6 * ( q )**( 2 ) \
    ) ) * ( S1 )**( 4 ) + ( -2 * ( 3 + ( -5 * q + 3 * ( q )**( 2 ) ) ) * \
    ( S1 )**( 2 ) * ( S2 )**( 2 ) + ( 6 + ( -6 * q + ( q )**( 2 ) ) ) * ( \
    S2 )**( 4 ) ) ) * ( u )**( 4 ) + ( -256 * ( q )**( 4 ) * ( ( 1 + q ) \
    )**( 2 ) * ( 1 + ( 3 * q + ( q )**( 2 ) ) ) * u * kappa + ( u )**( 2 \
    ) * ( -512 * ( q )**( 4 ) * ( ( 1 + q ) )**( 2 ) * ( ( 1 + ( -1 * q + \
    3 * ( q )**( 2 ) ) ) * ( S1 )**( 2 ) + ( 3 + ( -1 * q + ( q )**( 2 ) \
    ) ) * ( S2 )**( 2 ) ) * u * kappa + 128 * ( q )**( 4 ) * ( ( 1 + q ) \
    )**( 2 ) * ( ( -4 + ( 7 * q + ( q )**( 2 ) ) ) * ( S1 )**( 2 ) + ( ( \
    1 + ( 7 * q + -4 * ( q )**( 2 ) ) ) * ( S2 )**( 2 ) + 2 * ( 1 + ( 4 * \
    q + ( q )**( 2 ) ) ) * ( kappa )**( 2 ) ) ) ) ) ) )

    coeff5 = \
    ( 128 * ( q )**( 5 ) * ( ( 1 + q ) )**( 2 ) * u + ( u )**( 2 ) * ( \
    512 * ( q )**( 5 ) * ( 1 + q ) * ( ( -1 + 2 * q ) * ( S1 )**( 2 ) + \
    -1 * ( -2 + q ) * ( S2 )**( 2 ) ) * u + -512 * ( q )**( 5 ) * ( ( 1 + \
    q ) )**( 2 ) * kappa ) )

    coeff6 = \
    256 * ( q )**( 6 ) * ( u )**( 2 )

    return np.stack([coeff6, coeff5, coeff4, coeff3, coeff2, coeff1, coeff0])


def xiresonances(J,r,q,chi1,chi2):
    """
    Effective spin of the two spin-orbit resonances. The resonances minimizes and maximizes xi for a given value of J. The minimum corresponds to either DeltaPhi=0 or DeltaPhi=pi, the maximum always corresponds to DeltaPhi=pi.

    Call
    ----
    ximin,ximax = xiresonances(J,r,q,chi1,chi2)

    Parameters
    ----------
    J: float
    	Magnitude of the total angular momentum.
    r: float
    	Binary separation.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    ximin: float
    	Minimum value of the effective spin xi.
    ximax: float
    	Maximum value of the effective spin xi.
    """

    #Altough there are 6 solutions in general, we know that only two can lie between Smin and Smax.
    J=np.atleast_1d(J)
    r=np.atleast_1d(r)
    q=np.atleast_1d(q)
    chi1=np.atleast_1d(chi1)
    chi2=np.atleast_1d(chi2)

    kappa = eval_kappa(J, r, q)
    u = eval_u(r, q)

    Smin,Smax = Slimits_LJS1S2(J,r,q,chi1,chi2)
    xiroots= wraproots(xidiscriminant_coefficients,kappa,u,q,chi1,chi2)

    def _compute(Smin,Smax,J,r,xiroots,q,chi1,chi2):
        xiroots = xiroots[np.isfinite(xiroots)]
        Sroots = Satresonance(np.tile(J,xiroots.shape),np.tile(r,xiroots.shape),xiroots,np.tile(q,xiroots.shape),np.tile(chi1,xiroots.shape),np.tile(chi2,xiroots.shape))
        xires = xiroots[np.logical_and(Sroots>Smin, Sroots<Smax)]
        assert len(xires)<=2, "I found more than two resonances, this should not be possible."
        # If you didn't find enough solutions, append nans
        xires=np.concatenate([xires,np.repeat(np.nan,2-len(xires))])
        return xires

    ximin,ximax =np.array(list(map(_compute, Smin,Smax,J,r,xiroots,q,chi1,chi2))).T
    return np.stack([ximin,ximax])


def anglesresonances(J=None,r=None,xi=None,q=None,chi1=None,chi2=None):
    '''
    Compute the values of the angles corresponding to the two spin-orbit resonances. Provide either J or xi, not both.


    Provide either

    '''
    q=np.atleast_1d(q)

    if J is None and r is not None and xi is not None and q is not None and chi1 is not None and chi2 is not None:

        Jmin, Jmax = Jresonances(r,xi,q,chi1,chi2)
        Satmin = Satresonance(Jmin,r,xi,q,chi1,chi2)
        theta1atmin = eval_theta1(Satmin,Jmin,r,xi,q,chi1,chi2)
        theta2atmin = eval_theta2(Satmin,Jmin,r,xi,q,chi1,chi2)
        deltaphiatmin=np.tile(np.pi,q.shape)

        Satmax = Satresonance(Jmax,r,xi,q,chi1,chi2)
        theta1atmax = eval_theta1(Satmax,Jmax,r,xi,q,chi1,chi2)
        theta2atmax = eval_theta2(Satmax,Jmax,r,xi,q,chi1,chi2)
        deltaphiatmax=np.tile(0,q.shape)


    elif J is not None and r is not None and xi is None and q is not None and chi1 is not None and chi2 is not None:

        ximin, ximax = xiresonances(J,r,q,chi1,chi2)

        Satmin = Satresonance(J,r,ximin,q,chi1,chi2)
        theta1atmin = eval_theta1(Satmin,J,r,ximin,q,chi1,chi2)
        theta2atmin = eval_theta2(Satmin,J,r,ximin,q,chi1,chi2)
        # See Fig 5 in arxiv:1506.03492
        J=np.atleast_1d(J)
        S1,S2 = spinmags(q,chi1,chi2)
        L = eval_L(r,q)
        deltaphiatmin=np.where(J>np.abs(L-S1-S2), 0, np.pi)

        Satmax = Satresonance(J,r,ximax,q,chi1,chi2)
        theta1atmax = eval_theta1(Satmax,J,r,ximax,q,chi1,chi2)
        theta2atmax = eval_theta2(Satmax,J,r,ximax,q,chi1,chi2)
        deltaphiatmax=np.tile(np.pi,q.shape)

    else:
        raise TypeError("Provide either (r,xi,q,chi1,chi2) or (J,r,q,chi1,chi2).")

    return np.stack([theta1atmin,theta2atmin,deltaphiatmin,theta1atmax,theta2atmax,deltaphiatmax])


def xilimits(J=None,r=None,q=None,chi1=None,chi2=None):
    """
    Limits on the projected effective spin. The contraints considered depend on the inputs provided.
    - If q, chi1, and chi2 are provided, enforce xi = (1+q)S1.L + (1+1/q)S2.L.
    - If J, r, q, chi1, and chi2 are provided, the limits are given by the two spin-orbit resonances.

    Call
    ----
    ximin,ximax = xilimits(J=None,r=None,q=None,chi1=None,chi2=None)

    Parameters
    ----------
    J: float, optional (default: None)
    	Magnitude of the total angular momentum.
    r: float, optional (default: None)
    	Binary separation.
    q: float, optional (default: None)
    	Mass ratio: 0<=q<=1.
    chi1: float, optional (default: None)
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float, optional (default: None)
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    ximin: float
    	Minimum value of the effective spin xi.
    ximax: float
    	Maximum value of the effective spin xi.
    """

    if J is None and r is None and q is not None and chi1 is not None and chi2 is not None:
        ximin,ximax = xilimits_definition(q,chi1,chi2)

    elif J is not None and r is not None and q is not None and chi1 is not None and chi2 is not None:
        ximin,ximax = xiresonances(J,r,q,chi1,chi2)
        # Check precondition
        ximin_cond,ximax_cond = xilimits_definition(q,chi1,chi2)
        assert (ximin>ximin_cond).all() and (ximax<ximax_cond).all(), "Input values are incompatible."

    else:
        raise TypeError("Provide either (q,chi1,chi2) or (J,r,q,chi1,chi2).")

    return np.stack([ximin,ximax])


def Slimits_S1S2(q,chi1,chi2):
    """
    Limits on the total spin magnitude due to the vector relation S=S1+S2.

    Call
    ----
    Smin,Smax = Slimits_S1S2(q,chi1,chi2)

    Parameters
    ----------
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    Smin: float
    	Minimum value of the total spin S.
    Smax: float
    	Maximum value of the total spin S.
    """

    S1,S2= spinmags(q,chi1,chi2)
    Smin = np.abs(S1-S2)
    Smax = S1+S2

    return np.stack([Smin,Smax])


def Slimits_LJ(J,r,q):
    """
    Limits on the total spin magnitude due to the vector relation S=J-L.

    Call
    ----
    Smin,Smax = Slimits_LJ(J,r,q)

    Parameters
    ----------
    J: float
    	Magnitude of the total angular momentum.
    r: float
    	Binary separation.
    q: float
    	Mass ratio: 0<=q<=1.

    Returns
    -------
    Smin: float
    	Minimum value of the total spin S.
    Smax: float
    	Maximum value of the total spin S.
    """

    L= eval_L(r,q)
    Smin = np.abs(J-L)
    Smax = J+L

    return np.stack([Smin,Smax])


def Slimits_LJS1S2(J,r,q,chi1,chi2):
    """
    Limits on the total spin magnitude due to the vector relations S=S1+S2 and S=J-L.

    Call
    ----
    Smin,Smax = Slimits_LJS1S2(J,r,q,chi1,chi2)

    Parameters
    ----------
    J: float
    	Magnitude of the total angular momentum.
    r: float
    	Binary separation.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    Smin: float
    	Minimum value of the total spin S.
    Smax: float
    	Maximum value of the total spin S.
    """

    SminS1S2,SmaxS1S2 = Slimits_S1S2(q,chi1,chi2)
    SminLJ, SmaxLJ = Slimits_LJ(J,r,q)
    Smin = np.maximum(SminS1S2,SminLJ)
    Smax = np.minimum(SmaxS1S2,SmaxLJ)

    return np.stack([Smin,Smax])


def Scubic_coefficients(kappa,u,xi,q,chi1,chi2):
    """
    Coefficients of the cubic equation in S^2 that identifies the effective potentials.

    Call
    ----
    coeff3,coeff2,coeff1,coeff0 = Scubic_coefficients(kappa,u,xi,q,chi1,chi2)

    Parameters
    ----------
    kappa: float
    	Regularized angular momentum (J^2-L^2)/(2L).
    u: float
    	Compactified separation 1/(2L).
    xi: float
    	Effective spin.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    coeff3: float
    	Coefficient to the x^3 term in polynomial.
    coeff2: float
    	Coefficient to the x^2 term in polynomial.
    coeff1: float
    	Coefficient to the x^1 term in polynomial.
    coeff0: float
    	Coefficient to the x^0 term in polynomial.
    """

    kappa=np.atleast_1d(kappa)
    u=np.atleast_1d(u)
    xi=np.atleast_1d(xi)
    q=np.atleast_1d(q)
    S1,S2 = spinmags(q,chi1,chi2)

    coeff3 = \
    q * ( ( 1 + q ) )**( 2 ) * ( u )**( 2 )

    coeff2 = \
    ( 1/4 * ( ( 1 + q ) )**( 2 ) + ( -1/2 * q * ( ( 1 + q ) )**( 2 ) + ( \
    1/4 * ( q )**( 2 ) * ( ( 1 + q ) )**( 2 ) + ( ( -1 * q * ( ( 1 + q ) \
    )**( 2 ) * ( S1 )**( 2 ) + ( ( q )**( 2 ) * ( ( 1 + q ) )**( 2 ) * ( \
    S1 )**( 2 ) + ( ( ( 1 + q ) )**( 2 ) * ( S2 )**( 2 ) + -1 * q * ( ( 1 \
    + q ) )**( 2 ) * ( S2 )**( 2 ) ) ) ) * ( u )**( 2 ) + u * ( q * ( ( 1 \
    + q ) )**( 2 ) * xi + -2 * q * ( ( 1 + q ) )**( 2 ) * kappa ) ) ) ) )

    coeff1 = \
    ( -1/2 * ( 1 + -1 * ( q )**( 2 ) ) * ( S1 )**( 2 ) + ( 1/2 * ( q \
    )**( 2 ) * ( 1 + -1 * ( q )**( 2 ) ) * ( S1 )**( 2 ) + ( -1/2 * ( 1 + \
    -1 * ( q )**( 2 ) ) * ( S2 )**( 2 ) + ( 1/2 * ( q )**( 2 ) * ( 1 + -1 \
    * ( q )**( 2 ) ) * ( S2 )**( 2 ) + ( u * ( -1 * q * ( 1 + -1 * ( q \
    )**( 2 ) ) * ( S1 )**( 2 ) * ( xi + -2 * kappa ) + ( q * ( 1 + -1 * ( \
    q )**( 2 ) ) * ( S2 )**( 2 ) * ( xi + -2 * kappa ) + ( 2 * ( q )**( 2 \
    ) * ( 1 + -1 * ( q )**( 2 ) ) * ( S1 )**( 2 ) * kappa + -2 * ( 1 + -1 \
    * ( q )**( 2 ) ) * ( S2 )**( 2 ) * kappa ) ) ) + q * ( kappa * ( -1 * \
    xi + kappa ) + ( ( q )**( 2 ) * kappa * ( -1 * xi + kappa ) + q * ( ( \
    xi )**( 2 ) + ( -2 * xi * kappa + 2 * ( kappa )**( 2 ) ) ) ) ) ) ) ) \
    ) )

    coeff0 = \
    1/4 * ( -1 + ( q )**( 2 ) ) * ( ( -1 + ( q )**( 2 ) ) * ( S1 )**( 4 \
    ) + ( ( -1 + ( q )**( 2 ) ) * ( S2 )**( 4 ) + ( -4 * ( S2 )**( 2 ) * \
    kappa * ( -1 * q * xi + ( kappa + q * kappa ) ) + ( S1 )**( 2 ) * ( \
    -2 * ( -1 + ( q )**( 2 ) ) * ( S2 )**( 2 ) + 4 * q * kappa * ( -1 * \
    xi + ( kappa + q * kappa ) ) ) ) ) )

    return np.stack([coeff3, coeff2, coeff1, coeff0])


# TODO: this is a case where we use 2 for square.
def S2roots(J,r,xi,q,chi1,chi2):
    """
    Roots of the cubic equation in S^2 that identifies the effective potentials.

    Call
    ----
    Sminus2,Splus2,S32 = S2roots(J,r,xi,q,chi1,chi2)

    Parameters
    ----------
    J: float
    	Magnitude of the total angular momentum.
    r: float
    	Binary separation.
    xi: float
    	Effective spin.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    Sminus2: float
    	Lowest physical root, if present, of the effective potential equation.
    Splus2: float
    	Largest physical root, if present, of the effective potential equation.
    S32: float
    	Spurious root of the effective potential equation.
    """


    kappa = eval_kappa(J, r, q)
    u = eval_u(r, q)
    S32, Sminus2, Splus2 = wraproots(Scubic_coefficients,kappa,u,xi,q,chi1,chi2).T

    return np.stack([Sminus2,Splus2,S32])


def Slimits_plusminus(J,r,xi,q,chi1,chi2):
    """
    Limits on the total spin magnitude compatible with both J and xi.

    Call
    ----
    Smin,Smax = Slimits_plusminus(J,r,xi,q,chi1,chi2)

    Parameters
    ----------
    J: float
    	Magnitude of the total angular momentum.
    r: float
    	Binary separation.
    xi: float
    	Effective spin.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    Smin: float
    	Minimum value of the total spin S.
    Smax: float
    	Maximum value of the total spin S.
    """

    Sminus2,Splus2,_= S2roots(J,r,xi,q,chi1,chi2)
    with np.errstate(invalid='ignore'):
        Smin=Sminus2**0.5
        Smax=Splus2**0.5

    return np.stack([Smin,Smax])


def Satresonance(J,r,xi,q,chi1,chi2):
    """
    Assuming that the inputs correspond to a spin-orbit resonance, find the corresponding value of S. There will be two roots that are conincident if not for numerical errors: for concreteness, return the mean of the real part. This function does not check that the input is a resonance; it is up to the user.

    Call
    ----
    S = Satresonance(J,r,xi,q,chi1,chi2)

    Parameters
    ----------
    J: float
    	Magnitude of the total angular momentum.
    r: float
    	Binary separation.
    xi: float
    	Effective spin.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    S: float
    	Magnitude of the total spin.
    """

    kappa = eval_kappa(J, r, q)
    u = eval_u(r, q)
    coeffs = Scubic_coefficients(kappa,u,xi,q,chi1,chi2)
    with np.errstate(invalid='ignore'): # nan is ok here
        Sres = np.array([np.mean(np.real(np.sort_complex(np.roots(x))[1:]))**0.5 for x in coeffs.T])
    return Sres


def Slimits(J=None,r=None,xi=None,q=None,chi1=None,chi2=None):
    """
    Limits on the total spin magnitude. The contraints considered depend on the inputs provided.
    - If q, chi1, and chi2 are provided, enforce S=S1+S2.
    - If J, r, and q are provided, enforce S=J-L.
    - If J, r, q, chi1, and chi2 are provided, enforce S=S1+S2 and S=J-L.
    - If J, r, xi, q, chi1, and chi2 are provided, compute solve the cubic equation of the effective potentials (Sminus and Splus).

    Call
    ----
    Smin,Smax = Slimits(J=None,r=None,xi=None,q=None,chi1=None,chi2=None)

    Parameters
    ----------
    J: float, optional (default: None)
    	Magnitude of the total angular momentum.
    r: float, optional (default: None)
    	Binary separation.
    xi: float, optional (default: None)
    	Effective spin.
    q: float, optional (default: None)
    	Mass ratio: 0<=q<=1.
    chi1: float, optional (default: None)
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float, optional (default: None)
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    Smin: float
    	Minimum value of the total spin S.
    Smax: float
    	Maximum value of the total spin S.
    """

    if J is None and r is None and xi is None and q is not None and chi1 is not None and chi2 is not None:
        Smin,Smax = Slimits_S1S2(q,chi1,chi2)

    elif J is not None and r is not None and xi is None and q is not None and chi1 is None and chi2 is None:
        Smin,Smax = Slimits_LJ(J,r,q)

    elif J is not None and r is not None and xi is None and q is not None and chi1 is not None and chi2 is not None:
        Smin,Smax = Slimits_LJS1S2(J,r,q,chi1,chi2)

    elif J is not None and r is not None and xi is not None and q is not None and chi1 is not None and chi2 is not None:
        # Compute limits
        Smin,Smax = Slimits_plusminus(J,r,xi,q,chi1,chi2)
        # Check precondition
        Smin_cond,Smax_cond = Slimits_LJS1S2(J,r,q,chi1,chi2)
        assert (Smin>Smin_cond).all() and (Smax<Smax_cond).all(), "Input values are incompatible."

    else:
        raise TypeError("Provide one of the following: (q,chi1,chi2), (J,r,q), (J,r,q,chi1,chi2), (J,r,xi,q,chi1,chi2).")

    return np.stack([Smin,Smax])


# TODO: Check inter-compatibility of Slimits, Jlimits, xilimits
# TODO: check docstrings
# Tags for each limit check that fails?
# Davide: Does this function uses only Jlimits and xilimits or also Slimits? Move later?
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


#### Evaluations and conversions ####

# TODO Should this be called eval_xi?
def eval_xi(theta1=None,theta2=None,S=None,varphi=None,J=None,r=None,q=None,chi1=None,chi2=None):
    """
    Eftective spin. Provide either (theta1,theta2,q,chi1,chi2) or (S,varphi,J,r,q,chi1,chi2).

    Call
    ----
    xi = eval_xi(theta1=None,theta2=None,S=None,varphi=None,J=None,r=None,q=None,chi1=None,chi2=None)

    Parameters
    ----------
    theta1: float, optional (default: None)
    	Angle between orbital angular momentum and primary spin.
    theta2: float, optional (default: None)
    	Angle between orbital angular momentum and secondary spin.
    S: float, optional (default: None)
    	Magnitude of the total spin.
    varphi: float, optional (default: None)
    	Generalized nutation coordinate (Eq 9 in arxiv:1506.03492).
    J: float, optional (default: None)
    	Magnitude of the total angular momentum.
    r: float, optional (default: None)
    	Binary separation.
    q: float, optional (default: None)
    	Mass ratio: 0<=q<=1.
    chi1: float, optional (default: None)
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float, optional (default: None)
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    xi: float
    	Effective spin.
    """

    if theta1 is not None and theta2 is not None and S is None and varphi is None and J is None and r is None and q is not None and chi1 is not None and chi2 is not None:

        theta1=np.atleast_1d(theta1)
        theta2=np.atleast_1d(theta2)
        q=np.atleast_1d(q)
        S1,S2 = spinmags(q,chi1,chi2)
        xi=(1+q)*(q*S1*np.cos(theta1)+S2*np.cos(theta2))/q

    elif theta1 is None and theta2 is None and S is not None and varphi is not None and J is not None and r is not None and q is not None and chi1 is not None and chi2 is not None:

        S=np.atleast_1d(S)
        varphi=np.atleast_1d(varphi)
        J=np.atleast_1d(J)
        q=np.atleast_1d(q)
        S1,S2 = spinmags(q,chi1,chi2)
        L = eval_L(r,q)

        xi = \
        1/4 * ( L )**( -1 ) * ( q )**( -1 ) * ( S )**( -2 ) * ( ( ( J )**( 2 \
        ) + ( -1 * ( L )**( 2 ) + -1 * ( S )**( 2 ) ) ) * ( ( ( 1 + q ) )**( \
        2 ) * ( S )**( 2 ) + ( -1 + ( q )**( 2 ) ) * ( ( S1 )**( 2 ) + -1 * ( \
        S2 )**( 2 ) ) ) + -1 * ( 1 + -1 * ( q )**( 2 ) ) * ( ( ( J )**( 2 ) + \
        -1 * ( ( L + -1 * S ) )**( 2 ) ) )**( 1/2 ) * ( ( -1 * ( J )**( 2 ) + \
        ( ( L + S ) )**( 2 ) ) )**( 1/2 ) * ( ( ( S )**( 2 ) + -1 * ( ( S1 + \
        -1 * S2 ) )**( 2 ) ) )**( 1/2 ) * ( ( -1 * ( S )**( 2 ) + ( ( S1 + S2 \
        ) )**( 2 ) ) )**( 1/2 ) * np.cos( varphi ) )

    else:
        raise TypeError("Provide either (theta1,theta2,J,r,q,chi1,chi2) or (S,varphi,J,r,q,chi1,chi2).")

    return xi


def effectivepotential_plus(S,J,r,q,chi1,chi2):
    """
    Upper effective potential.

    Call
    ----
    xi = effectivepotential_plus(S,J,r,q,chi1,chi2)

    Parameters
    ----------
    S: float
    	Magnitude of the total spin.
    J: float
    	Magnitude of the total angular momentum.
    r: float
    	Binary separation.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    xi: float
    	Effective spin.
    """

    q=np.atleast_1d(q)
    varphi = np.tile(np.pi,q.shape)
    xi = eval_xi(S=S,varphi=varphi,J=J,r=r,q=q,chi1=chi1,chi2=chi2)

    return xi


def effectivepotential_minus(S,J,r,q,chi1,chi2):
    """
    Lower effective potential.

    Call
    ----
    xi = effectivepotential_minus(S,J,r,q,chi1,chi2)

    Parameters
    ----------
    S: float
    	Magnitude of the total spin.
    J: float
    	Magnitude of the total angular momentum.
    r: float
    	Binary separation.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    xi: float
    	Effective spin.
    """

    q=np.atleast_1d(q)
    varphi = np.tile(0,q.shape)
    xi = eval_xi(S=S,varphi=varphi,J=J,r=r,q=q,chi1=chi1,chi2=chi2)

    return xi


def eval_varphi(S, J, r, xi, q, chi1, chi2, sign=1):
    """
    Evaluate the nutation parameter varphi.

    Call
    ----
    varphi = eval_varphi(S,J,r,xi,q,chi1,chi2,sign = 1)

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
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    sign: integer, optional (default: 1)
    	Sign, either +1 or -1.

    Returns
    -------
    varphi: float
    	Generalized nutation coordinate (Eq 9 in arxiv:1506.03492).
    """

    S=np.atleast_1d(S)
    J=np.atleast_1d(J)
    xi=np.atleast_1d(xi)
    q=np.atleast_1d(q)
    sign=np.atleast_1d(sign)

    L = eval_L(r, q)
    S1, S2 = spinmags(q, chi1, chi2)

    cosvarphi = \
    ( xi /(1/4 * ( L )**( -1 ) * ( q )**( -1 ) * ( S )**( -2 ) ) - ( ( ( J \
    )**( 2 ) + ( -1 * ( L )**( 2 ) + -1 * ( S )**( 2 ) ) ) * ( ( ( 1 + q ) \
    )**( 2 ) * ( S )**( 2 ) + ( -1 + ( q )**( 2 ) ) * ( ( S1 )**( 2 ) + -1 \
    * ( S2 )**( 2 ) ) ) ) ) / (-1 * ( 1 + -1 * ( q )**( 2 ) ) * ( ( ( J    \
    )**( 2 ) + -1 * ( ( L + -1 * S ) )**( 2 ) ) )**( 1/2 ) * ( ( -1 * ( J  \
    )**( 2 ) + ( ( L + S ) )**( 2 ) ) )**( 1/2 ) * ( ( ( S )**( 2 ) + -1 * \
    ( ( S1 + -1 * S2 ) )**( 2 ) ) )**( 1/2 ) * ( ( -1 * ( S )**( 2 ) + ( ( \
    S1 + S2 ) )**( 2 ) ) )**( 1/2 ) )

    varphi = np.arccos(cosvarphi) * sign

    return varphi


def eval_costheta1(S,J,r,xi,q,chi1,chi2):
    """
    Cosine of the angle theta1 between the orbital angular momentum and the spin of the primary black hole.

    Call
    ----
    costheta1 = eval_costheta1(S,J,r,xi,q,chi1,chi2)

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
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    costheta1: float
    	Cosine of the angle between orbital angular momentum and primary spin.
    """

    S=np.atleast_1d(S)
    J=np.atleast_1d(J)
    q=np.atleast_1d(q)

    S1,S2 = spinmags(q,chi1,chi2)
    L = eval_L(r,q)

    costheta1= ( ((J**2-L**2-S**2)/L) - (2*q*xi)/(1+q) )/(2*(1-q)*S1)

    return costheta1


def eval_theta1(S,J,r,xi,q,chi1,chi2):
    """
    Angle theta1 between the orbital angular momentum and the spin of the primary black hole.

    Call
    ----
    theta1 = eval_theta1(S,J,r,xi,q,chi1,chi2)

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
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

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

    Call
    ----
    costheta2 = eval_costheta2(S,J,r,xi,q,chi1,chi2)

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
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    costheta2: float
    	Cosine of the angle between orbital angular momentum and secondary spin.
    """

    S=np.atleast_1d(S)
    J=np.atleast_1d(J)
    q=np.atleast_1d(q)

    S1,S2 = spinmags(q,chi1,chi2)
    L = eval_L(r,q)

    costheta2= ( ((J**2-L**2-S**2)*(-q/L)) + (2*q*xi)/(1+q) )/(2*(1-q)*S2)

    return costheta2


def eval_theta2(S,J,r,xi,q,chi1,chi2):
    """
    Angle theta2 between the orbital angular momentum and the spin of the secondary black hole.

    Call
    ----
    theta2 = eval_theta2(S,J,r,xi,q,chi1,chi2)

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
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

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

    Call
    ----
    costheta12 = eval_costheta12(S,q,chi1,chi2)

    Parameters
    ----------
    S: float
    	Magnitude of the total spin.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    costheta12: float
    	Cosine of the angle between the two spins.
    """

    S=np.atleast_1d(S)

    S1,S2 = spinmags(q,chi1,chi2)

    costheta12=(S**2-S1**2-S2**2)/(2*S1*S2)

    return costheta12


def eval_theta12(S,q,chi1,chi2):
    """
    Angle theta12 between the two spins.

    Call
    ----
    theta12 = eval_theta12(S,q,chi1,chi2)

    Parameters
    ----------
    S: float
    	Magnitude of the total spin.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

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

    Call
    ----
    cosdeltaphi = eval_cosdeltaphi(S,J,r,xi,q,chi1,chi2)

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
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    cosdeltaphi: float
    	Cosine of the angle between the projections of the two spins onto the orbital plane.
    """

    q=np.atleast_1d(q)

    S1,S2 = spinmags(q,chi1,chi2)
    costheta1=eval_costheta1(S,J,r,xi,q,chi1,chi2)
    costheta2=eval_costheta2(S,J,r,xi,q,chi1,chi2)
    costheta12=eval_costheta12(S,q,chi1,chi2)

    cosdeltaphi= (costheta12 - costheta1*costheta2)/((1-costheta1**2)*(1-costheta2**2))**0.5

    return cosdeltaphi


def eval_deltaphi(S,J,r,xi,q,chi1,chi2,sign=+1):
    """
    Angle deltaphi between the projections of the two spins onto the orbital plane. By default this is returned in [0,pi]. Setting sign=-1 returns the other half of the  precession cycle [-pi,0].

    Call
    ----
    deltaphi = eval_deltaphi(S,J,r,xi,q,chi1,chi2,sign = +1)

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
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    sign: integer, optional (default: +1)
    	Sign, either +1 or -1.

    Returns
    -------
    deltaphi: float
    	Angle between the projections of the two spins onto the orbital plane.
    """

    sign = np.atleast_1d(sign)
    cosdeltaphi=eval_cosdeltaphi(S,J,r,xi,q,chi1,chi2)
    deltaphi = np.sign(sign)*np.arccos(cosdeltaphi)

    return deltaphi


def eval_costhetaL(S,J,r,q,chi1,chi2):
    """
    Cosine of the angle thetaL betwen orbital angular momentum and total angular momentum.

    Call
    ----
    costhetaL = eval_costhetaL(S,J,r,q,chi1,chi2)

    Parameters
    ----------
    S: float
    	Magnitude of the total spin.
    J: float
    	Magnitude of the total angular momentum.
    r: float
    	Binary separation.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    costhetaL: float
    	Cosine of the angle betwen orbital angular momentum and total angular momentum.
    """

    S=np.atleast_1d(S)
    J=np.atleast_1d(J)

    S1,S2 = spinmags(q,chi1,chi2)
    L = eval_L(r,q)
    costhetaL=(J**2+L**2-S**2)/(2*J*L)

    return costhetaL


def eval_thetaL(S,J,r,q,chi1,chi2):
    """
    Angle thetaL betwen orbital angular momentum and total angular momentum.

    Call
    ----
    thetaL = eval_thetaL(S,J,r,q,chi1,chi2)

    Parameters
    ----------
    S: float
    	Magnitude of the total spin.
    J: float
    	Magnitude of the total angular momentum.
    r: float
    	Binary separation.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    thetaL: float
    	Angle betwen orbital angular momentum and total angular momentum.
    """

    costhetaL=eval_costhetaL(S,J,r,q,chi1,chi2)
    thetaL=np.arccos(costhetaL)

    return thetaL



def eval_J(theta1=None,theta2=None,deltaphi=None,kappa=None,r=None,q=None,chi1=None,chi2=None):
    """
    Magnitude of the total angular momentum. Provide either (theta1,theta,deltaphi,r,q,chi1,chhi2) or (kappa,r,q,chi1,chhi2).

    Call
    ----
    J = eval_J(theta1=None,theta2=None,deltaphi=None,kappa=None,r=None,q=None,chi1=None,chi2=None)

    Parameters
    ----------
    theta1: float, optional (default: None)
    	Angle between orbital angular momentum and primary spin.
    theta2: float, optional (default: None)
    	Angle between orbital angular momentum and secondary spin.
    deltaphi: float, optional (default: None)
    	Angle between the projections of the two spins onto the orbital plane.
    kappa: float, optional (default: None)
    	Regularized angular momentum (J^2-L^2)/(2L).
    r: float, optional (default: None)
    	Binary separation.
    q: float, optional (default: None)
    	Mass ratio: 0<=q<=1.
    chi1: float, optional (default: None)
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float, optional (default: None)
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    J: float
    	Magnitude of the total angular momentum.
    """

    if theta1 is not None and theta2 is not None and deltaphi is not None and kappa is None and r is not None and q is not None and chi1 is not None and chi2 is not None:

        theta1=np.atleast_1d(theta1)
        theta2=np.atleast_1d(theta2)
        deltaphi=np.atleast_1d(deltaphi)
        q=np.atleast_1d(q)

        S1,S2 = spinmags(q,chi1,chi2)
        L = eval_L(r,q)
        S=eval_S(theta1,theta2,deltaphi,q,chi1,chi2)

        J=(L**2+S**2+2*L*(S1*np.cos(theta1)+S2*np.cos(theta2)))**0.5

    elif theta1 is None and theta2 is None and deltaphi is None and kappa is not None and r is not None and q is not None and chi1 is None and chi2 is None:

        kappa=np.atleast_1d(kappa)

        L = eval_L(r,q)

        J = ( 2*L*kappa + L**2 )**0.5

    else:
        raise TypeError("Provide either (theta1,theta2,deltaphi,r,q,chi1,chi2) or (kappa,r,q,chi1,chi2).")

    return J


def eval_S(theta1,theta2,deltaphi,q,chi1,chi2):
    """
    Magnitude of the total spin from the spin angles.

    Call
    ----
    S = eval_S(theta1,theta2,deltaphi,q,chi1,chi2)

    Parameters
    ----------
    theta1: float
    	Angle between orbital angular momentum and primary spin.
    theta2: float
    	Angle between orbital angular momentum and secondary spin.
    deltaphi: float
    	Angle between the projections of the two spins onto the orbital plane.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    S: float
    	Magnitude of the total spin.
    """

    theta1=np.atleast_1d(theta1)
    theta2=np.atleast_1d(theta2)
    deltaphi=np.atleast_1d(deltaphi)

    S1,S2 = spinmags(q,chi1,chi2)

    S=(S1**2+S2**2+2*S1*S2*(np.sin(theta1)*np.sin(theta2)*np.cos(deltaphi)+np.cos(theta1)*np.cos(theta2)))**0.5

    return S


def eval_kappa(J, r, q):
    """
    Change of dependent variable to regularize the infinite orbital separation
    limit of the precession-averaged evolution equation.

    Call
    ----
    kappa = eval_kappa(J,r,q)

    Parameters
    ----------
    J: float
    	Magnitude of the total angular momentum.
    r: float
    	Binary separation.
    q: float
    	Mass ratio: 0<=q<=1.

    Returns
    -------
    kappa: float
    	Regularized angular momentum (J^2-L^2)/(2L).
    """

    J=np.atleast_1d(J)

    L = eval_L(r, q)
    kappa = (J**2 - L**2) / (2*L)

    return kappa


def eval_u(r, q):
    """
    Change of independent variable to regularize the infinite orbital separation
    limit of the precession-averaged evolution equation.

    Call
    ----
    u = eval_u(r,q)

    Parameters
    ----------
    r: float
    	Binary separation.
    q: float
    	Mass ratio: 0<=q<=1.

    Returns
    -------
    u: float
    	Compactified separation 1/(2L).
    """

    L = eval_L(r, q)
    u = 1 / (2*L)

    return u



def eval_kappainf(theta1inf, theta2inf, q, chi1, chi2):
    """
    Infinite orbital-separation limit of the regularized momentum kappa.

    Call
    ----
    kappainf = eval_kappainf(theta1inf,theta2inf,q,chi1,chi2)

    Parameters
    ----------
    theta1inf: float
    	Asymptotic value of the angle between orbital angular momentum and primary spin.
    theta2inf: float
    	Asymptotic value of the angle between orbital angular momentum and secondary spin.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    kappainf: float
    	Asymptotic value of the regularized momentum kappa.
    """

    theta1inf=np.atleast_1d(theta1inf)
    theta2inf=np.atleast_1d(theta2inf)

    S1, S2 = spinmags(q, chi1, chi2)
    kappainf = S1*np.cos(theta1inf) + S2*np.cos(theta2inf)

    return kappainf


def eval_costheta1inf(kappainf, xi, q, chi1, chi2):
    """
    Infinite orbital separation limit of the cosine of the angle between the
    orbital angular momentum and the primary spin.

    Call
    ----
    costheta1inf = eval_costheta1inf(kappainf,xi,q,chi1,chi2)

    Parameters
    ----------
    kappainf: float
    	Asymptotic value of the regularized momentum kappa.
    xi: float
    	Effective spin.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    costheta1inf: float
    	Cosine of the asymptotic angle between orbital angular momentum and primary spin.
    """

    kappainf=np.atleast_1d(kappainf)
    xi=np.atleast_1d(xi)
    q=np.atleast_1d(q)

    S1, S2 = spinmags(q, chi1, chi2)
    costheta1inf = (-xi + kappainf*(1+1/q)) / (S1*(1/q-q))

    return costheta1inf


def eval_theta1inf(kappainf, xi, q, chi1, chi2):
    """
    Infinite orbital separation limit of the angle between the orbital angular
    momentum and the primary spin.

    Call
    ----
    theta1inf = eval_theta1inf(kappainf,xi,q,chi1,chi2)

    Parameters
    ----------
    kappainf: float
    	Asymptotic value of the regularized momentum kappa.
    xi: float
    	Effective spin.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    theta1inf: float
    	Asymptotic value of the angle between orbital angular momentum and primary spin.
    """


    costheta1inf = eval_costheta1inf(kappainf, xi, q, chi1, chi2)
    theta1inf = np.arccos(costheta1inf)

    return theta1inf


def eval_costheta2inf(kappainf, xi, q, chi1, chi2):
    """
    Infinite orbital separation limit of the cosine of the angle between the
    orbital angular momentum and the secondary spin.

    Call
    ----
    theta1inf = eval_costheta2inf(kappainf,xi,q,chi1,chi2)

    Parameters
    ----------
    kappainf: float
    	Asymptotic value of the regularized momentum kappa.
    xi: float
    	Effective spin.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    theta1inf: float
    	Asymptotic value of the angle between orbital angular momentum and primary spin.
    """

    kappainf=np.atleast_1d(kappainf)
    xi=np.atleast_1d(xi)
    q=np.atleast_1d(q)

    S1, S2 = spinmags(q, chi1, chi2)
    costheta2inf = (xi - kappainf*(1+q)) / (S2*(1/q-q))

    return costheta2inf


def eval_theta2inf(kappainf, xi, q, chi1, chi2):
    """
    Infinite orbital separation limit of the angle between the orbital angular
    momentum and the secondary spin.

    Call
    ----
    theta2inf = eval_theta2inf(kappainf,xi,q,chi1,chi2)

    Parameters
    ----------
    kappainf: float
    	Asymptotic value of the regularized momentum kappa.
    xi: float
    	Effective spin.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    theta2inf: float
    	Asymptotic value of the angle between orbital angular momentum and secondary spin.
    """

    costheta2inf = eval_costheta2inf(kappainf, xi, q, chi1, chi2)
    theta2inf = np.arccos(costheta2inf)

    return theta2inf


def morphology(J,r,xi,q,chi1,chi2,simpler=False):
    """
    Evaluate the spin morphology and return `L0` for librating about DeltaPhi=0, `Lpi` for librating about DeltaPhi=pi, `C-` for circulating from DeltaPhi=pi to DeltaPhi=0, and `C+` for circulating from DeltaPhi=0 to DeltaPhi=pi. If simpler=True, do not distinguish between the two circulating morphologies and return `C` for both.

    Call
    ----
    morph = morphology(J,r,xi,q,chi1,chi2,simpler = False)

    Parameters
    ----------
    J: float
    	Magnitude of the total angular momentum.
    r: float
    	Binary separation.
    xi: float
    	Effective spin.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    simpler: boolean, optional (default: False)
    	If True simplifies output.

    Returns
    -------
    morph: string
    	Spin morphology.
    """

    Smin,Smax = Slimits_plusminus(J,r,xi,q,chi1,chi2)
    # Pairs of booleans based on the values of deltaphi at S- and S+
    status = np.transpose([eval_cosdeltaphi(Smin,J,r,xi,q,chi1,chi2) >0,eval_cosdeltaphi(Smax,J,r,xi,q,chi1,chi2) >0])
    # Map to labels
    dictlabel = {(False,False):"Lpi", (True,True):"L0", (False, True):"C-", (True, False):"C+"}
    # Subsitute pairs with labels
    morphs = np.zeros(Smin.shape)
    for k, v in dictlabel.items():
        morphs=np.where((status == k).all(axis=1),v,morphs)
    # Simplifies output, only one circulating morphology
    if simpler:
        morphs = np.where( np.logical_or(morphs == 'C+',morphs == 'C-'), 'C', morphs)

    return morphs


def conserved_to_angles(S,J,r,xi,q,chi1,chi2,sign=+1):
    """
    Convert conserved quantities (S,J,xi) into angles (theta1,theta2,deltaphi).
    Setting sign=+1 (default) returns deltaphi in [0, pi], setting sign=-1 returns deltaphi in [-pi,0].

    Call
    ----
    theta1,theta2,deltaphi = conserved_to_angles(S,J,r,xi,q,chi1,chi2,sign = +1)

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
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    sign: integer, optional (default: +1)
    	Sign, either +1 or -1.

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

    return np.stack([theta1,theta2,deltaphi])


def angles_to_conserved(theta1,theta2,deltaphi,r,q,chi1,chi2):
    """
    Convert angles (theta1,theta2,deltaphi) into conserved quantities (S,J,xi).

    Call
    ----
    S,J,xi = angles_to_conserved(theta1,theta2,deltaphi,r,q,chi1,chi2)

    Parameters
    ----------
    theta1: float
    	Angle between orbital angular momentum and primary spin.
    theta2: float
    	Angle between orbital angular momentum and secondary spin.
    deltaphi: float
    	Angle between the projections of the two spins onto the orbital plane.
    r: float
    	Binary separation.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    S: float
    	Magnitude of the total spin.
    J: float
    	Magnitude of the total angular momentum.
    xi: float
    	Effective spin.
    """

    S=eval_S(theta1,theta2,deltaphi,q,chi1,chi2)
    J=eval_J(theta1=theta1,theta2=theta2,deltaphi=deltaphi,r=r,q=q,chi1=chi1,chi2=chi2)
    xi=eval_xi(theta1=theta1,theta2=theta2,q=q,chi1=chi1,chi2=chi2)

    return np.stack([S,J,xi])


def angles_to_asymptotic(theta1inf, theta2inf, q, chi1, chi2):
    """
    Convert asymptotic angles (theta1, theta2) into regularized momentum and effective spin (kappa, xi).

    Call
    ----
    kappainf,xi = angles_to_asymptotic(theta1inf,theta2inf,q,chi1,chi2)

    Parameters
    ----------
    theta1inf: float
    	Asymptotic value of the angle between orbital angular momentum and primary spin.
    theta2inf: float
    	Asymptotic value of the angle between orbital angular momentum and secondary spin.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    kappainf: float
    	Asymptotic value of the regularized momentum kappa.
    xi: float
    	Effective spin.
    """

    kappainf = eval_kappainf(theta1inf, theta2inf, q, chi1, chi2)
    xi=eval_xi(theta1=theta1inf,theta2=theta2inf,q=q,chi1=chi1,chi2=chi2)

    return np.stack([kappainf, xi])


def asymptotic_to_angles(kappainf, xi, q, chi1, chi2):
    """
    Convert regularized momentum and effective spin (kappa, xi) into asymptotic angles (theta1, theta2).

    Call
    ----
    theta1inf,theta2inf = asymptotic_to_angles(kappainf,xi,q,chi1,chi2)

    Parameters
    ----------
    kappainf: float
    	Asymptotic value of the regularized momentum kappa.
    xi: float
    	Effective spin.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    theta1inf: float
    	Asymptotic value of the angle between orbital angular momentum and primary spin.
    theta2inf: float
    	Asymptotic value of the angle between orbital angular momentum and secondary spin.
    """

    theta1inf = eval_theta1inf(kappainf, xi, q, chi1, chi2)
    theta2inf = eval_theta2inf(kappainf, xi, q, chi1, chi2)

    return np.stack([theta1inf, theta2inf])



def vectors_to_conserved(Lvec, S1vec, S2vec, q):
    """
    Convert cartesian vectors (L,S1,S2) into conserved quantities (S,J,xi).

    Call
    ----
    S,J,xi = vectors_to_conserved(Lvec,S1vec,S2vec,q)

    Parameters
    ----------
    Lvec: array
    	Cartesian vector of the orbital angular momentum.
    S1vec: array
    	Cartesian vector of the primary spin.
    S2vec: array
    	Cartesian vector of the secondary spin.
    q: float
    	Mass ratio: 0<=q<=1.

    Returns
    -------
    S: float
    	Magnitude of the total spin.
    J: float
    	Magnitude of the total angular momentum.
    xi: float
    	Effective spin.
    """

    Lvec = np.atleast_2d(Lvec)
    S1vec = np.atleast_2d(S1vec)
    S2vec = np.atleast_2d(S2vec)

    S = norm_nested(S1vec+S2vec)
    J = norm_nested(S1vec+S2vec+Lvec)
    L = norm_nested(Lvec)
    m1, m2 = masses(q)

    xi = dot_nested(S1vec,Lvec)/(m1*L) + dot_nested(S2vec,Lvec)/(m2*L)

    return np.stack([S, J, xi])

# TODO: write function to get theta12 from theta1,theta2 and deltaphi

def vectors_to_angles(Lvec, S1vec, S2vec):
    """
    Convert cartesian vectors (L,S1,S2) into angles (theta1,theta2,deltaphi). The convention for the sign of deltaphi is given in Eq. (2d) of arxiv:1506.03492.

    Call
    ----
    theta1,theta2,deltaphi = vectors_to_angles(Lvec,S1vec,S2vec,q)

    Parameters
    ----------
    Lvec: array
    	Cartesian vector of the orbital angular momentum.
    S1vec: array
    	Cartesian vector of the primary spin.
    S2vec: array
    	Cartesian vector of the secondary spin.
    q: float
    	Mass ratio: 0<=q<=1.

    Returns
    -------
    theta1: float
    	Angle between orbital angular momentum and primary spin.
    theta2: float
    	Angle between orbital angular momentum and secondary spin.
    deltaphi: float
    	Angle between the projections of the two spins onto the orbital plane.
    """

    Lvec = np.atleast_2d(Lvec)
    S1vec = np.atleast_2d(S1vec)
    S2vec = np.atleast_2d(S2vec)

    S1vec = normalize_nested(S1vec)
    S2vec = normalize_nested(S2vec)
    Lvec = normalize_nested(Lvec)

    theta1 = np.arccos(dot_nested(S1vec,Lvec))
    theta2 = np.arccos(dot_nested(S2vec,Lvec))
    S1crL = np.cross(S1vec, Lvec)
    S2crL = np.cross(S2vec, Lvec)

    absdeltaphi = np.arccos(dot_nested(normalize_nested(S1crL), normalize_nested(S2crL)))
    signdeltaphi = np.sign(dot_nested(Lvec,np.cross(S1crL,S2crL)))
    deltaphi = absdeltaphi*signdeltaphi

    return np.stack([theta1, theta2, deltaphi])


def conserved_to_Jframe(S, J, r, xi, q, chi1, chi2):
    """
    Convert the conserved quantities (S,J,xi) to angular momentum vectors (L,S1,S2) in the frame
    aligned with the total angular momentum. In particular, we set Jx=Jy=Ly=0.

    Call
    ----
    Lvec,S1vec,S2vec = conserved_to_Jframe(S,J,r,xi,q,chi1,chi2)

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
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    Lvec: array
    	Cartesian vector of the orbital angular momentum.
    S1vec: array
    	Cartesian vector of the primary spin.
    S2vec: array
    	Cartesian vector of the secondary spin.
    """

    S=np.atleast_1d(S)
    J=np.atleast_1d(J)

    L = eval_L(r, q)
    S1, S2 = spinmags(q, chi1, chi2)
    varphi = eval_varphi(S, J, r, xi, q, chi1, chi2)
    thetaL = eval_thetaL(S, J, r, q, chi1, chi2)

    Lx = L * np.sin(thetaL)
    Ly = np.zeros(L.shape)
    Lz = L * np.cos(thetaL)
    Lvec = np.transpose([Lx, Ly, Lz])

    A1 = (J**2 - (L-S)**2)**0.5
    A2 = ((L+S)**2 - J**2)**0.5
    A3 = (S**2 - (S1-S2)**2)**0.5
    A4 = ((S1+S2)**2 - S**2)**0.5

    S1x = (-(S**2+S1**2-S2**2)*A1*A2 + (J**2-L**2+S**2)*A3*A4*np.cos(varphi)) \
        / (4*J*S**2)
    S1y = A3 * A4 * np.sin(varphi) / (2*S)
    S1z = ((S**2+S1**2-S2**2)*(J**2-L**2+S**2) + A1*A2*A3*A4*np.cos(varphi)) \
        / (4*J*S**2)
    S1vec = np.transpose([S1x, S1y, S1z])

    S2x = -((S**2+S2**2-S1**2)*A1*A2 + (J**2-L**2+S**2)*A3*A4*np.cos(varphi)) \
        / (4*J*S**2)
    S2y = -A3*A4*np.sin(varphi) / (2*S)
    S2z = ((S**2+S2**2-S1**2)*(J**2-L**2+S**2) - A1*A2*A3*A4*np.cos(varphi)) \
        / (4*J*S**2)
    S2vec = np.transpose([S2x, S2y, S2z])

    return np.stack([Lvec, S1vec, S2vec])


def angles_to_Jframe(theta1, theta2, deltaphi, r, q, chi1, chi2):
    """
    Convert the angles (theta1,theta2,deltaphi) to angular momentum vectors (L,S1,S2) in the frame
    aligned with the total angular momentum. In particular, we set Jx=Jy=Ly=0.

    Call
    ----
    Lvec,S1vec,S2vec = angles_to_Jframe(theta1,theta2,deltaphi,r,q,chi1,chi2)

    Parameters
    ----------
    theta1: float
    	Angle between orbital angular momentum and primary spin.
    theta2: float
    	Angle between orbital angular momentum and secondary spin.
    deltaphi: float
    	Angle between the projections of the two spins onto the orbital plane.
    r: float
    	Binary separation.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    Lvec: array
    	Cartesian vector of the orbital angular momentum.
    S1vec: array
    	Cartesian vector of the primary spin.
    S2vec: array
    	Cartesian vector of the secondary spin.
    """

    S, J, xi = angles_to_conserved(theta1, theta2, deltaphi, r, q, chi1, chi2)
    Lvec, S1vec, S2vec = conserved_to_Jframe(S, J, r, xi, q, chi1, chi2)

    return np.stack([Lvec, S1vec, S2vec])


def angles_to_Lframe(theta1, theta2, deltaphi, r, q, chi1, chi2):
    """
    Convert the angles (theta1,theta2,deltaphi) to angular momentum vectors (L,S1,S2) in the frame aligned with the orbital angular momentum. In particular, we set Lx=Ly=S1y=0.

    Call
    ----
    Lvec,S1vec,S2vec = angles_to_Lframe(theta1,theta2,deltaphi,r,q,chi1,chi2)

    Parameters
    ----------
    theta1: float
    	Angle between orbital angular momentum and primary spin.
    theta2: float
    	Angle between orbital angular momentum and secondary spin.
    deltaphi: float
    	Angle between the projections of the two spins onto the orbital plane.
    r: float
    	Binary separation.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    Lvec: array
    	Cartesian vector of the orbital angular momentum.
    S1vec: array
    	Cartesian vector of the primary spin.
    S2vec: array
    	Cartesian vector of the secondary spin.
    """

    L = eval_L(r, q)
    S1, S2 = spinmags(q, chi1, chi2)

    Lx = np.zeros(L.shape)
    Ly = np.zeros(L.shape)
    Lz = L
    Lvec = np.transpose([Lx, Ly, Lz])

    S1x = S1 * np.sin(theta1)
    S1y = np.zeros(S1.shape)
    S1z = S1 * np.cos(theta1)
    S1vec = np.transpose([S1x, S1y, S1z])

    S2x = S2 * np.sin(theta2) * np.cos(deltaphi)
    S2y = S2 * np.sin(theta2) * np.sin(deltaphi)
    S2z = S2 * np.cos(theta2)
    S2vec = np.transpose([S2x, S2y, S2z])

    return np.stack([Lvec, S1vec, S2vec])


def conserved_to_Lframe(S, J, r, xi, q, chi1, chi2):
    """
    Convert the angles (theta1,theta2,deltaphi) to angular momentum vectors (L,S1,S2) in the frame aligned with the orbital angular momentum. In particular, we set Lx=Ly=S1y=0.

    Call
    ----
    Lvec,S1vec,S2vec = conserved_to_Lframe(S,J,r,xi,q,chi1,chi2)

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
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    Lvec: array
    	Cartesian vector of the orbital angular momentum.
    S1vec: array
    	Cartesian vector of the primary spin.
    S2vec: array
    	Cartesian vector of the secondary spin.
    """

    theta1, theta2, deltaphi = conserved_to_angles(S, J, r, xi, q, chi1, chi2)
    Lvec, S1vec, S2vec = angles_to_Lframe(theta1, theta2, deltaphi, r, q, chi1, chi2)

    return np.stack([Lvec, S1vec, S2vec])


#### Precessional timescale dynamics ####

def Speriod_prefactor(r,xi,q):
    """
    Numerical prefactor to the precession period.

    Call
    ----
    coeff = Speriod_prefactor(r,xi,q)

    Parameters
    ----------
    r: float
    	Binary separation.
    xi: float
    	Effective spin.
    q: float
    	Mass ratio: 0<=q<=1.

    Returns
    -------
    coeff: float
    	Coefficient.
    """

    r=np.atleast_1d(r)
    xi=np.atleast_1d(xi)

    eta=eval_eta(q)
    coeff = (3/2)*(1/(r**3*eta**0.5))*(1-(xi/r**0.5))

    return coeff


# TODO: Here we use S2 for square...
def dS2dtsquared(S,J,r,xi,q,chi1,chi2):
    """
    Squared first time derivative of the squared total spin, on the precession timescale.

    Call
    ----
    dS2dt2 = dS2dtsquared(S,J,r,xi,q,chi1,chi2)

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
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    dS2dt2: float
    	Squared first derivative of the squared total spin.
    """

    mathcalA = Speriod_prefactor(r,xi,q)
    Sminus2,Splus2,S32 = S2roots(J,r,xi,q,chi1,chi2)
    dS2dt2 = - mathcalA**2 * (S**2-Splus2) * (S**2-Sminus2) * (S**2-S32)

    return dS2dt2


# Change name to this function, otherwise is identical to the returned variable.
def dS2dt(S,J,r,xi,q,chi1,chi2):
    """
    Time derivative of the squared total spin, on the precession timescale.

    Call
    ----
    dS2dt = dS2dt(S,J,r,xi,q,chi1,chi2)

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
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    dS2dt: float
    	Time derivative of the squared total spin.
    """

    return dS2dtsquared(S,J,r,xi,q,chi1,chi2)**0.5

# Change name to this function, otherwise is identical to the returned variable.
def dSdt(S,J,r,xi,q,chi1,chi2):
    """
    Time derivative of the total spin, on the precession timescale.

    Call
    ----
    dSdt = dSdt(S,J,r,xi,q,chi1,chi2)

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
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    dSdt: float
    	Time derivative of the total spin.
    """

    return dS2dt(S,J,r,xi,q,chi1,chi2) / (2*S)


def elliptic_parameter(Sminus2,Splus2,S32):
    """
    Parameter m entering elliptic functiosn for the evolution of S.

    Call
    ----
    m = elliptic_parameter(Sminus2,Splus2,S32)

    Parameters
    ----------
    Sminus2: float
    	Lowest physical root, if present, of the effective potential equation.
    Splus2: float
    	Largest physical root, if present, of the effective potential equation.
    S32: float
    	Spurious root of the effective potential equation.

    Returns
    -------
    m: float
    	Parameter of elliptic function(s).
    """

    m = (Splus2-Sminus2)/(Splus2-S32)

    return m


def Speriod(J,r,xi,q,chi1,chi2):
    """
    Period of S as it oscillates from S- to S+ and back to S-.

    Call
    ----
    tau = Speriod(J,r,xi,q,chi1,chi2)

    Parameters
    ----------
    J: float
    	Magnitude of the total angular momentum.
    r: float
    	Binary separation.
    xi: float
    	Effective spin.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    tau: float
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
    The broadcasting rules for this function are more general than those of the rest of the code. The variable t is allowed to have shapes (N,M) while all the other variables have shape (N,). This is useful to sample M precession configuration for each of the N binaries specified as inputs.

    Call
    ----
    S = Soft(t,J,r,xi,q,chi1,chi2)

    Parameters
    ----------
    t: float
    	Time.
    J: float
    	Magnitude of the total angular momentum.
    r: float
    	Binary separation.
    xi: float
    	Effective spin.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    S: float
    	Magnitude of the total spin.
    """

    t=np.atleast_1d(t)
    mathcalA=Speriod_prefactor(r,xi,q)
    Sminus2,Splus2,S32 = S2roots(J,r,xi,q,chi1,chi2)
    m = elliptic_parameter(Sminus2,Splus2,S32)

    sn,_,dn,_ = scipy.special.ellipj(t.T*mathcalA*(Splus2-S32)**0.5/2,m)
    Ssq = Sminus2 + (Splus2-Sminus2)*((Sminus2-S32)/(Splus2-S32)) *(sn/dn)**2
    S=Ssq.T**0.5

    return S


def Ssampling(J,r,xi,q,chi1,chi2,N=1):
    """
    Sample N values of S at fixed separation accoring to its PN-weighted distribution function.
    Can only be used to sample the *same* number of configuration for each binary. If the inputs J,r,xi,q,chi1, and chi2 have shape (M,) the output will have shape
    - (M,N) if M>1 and N>1;
    - (M,) if N=1;
    - (N,) if M=1.

    Call
    ----
    S = Ssampling(J,r,xi,q,chi1,chi2,N = 1)

    Parameters
    ----------
    J: float
    	Magnitude of the total angular momentum.
    r: float
    	Binary separation.
    xi: float
    	Effective spin.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    N: integer, optional (default: 1)
    	Number of samples.

    Returns
    -------
    S: float
    	Magnitude of the total spin.
    """

    tau = Speriod(J,r,xi,q,chi1,chi2)
    # For each binary, generate N samples between 0 and tau.
    t = np.random.uniform(size=tau.size*N).reshape((tau.size,N)) * tau[:,None]
    # Note the special broadcasting rules of Soft, see Soft.__docs__
    # S has shape (M,N).
    S = Soft(t,J,r,xi,q,chi1,chi2)

    # np.squeeze is necessary to return shape (M,) instead of (M,1) if N=1
    # np.atleast_1d is necessary to retun shape (1,) instead of (,) if M=N=1
    return np.atleast_1d(np.squeeze(S))


def S2av_mfactor(m):
    """
    Factor depending on the elliptic parameter in the precession averaged squared total spin. This is (1 - E(m)/K(m)) / m.

    Call
    ----
    coeff = S2av_mfactor(m)

    Parameters
    ----------
    m: float
    	Parameter of elliptic function(s).

    Returns
    -------
    coeff: float
    	Coefficient.
    """

    m=np.atleast_1d(m)
    # The limit of the S2av coefficient as m->0 is finite and equal to 1/2.
    # This is implementation is numerically stable up to m~1e-10.
    # For m=1e-7, the analytic m=0 limit is returned with a precision of 1e-9, which is enough.
    m=np.maximum(1e-7,m)
    coeff = (1- scipy.special.ellipe(m)/scipy.special.ellipk(m))/m

    return coeff


# TODO: change name to this function
def S2av(J, r, xi, q, chi1, chi2):
    """
    Analytic precession averaged expression for the squared total spin.

    Call
    ----
    Ssq = S2av(J,r,xi,q,chi1,chi2)

    Parameters
    ----------
    J: float
    	Magnitude of the total angular momentum.
    r: float
    	Binary separation.
    xi: float
    	Effective spin.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    Ssq: float
    	Squared magnitude of the total spin.
    """

    Sminus2, Splus2, S32 = S2roots(J, r, xi, q, chi1, chi2)
    m = elliptic_parameter(Sminus2, Splus2, S32)
    Ssq = Splus2 - (Splus2-Sminus2)*S2av_mfactor(m)

    return Ssq

# TODO again S2 instead of Ssq
def S2rootsinf(theta1inf, theta2inf, q, chi1, chi2):
    """
    Infinite orbital separation limit of the roots of the cubic equation in S^2.

    Call
    ----
    Sminus2inf,Splus2inf,S32inf = S2rootsinf(theta1inf,theta2inf,q,chi1,chi2)

    Parameters
    ----------
    theta1inf: float
    	Asymptotic value of the angle between orbital angular momentum and primary spin.
    theta2inf: float
    	Asymptotic value of the angle between orbital angular momentum and secondary spin.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    Sminus2inf: float
    	Asymptotic value of the lowest physical root, if present, of the effective potential equation.
    Splus2inf: float
    	Asymptotic value of the largest physical root, if present, of the effective potential equation.
    S32inf: float
    	Asymptotic value of the spurious root of the effective potential equation.
    """

    S1, S2 = spinmags(q, chi1, chi2)
    coscos = np.cos(theta1inf)*np.cos(theta2inf)
    sinsin = np.sin(theta1inf)*np.sin(theta2inf)
    Sminus2inf = S1**2 + S2**2 + 2*S1*S2*(coscos - sinsin)
    Splus2inf = S1**2 + S2**2 + 2*S1*S2*(coscos + sinsin)
    S32inf = -np.inf

    return toarray(Sminus2inf, Splus2inf, S32inf)


def S2avinf(theta1inf, theta2inf, q, chi1, chi2):
    """
    Infinite orbital separation limit of the precession averaged values of S^2
    from the asymptotic angles (theta1, theta2).

    Call
    ----
    Ssq = S2avinf(theta1inf,theta2inf,q,chi1,chi2)

    Parameters
    ----------
    theta1inf: float
    	Asymptotic value of the angle between orbital angular momentum and primary spin.
    theta2inf: float
    	Asymptotic value of the angle between orbital angular momentum and secondary spin.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    Ssq: float
    	Squared magnitude of the total spin.
    """

    theta1inf = np.atleast_1d(theta1inf)
    theta2inf = np.atleast_1d(theta2inf)

    S1, S2 = spinmags(q, chi1, chi2)
    S2avinf = S1**2 + S2**2 + 2*S1*S2*np.cos(theta1inf)*np.cos(theta2inf)

    return S2avinf


#### Precession-averaged evolution ####

def dkappadu(u, kappa, xi, q, chi1, chi2):
    """
    Right-hand side of the dkappa/du ODE describing precession-averaged inspiral. This is an internal function used by the ODE integrator and is not array-compatible. It is equivalent to S2av and S2avinf and it has been re-written for optimization purposes.

    Call
    ----
    RHS = dkappadu(kappa,u,xi,q,chi1,chi2)

    Parameters
    ----------
    kappa: float
    	Regularized angular momentum (J^2-L^2)/(2L).
    u: float
    	Compactified separation 1/(2L).
    xi: float
    	Effective spin.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    RHS: float
    	Right-hand side.
    """

    if u==0:
       # In this case use analytic result
       theta1inf,theta2inf = asymptotic_to_angles(kappa,xi,q,chi1,chi2)
       S2av = S2avinf(theta1inf, theta2inf, q, chi1, chi2)
    else:
        #This is equivalent to S2av, but we avoid multiple conversions J <--> kappa and repated calculation of the S^2 roots.
        S32, Sminus2, Splus2 = np.squeeze(wraproots(Scubic_coefficients,kappa,u,xi,q,chi1,chi2))
        m = elliptic_parameter(Sminus2, Splus2, S32)
        S2av = Splus2 - (Splus2-Sminus2)*S2av_mfactor(m)

    return S2av


# TODO: update docstrings
# TODO: change names to precav_integrator and precav_RHS
# If debug return ODE object
def kappaofu(kappainitial, uinitial, ufinal, xi, q, chi1, chi2):
    """
    Integration of ODE describing precession-averaged inspirals. Returns kappa(u) for a given initial condition kappa, sampled at given outputs u. The initial condition corresponds to the value of kappa at u[0].

    Call
    ----
    kappa = kappaofu(kappa,u,xi,q,chi1,chi2)

    Parameters
    ----------
    kappa: float
    	Regularized angular momentum (J^2-L^2)/(2L).
    u: float
    	Compactified separation 1/(2L).
    xi: float
    	Effective spin.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    kappa: float
    	Regularized angular momentum (J^2-L^2)/(2L).
    """

    kappainitial = np.atleast_1d(kappainitial)
    uinitial = np.atleast_1d(uinitial)
    ufinal = np.atleast_1d(ufinal)

    xi = np.atleast_1d(xi)
    q= np.atleast_1d(q)
    chi1 = np.atleast_1d(chi1)
    chi2 = np.atleast_1d(chi2)

    def _compute(kappainitial, uinitial, ufinal, xi, q, chi1, chi2):

        # h0 controls the first stepsize attempted. If integrating from finite separation, let the solver decide (h0=0). If integrating from infinity, prevent it from being too small.
        # TODO. This breaks down if r is very large but not infinite.
        #h0= 1e-3 if u[0]==0 else 0
        # TODO: I disabled the intial timestep check when I switched to solve_ivp. This needs to be tested!

        # As far as I understand by inspective the scipy code, the "vectorized" option is ignored if a jacobian is not provided. If you decide it's needed, a vectorized implementation of "dkappadu" requires substituting that if statement with np.where

        ODEsolution = scipy.integrate.solve_ivp(dkappadu, (uinitial, ufinal), np.atleast_1d(kappainitial), method='RK45', t_eval=(uinitial, ufinal), dense_output=True, args=(xi,q,chi1,chi2))

        # Also return ODE object. The key methods is .sol --callable, sol(t).
        return ODEsolution

    ODEsolution = np.array(list(map(_compute, kappainitial, uinitial,ufinal, xi, q, chi1, chi2)))

    return ODEsolution



def inspiral_precav(theta1=None,theta2=None,deltaphi=None,S=None,J=None,kappa=None,r=None,u=None,xi=None,q=None,chi1=None,chi2=None,requested_outputs=None):
    """
    Perform precession-averaged inspirals. The variables q, chi1, and chi2 must always be provided. The integration range must be specified using either r or u (and not both). The initial conditions correspond to the binary at either r[0] or u[0]. The vector r or u needs to monotonic increasing or decreasing, allowting to integrate forward and backward in time. In addition, integration can be be done between finite separation, forward from infinite to finite separation, or backward from finite to infinite separation. For infinity, use r=np.inf or u=0.
    The initial conditions must be specified in terms of one an only one of the following:
    - theta1,theta2, and deltaphi (but note that deltaphi is not necessary if integrating from infinite separation).
    - J, xi (only if integrating from finite separations because J otherwise diverges).
    - kappa, xi.
    The desired outputs can be specified with a list e.g. requested_outputs=['theta1','theta2','deltaphi']. All the available variables are returned by default.

    Call
    ----
    outputs = inspiral_precav(theta1=None,theta2=None,deltaphi=None,S=None,J=None,kappa=None,r=None,u=None,xi=None,q=None,chi1=None,chi2=None,requested_outputs=None)

    Parameters
    ----------
    theta1: float, optional (default: None)
    	Angle between orbital angular momentum and primary spin.
    theta2: float, optional (default: None)
    	Angle between orbital angular momentum and secondary spin.
    deltaphi: float, optional (default: None)
    	Angle between the projections of the two spins onto the orbital plane.
    S: float, optional (default: None)
    	Magnitude of the total spin.
    J: float, optional (default: None)
    	Magnitude of the total angular momentum.
    kappa: float, optional (default: None)
    	Regularized angular momentum (J^2-L^2)/(2L).
    r: float, optional (default: None)
    	Binary separation.
    u: float, optional (default: None)
    	Compactified separation 1/(2L).
    xi: float, optional (default: None)
    	Effective spin.
    q: float, optional (default: None)
    	Mass ratio: 0<=q<=1.
    chi1: float, optional (default: None)
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float, optional (default: None)
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    requested_outputs: list, optional (default: None)
    	Set of outputs.

    Returns
    -------
    outputs: dictionary
    	Set of outputs.
    """

    def _compute(theta1,theta2,deltaphi,S,J,kappa,r,u,xi,q,chi1,chi2):

        if q is None:
            raise TypeError("Please provide q.")
        if chi1 is None:
            raise TypeError("Please provide chi1.")
        if chi2 is None:
            raise TypeError("Please provide chi2.")

        if r is not None and u is None:
            r=toarray(r)
            u = eval_u(r, np.repeat(q,flen(r)) )
        elif r is None and u is not None:
            u=toarray(u)
            r = eval_r(u=u, q=np.repeat(q,flen(u)) )
        else:
            raise TypeError("Please provide either r or u. Use np.inf for infinity.")

        assert np.sum(u==0)<=1 and np.sum(u[1:-1]==0)==0, "There can only be one r=np.inf location, either at the beginning or at the end."


        # Start from r=infinity
        if u[0]==0:

            if theta1 is not None and theta2 is not None and S is None and J is None and kappa is None and xi is None:
                kappa, xi = angles_to_asymptotic(theta1,theta2,q,chi1,chi2)
                theta1inf, theta2inf = theta1, theta2

            elif theta1 is None and theta2 is None and deltaphi is None and J is None and kappa is not None and xi is not None:
                theta1inf,theta2inf = asymptotic_to_angles(kappa,xi,q,chi1,chi2)

            else:
                raise TypeError("Integrating from infinite separation. Please provide either (theta1,theta2) or (kappa,xi).")


        # Start from finite separations
        else:

            # User provides theta1,theta2, and deltaphi.
            if theta1 is not None and theta2 is not None and deltaphi is not None and S is None and J is None and kappa is None and xi is None:
                S, J, xi = angles_to_conserved(theta1, theta2, deltaphi, r[0], q, chi1, chi2)
                kappa = eval_kappa(J, r[0], q)

            # User provides J, xi, and maybe S.
            elif theta1 is None and theta2 is None and deltaphi is None and J is not None and kappa is None and xi is not None:
                kappa = eval_kappa(J, r[0], q)

            # User provides kappa, xi, and maybe S.
            elif theta1 is None and theta2 is None and deltaphi is None and J is None and kappa is not None and xi is not None:
                pass

            else:
                TypeError("Integrating from finite separations. Please provide one and not more of the following: (theta1,theta2,deltaphi), (J,xi), (S,J,xi), (kappa,xi), (S,kappa,xi).")

        # Integration. Return interpolant along the solution
        ODEsolution = kappaofu(kappa, u[0],u[-1], xi, q, chi1, chi2)
        # Evaluate the interpolant at the requested values of u
        kappa=np.squeeze(ODEsolution.item().sol(u))

        # Select finite separations
        rok = r[u!=0]
        kappaok = kappa[u!=0]

        # Resample S and assign random sign to deltaphi
        J = eval_J(kappa=kappaok,r=rok,q=np.repeat(q,flen(rok)))
        S = Ssampling(J, rok, np.repeat(xi,flen(rok)), np.repeat(q,flen(rok)),
        np.repeat(chi1,flen(rok)), np.repeat(chi2,flen(rok)), N=1)
        theta1,theta2,deltaphi = conserved_to_angles(S, J, rok, xi, np.repeat(q,flen(rok)), np.repeat(chi1,flen(rok)), np.repeat(chi2,flen(rok)))
        deltaphi = deltaphi * np.random.choice([-1,1],flen(deltaphi))

        # Integrating from infinite separation.
        if u[0]==0:
            J = np.concatenate(([np.inf],J))
            S = np.concatenate(([np.nan],S))
            theta1 = np.concatenate(([theta1inf],theta1))
            theta2 = np.concatenate(([theta2inf],theta2))
            deltaphi = np.concatenate(([np.nan],deltaphi))
        # Integrating backwards to infinity
        elif u[-1]==0:
            J = np.concatenate((J,[np.inf]))
            S = np.concatenate((S,[np.nan]))
            theta1inf,theta2inf = asymptotic_to_angles(kappa[-1],xi,q,chi1,chi2)
            theta1 = np.concatenate((theta1,[theta1inf]))
            theta2 = np.concatenate((theta2,[theta2inf]))
            deltaphi = np.concatenate((deltaphi,[np.nan]))
        else:
            pass

        return toarray(theta1,theta2,deltaphi,S,J,kappa,r,u,xi,q,chi1,chi2)

    #This array has to match the outputs of _compute (in the right order!)
    alloutputs = np.array(['theta1','theta2','deltaphi','S','J','kappa','r','u','xi','q','chi1','chi2'])

    # allresults is an array of dtype=object because different variables have different shapes
    if flen(q)==1:
        allresults =_compute(theta1,theta2,deltaphi,S,J,kappa,r,u,xi,q,chi1,chi2)
    else:
        inputs = np.array([theta1,theta2,deltaphi,S,J,kappa,r,u,xi,q,chi1,chi2])
        for k,v in enumerate(inputs):
            if v==None:
                inputs[k] = itertools.repeat(None) #TODO: this could be np.repeat(None,flen(q)) if you need to get rid of the itertools dependence

        theta1,theta2,deltaphi,S,J,kappa,r,u,xi,q,chi1,chi2= inputs
        allresults = np.array(list(map(_compute, theta1,theta2,deltaphi,S,J,kappa,r,u,xi,q,chi1,chi2))).T

    # Handle the outputs.
    # Return all
    if requested_outputs is None:
        requested_outputs = alloutputs
    # Return only those requested (in1d return boolean array)
    wantoutputs = np.in1d(alloutputs,requested_outputs)

    # Store into a dictionary
    outcome={}

    for k,v in zip(alloutputs[wantoutputs],np.array(allresults)[wantoutputs]):
        # np.stack fixed shapes and object types
        outcome[k]=np.stack(np.atleast_1d(v))

    return outcome




#TODO: does this work on arrays?
# TODO: docstrings
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

    Sminus2, Splus2, S32 = S2roots(J, r, xi, q, chi1, chi2)
    a = Speriod_prefactor(r, xi ,q)

    def _integrand(Ssq):
        return func(Ssq, *args, **kwargs) / np.abs(dS2dt(Ssq, Sminus2, Splus2, S32, a))

    tau = Speriod(J, r, xi, q, chi1, chi2)
    func_av = (2/tau) * scipy.integrate.quad(_integrand, Sminus2, Splus2)[0]

    return func_av


def rupdown(q, chi1, chi2):
    """
    The critical separations r_ud+/- marking the region of the up-down precessional instability.

    Call
    ----
    rudp,rudm = rupdown(q,chi1,chi2)

    Parameters
    ----------
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    rudp: float
    	Outer orbital separation in the up-down instability.
    rudm: float
    	Inner orbital separation in the up-down instability.
    """

    q=np.atleast_1d(q)
    chi1=np.atleast_1d(chi1)
    chi2=np.atleast_1d(chi2)

    q, chi1, chi2 = toarray(q, chi1, chi2)
    rudp = (chi1**0.5+(q*chi2)**0.5)**4/(1-q)**2
    rudm = (chi1**0.5-(q*chi2)**0.5)**4/(1-q)**2

    return np.stack([rudp, rudm])


def omegasq_aligned(r, q, chi1, chi2, which):
    """
    Squared oscillation frequency of a given perturbed aligned-spin binary. The flag which needs to be set to `uu` for up-up, `ud` for up-down, `du` for down-up or `dd` for down-down where the term before (after) the hyphen refers to the spin of the heavier (lighter) black hole.

    Call
    ----
    omegasq = omegasq_aligned(r,q,chi1,chi2,which)

    Parameters
    ----------
    r: float
    	Binary separation.
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    which: string
    	Select function behavior.

    Returns
    -------
    omegasq: float
    	Squared frequency.
    """

    q=np.atleast_1d(q)

    # These are all the valid input flags
    uulabels=np.array(['uu','up-up','upup','++'])
    udlabels=np.array(['ud','up-down','updown','+-'])
    dulabels=np.array(['du','down-up','downup','-+'])
    ddlabels=np.array(['dd','down-down','downdown','--'])

    assert np.isin(which,np.concatenate([uulabels,udlabels,dulabels,ddlabels])).all(), "Set `which` flag to either uu, ud, du, or dd."

    #+1 if primary is co-aligned, -1 if primary is counter-aligned
    alpha1 = np.where(np.isin(which,np.concatenate([uulabels,udlabels])), 1,-1)
    #+1 if secondary is co-aligned, -1 if secondary is counter-aligned
    alpha2 = np.where(np.isin(which,np.concatenate([uulabels,dulabels])), 1,-1)

    L = eval_L(r, q)
    S1, S2 = spinmags(q, chi1, chi2)
    # Slightly rewritten from Eq. 18 in arXiv:2003.02281, regularized for q=1
    omegasq = ( 3 * q**5 / ( 2 * ( 1 + q )**11 * L**7 ) )**2 * ( L - ( q *     \
        alpha1 * S1 + alpha2 * S2 ) / ( 1 + q ) )**2 * ( L**2 * ( 1 - q )**2 - \
        2 * L * ( q * alpha1 * S1 - alpha2 * S2 ) * ( 1 - q ) + ( q * alpha1 * \
        S1 + alpha2 * S2 )**2 )

    return omegasq


def widenutation(q, chi1, chi2):
    """
    The critical separation r_wide below which the binary component with
    smaller dimensionless spin may undergo wide nutations.

    Call
    ----
    r_wide = widenutation(q,chi1,chi2)

    Parameters
    ----------
    q: float
    	Mass ratio: 0<=q<=1.
    chi1: float
    	Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1.
    chi2: float
    	Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    r_wide: float
    	Orbital separation where wide nutations becomes possible.
    """

    q=np.atleast_1d(q)
    chi1=np.atleast_1d(chi1)
    chi2=np.atleast_1d(chi2)

    rwide = ((q*chi2 - chi1) / (1-q))**2

    return rwide


#### Orbit averaged things ####

# TODO: this comes straight from precession_V1. Update docstrings. It's not necesssary that this function works on arrays
# TODO: replace quadrupole_formula flag with parameter to select a given PN order
def orbav_RHS(v,allvars,q,m1,m2,eta,chi1,chi2,S1,S2,tracktime=False,quadrupole_formula=False):

    '''
    Right-hand side of the orbit-averaged PN equations: d[allvars]/dv=RHS, where
    allvars is an array with the cartesian components of the unit vectors L, S1
    and S2. This function is only the actual system of equations, not the ODE
    solver.

    Equations are the ones reported in Gerosa et al. [Phys.Rev. D87 (2013) 10,
    104028](http://journals.aps.org/prd/abstract/10.1103/PhysRevD.87.104028);
    see references therein. In particular, the quadrupole-monopole term computed
    by Racine is included. The results presented in Gerosa et al. 2013 actually
    use additional unpublished terms, that are not listed in the published
    equations and are NOT included here. Radiation reaction is included up to
    3.5PN.

    The internal quadrupole_formula flag switches off all PN corrections in
    radiation reaction.

    The integration is carried over in the orbital velocity v (equivalent to the
    separation), not in time. If an expression for v(t) is needed, the code can
    be easiliy modified to return time as well.

    **Call:**

        allders=precession.orbav_RHS(allvars,v,q,S1,S2,eta,m1,m2,chi1,chi2,time=False)

    **Parameters:**
    - `allvars`: array of lenght 9 cointaining the initial condition for numerical integration for the components of the unit vectors L, S1 and S2.
    - `v`: orbital velocity.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `eta`: symmetric mass ratio.
    - `m1`: mass of the primary BH.
    - `m2`: mass of the secondary BH.
    - `chi1`: dimensionless spin magnitude of the primary BH. Must be 0<=chi1<=1
    - `chi2`: dimensionless spin magnitude of the secondary BH. Must be 0<=chi2<=1
    - `time`: if `True` also integrate t(r).

    **Returns:**

    - `allders`: array of lenght 9 cointaining the derivatives of allvars with respect to the orbital velocity v.
    '''

    # Unpack inputs
    Lh = allvars[0:3]
    S1h = allvars[3:6]
    S2h = allvars[6:9]
    if tracktime:
        t = allvars[9]

    # Angles
    ct1 = np.dot(S1h,Lh)
    ct2 = np.dot(S2h,Lh)
    ct12 = np.dot(S1h,S2h)

    # Spin precession for S1
    Omega1= eta*v**5*(2+3*q/2)*Lh  \
            + v**6*(S2*S2h-3*S2*ct2*Lh-3*q*S1*ct1*Lh)/2
    dS1hdt = np.cross(Omega1,S1h)

    # Spin precession for S2
    Omega2= eta*v**5*(2+3/(2*q))*Lh  \
            + v**6*(S1*S1h-3*S1*ct1*Lh-3*S2*ct2*Lh/q)/2
    dS2hdt = np.cross(Omega2,S2h)

    # Conservation of angular momentum
    dLhdt= -v*(S1*dS1hdt+S2*dS2hdt)/eta

    # Radiation reaction
    if quadrupole_formula: # Use to switch off higher-order terms
        dvdt= (32*eta*v**9/5)
    else:
        dvdt= (32*eta*v**9/5)* ( 1                                  \
            - v**2* (743+924*eta)/336                               \
            + v**3* (4*np.pi                                        \
                     - chi1*ct1*(113*m1**2/12 + 25*eta/4 )   \
                     - chi2*ct2*(113*m2**2/12 + 25*eta/4 ))  \
            + v**4* (34103/18144 + 13661*eta/2016 + 59*eta**2/18    \
                     + eta*chi1*chi2* (721*ct1*ct2 - 247*ct12) /48  \
                     + ((m1*chi1)**2 * (719*ct1**2-233))/96       \
                     + ((m2*chi2)**2 * (719*ct2**2-233))/96)      \
            - v**5* np.pi*(4159+15876*eta)/672                      \
            + v**6* (16447322263/139708800 + 16*np.pi**2/3          \
                     -1712*(0.5772156649+np.log(4*v))/105           \
                     +(451*np.pi**2/48 - 56198689/217728)*eta       \
                     +541*eta**2/896 - 5605*eta**3/2592)            \
            + v**7* np.pi*( -4415/4032 + 358675*eta/6048            \
                     + 91495*eta**2/1512)                           \
            )

    # Integrate in v, not in time
    dtdv=1./dvdt
    dLhdv=dLhdt*dtdv
    dS1hdv=dS1hdt*dtdv
    dS2hdv=dS2hdt*dtdv

    # Pack outputs
    if tracktime:
        return np.concatenate([dLhdv,dS1hdv,dS2hdv,[dtdv]])
    else:
        return np.concatenate([dLhdv,dS1hdv,dS2hdv])

# TODO: this comes straight from precession_V1. Update docstrings
def orbav_integrator(Lhinitial,S1hinitial,S2hinitial,rinitial,rfinal,q,chi1,chi2,tracktime=False,quadrupole_formula=False,rsteps = None):

    '''
    Single orbit-averaged integration. Integrate the system of ODEs specified in
    `precession.orbav_RHS`. The initial configuration (at r_vals[0]) is
    specified through J, xi and S. The components of the unit vectors L, S1 and
    S2 are returned at the output separations specified by r_vals. The initial
    values of J and S must be compatible with the initial separation r_vals[0],
    otherwise an error is raised. Integration is performed in a reference frame
    in which the z axis is along J and L lies in the x-z plane at the initial
    separation. Equations are integrated in v (orbital velocity) but outputs are
    converted to r (separation).

    Of course, this can only integrate to/from FINITE separations.

    Bear in mind that orbit-averaged integrations are tpically possible from
    r<10000; integrations from larger separations take a very long time and can
    occasionally crash. If q=1, the initial binary configuration is specified
    through cos(varphi), not S.

    We recommend to use one of the wrappers `precession.orbit_averaged` and
    `precession.orbit_angles` provided.

    **Call:**
        Lhx_fvals,Lhy_fvals,Lhz_fvals,S1hx_fvals,S1hy_fvals,S1hz_fvals,S2hx_fvals,S2hy_fvals,S2hz_fvals=precession.orbav_integrator(J,xi,S,r_vals,q,S1,S2,time=False)

    **Parameters:**

    - `J`: magnitude of the total angular momentum.
    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `S`: magnitude of the total spin.
    - `r_vals`: binary separation (array).
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `time`: if `True` also integrate t(r).

    **Returns:**

    - `Lhx_vals`: x component of the unit vector L/|L|.
    - `Lhy_vals`: y component of the unit vector L/|L|.
    - `Lhz_vals`: z component of the unit vector L/|L|.
    - `S1hx_vals`: x component of the unit vector S1/|S1|.
    - `S1hy_vals`: y component of the unit vector S1/|S1|.
    - `S1hz_vals`: z component of the unit vector S1/|S1|.
    - `S2hx_vals`: x component of the unit vector S2/|S2|.
    - `S2hy_vals`: y component of the unit vector S2/|S2|.
    - `S2hz_vals`: z component of the unit vector S2/|S2|.
    - `t_fvals`: (optional) time as a function of the separation.
    '''

    Lhinitial=np.atleast_2d(Lhinitial)
    S1hinitial=np.atleast_2d(S1hinitial)
    S2hinitial=np.atleast_2d(S2hinitial)
    rinitial= np.atleast_1d(rinitial)
    rfinal=np.atleast_1d(rfinal)
    q= np.atleast_1d(q)

    q= np.atleast_1d(q)
    chi1 = np.atleast_1d(chi1)
    chi2 = np.atleast_1d(chi2)

    if rsteps is None:
        rsteps = np.stack([rinitial,rfinal],axis=1)
    else:
        rsteps=np.atleast_2d(rsteps)


    def _compute(Lhinitial,S1hinitial,S2hinitial,rinitial,rfinal,rsteps,q,chi1,chi2):

        # I need unit vectors
        assert np.isclose(np.linalg.norm(Lh0),1)
        assert np.isclose(np.linalg.norm(S1h0),1)
        assert np.isclose(np.linalg.norm(S2h0),1)

        # Pack inputs
        if tracktime:
            ic = np.concatenate([Lh0,S1h0,S2h0,[0]])
        else:
            ic = np.concatenate([Lh0,S1h0,S2h0])

        # Compute these quantities here instead of inside the RHS for speed
        vinitial=eval_v(rinitial)
        vfinal=eval_v(rfinal)
        vsteps=eval_v(rsteps)
        m1=eval_m1(q)
        m2=eval_m2(q)
        S1,S2 = spinmags(q,chi1,chi2)
        eta=eval_eta(q)

        # Integration
        #t0=time.time()
        #res =scipy.integrate.odeint(orbav_RHS, ic, v, args=(q,m1,m2,eta,chi1,chi2,S1,S2,tracktime,quadrupole_formula), mxstep=5000000, full_output=0, printmessg=0,rtol=1e-12,atol=1e-12)
        #print(time.time()-t0)
        print(q,m1,m2,eta,chi1,chi2,S1,S2,tracktime,quadrupole_formula)

        ODEsolution = scipy.integrate.solve_ivp(orbav_RHS, (vinitial, vfinal), ic, method='RK45', t_eval=vsteps, dense_output=True, args=(q,m1,m2,eta,chi1,chi2,S1,S2,tracktime,quadrupole_formula))

        # Returned output is
        # Lx, Ly, Lz, S1x, S1y, S1z, S2x, S2y, S2z, (t)
        evaluations = np.squeeze(ODEsolution.y).T

        # Also return ODE object. The key methods is .sol --callable, sol(t).
        return evaluations, ODEsolution



    evaluations, ODEsolution  = np.squeeze(list(map(_compute, Lhinitial,S1hinitial,S2hinitial,rinitial,rfinal,rsteps,q,chi1,chi2))).T

    print(evaluations.shape)

    Lh = np.squeeze(np.swapaxes(evaluations[:,0:3],1,2))
    S1h = np.squeeze(np.swapaxes(evaluations[:,3:6],1,2))
    S2h = np.squeeze(np.swapaxes(evaluations[:,6:9],1,2))

    if tracktime:
        t = np.squeeze(evaluations[:,9])
        return toarray(Lh,S1h,S2h,t)
    else:
        return toarray(Lh,S1h,S2h)


def inspiral_orbav(theta1=None,theta2=None,deltaphi=None,S=None,Lh=None,S1h=None,S2h=None, J=None,kappa=None,r=None,u=None,xi=None,q=None,chi1=None,chi2=None,tracktime=False,quadrupole_formula=False,requested_outputs=None):
    '''
    TODO: docstrings. Orbit average evolution; this is the function the user should call (I think)
    '''

    # Overwrite the tracktime flag if the user explicitely asked for the time output
    try:
        if 't' in requested_outputs:
            tracktime=True
    except:
        pass

    def _compute(theta1,theta2,deltaphi,S,Lh,S1h,S2h,J,kappa,r,u,xi,q,chi1,chi2):

        if q is None:
            raise TypeError("Please provide q.")
        if chi1 is None:
            raise TypeError("Please provide chi1.")
        if chi2 is None:
            raise TypeError("Please provide chi2.")

        if r is not None and u is None:
            r=toarray(r)
            u = eval_u(r, np.repeat(q,flen(r)) )
        elif r is None and u is not None:
            u=toarray(u)
            r = eval_r(u=u, q=np.repeat(q,flen(u)) )
        else:
            raise TypeError("Please provide either r or u.")


        # User provides Lh, S1h, and S2h
        if Lh is not None and S1h is not None and S2h is not None and theta1 is None and theta2 is None and deltaphi is None and S is None and J is None and kappa is None and xi is None:
            pass

        # User provides theta1,theta2, and deltaphi.
        elif Lh is None and S1h is None and S2h is None and theta1 is not None and theta2 is not None and deltaphi is not None and S is None and J is None and kappa is None and xi is None:
            Lh, S1h, S2h = angles_to_Jframe(theta1, theta2, deltaphi, r[0], q, chi1, chi2)

        # User provides J, xi, and S.
        elif Lh is None and S1h is None and S2h is None and theta1 is None and theta2 is None and deltaphi is None and S is not None and J is not None and kappa is None and xi is not None:
            Lh, S1h, S2h = conserved_to_Jframe(S, J, r[0], xi, q, chi1, chi2)

        # User provides kappa, xi, and maybe S.
        elif Lh is None and S1h is None and S2h is None and theta1 is None and theta2 is None and deltaphi is None and S is not None and J is None and kappa is not None and xi is not None:
            J = eval_J(kappa=kappa,r=r[0],q=q)
            Lh, S1h, S2h = conserved_to_Jframe(S, J, r[0], xi, q, chi1, chi2)

        else:
            TypeError("Please provide one and not more of the following: (Lh,S1h,S2h), (theta1,theta2,deltaphi), (S,J,xi), (S,kappa,xi).")

        # Make sure vectors are normalized
        Lh = normalize_nested(Lh)
        S1h = normalize_nested(S1h)
        S2h = normalize_nested(S2h)

        # Integration
        outcome = orbav_integrator(Lh,S1h,S2h,r,q,chi1,chi2,tracktime=tracktime,quadrupole_formula=quadrupole_formula)
        Lh,S1h,S2h = outcome[0:3]
        if tracktime:
            t=outcome[3]
        else:
            t=None

        S1,S2= spinmags(q,chi1,chi2)
        L = eval_L(r,np.repeat(q,flen(r)))
        Lvec= (L*Lh.T).T
        S1vec= S1*S1h
        S2vec= S2*S2h

        theta1, theta2, deltaphi = vectors_to_angles(Lvec, S1vec, S2vec)
        S, J, xi = vectors_to_conserved(Lvec, S1vec, S2vec, q)
        kappa = eval_kappa(J, r, q)

        return toarray(t,theta1,theta2,deltaphi,S,Lh,S1h,S2h,J,kappa,r,u,xi,q,chi1,chi2)

    #This array has to match the outputs of _compute (in the right order!)
    alloutputs = np.array(['t','theta1','theta2','deltaphi','S','Lh','S1h','S2h','J','kappa','r','u','xi','q','chi1','chi2'])

    # allresults is an array of dtype=object because different variables have different shapes
    if flen(q)==1:
        allresults =_compute(theta1,theta2,deltaphi,S,Lh,S1h,S2h,J,kappa,r,u,xi,q,chi1,chi2)
    else:
        inputs = np.array([theta1,theta2,deltaphi,S,Lh,S1h,S2h,J,kappa,r,u,xi,q,chi1,chi2])
        for k,v in enumerate(inputs):
            if v==None:
                inputs[k] = itertools.repeat(None) #TODO: this could be np.repeat(None,flen(q)) if you need to get rid of the itertools dependence

        theta1,theta2,deltaphi,S,Lh,S1h,S2h,J,kappa,r,u,xi,q,chi1,chi2= inputs
        allresults = np.array(list(map(_compute, theta1,theta2,deltaphi,S,Lh,S1h,S2h,J,kappa,r,u,xi,q,chi1,chi2))).T

    # Handle the outputs.
    # Return all
    if requested_outputs is None:
        requested_outputs = alloutputs
    # Return only those requested (in1d return boolean array)
    wantoutputs = np.in1d(alloutputs,requested_outputs)

    # Store into a dictionary
    outcome={}
    for k,v in zip(alloutputs[wantoutputs],allresults[wantoutputs]):
        if not tracktime and k=='t':
            continue
        # np.stack fixed shapes and object types
        outcome[k]=np.stack(np.atleast_1d(v))

    return outcome


def inspiral(*args, which=None,**kwargs):
    '''
    TODO write docstings. This is the ultimate wrapper the user should call.
    '''

    # Precession-averaged integrations
    if which in ['precession','precav','precessionaveraged','precessionaverage','precession-averaged','precession-average']:
        return inspiral_precav(*args, **kwargs)

    elif which in ['orbit','orbav','orbitaveraged','orbitaverage','orbit-averaged','orbit-average']:
        return inspiral_orbav(*args, **kwargs)

    else:
        raise ValueError("kind need to be either 'precav' or 'orbav'.")


if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)

    #print(eval_r(u=1, L=None, q=1))

    #print(spinmags([0.5,0.5],[1,1],[1,1]))
    #print(spinmags(0.5,1,1))
    #print(eval_S1(0.5,1))

    #print(eval_S2([0.5,0.5],[1,1]))

    #print(masses([0.5,0.6]))

    #
    # r=[10,10]
    # xi=[0.35,0.35]
    # q=[0.8,0.8]
    # chi1=[1,1]
    # chi2=[1,1]
    # J=[1,1]
    # u=[1/10,1/10]
    # theta1=[1,1]
    # theta2=[1,1]
    # S=[0.3,0.3]
    # t=[1,100]
    #
    # print(omegasq_aligned(r, q, chi1, chi2, ['uu','ud']))

    # print("on many", Jresonances(r,xi,q,chi1,chi2))
    #
    # print("on one", Jresonances(r[0],xi[0],q[0],chi1[0],chi2[0]))
    #
    #
    # sys.exit()
    # #
    #
    # for x in np.linspace()
    #
    # print(S2av_mfactor([0,1e-,0.2]))

    #print(morphology(J,r,xi,q,chi1,chi2,simpler=False))
    #print(morphology(J[0],r[0],xi[0],q[0],chi1[0],chi2[0],simpler=True))

    # print(Soft(t[0],J[0],r[0],xi[0],q[0],chi1[0],chi2[0]))
    # print(Soft(t[1],J[0],r[0],xi[0],q[0],chi1[0],chi2[0]))
    # print(Soft(t[1],J[1],r[1],xi[1],q[1],chi1[1],chi2[1]))
    #
    # print(Soft(t,J,r,xi,q,chi1,chi2))
    #
    # print(Soft(t,J[0],r[0],xi[0],q[0],chi1[0],chi2[0]))
    #
    #
    # print(Soft([[1,100,1,100],[500,600,500,600]],J,r,xi,q,chi1,chi2))

    #print(Ssampling(J,r,xi,q,chi1,chi2,N=10).shape)
    #print(Ssampling(J,r,xi,q,chi1,chi2,N=1).shape)
    #print(Ssampling(J[0],r[0],xi[0],q[0],chi1[0],chi2[0],N=1).shape)
    #print(Ssampling(J[0],r[0],xi[0],q[0],chi1[0],chi2[0],N=10).shape)



    #Lvec = [[1,2454,3],[1,2,334]]
    #S1vec = [[13,20,30],[1,21,3]]
    #S2vec = [[12,23,33],[1,23,3]]

    #v1,v2,v3 = conserved_to_Jframe(S[1], J[1], r[1], xi[1], q[1], chi1[1], chi2[1])
    #print(v1)

    #v1,v2,v3 = conserved_to_Jframe(S, J, r, xi, q, chi1, chi2)
    #print(v1)


    #print(kappadiscriminant_coefficients(u,xi,q,chi1,chi2))
    #print(kappadiscriminant_coefficients(0.1,0.2,0.8,1,1))
    #print("on one", Jresonances(r[0],xi[0],q[0],chi1[0],chi2[0]))
    #print(Jresonances(r[1],xi[1],q[1],chi1[1],chi2[1]))
    #print("on many", Jresonances(r,xi,q,chi1,chi2))

    #print("on one", xiresonances(J[0],r[0],q[0],chi1[0],chi2[0]) )

    #print("on many", xiresonances(J,r,q,chi1,chi2) )

    #print(anglesresonances(J=J[0],r=r[0],xi=None,q=q[0],chi1=chi1[0],chi2=chi2[0]))

    #print(anglesresonances(J=J,r=r,xi=None,q=q,chi1=chi1,chi2=chi2))
    #print(Slimits(J,r,xi,q,chi1,chi2))
    #print(Slimits(J[0],r[0],xi[0],q[0],chi1[0],[chi2[0]]))

    #print(xilimits(J=J, r=r,q=q,chi1=chi1,chi2=chi2))
    #print(eval_xi(theta1=theta1,theta2=theta2,S=[1,1],varphi=[1,1],J=J,r=r,q=q,chi1=chi1,chi2=chi2))
    #print(effectivepotential_minus(S[0],J[0],r[0],q[0],chi1[0],chi2[0]))

    #print(effectivepotential_minus(S,J,r,q,chi1,chi2))
    #print(Slimits_plusminus(J,r,xi,q,chi1,chi2))
    #t0=time.time()
    #print(Jofr(ic=1.8, r=np.linspace(100,10,100), xi=-0.5, q=0.4, chi1=0.9, chi2=0.8))
    #print(time.time()-t0)

    # t0=time.time()
    #print(repr(Jofr(ic=203.7430728810311, r=np.logspace(6,1,100), xi=-0.5, q=0.4, chi1=0.9, chi2=0.8)))
    # print(time.time()-t0)



    # theta1inf=0.5
    # theta2inf=0.5
    # q=0.5
    # chi1=0.6
    # chi2=0.7
    # kappainf, xi = angles_to_asymptotic(theta1inf,theta2inf,q,chi1,chi2)
    # r = np.concatenate(([np.inf],np.logspace(10,1,100)))
    # print(repr(Jofr(kappainf, r, xi, q, chi1, chi2)))


    # r=1e2
    # xi=-0.5
    # q=0.4
    # chi1=0.9
    # chi2=0.8
    #
    # Jmin,Jmax = Jlimits(r=r,xi=xi,q=q,chi1=chi1,chi2=chi2)
    # J0=(Jmin+Jmax)/2
    # #print(J)
    # #print(Jmin,Jmax)
    # r = np.logspace(np.log10(r),1,100)
    # J=Jofr(J0, r, xi, q, chi1, chi2)
    # print(J)
    #
    # J=Jofr([J0,J0], [r,r], [xi,xi], [q,q], [chi1,chi1], [chi2,chi2])
    #
    # print(J)



    #S = Ssampling(J,r,xi,q,chi1,chi2,N=1)

    #S = Ssampling([J,J],[r,r],[xi,xi],[q,q],[chi1,chi1],[chi2,chi2],N=[10,10])

    #print(repr(S))

    ##### INSPIRAL TESTING: precav, to/from finite #######
    # q=0.5
    # chi1=1
    # chi2=1
    # theta1=0.4
    # theta2=0.45
    # deltaphi=0.46
    # S = 0.5538768649231461
    # J = 2.740273008918153
    # xi = 0.9141896967861489
    # kappa = 0.5784355256550922
    # r=np.logspace(2,1,6)
    # #d=inspiral_precav(theta1=theta1,theta2=theta2,deltaphi=deltaphi,q=q,chi1=chi1,chi2=chi2,r=r,requested_outputs=None)
    #
    #
    # #
    # #
    # # print(d['theta1'])
    # #
    # d=inspiral_precav(theta1=[theta1,theta1],theta2=[theta2,theta2],deltaphi=[deltaphi,deltaphi],q=[q,q],chi1=[chi1,chi1],chi2=[chi2,chi2],r=[r,r],requested_outputs=None)
    #
    #
    # #print(d)
    #
    # #
    # print(d['theta1'])
    #
    #
    # sys.exit()
    #
    # d=inspiral(which='precav',theta1=theta1,theta2=theta2,deltaphi=deltaphi,q=q,chi1=chi1,chi2=chi2,r=r,outputs=['J'])
    #
    # print(d)
    #
    # d=inspiral_orbav(theta1=theta1,theta2=theta2,deltaphi=deltaphi,q=q,chi1=chi1,chi2=chi2,r=r,outputs=['J'])
    # print(d)
    #
    # d=inspiral(which='orbav',theta1=theta1,theta2=theta2,deltaphi=deltaphi,q=q,chi1=chi1,chi2=chi2,r=r,outputs=['J'])
    #
    # print(d)

    #print('')

    # d=inspiral_precav(theta1=[theta1,theta1],theta2=[theta2,theta2],deltaphi=[deltaphi,deltaphi],q=[q,q],chi1=[chi1,chi1],chi2=[chi2,chi2],r=[r,r])
    #
    # # #d=inspiral_precav(theta1=theta1,theta2=theta2,deltaphi=deltaphi,q=q,chi1=chi1,chi2=chi2,r=r,outputs=['r','theta1'])
    # #
    # # #d=inspiral_precav(S=S,J=J,xi=xi,q=q,chi1=chi1,chi2=chi2,r=r)
    # # #d=inspiral_precav(J=J,xi=xi,q=q,chi1=chi1,chi2=chi2,r=r)
    # # #d=inspiral_precav(S=S,kappa=kappa,xi=xi,q=q,chi1=chi1,chi2=chi2,r=r)
    # # #d=inspiral_precav(kappa=kappa,xi=xi,q=q,chi1=chi1,chi2=chi2,r=r)
    # #
    # print(d)

    ###### INSPIRAL TESTING: precav, from infinite #######
    # q=0.5
    # chi1=1
    # chi2=1
    # theta1=0.4
    # theta2=0.45
    # kappa = 0.50941012
    # xi = 0.9141896967861489
    # r=np.concatenate(([np.inf],np.logspace(2,1,6)))
    #


    #d=inspiral_precav(theta1=theta1,theta2=theta2,q=q,chi1=chi1,chi2=chi2,r=r)
    # d=inspiral_precav(kappa=kappa,xi=xi,q=q,chi1=chi1,chi2=chi2,r=r,outputs=['J','theta1'])
    #
    # print(d)
    #
    # d=inspiral_precav(kappa=[kappa,kappa],xi=[xi,xi],q=[q,q],chi1=[chi1,chi1],chi2=[chi2,chi2],r=[r,r],outputs=['J','theta1'])
    #
    # print(d)
    # ###### INSPIRAL TESTING #######
    # q=0.5
    # chi1=1
    # chi2=1
    # theta1=0.4
    # theta2=0.45
    # deltaphi=0.46
    # S = 0.5538768649231461
    # J = 1.2314871608018418
    # xi = 0.9141896967861489
    # kappa=0.7276876186801603
    #
    # #kappa = 0.5784355256550922
    # r=np.concatenate((np.logspace(1,4,6),[np.inf]))
    # d=inspiral_precav(theta1=theta1,theta2=theta2,deltaphi=deltaphi,q=q,chi1=chi1,chi2=chi2,r=r)
    #sys.exit()
    # #d=inspiral_precav(S=S,J=J,xi=xi,q=q,chi1=chi1,chi2=chi2,r=r)
    # #d=inspiral_precav(J=J,xi=xi,q=q,chi1=chi1,chi2=chi2,r=r)
    # #d=inspiral_precav(S=S,kappa=kappa,xi=xi,q=q,chi1=chi1,chi2=chi2,r=r)
    # d=inspiral_precav(kappa=kappa,xi=xi,q=q,chi1=chi1,chi2=chi2,r=r)
    #
    # print(d)
    #

    # q=0.5
    # chi1=1
    # chi2=1
    # theta1=0.4
    # theta2=0.45
    # deltaphi=0.46
    # S = 0.5538768649231461
    # J = 2.740273008918153
    # xi = 0.9141896967861489
    # kappa0 = 0.5784355256550922
    # r=np.logspace(2,1,3)
    # u=eval_u(r,q)
    # #print(kappaofu(kappa0, u[0],u[-1], xi, q, chi1, chi2))
    # sols = kappaofu([kappa0,kappa0], [u[0],u[0]], [u[-1],u[-1]], [xi,xi], [q,q], [chi1,chi1], [chi2,chi2])
    # print(sols)
    #print(sols[0])

    #ode_kappaofu(kappa0, uinitial, ufinal, xi, q, chi1, chi2)
    #
    # xi=-0.5
    # q=0.4
    # chi1=0.9
    # chi2=0.8
    # r=np.logspace(2,1,100)
    # Jmin,Jmax = Jlimits(r=r[0],xi=xi,q=q,chi1=chi1,chi2=chi2)
    # J=(Jmin+Jmax)/2
    # Smin,Smax= Slimits(J=J,r=r[0],xi=xi,q=q,chi1=chi1,chi2=chi2)
    # S=(Smin+Smax)/2
    # Svec, S1vec, S2vec, Jvec, Lvec = conserved_to_Jframe(S, J, r[0], xi, q, chi1, chi2)
    # S1h0=S1vec/eval_S1(q,chi1)
    # S2h0=S2vec/eval_S2(q,chi2)
    # Lh0=Lvec/eval_L(r[0],q)
    #
    # print(J,S)

    xi=-0.5
    q=0.4
    chi1=0.9
    chi2=0.8
    r=np.logspace(2,1,5)
    Lh0,S1h0,S2h0 = sample_unitsphere(3)
    #print(Lh0,S1h0,S2h0)
    #t0=time.time()
    #Lh,S1h,S2h = orbav_integrator(Lh0,S1h0,S2h0,r[0],r[-1],q,chi1,chi2,rsteps=r, tracktime=False)

    Lh,S1h,S2h = orbav_integrator([Lh0,Lh0],[S1h0,S1h0],[S2h0,S2h0],[r[0],r[0]],[r[-1],r[-1]],[q,q],[chi1,chi1],[chi2,chi2],rsteps=[r,r], tracktime=False)
    #
    #
    # print(Lh)
    #
    # print( [orbav_integrator([Lh0,Lh0],[S1h0,S1h0],[S2h0,S2h0],[r,r],[q,q],[chi1,chi1],[chi2,chi2],tracktime=True)])
    # #
    # print(t)
    #
    # print(time.time()-t0)
    # #print(Lh)

    # ### ORBAV TESTING ####
    # xi=-0.5
    # q=0.4
    # chi1=0.9
    # chi2=0.8
    # r=np.logspace(2,1,5)
    # Lh,S1h,S2h = sample_unitsphere(3)
    #
    # d= inspiral_orbav(Lh=Lh,S1h=S1h,S2h=S2h,r=r,q=q,chi1=chi1,chi2=chi2,tracktime=True)
    # print(d)
    # print(" ")
    #
    # theta1,theta2,deltaphi = vectors_to_angles(Lh,S1h,S2h)
    # d= inspiral_orbav(theta1=theta1,theta2=theta2,deltaphi=deltaphi,r=r,q=q,chi1=chi1,chi2=chi2)
    # print(d)
    # print(" ")
    #
    # S,J,xi = angles_to_conserved(theta1,theta2,deltaphi,r[0],q,chi1,chi2)
    # d= inspiral_orbav(S=S,J=J,xi=xi,r=r,q=q,chi1=chi1,chi2=chi2)
    # print(d)
    # print(" ")
    #
    #
    # kappa=eval_kappa(J,r[0],q)
    # d= inspiral_orbav(S=S,kappa=kappa,xi=xi,r=r,q=q,chi1=chi1,chi2=chi2)
    # print(d)
    # print(" ")
    #
    # d= inspiral_orbav(Lh=Lh,S1h=S1h,S2h=S2h,r=r,q=q,chi1=chi1,chi2=chi2)
    # print(d)
    # print(" ")
    #
    #
    # d= inspiral_orbav(Lh=[Lh,Lh],S1h=[S1h,S1h],S2h=[S2h,S2h],r=[r,r],q=[q,q],chi1=[chi1,chi1],chi2=[chi2,chi2],tracktime=True)
    # print(d)
    # print(" ")


    # J=6.1
    # print("LS",Slimits_LJS1S2(J,r,q,chi1,chi2)**2)
    # print(S2roots(J,r,xi,q,chi1,chi2))
    #
    # J=6.6
    # print(Slimits_LJS1S2(J,r,q,chi1,chi2)**2)
    # print(S2roots(J,r,xi,q,chi1,chi2))
    #
    # # print(repr(Jofr(ic=(Jmin+Jmax)/2, r=np.logspace(6,1,100), xi=-0.5, q=0.4, chi1=0.9, chi2=0.8)))
    # for J in [5.99355616 ,6.0354517,6.20850742,6.57743474,6.94028614]:
    #     ssol = Slimits_plusminus(J,r,xi,q,chi1,chi2,coincident=True)[0]**2
    #     smin,smax = Slimits_LJS1S2(J,r,q,chi1,chi2)**2
    #     print(ssol>smin,ssol<smax)
    #


    # print( dSdtprefactor(r,xi,q) )
    # kappa=eval_kappa(J,r,q)
    # u=eval_u(r,q)
    # print(S2roots_NEW(kappa,u,xi,q,chi1,chi2))


    #print(Jresonances(r[0],xi[0],q[0],chi1[0],chi2[0]))
    #print(Jresonances(r[1],xi[1],q[1],chi1[1],chi2[1]))
    #  print(Jresonances(r,xi,q,chi1,chi2))
    #print(Jlimits(r=r,xi=xi,q=q,chi1=chi1,chi2=chi2))
    #print(Jlimits(r=r,q=q,chi1=chi1,chi2=chi2))


    #
    # r=1e14
    # xi=-0.5
    # q=0.4
    # chi1=0.9
    # chi2=0.8
    #
    #
    # Jmin,Jmax = Jlimits(r=r,xi=xi,q=q,chi1=chi1,chi2=chi2)
    # print(Jmin,Jmax)
    #
    # print(Satresonance([Jmin,Jmax],[r,r],[xi,xi],[q,q],[chi1,chi1],[chi2,chi2]))
    #
    #
    # print(xiresonances((Jmin+Jmax)/2,r,q,chi1,chi2))
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

    # q=0.7
    # chi1=0.7
    # chi2=0.9
    # r=30
    # J=1.48
    # xi=0.25
    # S = 0.3
    # #print("stillworks",S2roots(J,r,xi,q,chi1,chi2)**0.5)
    #
    # #print(eval_deltaphi(S,J,r,xi,q,chi1,chi2, sign=1))
    #
    # #print(eval_deltaphi([S,S], [J,J], [r,r], [xi,xi], [q,q], [chi1,chi1], [chi2,chi2], sign=[1,1]))
    # #print(eval_deltaphi([S,S], [J,J], [r,r], [xi,xi], [q,q], [chi1,chi1], [chi2,chi2], sign=1))
    #
    # print(morphology(J,r,xi,q,chi1,chi2,simpler=False))
    #
    #
    # #print(morphology(J,r,xi,q,chi1,chi2))
    # print(morphology([J,J],[r,r],[xi,xi],[q,q],[chi1,chi1],[chi2,chi2]))



    #print(spinorbitresonances(J=0.0001,r=10,xi=None,q=0.32,chi1=1,chi2=1))
    #print(spinorbitresonances(J=[0.0001,0.0001],r=[10,10],xi=None,q=[0.32,0.32],chi1=[1,1],chi2=[1,1]))

    #print(xilimits(J=0.05,r=10,q=0.32,chi1=1,chi2=1))

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

    #
    # chi1=0.9
    # chi2=0.8
    # q=0.8
    # Lh,S1h,S2h = sample_unitsphere(3)
    # S1,S2= spinmags(q,chi1,chi2)
    # r=10
    # L = eval_L(r,q)
    # S1vec = S1*S1h
    # S2vec = S2*S2h
    # Lvec = L*Lh
    #
    # S, J, xi = vectors_to_conserved(Lvec, S1vec, S2vec, q)
    # theta1,theta2,deltaphi = conserved_to_angles(S,J,r,xi,q,chi1,chi2,sign=+1)
    # #print(theta1,theta2,deltaphi)
    # #print(vectors_to_conserved([S1vec,S1vec], [S2vec,S2vec], [Lvec,Lvec], [q,q+0.1]))
    # #print(' ')
    # #print(vectors_to_angles(S1vec, S2vec, Lvec))
    # #print(vectors_to_angles([S1vec,S1vec], [S2vec,S2vec], [Lvec,Lvec]))
    # # print(conserved_to_Jframe(S, J, r, xi, q, chi1, chi2))
    # # print(conserved_to_Jframe([S,S], [J,J], [r,r], [xi,xi], [q,q], [chi1,chi1], [chi2,chi2]))
    # #
    # # print(angles_to_Jframe(theta1, theta2, deltaphi, r, q, chi1, chi2))
    # #print(angles_to_Jframe([theta1,theta1], [theta2,theta2], [deltaphi,deltaphi], [r,r], [q,q], [chi1,chi1], [chi2,chi2]))
    #
    # #print(angles_to_Lframe(theta1, theta2, deltaphi, r, q, chi1, chi2))
    # print(angles_to_Lframe([theta1,theta1], [theta2,theta2], [deltaphi,deltaphi], [r,r], [q,q], [chi1,chi1], [chi2,chi2]))
    #
    # print(conserved_to_Lframe([S,S], [J,J], [r,r], [xi,xi], [q,q], [chi1,chi1], [chi2,chi2]))

    # r=10
    # q=0.5
    # chi1=2
    # chi2=2
    # which='uu'
    # print(omega2_aligned([r,r], [q,q], [chi1,chi1], [chi2,chi2], 'dd'))
