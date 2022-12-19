"""
precession. TODO: write me here
"""

import warnings
import numpy as np
# TODO: remember to require scipy>=1.8.0
import scipy.special
import scipy.integrate
from itertools import repeat


# TODO: new algorithm! Needs to be documented!
def roots_vec(p, enforce=False):
    """
    Locate roots of polynomial using a vectorized version of numpy.roots. Equivalent to [np.roots(x) for x in p].
    Credits: stackoverflow user `pv`, see https://stackoverflow.com/a/35853977

    Call
    ----
    roots = roots_vec(p)

    Parameters
    ----------
    p: array
        Polynomial coefficients.

    Returns
    -------
    roots: array
        Polynomial roots.
    """

    p = np.atleast_1d(p)
    non_zeros = np.count_nonzero(p, axis=1)

    if not non_zeros.all()!=0:
        if enforce:
            raise ValueError("There is at least one coefficients line with all zeros [roots_vec_zeros].")
        else:
            warnings.warn("There is at least one coefficients line with all zeros [roots_vec_zeros].", Warning)

    #https://stackoverflow.com/a/20361561
    B = np.append(p, np.ones(p.shape[0])[:,None], axis=1)
    nz = np.argmax(B!=0,axis=1)
    rows, columns = np.ogrid[:p.shape[0], :p.shape[1]]
    shift = np.copy(nz)
    shift[shift > 0] -= p.shape[1]
    columns = columns + shift[:, np.newaxis]
    p = p[rows, columns]

    n = p.shape[-1]
    A = np.zeros(p.shape[:1] + (n-1, n-1), float)
    A[..., 1:, :-1] = np.eye(n-2)
    A[..., 0, :] = -p[..., 1:]/p[..., None, 0]

    results = np.linalg.eigvals(A)

    nansol = np.reshape(np.repeat(nz, results.shape[1], axis=0), results.shape)
    resind = np.mgrid[0:results.shape[0], 0:results.shape[1]][1]

    return np.where(resind<nansol, np.nan, results)

def norm_nested(x):
    """
    Norm of 2D array of shape (N,3) along last axis.

    Call
    ----
    n = norm_nested(x)

    Parameters
    ----------
    x : array
        Input array.

    Returns
    -------
    n : array
        Norm of the input arrays.
    """

    return np.linalg.norm(x, axis=1)


def normalize_nested(x):
    """
    Normalize 2D array of shape (N,3) along last axis.

    Call
    ----
    y = normalize_nested(x)

    Parameters
    ----------
    x : array
        Input array.

    Returns
    -------
    y : array
        Normalized array.
    """

    return x/norm_nested(x)[:, None]


def dot_nested(x, y):
    """
    Dot product between 2D arrays along last axis.

    Call
    ----
    z = dot_nested(x, y)

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

    return np.einsum('ij, ij->i', x, y)


def scalar_nested(k, x):
    """
    Nested scalar product between a 1D and a 2D array.

    Call
    ----
    y = scalar_nested(k, x)

    Parameters
    ----------
    k : float
        Input scalar.
    x : array
        Input array.

    Returns
    -------
    y : array
        Scalar product array.
    """

    return k[:,np.newaxis]*x


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


def tiler(thing,shaper):
    thing =np.atleast_1d(thing)
    shaper =np.atleast_1d(shaper)
    assert thing.ndim == 1 and shaper.ndim==1

    return np.squeeze(np.tile(thing, np.shape(shaper)).reshape(len(shaper),len(thing)))



def affine(vec, low, up):
    vec = np.atleast_1d(vec)
    up = np.atleast_1d(up)
    low = np.atleast_1d(low)

    rescaled = ( vec - low ) / (up - low)

    return rescaled


def inverseaffine(rescaled, low, up):
    
    rescaled = np.atleast_1d(rescaled)
    up = np.atleast_1d(up)
    low = np.atleast_1d(low)

    vec = low + rescaled*(up-low)

    return vec


def wraproots(coefficientfunction, *args, **kwargs):
    """
    Find roots of a polynomial given coefficients, ordered according to their real part. Complex roots are masked with nans. This is essentially a wrapper of numpy.roots.

    Call
    ----
    sols = precession.wraproots(coefficientfunction, *args, **kwargs)

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

    coeffs = coefficientfunction(*args, **kwargs)
    sols = np.sort_complex(roots_vec(coeffs.T))
    sols = np.real(np.where(np.isreal(sols), sols, np.nan))

    return sols


def ellippi(n, phi, m):
    """
    Incomplete elliptic integral of the third kind. This is reconstructed using scipy's implementation of Carlson's R integrals (arxiv:math/9409227).

    Call
    ----
    piintegral = precession.ellippi(n, phi, m)

    Parameters
    ----------
    n: foat
        Characheristic of the elliptic integral.
    phi: float
        Amplitude of the elliptic integral.
    m: float
        Parameter of the elliptic integral

    Returns
    -------
    piintegral: float
        Incomplete elliptic integral of the third kind
    """

    # Important: this requires scipy>=1.8.0
    # https://docs.scipy.org/doc/scipy/release.1.8.0.html

    # Notation used here:
    # https://reference.wolfram.com/language/ref/EllipticPi.html

    # A much slower implementation using simpy
    # from sympy import elliptic_pi
    # return float(elliptic_pi(n, phi, m))

    n = np.array(n)
    phi = np.array(phi)
    m = np.array(m)

    if ~np.all(n>0) or ~np.all(n<1) or ~np.all(phi>0) or ~np.all(phi<np.pi/2) or ~np.all(m>0) or ~np.all(m<1):
        warnings.warn("Elliptic intergal of the third kind evaluated outside of the expected domain. Our implementation has not been tested in this regime!", Warning)

    # Eq (61) in Carlson 1994 (arxiv:math/9409227v1). Careful with the notation: one has k^2 --> m and n --> -n.
    c = (1/np.sin(phi))**2
    return scipy.special.elliprf(c-1,c-m,c) +(np.array(n)/3)*scipy.special.elliprj(c-1,c-m,c,c-n)


def rotate_zaxis(vec, angle):
    """
    Rotate series of arrays along the z axis of a given angle. Input vec has shape (N,3) and input angle has shape (N,).

    Call
    ----
        newvec = rotate_zaxis(vec,angle)

    Parameters
    ----------
    vec: array
        Input array.
    angle: float
        Rotation angle.

    Returns
    -------
    newvec: array
        Rotated array.
    """

    newx = vec[:, 0]*np.cos(angle) - vec[:, 1]*np.sin(angle)
    newy = vec[:, 0]*np.sin(angle) + vec[:, 1]*np.cos(angle)
    newz = vec[:, 2]
    newvec = np.transpose([newx, newy, newz])

    return newvec


def ismonotonic(vec, which):
    """
    Check if an array is monotonic. The parameter `which` can takes the following values:
    - `<` check array is strictly increasing.
    - `<=` check array is increasing.
    - `>` check array is strictly decreasing.
    - `>=` check array is decreasing.

    Call
    ----
        check = ismonotonic(vec, which):

    Parameters
    ----------
    vec: array
        Input array.
    which: string
        Select function behavior.

    Returns
    -------
    check: boolean
        Result
    """

    if which == '<':
        return np.all(vec[:-1] < vec[1:])
    elif which == '<=':
        return np.all(vec[:-1] <= vec[1:])
    elif which == '>':
        return np.all(vec[:-1] > vec[1:])
    elif which == '>=':
        return np.all(vec[:-1] >= vec[1:])
    else:
        raise ValueError("`which` needs to be one of the following: `>`, `>=`, `<`, `<=`.")


def squarewithsign(x):
    return x*np.abs(x)

# Definitions

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

    return np.stack([m1, m2])


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
    q = m2/m1
    assert (q < 1).all(), "The convention used in this code is q=m2/m1<1."

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


def eval_S1(q, chi1):
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
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.

    Returns
    -------
    S1: float
        Magnitude of the primary spin.
    """

    chi1 = np.atleast_1d(chi1)
    S1 = chi1*(eval_m1(q))**2

    return S1


def eval_S2(q, chi2):
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


# TODO: remove
def spinmags(q, chi1, chi2):
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
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    S1: float
        Magnitude of the primary spin.
    S2: float
        Magnitude of the secondary spin.
    """

    S1 = eval_S1(q, chi1)
    S2 = eval_S2(q, chi2)

    return np.stack([S1, S2])


def eval_L(r, q):
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
    v = 1/r**0.5

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
        r = (2*eval_m1(q)*eval_m2(q)*u)**(-2)

    else:
        raise TypeError("Provide either (L,q) or (u,q).")

    return r


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


# TODO: Do we need this?
def Jlimits_LS1S2(r, q, chi1, chi2):
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
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    Jmin: float
        Minimum value of the total angular momentum J.
    Jmax: float
        Maximum value of the total angular momentum J.
    """

    S1, S2 = spinmags(q, chi1, chi2)
    L = eval_L(r, q)
    Jmin = np.maximum.reduce([np.zeros(L.shape), L-S1-S2, np.abs(S1-S2)-L])
    Jmax = L+S1+S2

    return np.stack([Jmin, Jmax])


def kappadiscriminant_coefficients(u, chieff, q, chi1, chi2):
    """
    Coefficients of the quintic equation in kappa that defines the spin-orbit resonances.

    Call
    ----
    coeff5,coeff4,coeff3,coeff2,coeff1,coeff0 = kappadiscriminant_coefficients(u,chieff,q,chi1,chi2)

    Parameters
    ----------
    u: float
        Compactified separation 1/(2L).
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
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

    u = np.atleast_1d(u).astype(float)
    chieff = np.atleast_1d(chieff).astype(float)
    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)

    coeff5 = -u

    # Machine generated with eq_generator.nb
    coeff4 = 1/16 * q**(-1) * ((1 + q))**(-4) * (1 + (q**6 + (q * (2 + \
    (80 * u**2 * chi1**2 + 40 * u * chieff)) + (q**5 * (2 + (80 * u**2 * \
    chi2**2 + 40 * u * chieff)) + (4 * q**3 * (-1 + (60 * u * chieff + 8 \
    * u**2 * chieff**2)) + (q**2 * (-1 + (160 * u * chieff + 16 * u**2 * \
    (-3 * chi1**2 + chieff**2))) + q**4 * (-1 + (160 * u * chieff + 16 * \
    u**2 * (-3 * chi2**2 + chieff**2)))))))))

    # Machine generated with eq_generator.nb
    coeff3 = -1/8 * q**(-1) * ((1 + q))**(-8) * (((-1 + q))**2 * ((1 + \
    q))**8 * chieff + (8 * q * u**3 * ((10 + (-12 * q + 3 * q**2)) * \
    chi1**4 + (-2 * q * chi1**2 * (6 * q**2 * chi2**2 + (6 * q**4 * \
    chi2**2 + (-2 * chieff**2 + (-3 * q * chieff**2 + q**3 * (-11 * \
    chi2**2 + chieff**2))))) + q**4 * chi2**2 * (10 * q**4 * chi2**2 + \
    (-2 * chieff**2 + (4 * q**3 * (-3 * chi2**2 + chieff**2) + 3 * q**2 * \
    (chi2**2 + 2 * chieff**2)))))) + (4 * q * ((1 + q))**3 * u**2 * \
    chieff * (-1 * (-20 + (3 * q + q**2)) * chi1**2 + q * (20 * q**4 * \
    chi2**2 + (4 * chieff**2 + (12 * q * chieff**2 + (-1 * q**2 * \
    (chi2**2 + -12 * chieff**2) + q**3 * (-3 * chi2**2 + 4 * \
    chieff**2)))))) + 2 * ((1 + q))**4 * u * (-1 * ((-1 + q))**2 * (-1 + \
    5 * q) * chi1**2 + q * (q**5 * chi2**2 + (8 * chieff**2 + (40 * q * \
    chieff**2 + (q**4 * (-7 * chi2**2 + 8 * chieff**2) + (q**3 * (11 * \
    chi2**2 + 40 * chieff**2) + q**2 * (-5 * chi2**2 + 64 * \
    chieff**2))))))))))

    # Machine generated with eq_generator.nb
    coeff2 = 1/16 * q**(-1) * ((1 + q))**(-12) * (-16 * q * (-10 + (18 * \
    q + (-9 * q**2 + q**3))) * u**4 * chi1**6 + (chieff**2 + (4 * q * \
    chieff**2 * (3 + 2 * u * chieff) + (q**(14) * (6 * u**2 * chi2**4 + \
    (chieff**2 + chi2**2 * (-1 + 6 * u * chieff))) + (q**2 * (-1 * \
    chi2**2 + chieff**2 * (59 + (144 * u * chieff + 16 * u**2 * \
    chieff**2))) + (4 * q**(13) * (40 * u**4 * chi2**6 + (chieff**2 * (3 \
    + 2 * u * chieff) + (12 * u**2 * chi2**4 * (-1 + 5 * u * chieff) + \
    chi2**2 * (-1 + (-4 * u * chieff + 24 * u**2 * chieff**2))))) + (-4 * \
    q**3 * (chi2**2 * (1 + 2 * u * chieff) + -2 * chieff**2 * (19 + (126 \
    * u * chieff + 24 * u**2 * chieff**2))) + (q**4 * (-2 * chi2**2 * (1 \
    + (43 * u * chieff + 4 * u**2 * chieff**2)) + chieff**2 * (201 + \
    (3920 * u * chieff + 976 * u**2 * chieff**2))) + (4 * q**5 * \
    (chieff**2 * (13 + (2430 * u * chieff + 704 * u**2 * chieff**2)) + \
    chi2**2 * (3 + (-72 * u * chieff + (10 * u**2 * chieff**2 + 8 * u**3 \
    * chieff**3)))) + (-4 * q**7 * (u**2 * chi2**4 * (19 + 8 * u * \
    chieff) + (-4 * chieff**2 * (-27 + (1218 * u * chieff + 392 * u**2 * \
    chieff**2)) + -2 * chi2**2 * (-1 + (20 * u * chieff + (169 * u**2 * \
    chieff**2 + 32 * u**3 * chieff**3))))) + (q**(12) * (-288 * u**4 * \
    chi2**6 + (chieff**2 * (59 + (144 * u * chieff + 16 * u**2 * \
    chieff**2)) + (2 * u**2 * chi2**4 * (-67 + (204 * u * chieff + 48 * \
    u**2 * chieff**2)) + 2 * chi2**2 * (-1 + (-95 * u * chieff + (296 * \
    u**2 * chieff**2 + 48 * u**3 * chieff**3)))))) + (4 * q**9 * (2 * \
    u**2 * chi2**4 * (13 + (6 * u * chieff + -8 * u**2 * chieff**2)) + \
    (chieff**2 * (13 + (2430 * u * chieff + 704 * u**2 * chieff**2)) + 2 \
    * chi2**2 * (-1 + (70 * u * chieff + (379 * u**2 * chieff**2 + 100 * \
    u**3 * chieff**3))))) + (4 * q**(11) * (36 * u**4 * chi2**6 + (u**2 * \
    chi2**4 * (5 + (-16 * u * chieff + 24 * u**2 * chieff**2)) + (2 * \
    chieff**2 * (19 + (126 * u * chieff + 24 * u**2 * chieff**2)) + \
    chi2**2 * (3 + (-102 * u * chieff + (406 * u**2 * chieff**2 + 112 * \
    u**3 * chieff**3)))))) + (q**8 * (2 * u**2 * chi2**4 * (-53 + (28 * u \
    * chieff + 8 * u**2 * chieff**2)) + (chieff**2 * (-261 + (16416 * u * \
    chieff + 5152 * u**2 * chieff**2)) + 4 * chi2**2 * (-7 + (189 * u * \
    chieff + (614 * u**2 * chieff**2 + 120 * u**3 * chieff**3))))) + \
    (q**6 * (-8 * u**2 * chi2**4 + (chieff**2 * (-261 + (16416 * u * \
    chieff + 5152 * u**2 * chieff**2)) + chi2**2 * (17 + (-338 * u * \
    chieff + (424 * u**2 * chieff**2 + 128 * u**3 * chieff**3))))) + \
    (q**10 * (-16 * u**4 * chi2**6 + (-2 * u**2 * chi2**4 * (-121 + (136 \
    * u * chieff + 40 * u**2 * chieff**2)) + (chieff**2 * (201 + (3920 * \
    u * chieff + 976 * u**2 * chieff**2)) + chi2**2 * (17 + (-148 * u * \
    chieff + (2680 * u**2 * chieff**2 + 832 * u**3 * chieff**3)))))) + (2 \
    * u**2 * chi1**4 * (3 + (-4 * q**8 + (2 * q**7 * (-19 + (36 * u**2 * \
    chi2**2 + -8 * u * chieff)) + (24 * q * (-1 + 5 * u * chieff) + (2 * \
    q**3 * (5 + (-16 * u * chieff + 24 * u**2 * chieff**2)) + (q**2 * \
    (-67 + (204 * u * chieff + 48 * u**2 * chieff**2)) + (4 * q**5 * (13 \
    + (6 * u * chieff + 8 * u**2 * (9 * chi2**2 + -1 * chieff**2))) + \
    (q**6 * (-53 + (28 * u * chieff + 8 * u**2 * (-27 * chi2**2 + \
    chieff**2))) + -1 * q**4 * (-121 + (136 * u * chieff + 8 * u**2 * (18 \
    * chi2**2 + 5 * chieff**2))))))))))) + -1 * chi1**2 * (1 + (q**(12) + \
    (-6 * u * chieff + (q**(11) * (4 + (60 * u**2 * chi2**2 + 8 * u * \
    chieff)) + (q * (4 + (16 * u * chieff + -96 * u**2 * chieff**2)) + \
    (q**2 * (2 + (190 * u * chieff + (-592 * u**2 * chieff**2 + -96 * \
    u**3 * chieff**3))) + (q**10 * (2 + (288 * u**4 * chi2**4 + (86 * u * \
    chieff + (24 * u**3 * chi2**2 * chieff + 4 * u**2 * (chi2**2 + 2 * \
    chieff**2))))) + (-4 * q**3 * (3 + (-102 * u * chieff + (112 * u**3 * \
    chieff**3 + u**2 * (-15 * chi2**2 + 406 * chieff**2)))) + (-4 * q**9 \
    * (3 + (-72 * u * chieff + (8 * u**3 * chieff * (-3 * chi2**2 + \
    chieff**2) + (2 * u**2 * (29 * chi2**2 + 5 * chieff**2) + 24 * u**4 * \
    (6 * chi2**4 + -1 * chi2**2 * chieff**2))))) + (q**4 * (-17 + (148 * \
    u * chieff + (4 * u**2 * (chi2**2 + -670 * chieff**2) + 8 * u**3 * (3 \
    * chi2**2 * chieff + -104 * chieff**3)))) + (8 * q**5 * (1 + (-70 * u \
    * chieff + (12 * u**4 * chi2**2 * chieff**2 + (-1 * u**2 * (29 * \
    chi2**2 + 379 * chieff**2) + 4 * u**3 * (3 * chi2**2 * chieff + -25 * \
    chieff**3))))) + (4 * q**6 * (7 + (-189 * u * chieff + (8 * u**4 * \
    chi2**2 * chieff**2 + (-1 * u**2 * (chi2**2 + 614 * chieff**2) + 6 * \
    u**3 * (7 * chi2**2 * chieff + -20 * chieff**3))))) + (q**8 * (-17 + \
    (338 * u * chieff + (-4 * u**2 * (chi2**2 + 106 * chieff**2) + (16 * \
    u**4 * (27 * chi2**4 + 2 * chi2**2 * chieff**2) + 8 * u**3 * (21 * \
    chi2**2 * chieff + -16 * chieff**3))))) + -8 * q**7 * (-1 + (20 * u * \
    chieff + (u**2 * (-43 * chi2**2 + 169 * chieff**2) + (2 * u**4 * (9 * \
    chi2**4 + 8 * chi2**2 * chieff**2) + -8 * u**3 * (3 * chi2**2 * \
    chieff + -4 * chieff**3)))))))))))))))))))))))))))))))))))

    # Machine generated with eq_generator.nb
    coeff1 = -1/8 * q**(-1) * ((1 + q))**(-16) * (-1 * ((-1 + q))**2 * \
    ((1 + q))**(11) * chieff * (((-1 + q))**2 * chi1**2 + q * (q**4 * \
    chi2**2 + (-1 * chieff**2 + (-3 * q * chieff**2 + (q**2 * (chi2**2 + \
    -3 * chieff**2) + -1 * q**3 * (2 * chi2**2 + chieff**2)))))) + (8 * \
    (-1 + q) * q * u**5 * (-1 * chi1**2 + q**3 * chi2**2) * ((5 + (-7 * q \
    + 2 * q**2)) * chi1**6 + (q * chi1**4 * (-7 * q**2 * chi2**2 + (-2 * \
    q**4 * chi2**2 + (2 * q**5 * chi2**2 + (4 * chieff**2 + (6 * q * \
    chieff**2 + q**3 * (7 * chi2**2 + -2 * chieff**2)))))) + (-1 * q**4 * \
    chi1**2 * chi2**2 * (7 * q**5 * chi2**2 + (2 * chieff**2 + (4 * q * \
    chieff**2 + (-2 * q**2 * (chi2**2 + -2 * chieff**2) + (q**4 * (-7 * \
    chi2**2 + 2 * chieff**2) + 2 * q**3 * (chi2**2 + 2 * chieff**2)))))) \
    + q**8 * chi2**4 * (5 * q**4 * chi2**2 + (-2 * chieff**2 + (2 * q**2 \
    * (chi2**2 + 3 * chieff**2) + q**3 * (-7 * chi2**2 + 4 * \
    chieff**2))))))) + (4 * q * ((1 + q))**3 * u**4 * chieff * ((20 + \
    (-49 * q + (41 * q**2 + -12 * q**3))) * chi1**6 + (q**4 * chi1**2 * \
    chi2**2 * (-3 * q**6 * chi2**2 + (8 * chieff**2 + (4 * q * chieff**2 \
    + (q**3 * (22 * chi2**2 + -28 * chieff**2) + (q**4 * (-14 * chi2**2 + \
    4 * chieff**2) + (-4 * q**2 * (2 * chi2**2 + 7 * chieff**2) + q**5 * \
    (3 * chi2**2 + 8 * chieff**2))))))) + (q**8 * chi2**4 * (20 * q**5 * \
    chi2**2 + (12 * chieff**2 + (4 * q * chieff**2 + (-4 * q**2 * (3 * \
    chi2**2 + 4 * chieff**2) + (q**3 * (41 * chi2**2 + 4 * chieff**2) + \
    q**4 * (-49 * chi2**2 + 12 * chieff**2)))))) + q * chi1**4 * (22 * \
    q**5 * chi2**2 + (-8 * q**6 * chi2**2 + (12 * chieff**2 + (4 * q * \
    chieff**2 + (-2 * q**4 * (7 * chi2**2 + -6 * chieff**2) + (q**3 * (3 \
    * chi2**2 + 4 * chieff**2) + -1 * q**2 * (3 * chi2**2 + 16 * \
    chieff**2)))))))))) + (-1 * ((1 + q))**8 * u * (((-1 + q))**4 * \
    chi1**4 + (((-1 + q))**2 * chi1**2 * (-14 * q**5 * chi2**2 + (q**6 * \
    chi2**2 + (-1 * chieff**2 + (16 * q * chieff**2 + (q**4 * (26 * \
    chi2**2 + 7 * chieff**2) + (q**3 * (-14 * chi2**2 + 32 * chieff**2) + \
    q**2 * (chi2**2 + 42 * chieff**2))))))) + q**2 * (-8 * chieff**4 + \
    (-56 * q * chieff**4 + (q**8 * (chi2**4 + -1 * chi2**2 * chieff**2) + \
    (q**7 * (-4 * chi2**4 + 18 * chi2**2 * chieff**2) + (q**4 * (chi2**4 \
    + (-15 * chi2**2 * chieff**2 + -152 * chieff**4)) + (q**2 * (7 * \
    chi2**2 * chieff**2 + -152 * chieff**4) + (2 * q**3 * (9 * chi2**2 * \
    chieff**2 + -104 * chieff**4) + (q**6 * (6 * chi2**4 + (9 * chi2**2 * \
    chieff**2 + -8 * chieff**4)) + -4 * q**5 * (chi2**4 + (9 * chi2**2 * \
    chieff**2 + 14 * chieff**4)))))))))))) + (2 * ((1 + q))**4 * u**3 * \
    (-1 * ((-1 + q))**2 * (-1 + (15 * q + 4 * q**2)) * chi1**6 + (-1 * q \
    * chi1**4 * (10 * q**6 * chi2**2 + (4 * q**7 * chi2**2 + (-24 * \
    chieff**2 + (16 * q * chieff**2 + (q**4 * (143 * chi2**2 + -32 * \
    chieff**2) + (-5 * q**3 * (17 * chi2**2 + 12 * chieff**2) + (q**5 * \
    (-87 * chi2**2 + 20 * chieff**2) + q**2 * (15 * chi2**2 + 32 * \
    chieff**2)))))))) + (q**2 * chi1**2 * (-15 * q**9 * chi2**4 + (8 * \
    chieff**4 + (24 * q * chieff**4 + (q**8 * (85 * chi2**4 + -4 * \
    chi2**2 * chieff**2) + (q**7 * (-143 * chi2**4 + 24 * chi2**2 * \
    chieff**2) + (-2 * q**5 * (5 * chi2**4 + (48 * chi2**2 * chieff**2 + \
    -44 * chieff**4)) + (-4 * q**4 * (chi2**4 + (5 * chi2**2 * chieff**2 \
    + -30 * chieff**4)) + (8 * q**3 * (3 * chi2**2 * chieff**2 + 10 * \
    chieff**4) + (q**6 * (87 * chi2**4 + (-20 * chi2**2 * chieff**2 + 24 \
    * chieff**4)) + q**2 * (-4 * chi2**2 * chieff**2 + 40 * \
    chieff**4)))))))))) + q**6 * chi2**2 * (q**8 * chi2**4 + (24 * \
    chieff**4 + (88 * q * chieff**4 + (-20 * q**2 * chieff**2 * (chi2**2 \
    + -6 * chieff**2) + (q**7 * (-17 * chi2**4 + 24 * chi2**2 * \
    chieff**2) + (16 * q**3 * (2 * chi2**2 * chieff**2 + 5 * chieff**4) + \
    (q**6 * (27 * chi2**4 + (-16 * chi2**2 * chieff**2 + 8 * chieff**4)) \
    + (q**5 * (-7 * chi2**4 + (-32 * chi2**2 * chieff**2 + 24 * \
    chieff**4)) + q**4 * (-4 * chi2**4 + (60 * chi2**2 * chieff**2 + 40 * \
    chieff**4))))))))))))) + ((1 + q))**7 * u**2 * chieff * (-1 * ((-1 + \
    q))**2 * (-3 + (49 * q + 16 * q**2)) * chi1**4 + (-2 * q * (1 + q) * \
    chi1**2 * (22 * q**4 * chi2**2 + (-15 * q**5 * chi2**2 + (4 * q**6 * \
    chi2**2 + (-4 * chieff**2 + (6 * q * chieff**2 + (4 * q**2 * (chi2**2 \
    + -7 * chieff**2) + -1 * q**3 * (15 * chi2**2 + 38 * chieff**2))))))) \
    + q**3 * (3 * q**8 * chi2**4 + (16 * chieff**4 + (80 * q * chieff**4 \
    + (160 * q**2 * chieff**4 + (q**6 * (85 * chi2**4 + -4 * chi2**2 * \
    chieff**2) + (q**7 * (-55 * chi2**4 + 8 * chi2**2 * chieff**2) + \
    (q**5 * (-17 * chi2**4 + (44 * chi2**2 * chieff**2 + 16 * chieff**4)) \
    + (4 * q**3 * (19 * chi2**2 * chieff**2 + 40 * chieff**4) + q**4 * \
    (-16 * chi2**4 + (132 * chi2**2 * chieff**2 + 80 * \
    chieff**4)))))))))))))))))


    # Machine generated with eq_generator.nb
    coeff0 = -1/16 * q**(-1) * ((1 + q))**(-20) * (((-1 + q))**2 * ((1 + \
    q))**(12) * (-1 * ((-1 + q))**2 * chi1**2 + q**2 * ((1 + q))**2 * \
    chieff**2) * (-2 * q**3 * chi2**2 + (q**4 * chi2**2 + (-1 * chieff**2 \
    + (-2 * q * chieff**2 + q**2 * (chi2**2 + -1 * chieff**2))))) + (-16 \
    * ((-1 + q))**2 * q * u**6 * ((chi1**4 + (-1 * q**3 * (1 + q) * \
    chi1**2 * chi2**2 + q**7 * chi2**4)))**2 * (-1 * (-1 + q) * chi1**2 + \
    q * (q**3 * chi2**2 + (chieff**2 + (2 * q * chieff**2 + q**2 * (-1 * \
    chi2**2 + chieff**2))))) + (-8 * (-1 + q) * q * ((1 + q))**3 * u**5 * \
    (-1 * chi1**2 + q**4 * chi2**2) * chieff * ((5 + (-13 * q + 8 * \
    q**2)) * chi1**6 + (q**8 * chi2**4 * (8 * q**2 * chi2**2 + (5 * q**4 \
    * chi2**2 + (-8 * chieff**2 + (-12 * q * chieff**2 + q**3 * (-13 * \
    chi2**2 + 4 * chieff**2))))) + (q**4 * chi1**2 * chi2**2 * (-1 * q**5 \
    * chi2**2 + (4 * chieff**2 + (8 * q * chieff**2 + (-2 * q**3 * \
    (chi2**2 + -4 * chieff**2) + (-4 * q**2 * (chi2**2 + -2 * chieff**2) \
    + q**4 * (7 * chi2**2 + 4 * chieff**2)))))) + -1 * chi1**4 * (2 * \
    q**5 * chi2**2 + (4 * q**6 * chi2**2 + (-4 * q * chieff**2 + (q**4 * \
    (-7 * chi2**2 + 8 * chieff**2) + q**3 * (chi2**2 + 12 * \
    chieff**2)))))))) + (2 * ((1 + q))**(11) * u * chieff * (((-1 + \
    q))**4 * chi1**4 + (-1 * ((-1 + q))**2 * q * (1 + q) * chi1**2 * (-10 \
    * q**3 * chi2**2 + (5 * q**4 * chi2**2 + (-5 * chieff**2 + (-8 * q * \
    chieff**2 + q**2 * (5 * chi2**2 + -3 * chieff**2))))) + q**3 * (q**8 \
    * chi2**4 + (-4 * chieff**4 + (-20 * q * chieff**4 + (5 * q**3 * \
    chieff**2 * (chi2**2 + -8 * chieff**2) + (3 * q**6 * chi2**2 * (2 * \
    chi2**2 + chieff**2) + (q**7 * (-4 * chi2**4 + 5 * chi2**2 * \
    chieff**2) + (q**2 * (3 * chi2**2 * chieff**2 + -40 * chieff**4) + \
    (q**4 * (chi2**4 + (-6 * chi2**2 * chieff**2 + -20 * chieff**4)) + -2 \
    * q**5 * (2 * chi2**4 + (5 * chi2**2 * chieff**2 + 2 * \
    chieff**4)))))))))))) + (2 * ((1 + q))**7 * u**3 * chieff * (((-1 + \
    q))**2 * (-1 + (25 * q + 12 * q**2)) * chi1**6 + (q * chi1**4 * (85 * \
    q**6 * chi2**2 + (-32 * q**7 * chi2**2 + (-4 * chieff**2 + (48 * q * \
    chieff**2 + (q**4 * (83 * chi2**2 + -40 * chieff**2) + (4 * q**2 * \
    (chi2**2 + 7 * chieff**2) + (q**5 * (-103 * chi2**2 + 20 * chieff**2) \
    + -1 * q**3 * (37 * chi2**2 + 84 * chieff**2)))))))) + (q**3 * \
    chi1**2 * (-37 * q**8 * chi2**4 + (4 * q**9 * chi2**4 + (16 * \
    chieff**4 + (32 * q * chieff**4 + (16 * q**2 * chieff**2 * (chi2**2 + \
    -2 * chieff**2) + (q**7 * (83 * chi2**4 + 16 * chi2**2 * chieff**2) + \
    (q**6 * (-103 * chi2**4 + 24 * chi2**2 * chieff**2) + (q**5 * (85 * \
    chi2**4 + (-8 * chi2**2 * chieff**2 + -32 * chieff**4)) + (8 * q**3 * \
    (3 * chi2**2 * chieff**2 + -16 * chieff**4) + -8 * q**4 * (4 * \
    chi2**4 + (chi2**2 * chieff**2 + 14 * chieff**4))))))))))) + q**7 * \
    chi2**2 * (-1 * q**8 * chi2**4 + (-32 * chieff**4 + (-112 * q * \
    chieff**4 + (q**7 * (27 * chi2**4 + -4 * chi2**2 * chieff**2) + (q**6 \
    * (-39 * chi2**4 + 48 * chi2**2 * chieff**2) + (4 * q**2 * (5 * \
    chi2**2 * chieff**2 + -32 * chieff**4) + (-8 * q**3 * (5 * chi2**2 * \
    chieff**2 + 4 * chieff**4) + (4 * q**4 * (3 * chi2**4 + (-21 * \
    chi2**2 * chieff**2 + 8 * chieff**4)) + q**5 * (chi2**4 + (28 * \
    chi2**2 * chieff**2 + 16 * chieff**4))))))))))))) + (((1 + q))**4 * \
    u**4 * (((-1 + q))**2 * (-1 + (20 * q + 8 * q**2)) * chi1**8 + (-4 * \
    (-1 + q) * q * chi1**6 * (-11 * q**5 * chi2**2 + (8 * q**6 * chi2**2 \
    + (-8 * chieff**2 + (20 * q * chieff**2 + (q**4 * (30 * chi2**2 + -22 \
    * chieff**2) + (-8 * q**3 * (4 * chi2**2 + chieff**2) + q**2 * (5 * \
    chi2**2 + 42 * chieff**2))))))) + (q**10 * chi2**4 * (-1 * q**8 * \
    chi2**4 + (-96 * chieff**4 + (-288 * q * chieff**4 + (q**7 * (22 * \
    chi2**4 + -32 * chi2**2 * chieff**2) + (8 * q**2 * (11 * chi2**2 * \
    chieff**2 + -26 * chieff**4) + (-8 * q**3 * (7 * chi2**2 * chieff**2 \
    + -16 * chieff**4) + (q**6 * (-33 * chi2**4 + (112 * chi2**2 * \
    chieff**2 + -16 * chieff**4)) + (4 * q**5 * (chi2**4 + (22 * chi2**2 \
    * chieff**2 + 8 * chieff**4)) + 8 * q**4 * (chi2**4 + (-25 * chi2**2 \
    * chieff**2 + 24 * chieff**4)))))))))) + (4 * q**6 * chi1**2 * \
    chi2**2 * (5 * q**9 * chi2**4 + (24 * chieff**4 + (56 * q * chieff**4 \
    + (12 * q**3 * chieff**2 * (chi2**2 + -4 * chieff**2) + (8 * q**2 * \
    chieff**2 * (-2 * chi2**2 + chieff**2) + (q**7 * (62 * chi2**4 + -6 * \
    chi2**2 * chieff**2) + (q**8 * (-37 * chi2**4 + 2 * chi2**2 * \
    chieff**2) + (q**4 * (-8 * chi2**4 + (52 * chi2**2 * chieff**2 + 8 * \
    chieff**4)) + (q**6 * (-41 * chi2**4 + (-38 * chi2**2 * chieff**2 + \
    24 * chieff**4)) + q**5 * (19 * chi2**4 + (-6 * chi2**2 * chieff**2 + \
    56 * chieff**4))))))))))) + 2 * q**2 * chi1**4 * (-2 * q**9 * chi2**4 \
    + (4 * q**10 * chi2**4 + (-8 * chieff**4 + (16 * q * chieff**4 + (4 * \
    q**2 * chieff**2 * (chi2**2 + 24 * chieff**2) + (q**8 * (53 * chi2**4 \
    + -32 * chi2**2 * chieff**2) + (q**7 * (-110 * chi2**4 + 24 * chi2**2 \
    * chieff**2) + (q**6 * (53 * chi2**4 + (104 * chi2**2 * chieff**2 + \
    -48 * chieff**4)) + (4 * q**4 * (chi2**4 + (-19 * chi2**2 * chieff**2 \
    + -26 * chieff**4)) + (q**3 * (-12 * chi2**2 * chieff**2 + 64 * \
    chieff**4) + -2 * q**5 * (chi2**4 + (6 * chi2**2 * chieff**2 + 72 * \
    chieff**4)))))))))))))))) + ((1 + q))**8 * u**2 * (((-1 + q))**4 * \
    chi1**6 + (-1 * ((-1 + q))**2 * chi1**4 * (4 * q**5 * chi2**2 + (10 * \
    q**6 * chi2**2 + (chieff**2 + (-38 * q * chieff**2 + (q**3 * (26 * \
    chi2**2 + -86 * chieff**2) + (-1 * q**4 * (39 * chi2**2 + 23 * \
    chieff**2) + -1 * q**2 * (chi2**2 + 102 * chieff**2))))))) + (q**2 * \
    chi1**2 * (-28 * q**9 * chi2**4 + (q**10 * chi2**4 + (32 * chieff**4 \
    + (72 * q * chieff**4 + (48 * q**3 * chieff**2 * (chi2**2 + -5 * \
    chieff**2) + (q**8 * (92 * chi2**4 + -22 * chi2**2 * chieff**2) + \
    (-12 * q**7 * (9 * chi2**4 + -4 * chi2**2 * chieff**2) + (8 * q**5 * \
    (2 * chi2**4 + (-12 * chi2**2 * chieff**2 + -11 * chieff**4)) + (q**6 \
    * (37 * chi2**4 + (22 * chi2**2 * chieff**2 + -8 * chieff**4)) + (-2 \
    * q**2 * (11 * chi2**2 * chieff**2 + 20 * chieff**4) + -2 * q**4 * (5 \
    * chi2**4 + (-11 * chi2**2 * chieff**2 + 120 * chieff**4)))))))))))) \
    + q**4 * (-16 * chieff**6 + (-96 * q * chieff**6 + (-8 * q**2 * \
    chieff**4 * (chi2**2 + 30 * chieff**2) + (-4 * q**9 * (chi2**6 + -10 \
    * chi2**4 * chieff**2) + (q**10 * (chi2**6 + -1 * chi2**4 * \
    chieff**2) + (-4 * q**7 * (chi2**6 + (20 * chi2**4 * chieff**2 + -18 \
    * chi2**2 * chieff**4)) + (q**8 * (6 * chi2**6 + (25 * chi2**4 * \
    chieff**2 + 32 * chi2**2 * chieff**4)) + (q**4 * (23 * chi2**4 * \
    chieff**2 + (-240 * chi2**2 * chieff**4 + -240 * chieff**6)) + (q**6 \
    * (chi2**6 + (-47 * chi2**4 * chieff**2 + (-40 * chi2**2 * chieff**4 \
    + -16 * chieff**6))) + (8 * q**5 * (5 * chi2**4 * chieff**2 + (-30 * \
    chi2**2 * chieff**4 + -12 * chieff**6)) + -8 * q**3 * (11 * chi2**2 * \
    chieff**4 + 40 * chieff**6))))))))))))))))))))

    return np.stack([coeff5, coeff4, coeff3, coeff2, coeff1, coeff0])


def kappalimits_geometrical(r , q, chi1, chi2):

    r = np.atleast_1d(r)
    q = np.atleast_1d(q)
    chi1 = np.atleast_1d(chi1)
    chi2 = np.atleast_1d(chi2)
    
    kappamin= q*r**(1/2)/(2*(1+q)**2)*(
         np.maximum.reduce([np.zeros(q.shape), 
            squarewithsign( 1- (chi1+chi2*q**2) / (q*r**(1/2))),
            squarewithsign( np.abs(chi1-chi2*q**2) / (q*r**(1/2)) - 1 )] )
         -1)

    kappamax = (chi1+chi2*q**2) / (1+q)**2 * ( (chi1+chi2*q**2) / (2*q*r**(1/2)) +1 )


    return np.stack([kappamin,kappamax])


def kapparesonances(r, chieff, q, chi1, chi2,tol=1e-4):
    """
    Regularized angular momentum of the two spin-orbit resonances. The resonances minimizes and maximizes kappa for a given value of chieff. The minimum corresponds to deltaphi=pi and the maximum corresponds to deltaphi=0.
    Call
    ----
    kappamin,kappamax = kapparesonances(u,chieff,q,chi1,chi2)
    Parameters
    ----------
    u: float
        Compactified separation 1/(2L).
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    tol: FIX ME
    Returns
    -------
    kappamin: float
        Minimum value of the regularized angular momentum kappa.
    kappamax: float
        Maximum value of the regularized angular momentum kappa.
    """

    chieff = np.atleast_1d(chieff)
    q = np.atleast_1d(q)
    chi1 = np.atleast_1d(chi1)
    chi2 = np.atleast_1d(chi2)

    u = eval_u(r,q)

    #kapparoots = wraproots(kappadiscriminant_coefficients, u, chieff, q, chi1, chi2)
    
    coeffs = kappadiscriminant_coefficients(u, chieff, q, chi1, chi2)
    kapparootscomplex = np.sort_complex(roots_vec(coeffs.T))
    #sols = np.real(np.where(np.isreal(sols), sols, np.nan))


    # There are in principle five solutions, but only two are physical.
    def _compute(kapparootscomplex, u, chieff, q, chi1, chi2):
        kappares=None

        # At infinitely large separations the resonances are analytic...
        if u==0:
            kappaminus = np.maximum((q*(1+q)*chieff - (1-q)*chi1)/(1+q)**2 , ((1+q)*chieff - q*(1-q)*chi2)/(1+q)**2)
            kappaplus = np.minimum((q*(1+q)*chieff + (1-q)*chi1)/(1+q)**2, ((1+q)*chieff + q*(1-q)*chi2)/(1+q)**2)
            kappares = np.array([kappaminus,kappaplus])
            return kappares

        kapparoots = np.real(kapparootscomplex[np.isreal(kapparootscomplex)])

        upup,updown,downup,downdown=eval_chieff(theta1=[0,0,np.pi,np.pi], theta2=[0,np.pi,0,np.pi], q=np.repeat(q,4), chi1=np.repeat(chi1,4), chi2=np.repeat(chi2,4))

        # If too close to perfect alignment, return the analytical result.
        if np.isclose(np.repeat(chieff,2),np.squeeze([upup,downdown])).any():
            warnings.warn("Close to either up-up or down-down configuration. Using analytical results.", Warning)

            S1,S2 = spinmags(q,chi1,chi2)
            L=1/(2*u)
            kappar = ((L+np.sign(chieff)*(S1+S2))**2 - L**2) / (2*L)
            kappares=np.squeeze([kappar,kappar])



        # In this case, the spurious solution is always the smaller one. Just leave it out.
        elif len(kapparoots)==3:
            kappares = kapparoots[1:]

        # Here we have two candidate pairs of resonances...
        elif len(kapparoots)==5:
            # Compute the corresponding values of deltachi at the resonances
            deltachires = deltachiresonance(kappa=kapparoots, u=tiler(u,kapparoots), chieff=tiler(chieff,kapparoots), q=tiler(q,kapparoots), chi1=tiler(chi1,kapparoots), chi2=tiler(chi2,kapparoots))
            # Check which of those values is within the allowed region
            deltachimin,deltachimax = deltachilimits_rectangle(chieff, q, chi1, chi2)
            check = np.squeeze(np.logical_and(deltachires>deltachimin,deltachires<deltachimax))
            # The first root cannot possibly be right
            if check[0]:
                raise ValueError("Input values are not compatible [kapparesonances].")
            elif check[1] and check[2] and not check[3] and not check[4]:
                kappares = kapparoots[1:3]
            elif not check[1] and not check[2] and check[3] and check[4]:
                kappares = kapparoots[3:5]
            elif check[1] and check[2] and check[3] and check[4]:
                warnings.warn("Unphysical resonances detected and removed", Warning)
                err = np.abs( (kapparoots[2]-kapparoots[3])/np.mean(kapparoots[2:4]))
                warnings.warn("Unphysical resonances detected and removed. Relative accuracy Delta_kappa/kappa="+str(err)+", [kapparesonances].", Warning)
                kappares=np.array([kapparoots[1],kapparoots[4]])

        # This is an edge (and hopefully rare) case where the resonances are missed because of numerical errors
        elif len(kapparoots)==1:
            warnings.warn("Resonances not detected, best guess returned", Warning)
            # 5 complex solutions correspond to 3 real parts (i.e. thare are two conjugate pairs)
            kapparoots = np.unique(np.real(kapparootscomplex))
            deltachires = deltachiresonance(kappa=kapparoots, u=tiler(u,kapparoots), chieff=tiler(chieff,kapparoots), q=tiler(q,kapparoots), chi1=tiler(chi1,kapparoots), chi2=tiler(chi2,kapparoots))

            deltachimin,deltachimax = deltachilimits_rectangle(chieff, q, chi1, chi2)
            check = np.squeeze(np.logical_and(deltachires>deltachimin,deltachires<deltachimax))
            if not check[2]:
                raise ValueError("Input values are not compatible [kapparesonances].")
            elif check[0] and check[1]:
                raise ValueError("Input values are not compatible [kapparesonances].")
            elif check[1] and check[2]:
                kappares = kapparoots[1:3]
            elif check[0] and check[2]:
                kappares = kapparoots[0:2]

            # # Find values of kappa that are within the two candidate intervals. A simple average is enough.
            # avs = np.array([np.mean(kapparoots[1:3]),np.mean(kapparoots[3:5])])
            # # Find the corresponding deltachi limits
            # deltachiminus, deltachiplus, _ = wraproots(deltachicubic_coefficients, avs, [u,u], [chieff,chieff], [q,q], [chi1,chi1], [chi2,chi2]).T
            # # Find deltachi values in those intervals. Again an average is enough
            # avs = np.mean([deltachiminus, deltachiplus],axis=0)
            # # Check which of those values is within the allowed region
            # deltachimin,deltachimax = deltachilimits_rectangle(chieff, q, chi1, chi2)
            # check = np.squeeze(np.logical_and(avs>deltachimin,avs<deltachimax))
            # #
            # #print(check, q, kapparoots,avs,deltachimin,deltachimax)
            # if check[0] and not check[1]:
            #     kappares = kapparoots[1:3]
            # elif check[1] and not check[0]:
            #     kappares = kapparoots[3:5]
            #
            # # The code finds solutions, but the 3rd and the 4th are so close that they are likely to be a numerical fluke. Remove them.
            # elif np.abs( (kapparoots[2]-kapparoots[3])/np.mean(kapparoots[2:4])) <tol:
            #         warnings.warn("Unphysical resonances detected and removed", Warning)
            #         kappares=np.array([kapparoots[1],kapparoots[4]])
            #
            # else:
            #     print(eval_r(u=u,q=q), chieff, q, chi1, chi2)
            #     raise ValueError("Input values are not compatible [kapparesonances].")

            #TODO: Implement a tolerance?
            #warnings.warn("There are additional resonances but I believe are spurious and I removed them", Warning)
        # if len(kappares) > 2:
        #
        #     #print('xxxxx', kappares, np.diff(kappares),np.argmin(np.diff(kappares)),)
        #     diff = np.diff(kappares)
        #     if np.min(diff)<tol:
        #         warnings.warn("There are additional resonances but I believe are spurious and I removed them", Warning)
        #         kappares = np.delete(kappares, [np.argmin(diff),np.argmin(diff)+1])
        # We know there are always two resonances

            #upup,downdown = chiefflimits_definition(q,chi1,chi2)
            #updown,downup = deltachilimits_definition(q,chi1,chi2)
            #print(chieff,upup,downdown,updown,downup)
            #print(np.isclose(np.repeat(chieff,4),np.squeeze([upup,downdown,updown,downup])))


            #kappares2 = _compute(kapparootscomplex, u, chieff-tol/2, q, chi1, chi2)
            #print("k1", kappares1)
            #print("k2", kappares2)
        #     warnings.warn("Close to aligned configuration. Cleaning with tolerance DeltaImag(kappa)="+str(tol), Warning)
        #     kapparoots = np.real(kapparootscomplex)[np.abs(np.imag(kapparootscomplex))<tol]
        #     print("UD")
        #     print(kapparoots)


        # if len(kapparoots)==3:
        #     kappares = kapparoots[1:3]
        # if len(kapparoots)==5:
        #     kappares=np.array([kapparoots[1],kapparoots[4]])


        # Up-down and down-up are challenging. 
        # Evaluate the resonances outside of those points and interpolate linearly.
        # Note usage of recursive functions.
        elif np.isclose(np.repeat(chieff,2),np.squeeze([updown,downup])).any():
            warnings.warn("Close to either up-down or down-up configuration. Using recursive approach (tol="+str(tol)+") and analytical results.", Warning)
            chieff1 = max(min(chieff+tol/2,upup),downdown)
            coeffs = kappadiscriminant_coefficients(u, chieff1, q, chi1, chi2)
            kapparootscomplex = np.sort_complex(roots_vec(coeffs.T))
            kappares1 = _compute(kapparootscomplex, u, chieff1, q, chi1, chi2)
            chieff2 = max(min(chieff-tol/2,upup),downdown)
            coeffs = kappadiscriminant_coefficients(u, chieff2, q, chi1, chi2)
            kapparootscomplex = np.sort_complex(roots_vec(coeffs.T))
            kappares2 = _compute(kapparootscomplex, u, chieff2, q, chi1, chi2)
 
            kappares = np.mean([kappares1,kappares2],axis=0)

        # For stable configurations, we know some resonances analytically. 
        # Use those instead of the interpolated results above.
        rudplus = rupdown(q, chi1, chi2)[0]
        if np.isclose(chieff,updown) and u<eval_u(r=rudplus,q=q):
            S1,S2 = spinmags(q,chi1,chi2)
            L=1/(2*u)
            kappares[1]= ((L+S1-S2)**2 - L**2) / (2*L)
        if np.isclose(chieff,downup):
            S1,S2 = spinmags(q,chi1,chi2)
            L=1/(2*u)
            kappares[0]= ((L-S1+S2)**2 - L**2) / (2*L)

        if kappares is None:
            raise ValueError("Input values are not compatible [kapparesonances].")

        # If you didn't find enough solutions, append nans
        #kappares = np.concatenate([kappares, np.repeat(np.nan, 2-len(kappares))])
        #print(chieff, kappares)
        return kappares

    kappamin, kappamax = np.array(list(map(_compute, kapparootscomplex, u, chieff, q, chi1, chi2))).T

    return np.stack([kappamin, kappamax])



# Remove this
def kappainfresonances(chieff, q, chi1, chi2):
    """
    Regularized angular momentum of the two spin-orbit resonances. The resonances minimizes and maximizes kappa for a given value of chieff. The minimum corresponds to deltaphi=pi and the maximum corresponds to deltaphi=0.

    Call
    ----
    kappainfmin,kappainfmax = kappainfresonances(chieff,q,chi1,chi2)

    Parameters
    ----------
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    kappainfmin: float
        Minimum value of the asymptotic angular momentum kappainf.
    kappainfmax: float
        Maximum value of the asymptotic angular momentum kappainf.
    """

    chieff = np.atleast_1d(chieff)
    q = np.atleast_1d(q)

    S1, S2 = spinmags(q, chi1, chi2)
    kappainfmin = np.maximum((chieff - (q**-1-q)*S2)/(1+q), (chieff - (q**-1-q)*S1)/(1+q**-1))
    kappainfmax = np.minimum((chieff + (q**-1-q)*S2)/(1+q), (chieff + (q**-1-q)*S1)/(1+q**-1))

    return np.stack([kappainfmin, kappainfmax])


def kapparescaling(kappatilde, r, chieff, q, chi1, chi2):

    kappatilde = np.atleast_1d(kappatilde)
    kappaminus, kappaplus = kapparesonances(r, chieff, q, chi1, chi2)
    kappa = inverseaffine(kappatilde,kappaminus,kappaplus)
    return kappa


# TODO: rewrite, needs to become kappalimits
def Jlimits(r=None, chieff=None, q=None, chi1=None, chi2=None, enforce=False):
    """
    Limits on the magnitude of the total angular momentum. The contraints considered depend on the inputs provided.
    - If r, q, chi1, and chi2 are provided, the limits are given by J=L+S1+S2.
    - If r, chieff, q, chi1, and chi2 are provided, the limits are given by the two spin-orbit resonances.
    The boolean flag enforce allows raising an error in case the inputs are not compatible.

    Call
    ----
    Jmin,Jmax = Jlimits(r=None,chieff=None,q=None,chi1=None,chi2=None,enforce=False)

    Parameters
    ----------
    r: float, optional (default: None)
        Binary separation.
    chieff: float, optional (default: None)
        Effective spin.
    q: float, optional (default: None)
        Mass ratio: 0<=q<=1.
    chi1: float, optional (default: None)
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float, optional (default: None)
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    enforce: boolean, optional (default: False)
        If True raise errors, if False raise warnings.

    Returns
    -------
    Jmin: float
        Minimum value of the total angular momentum J.
    Jmax: float
        Maximum value of the total angular momentum J.
    """

    if r is not None and chieff is None and q is not None and chi1 is not None and chi2 is not None:
        Jmin, Jmax = Jlimits_LS1S2(r, q, chi1, chi2)

    elif r is not None and chieff is not None and q is not None and chi1 is not None and chi2 is not None:
        Jmin, Jmax = Jresonances(r, chieff, q, chi1, chi2)
        # Check precondition
        Jmin_cond, Jmax_cond = Jlimits_LS1S2(r, q, chi1, chi2)

        if (Jmin > Jmin_cond).all() and (Jmax < Jmax_cond).all():
            pass
        else:
            if enforce:
                raise ValueError("Input values are not compatible [Jlimits].")
            else:
                warnings.warn("Input values are not compatible [Jlimits].", Warning)

    else:
        raise TypeError("Provide either (r,q,chi1,chi2) or (r,chieff,q,chi1,chi2).")

    return np.stack([Jmin, Jmax])

# TODO: check when rewriting the large separation limit
def kappainflimits(chieff=None, q=None, chi1=None, chi2=None, enforce=False):
    """
    Limits on the asymptotic angular momentum. The contraints considered depend on the inputs provided.
    - If r, q, chi1, and chi2 are provided, the limits are given by kappa=S1+S2.
    - If r, chieff, q, chi1, and chi2 are provided, the limits are given by the two spin-orbit resonances.
    The boolean flag enforce allows raising an error in case the inputs are not compatible.

    Call
    ----
        kappainfmin,kappainfmin = kappainflimits(r=None,chieff=None,q=None,chi1=None,chi2=None,enforce=False)

    Parameters
    ----------
    r: float, optional (default: None)
        Binary separation.
    chieff: float, optional (default: None)
        Effective spin.
    q: float, optional (default: None)
        Mass ratio: 0<=q<=1.
    chi1: float, optional (default: None)
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float, optional (default: None)
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    enforce: boolean, optional (default: False)
        If True raise errors, if False raise warnings.

    Returns
    -------
    kappainfmin: float
        Minimum value of the asymptotic angular momentum kappainf.
    kappainfmin: float
        Minimum value of the asymptotic angular momentum kappainf.
    """

    if chieff is None and q is not None and chi1 is not None and chi2 is not None:
        kappainflim = Slimits_S1S2(q, chi1, chi2)[1]
        kappainfmin, kappainfmax = -kappainflim, kappainflim



    elif chieff is not None and q is not None and chi1 is not None and chi2 is not None:
        kappainfmin, kappainfmax = kappainfresonances(chieff, q, chi1, chi2)
        # Check precondition
        kappainflim = Slimits_S1S2(q, chi1, chi2)[1]
        kappainfmin_cond, kappainfmax_cond = -kappainflim, kappainflim

        if (kappainfmin > kappainfmin_cond).all() and (kappainfmax < kappainfmax_cond).all():
            pass
        else:
            if enforce:
                raise ValueError("Input values are not compatible [kappainflimits].")
            else:
                warnings.warn("Input values are not compatible [kappainflimits].", Warning)

    else:
        raise TypeError("Provide either (q,chi1,chi2) or (chieff,q,chi1,chi2).")

    return np.stack([kappainfmin, kappainfmax])


def chiefflimits_definition(q, chi1, chi2):
    """
    Limits on the effective spin based only on the definition chieff = (1+q)S1.L + (1+1/q)S2.L.

    Call
    ----
    chieffmin,chieffmax = chiefflimits_definition(q,chi1,chi2)

    Parameters
    ----------
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    chieffmin: float
        Minimum value of the effective spin chieff.
    chieffmax: float
        Maximum value of the effective spin chieff.
    """

    q = np.atleast_1d(q)
    chi1 = np.atleast_1d(chi1)
    chi2 = np.atleast_1d(chi2)
    chiefflim = (chi1+q*chi2)/(1+q)

    return np.stack([-chiefflim, chiefflim])



def deltachilimits_definition(q, chi1, chi2):
    """
    Limits on the effective spin based only on the definition chieff = (1+q)S1.L + (1+1/q)S2.L.

    Call
    ----
    chieffmin,chieffmax = chiefflimits_definition(q,chi1,chi2)

    Parameters
    ----------
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    chieffmin: float
        Minimum value of the effective spin chieff.
    chieffmax: float
        Maximum value of the effective spin chieff.
    """

    q = np.atleast_1d(q)
    chi1 = np.atleast_1d(chi1)
    chi2 = np.atleast_1d(chi2)
    deltachilim = np.abs((chi1-q*chi2)/(1+q))

    return np.stack([-deltachilim, deltachilim])



# Change. This needs to be kappa. All the Js should disappear and be left to the use if they want.
# Change. Only allow fixing chieff, not J or kappa
def anglesresonances(J=None, r=None, chieff=None, q=None, chi1=None, chi2=None):
    """
    Compute the values of the angles corresponding to the two spin-orbit resonances. Provide either J or chieff, not both.

    Call
    ----
    theta1atmin,theta2atmin,deltaphiatmin,theta1atmax,theta2atmax,deltaphiatmax = anglesresonances(J=None,r=None,chieff=None,q=None,chi1=None,chi2=None)

    Parameters
    ----------
    J: float, optional (default: None)
        Magnitude of the total angular momentum.
    r: float, optional (default: None)
        Binary separation.
    chieff: float, optional (default: None)
        Effective spin.
    q: float, optional (default: None)
        Mass ratio: 0<=q<=1.
    chi1: float, optional (default: None)
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float, optional (default: None)
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    theta1atmin: float
        Value of the angle theta1 at the resonance that minimizes either J or chieff, depending on the input.
    theta2atmin: float
        Value of the angle theta2 at the resonance that minimizes either J or chieff, depending on the input.
    deltaphiatmin: float
        Value of the angle deltaphi at the resonance that minimizes either J or chieff, depending on the input.
    theta1atmax: float
        Value of the angle theta1 at the resonance that maximizes either J or chieff, depending on the input.
    theta2atmax: float
        Value of the angle theta2 at the resonance that maximizes either J or chieff, depending on the input.
    deltaphiatmax: float
        Value of the angle deltaphi at the resonance that maximizes either J or chieff, depending on the input.
    """

    q = np.atleast_1d(q)

    if J is None and r is not None and chieff is not None and q is not None and chi1 is not None and chi2 is not None:

        Jmin, Jmax = Jresonances(r, chieff, q, chi1, chi2)
        #sys.exit()
        Satmin = Satresonance(J=Jmin, r=r, chieff=chieff, q=q, chi1=chi1, chi2=chi2)
        theta1atmin = eval_theta1(Satmin, Jmin, r, chieff, q, chi1, chi2)
        theta2atmin = eval_theta2(Satmin, Jmin, r, chieff, q, chi1, chi2)
        deltaphiatmin = tilere(np.pi, q)

        Satmax = Satresonance(J=Jmax, r=r, chieff=chieff, q=q, chi1=chi1, chi2=chi2)
        theta1atmax = eval_theta1(Satmax, Jmax, r, chieff, q, chi1, chi2)
        theta2atmax = eval_theta2(Satmax, Jmax, r, chieff, q, chi1, chi2)
        deltaphiatmax = tiler(0, q)

    elif J is not None and r is not None and chieff is None and q is not None and chi1 is not None and chi2 is not None:

        chieffmin, chieffmax = chieffresonances(J, r, q, chi1, chi2)

        Satmin = Satresonance(J=J, r=r, chieff=chieffmin, q=q, chi1=chi1, chi2=chi2)
        theta1atmin = eval_theta1(Satmin, J, r, chieffmin, q, chi1, chi2)
        theta2atmin = eval_theta2(Satmin, J, r, chieffmin, q, chi1, chi2)
        # See Fig 5 in arxiv:1506.03492
        J = np.atleast_1d(J)
        S1, S2 = spinmags(q, chi1, chi2)
        L = eval_L(r, q)
        deltaphiatmin = np.where(J > np.abs(L-S1-S2), 0, np.pi)

        Satmax = Satresonance(J=J, r=r, chieff=chieffmax, q=q, chi1=chi1, chi2=chi2)
        theta1atmax = eval_theta1(Satmax, J, r, chieffmax, q, chi1, chi2)
        theta2atmax = eval_theta2(Satmax, J, r, chieffmax, q, chi1, chi2)
        deltaphiatmax = tiler(np.pi, q)

    else:
        raise TypeError("Provide either (r,chieff,q,chi1,chi2) or (J,r,q,chi1,chi2).")

    return np.stack([theta1atmin, theta2atmin, deltaphiatmin, theta1atmax, theta2atmax, deltaphiatmax])


# I think this is not necessary anymore, because chieffresonances is gone
def chiefflimits(J=None, r=None, q=None, chi1=None, chi2=None, enforce=False):
    """
    Limits on the projected effective spin. The contraints considered depend on the inputs provided.
    - If q, chi1, and chi2 are provided, enforce chieff = (1+q)S1.L + (1+1/q)S2.L.
    - If J, r, q, chi1, and chi2 are provided, the limits are given by the two spin-orbit resonances.
    The boolean flag enforce allows raising an error in case the inputs are not compatible.

    Call
    ----
    chieffmin,chieffmax = chiefflimits(J=None,r=None,q=None,chi1=None,chi2=None,enforce=False)

    Parametersell
    ----------
    J: float, optional (default: None)
        Magnitude of the total angular momentum.
    r: float, optional (default: None)
        Binary separation.
    q: float, optional (default: None)
        Mass ratio: 0<=q<=1.
    chi1: float, optional (default: None)
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float, optional (default: None)
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    enforce: boolean, optional (default: False)
        If True raise errors, if False raise warnings.

    Returns
    -------
    chieffmin: float
        Minimum value of the effective spin chieff.
    chieffmax: float
        Maximum value of the effective spin chieff.
    """

    if J is None and r is None and q is not None and chi1 is not None and chi2 is not None:
        chieffmin, chieffmax = chiefflimits_definition(q, chi1, chi2)

    elif J is not None and r is not None and q is not None and chi1 is not None and chi2 is not None:
        chieffmin, chieffmax = chieffresonances(J, r, q, chi1, chi2)
        # Check precondition
        chieffmin_cond, chieffmax_cond = chiefflimits_definition(q, chi1, chi2)
        if (chieffmin > chieffmin_cond).all() and (chieffmax < chieffmax_cond).all():
            pass
        else:
            if enforce:
                raise ValueError("Input values are not compatible [chiefflimits].")
            else:
                warnings.warn("Input values are not compatible [chiefflimits].", Warning)

    else:
        raise TypeError("Provide either (q,chi1,chi2) or (J,r,q,chi1,chi2).")

    return np.stack([chieffmin, chieffmax])


def Slimits_S1S2(q, chi1, chi2):
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
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    Smin: float
        Minimum value of the total spin S.
    Smax: float
        Maximum value of the total spin S.
    """

    S1, S2 = spinmags(q, chi1, chi2)
    Smin = np.abs(S1-S2)
    Smax = S1+S2

    return np.stack([Smin, Smax])


def Slimits_LJ(J, r, q):
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

    L = eval_L(r, q)
    Smin = np.abs(J-L)
    Smax = J+L

    return np.stack([Smin, Smax])


def Slimits_LJS1S2(J, r, q, chi1, chi2):
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
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    Smin: float
        Minimum value of the total spin S.
    Smax: float
        Maximum value of the total spin S.
    """

    SminS1S2, SmaxS1S2 = Slimits_S1S2(q, chi1, chi2)
    SminLJ, SmaxLJ = Slimits_LJ(J, r, q)
    Smin = np.maximum(SminS1S2, SminLJ)
    Smax = np.minimum(SmaxS1S2, SmaxLJ)

    return np.stack([Smin, Smax])


def Scubic_coefficients(kappa, u, chieff, q, chi1, chi2):
    """
    Coefficients of the cubic equation in S^2 that identifies the effective potentials.

    Call
    ----
    coeff3,coeff2,coeff1,coeff0 = Scubic_coefficients(kappa,u,chieff,q,chi1,chi2)

    Parameters
    ----------
    kappa: float
        Regularized angular momentum (J^2-L^2)/(2L).
    u: float
        Compactified separation 1/(2L).
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
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

    kappa = np.atleast_1d(kappa)
    u = np.atleast_1d(u)
    chieff = np.atleast_1d(chieff)
    q = np.atleast_1d(q)
    S1, S2 = spinmags(q, chi1, chi2)

    # Machine generated with polycoefficients.nb
    coeff3 = q * ((1 + q))**2 * u**2

    # Machine generated with polycoefficients.nb
    coeff2 = 1/4 * ((1 + q))**2 * (1 + (4 * S2**2 * u**2 + (q**2 * (1 + 4
    * S1**2 * u**2) + -2 * q * (1 + 2 * u * ((S1**2 + S2**2) * u + (2 *
    kappa + -1 * chieff))))))

    # Machine generated with polycoefficients.nb
    coeff1 = (q * (kappa + (q * kappa + -1 * chieff)) * (kappa + (q *
    kappa + -1 * q * chieff)) + (-1/2 * (-1 + q**2) * S1**2 * ((1 + q) *
    (-1 + (q + 4 * q * u * kappa)) + -2 * q * u * chieff) + -1/2 * (-1 +
    q**2) * S2**2 * ((1 + q) * (-1 + (q + -4 * u * kappa)) + 2 * q * u *
    chieff)))

    # Machine generated with polycoefficients.nb
    coeff0 = 1/4 * (-1 + q) * (1 + q) * ((1 + q) * ((-1 + q) * ((S1**2 +
    -1 * S2**2))**2 + 4 * (q * S1**2 + -1 * S2**2) * kappa**2) + -4 * q *
    (S1 + -1 * S2) * (S1 + S2) * kappa * chieff)

    return np.stack([coeff3, coeff2, coeff1, coeff0])


# TODO: precomputedroots is not implemented consistently. Check that all functions that can use it have the option to do it
def Ssroots(J, r, chieff, q, chi1, chi2, precomputedroots=None):
    """
    Roots of the cubic equation in S^2 that identifies the effective potentials.

    Call
    ----
    Sminuss,Spluss,S3s = Ssroots(J,r,chieff,q,chi1,chi2,precomputedroots=None)

    Parameters
    ----------
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    precomputedroots: array, optional (default: None)
        Pre-computed output of Ssroots for computational efficiency.

    Returns
    -------
    Sminuss: float
        Lowest physical root, if present, of the effective potential equation.
    Spluss: float
        Largest physical root, if present, of the effective potential equation.
    S3s: float
        Spurious root of the effective potential equation.
    """

    if precomputedroots is None:

        kappa = eval_kappa(J, r, q)
        u = eval_u(r, q)
        S3s, Sminuss, Spluss = wraproots(Scubic_coefficients, kappa, u, chieff, q, chi1, chi2).T

        return np.stack([Sminuss, Spluss, S3s])

    else:
        precomputedroots=np.array(precomputedroots)
        assert precomputedroots.shape[0] == 3, "Shape of precomputedroots must be (3,N), i.e. Sminuss, Spluss, S3s. [Ssroots]"
        return precomputedroots


def Slimits_plusminus(J, r, chieff, q, chi1, chi2, precomputedroots=None):
    """
    Limits on the total spin magnitude compatible with both J and chieff.

    Call
    ----
    Smin,Smax = Slimits_plusminus(J,r,chieff,q,chi1,chi2,precomputedroots=None)

    Parameters
    ----------
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    precomputedroots: array, optional (default: None)
        Pre-computed output of Ssroots for computational efficiency.

    Returns
    -------
    Smin: float
        Minimum value of the total spin S.
    Smax: float
        Maximum value of the total spin S.
    """

    Sminuss, Spluss, _ = Ssroots(J, r, chieff, q, chi1, chi2, precomputedroots=precomputedroots)
    with np.errstate(invalid='ignore'):
        Smin = Sminuss**0.5
        Smax = Spluss**0.5

    return np.stack([Smin, Smax])


def Satresonance(J=None, kappa=None, r=None, u=None, chieff=None, q=None, chi1=None, chi2=None):
    """
    Assuming that the inputs correspond to a spin-orbit resonance, find the corresponding value of S. There will be two roots that are conincident if not for numerical errors: for concreteness, return the mean of the real part. This function does not check that the input is a resonance; it is up to the user. Provide either J or kappa and either r or u.

    Call
    ----
    S = Satresonance(J=None,kappa=None,r=None,u=None,chieff=None,q=None,chi1=None,chi2=None)

    Parameters
    ----------
    J: float, optional (default: None)
        Magnitude of the total angular momentum.
    kappa: float, optional (default: None)
        Regularized angular momentum (J^2-L^2)/(2L).
    r: float, optional (default: None)
        Binary separation.
    u: float, optional (default: None)
        Compactified separation 1/(2L).
    chieff: float, optional (default: None)
        Effective spin.
    q: float, optional (default: None)
        Mass ratio: 0<=q<=1.
    chi1: float, optional (default: None)
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float, optional (default: None)
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    S: float
        Magnitude of the total spin.
    """

    if q is None or chi1 is None or chi2 is None:
        raise TypeError("Please provide q, chi1, and chi2.")

    if r is None and u is None:
        raise TypeError("Please provide either r or u.")
    elif r is not None and u is None:
        u = eval_u(r=r, q=q)

    if J is None and kappa is not None:
        pass  # I don't need J
    elif J is not None and kappa is None:
        if r is None and u is not None:
            r = eval_r(u=u, q=q)
        kappa = eval_kappa(J, r, q)
    else:
        raise TypeError("Please provide either J or kappa.")

    #print('Satres', kappa)
    coeffs = Scubic_coefficients(kappa, u, chieff, q, chi1, chi2)
    with np.errstate(invalid='ignore'):  # nan is ok here
        # This is with a simple for loop
        # Sres = np.array([np.mean(np.real(np.sort_complex(np.roots(x))[1:]))**0.5 for x in coeffs.T])
        Sres = np.mean(np.real(np.sort_complex(roots_vec(coeffs.T))[:, 1:])**0.5, axis=1)

    return Sres


def Slimits(J=None, r=None, chieff=None, q=None, chi1=None, chi2=None, enforce=False, precomputedroots=None):
    """
    Limits on the total spin magnitude. The contraints considered depend on the inputs provided.
    - If q, chi1, and chi2 are provided, the limits are given by S=S1+S2.
    - If J, r, and q are provided, the limits are given by S=J-L.
    - If J, r, q, chi1, and chi2 are provided, the limits are given by S=S1+S2 and S=J-L.
    - If J, r, chieff, q, chi1, and chi2 are provided, compute solve the cubic equation of the effective potentials (Sminus and Splus).
    The boolean flag enforce allows raising an error in case the inputs are not compatible.

    Call
    ----
    Smin,Smax = Slimits(J=None,r=None,chieff=None,q=None,chi1=None,chi2=None,enforce=False,precomputedroots=None)

    Parameters
    ----------
    J: float, optional (default: None)
        Magnitude of the total angular momentum.
    r: float, optional (default: None)
        Binary separation.
    chieff: float, optional (default: None)
        Effective spin.
    q: float, optional (default: None)
        Mass ratio: 0<=q<=1.
    chi1: float, optional (default: None)
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float, optional (default: None)
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    enforce: boolean, optional (default: False)
        If True raise errors, if False raise warnings.
    precomputedroots: array, optional (default: None)
        Pre-computed output of Ssroots for computational efficiency.

    Returns
    -------
    Smin: float
        Minimum value of the total spin S.
    Smax: float
        Maximum value of the total spin S.
    """

    if J is None and r is None and chieff is None and q is not None and chi1 is not None and chi2 is not None:
        Smin, Smax = Slimits_S1S2(q, chi1, chi2)

    elif J is not None and r is not None and chieff is None and q is not None and chi1 is None and chi2 is None:
        Smin, Smax = Slimits_LJ(J, r, q)

    elif J is not None and r is not None and chieff is None and q is not None and chi1 is not None and chi2 is not None:
        Smin, Smax = Slimits_LJS1S2(J, r, q, chi1, chi2)

    elif J is not None and r is not None and chieff is not None and q is not None and chi1 is not None and chi2 is not None:
        # Compute limits
        Smin, Smax = Slimits_plusminus(J, r, chieff, q, chi1, chi2, precomputedroots=precomputedroots)
        # Check precondition
        Smin_cond, Smax_cond = Slimits_LJS1S2(J, r, q, chi1, chi2)
        if (Smin > Smin_cond).all() and (Smax < Smax_cond).all():
            pass
        else:
            if enforce:
                raise ValueError("Input values are not compatible [Slimits].")
            else:
                warnings.warn("Input values are not compatible [Slimits].", Warning)

    else:
        raise TypeError("Provide one of the following: (q,chi1,chi2), (J,r,q), (J,r,q,chi1,chi2), (J,r,chieff,q,chi1,chi2).")

    return np.stack([Smin, Smax])


def deltachicubic_coefficients(kappa, u, chieff, q, chi1, chi2):
    kappa = np.atleast_1d(kappa).astype(float)
    u = np.atleast_1d(u).astype(float)
    chieff = np.atleast_1d(chieff).astype(float)
    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)

    coeff3 = u*(1-q)

    # Machine generated with eq_generator.nb
    coeff2 = -1/2 * q**(-1) * ((1 + q))**(-3) * (1 + (q**6 + (q**2 * (-1 \
    + (16 * u * kappa + (4 * u**2 * chi1**2 + -8 * u * chieff))) + (q**4 \
    * (-1 + (16 * u * kappa + (4 * u**2 * chi2**2 + -8 * u * chieff))) + \
    (4 * q**3 * (-1 + (6 * u * kappa + -3 * u * chieff)) + (q * (2 + (4 * \
    u * kappa + (-4 * u**2 * chi1**2 + -2 * u * chieff))) + q**5 * (2 + \
    (4 * u * kappa + (-4 * u**2 * chi2**2 + -2 * u * chieff)))))))))

    # Machine generated with eq_generator.nb
    coeff1 = q**(-1) * ((1 + q))**(-3) * (-2 * (-1 + q) * ((1 + q))**5 * \
    kappa + ((-1 + q) * ((1 + q))**5 * chieff + (4 * q**2 * u**2 * \
    (chi1**2 + -1 * q**2 * chi2**2) * chieff + (-1 + q**2) * u * (2 * \
    chi1**2 + q * (2 * q**3 * chi2**2 + (chieff**2 + (2 * q * chieff**2 + \
    q**2 * chieff**2)))))))

    # Machine generated with eq_generator.nb
    coeff0 = 1/2 * q**(-1) * ((1 + q))**(-1) * ((1 + -2 * q * ((1 + \
    q))**(-2) * u * chieff))**2 * (-4 * u**2 * chi1**4 + (8 * q**4 * u**2 \
    * chi1**2 * chi2**2 + (-4 * q**8 * u**2 * chi2**4 + (4 * ((1 + q))**4 \
    * u * chi1**2 * (2 * kappa + -1 * chieff) + (4 * q**4 * ((1 + q))**4 \
    * u * chi2**2 * (2 * kappa + -1 * chieff) + (-4 * q * ((1 + q))**2 * \
    u**2 * chi1**2 * chieff**2 + (-4 * q**2 * ((1 + q))**2 * u**2 * \
    chi1**2 * chieff**2 + (-4 * q**4 * ((1 + q))**2 * u**2 * chi2**2 * \
    chieff**2 + (-4 * q**5 * ((1 + q))**2 * u**2 * chi2**2 * chieff**2 + \
    (2 * q * ((1 + q))**6 * u * (2 * kappa + -1 * chieff) * chieff**2 + \
    -1 * ((1 + q))**8 * ((-2 * kappa + chieff))**2)))))))))) * ((1 + \
    (q**2 + q * (2 + -2 * u * chieff))))**(-2)

    return np.stack([coeff3, coeff2, coeff1, coeff0])


def deltachicubic_rescaled_coefficients(kappa, u, chieff, q, chi1, chi2):
    kappa = np.atleast_1d(kappa).astype(float)
    u = np.atleast_1d(u).astype(float)
    chieff = np.atleast_1d(chieff).astype(float)
    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)

    coeff3 = u

    # Machine generated with eq_generator.nb
    coeff2 = -1/2 * q**(-1) * ((1 + q))**(-3) * (1 + (q**6 + (q**2 * (-1 \
    + (16 * u * kappa + (4 * u**2 * chi1**2 + -8 * u * chieff))) + (q**4 \
    * (-1 + (16 * u * kappa + (4 * u**2 * chi2**2 + -8 * u * chieff))) + \
    (4 * q**3 * (-1 + (6 * u * kappa + -3 * u * chieff)) + (q * (2 + (4 * \
    u * kappa + (-4 * u**2 * chi1**2 + -2 * u * chieff))) + q**5 * (2 + \
    (4 * u * kappa + (-4 * u**2 * chi2**2 + -2 * u * chieff)))))))))

    # Machine generated with eq_generator.nb
    coeff1 = (1 + -1 * q) * q**(-1) * ((1 + q))**(-3) * (-2 * (-1 + q) * \
    ((1 + q))**5 * kappa + ((-1 + q) * ((1 + q))**5 * chieff + (4 * q**2 \
    * u**2 * (chi1**2 + -1 * q**2 * chi2**2) * chieff + (-1 + q**2) * u * \
    (2 * chi1**2 + q * (2 * q**3 * chi2**2 + (chieff**2 + (2 * q * \
    chieff**2 + q**2 * chieff**2)))))))

    # Machine generated with eq_generator.nb
    coeff0 = 1/2 * ((-1 + q))**2 * q**(-1) * ((1 + q))**(-1) * ((1 + -2 * \
    q * ((1 + q))**(-2) * u * chieff))**2 * (-4 * u**2 * chi1**4 + (8 * \
    q**4 * u**2 * chi1**2 * chi2**2 + (-4 * q**8 * u**2 * chi2**4 + (4 * \
    ((1 + q))**4 * u * chi1**2 * (2 * kappa + -1 * chieff) + (4 * q**4 * \
    ((1 + q))**4 * u * chi2**2 * (2 * kappa + -1 * chieff) + (-4 * q * \
    ((1 + q))**2 * u**2 * chi1**2 * chieff**2 + (-4 * q**2 * ((1 + q))**2 \
    * u**2 * chi1**2 * chieff**2 + (-4 * q**4 * ((1 + q))**2 * u**2 * \
    chi2**2 * chieff**2 + (-4 * q**5 * ((1 + q))**2 * u**2 * chi2**2 * \
    chieff**2 + (2 * q * ((1 + q))**6 * u * (2 * kappa + -1 * chieff) * \
    chieff**2 + -1 * ((1 + q))**8 * ((-2 * kappa + chieff))**2)))))))))) \
    * ((1 + (q**2 + q * (2 + -2 * u * chieff))))**(-2)

    return np.stack([coeff3, coeff2, coeff1, coeff0])



# TODO: precomputedroots is not implemented consistently. Check that all functions that can use it have the option to do it
# Docstrings must be changed for kappa and deltachi everywhere
def deltachiroots(kappa, u, chieff, q, chi1, chi2, full_output=True, precomputedroots=None):
    """
    Roots of the cubic equation in S^2 that identifies the effective potentials.

    Call
    ----
    Sminuss,Spluss,S3s = Ssroots(J,r,chieff,q,chi1,chi2,precomputedroots=None)

    Parameters
    ----------
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    precomputedroots: array, optional (default: None)
        Pre-computed output of Ssroots for computational efficiency.

    Returns
    -------
    Sminuss: float
        Lowest physical root, if present, of the effective potential equation.
    Spluss: float
        Largest physical root, if present, of the effective potential equation.
    S3s: float
        Spurious root of the effective potential equation.
    """

    if precomputedroots is None:
        deltachiminus, deltachiplus, _ = wraproots(deltachicubic_coefficients, kappa, u, chieff, q, chi1, chi2).T

        # If you need the spurious root as well.
        if full_output:
            _, _, deltachi3 = wraproots(deltachicubic_rescaled_coefficients, kappa, u, chieff, q, chi1, chi2).T
        # Otherwise avoid (for computational efficiency)
        else:
            deltachi3 = tiler(np.nan,deltachiminus)

        return np.stack([deltachiminus, deltachiplus, deltachi3])

    else:
        precomputedroots=np.array(precomputedroots)
        assert precomputedroots.shape[0] == 3, "Shape of precomputedroots must be (3,N), i.e. deltachiminus, deltachiplus, deltachi3. [deltachiroots]"
        return precomputedroots



# TODO: check when rewriting the large separation limit
def deltachilimits_rectangle(chieff, q, chi1, chi2):
    """
    Limits on the asymptotic angular momentum. The contraints considered depend on the inputs provided.
    - If r, q, chi1, and chi2 are provided, the limits are given by kappa=S1+S2.
    - If r, chieff, q, chi1, and chi2 are provided, the limits are given by the two spin-orbit resonances.
    The boolean flag enforce allows raising an error in case the inputs are not compatible.

    Call
    ----
        kappainfmin,kappainfmin = kappainflimits(r=None,chieff=None,q=None,chi1=None,chi2=None,enforce=False)

    Parameters
    ----------
    r: float, optional (default: None)
        Binary separation.
    chieff: float, optional (default: None)
        Effective spin.
    q: float, optional (default: None)
        Mass ratio: 0<=q<=1.
    chi1: float, optional (default: None)
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float, optional (default: None)
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    enforce: boolean, optional (default: False)
        If True raise errors, if False raise warnings.

    Returns
    -------
    kappainfmin: float
        Minimum value of the asymptotic angular momentum kappainf.
    kappainfmin: float
        Minimum value of the asymptotic angular momentum kappainf.
    """

    chieff = np.atleast_1d(chieff)
    q = np.atleast_1d(q)
    chi1 = np.atleast_1d(chi1)
    chi2 = np.atleast_1d(chi2)


    deltachimin = np.maximum( -chieff - 2*chi1/(1+q), chieff - 2*q*chi2/(1+q))
    deltachimax = np.minimum( -chieff + 2*chi1/(1+q), chieff + 2*q*chi2/(1+q))

    return np.stack([deltachimin, deltachimax])



def deltachilimits_plusminus(kappa, r, chieff, q, chi1, chi2, precomputedroots=None):

    u = eval_u(r,q)
    deltachiminus, deltachiplus, _ = deltachiroots(kappa, u, chieff, q, chi1, chi2, full_output=False, precomputedroots=precomputedroots)

    # Correct when too close to perfect alignment
    angleup=tiler(0,q)
    angledown=tiler(np.pi,q)

    chieffupup = eval_chieff(theta1=angleup, theta2=angleup, q=q, chi1=chi1, chi2=chi2)
    deltachiupup = eval_deltachi(theta1=angleup, theta2=angleup, q=q, chi1=chi1, chi2=chi2)
    deltachiminus = np.where(np.isclose(chieff,chieffupup), deltachiupup,deltachiminus)
    deltachiplus = np.where(np.isclose(chieff,chieffupup), deltachiupup,deltachiplus)

    chieffdowndown = eval_chieff(theta1=angledown, theta2=angledown, q=q, chi1=chi1, chi2=chi2)
    deltachidowndown = eval_deltachi(theta1=angledown, theta2=angledown, q=q, chi1=chi1, chi2=chi2)
    deltachiminus = np.where(np.isclose(chieff,chieffdowndown), deltachidowndown,deltachiminus)
    deltachiplus = np.where(np.isclose(chieff,chieffdowndown), deltachidowndown,deltachiplus)

    return deltachiminus, deltachiplus




def deltachirescaling(deltachitilde, kappa, r, chieff, q, chi1, chi2,precomputedroots=None):

    deltachiminus, deltachiplus = deltachilimits_plusminus(kappa, r, chieff, q, chi1, chi2,precomputedroots=precomputedroots)
    deltachi =  inverseaffine(deltachitilde, deltachiminus, deltachiplus)

    return deltachi





def deltachiresonance(kappa=None, r=None, u=None, chieff=None, q=None, chi1=None, chi2=None):
    """
    Assuming that the inputs correspond to a spin-orbit resonance, find the corresponding value of S. There will be two roots that are conincident if not for numerical errors: for concreteness, return the mean of the real part. This function does not check that the input is a resonance; it is up to the user. Provide either J or kappa and either r or u.

    Call
    ----
    S = Satresonance(J=None,kappa=None,r=None,u=None,chieff=None,q=None,chi1=None,chi2=None)

    Parameters
    ----------
    J: float, optional (default: None)
        Magnitude of the total angular momentum.
    kappa: float, optional (default: None)
        Regularized angular momentum (J^2-L^2)/(2L).
    r: float, optional (default: None)
        Binary separation.
    u: float, optional (default: None)
        Compactified separation 1/(2L).
    chieff: float, optional (default: None)
        Effective spin.
    q: float, optional (default: None)
        Mass ratio: 0<=q<=1.
    chi1: float, optional (default: None)
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float, optional (default: None)
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    S: float
        Magnitude of the total spin.
    """

    if q is None or chi1 is None or chi2 is None or kappa is None:
        raise TypeError("Please provide q, chi1, and chi2.")

    if r is None and u is None:
        raise TypeError("Please provide either r or u.")
    elif r is not None and u is None:
        u = eval_u(r=r, q=q)

    #print('Satres', kappa)
    coeffs = deltachicubic_coefficients(kappa, u, chieff, q, chi1, chi2)
    with np.errstate(invalid='ignore'):  # nan is ok here
        # This is with a simple for loop
        deltachires = np.mean(np.real(np.sort_complex(roots_vec(coeffs.T))[:,:-1]),axis=1)
        # Sres = np.array([np.mean(np.real(np.sort_complex(np.roots(x))[1:]))**0.5 for x in coeffs.T])

        #deltachires = np.mean(np.real(np.sort_complex(roots_vec(coeffs.T))[:, 1:])**0.5, axis=1)

    return deltachires


# TODO: Check inter-compatibility of Slimits, Jlimits, chiefflimits
# TODO: check docstrings
# Tags for each limit check that fails?
# Davide: Does this function uses only Jlimits and chiefflimits or also Slimits? Move later?
def limits_check(S=None, J=None, r=None, chieff=None, q=None, chi1=None, chi2=None):
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
    chieff: float, optional
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

    # J: r, chieff, q, chi1, chi2
    # r, q, chi1, chi2 -> Jlimits_LS1S2
    # r, chieff, q, chi1, chi2 -> Jresonances

    # chieff: J, r, q, chi1, chi2
    # q, chi1, chi2 -> chiefflimits_definition
    # J, r, q, chi1, chi2 -> chieffresonances

    # S: J, r, chieff, q, chi1, chi2
    # q, chi1, chi2 -> Slimits_S1S2
    # J, r, q -> Slimits_LJ
    # J, r, q, chi1, chi2 -> Slimits_LJS1S2
    # J, r, chieff, q, chi1, chi2 -> Slimits_plusminus

    def _limits_check(testvalue, interval):
        """Check if a value is within a given interval"""
        return np.logical_and(testvalue >= interval[0], testvalue <= interval[1])

    Slim = Slimits(J, r, chieff, q, chi1, chi2)
    Sbool = _limits_check(S, Slim)

    Jlim = Jlimits(r, chieff, q, chi1, chi2)
    Jbool = _limits_check(J, Jlim)

    chiefflim = chiefflimits(J, r, q, chi1, chi2)
    chieffbool = _limits_check(chieff, chiefflim)

    check = all((Sbool, Jbool, chieffbool))

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


# Evaluations and conversions

def eval_chieff(theta1=None, theta2=None, S=None, varphi=None, J=None, r=None, q=None, chi1=None, chi2=None):
    """
    Eftective spin. Provide either (theta1,theta2,q,chi1,chi2) or (S,varphi,J,r,q,chi1,chi2).

    Call
    ----
    chieff = eval_chieff(theta1=None,theta2=None,S=None,varphi=None,J=None,r=None,q=None,chi1=None,chi2=None)

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
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float, optional (default: None)
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    chieff: float
        Effective spin.
    """

    if theta1 is not None and theta2 is not None and S is None and varphi is None and J is None and r is None and q is not None and chi1 is not None and chi2 is not None:

        theta1 = np.atleast_1d(theta1)
        theta2 = np.atleast_1d(theta2)
        q = np.atleast_1d(q)
        S1, S2 = spinmags(q, chi1, chi2)
        chieff = (1+q)*(q*S1*np.cos(theta1)+S2*np.cos(theta2))/q

    elif theta1 is None and theta2 is None and S is not None and varphi is not None and J is not None and r is not None and q is not None and chi1 is not None and chi2 is not None:

        S = np.atleast_1d(S)
        varphi = np.atleast_1d(varphi)
        J = np.atleast_1d(J)
        q = np.atleast_1d(q)
        S1, S2 = spinmags(q, chi1, chi2)
        L = eval_L(r, q)

        # Machine generated with polycoefficients.nb
        chieff = 1/4 * L**(-1) * q**(-1) * S**(-2) * ((J**2 + (-1 * L**2 + -1
        * S**2)) * (((1 + q))**2 * S**2 + (-1 + q**2) * (S1**2 + -1 * S2**2))
        + -1 * (1 + -1 * q**2) * ((J**2 + -1 * ((L + -1 * S))**2))**(1/2) *
        ((-1 * J**2 + ((L + S))**2))**(1/2) * ((S**2 + -1 * ((S1 + -1 *
        S2))**2))**(1/2) * ((-1 * S**2 + ((S1 + S2))**2))**(1/2) *
        np.cos(varphi))

    else:
        raise TypeError("Provide either (theta1,theta2,q,chi1,chi2) or (S,varphi,J,r,q,chi1,chi2).")

    return chieff


# TODO: update this function to evaluate from S
def eval_deltachi(theta1, theta2, q, chi1, chi2):
    """
    Eftective spin. Provide either (theta1,theta2,q,chi1,chi2) or (S,varphi,J,r,q,chi1,chi2).

    Call
    ----
    chieff = eval_chieff(theta1=None,theta2=None,S=None,varphi=None,J=None,r=None,q=None,chi1=None,chi2=None)

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
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float, optional (default: None)
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    chieff: float
        Effective spin.
    """


    theta1 = np.atleast_1d(theta1)
    theta2 = np.atleast_1d(theta2)
    q = np.atleast_1d(q)
    chi1 = np.atleast_1d(chi1)
    chi2 = np.atleast_1d(chi2)

    deltachi = (chi1*np.cos(theta1) -q*chi2*np.cos(theta2))/(1+q)

    return deltachi


# TODO: change S,J to deltachi, kappa in all these angle functions below!

def eval_costheta1_old(S, J, r, chieff, q, chi1, chi2):
    """
    Cosine of the angle theta1 between the orbital angular momentum and the spin of the primary black hole.

    Call
    ----
    costheta1 = eval_costheta1(S,J,r,chieff,q,chi1,chi2)

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    costheta1: float
        Cosine of the angle between orbital angular momentum and primary spin.
    """

    S = np.atleast_1d(S)
    J = np.atleast_1d(J)
    q = np.atleast_1d(q)

    S1, S2 = spinmags(q, chi1, chi2)
    L = eval_L(r, q)

    costheta1 = (((J**2-L**2-S**2)/L) - (2*q*chieff)/(1+q))/(2*(1-q)*S1)

    return costheta1

def eval_costheta1(deltachi, chieff, q, chi1):
    """
    Cosine of the angle theta1 between the orbital angular momentum and the spin of the primary black hole.

    Call
    ----
    costheta1 = eval_costheta1(S,J,r,chieff,q,chi1,chi2)

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    costheta1: float
        Cosine of the angle between orbital angular momentum and primary spin.
    """

    deltachi = np.atleast_1d(deltachi)
    chieff = np.atleast_1d(chieff)
    q = np.atleast_1d(q)
    chi1 = np.atleast_1d(chi1)

    costheta1 = (1+q)/(2*chi1)*(chieff+deltachi)

    return costheta1


def eval_theta1(deltachi, chieff, q, chi1):


    costheta1 = eval_costheta1(deltachi, chieff, q, chi1)
    theta1 = np.arccos(costheta1)

    return theta1

def eval_theta1_old(S, J, r, chieff, q, chi1, chi2):
    """
    Angle theta1 between the orbital angular momentum and the spin of the primary black hole.

    Call
    ----
    theta1 = eval_theta1(S,J,r,chieff,q,chi1,chi2)

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    theta1: float
        Angle between orbital angular momentum and primary spin.
    """

    costheta1 = eval_costheta1(S, J, r, chieff, q, chi1, chi2)
    theta1 = np.arccos(costheta1)

    return theta1


def eval_costheta2_old(S, J, r, chieff, q, chi1, chi2):
    """
    Cosine of the angle theta2 between the orbital angular momentum and the spin of the secondary black hole.

    Call
    ----
    costheta2 = eval_costheta2(S,J,r,chieff,q,chi1,chi2)

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    costheta2: float
        Cosine of the angle between orbital angular momentum and secondary spin.
    """

    S = np.atleast_1d(S)
    J = np.atleast_1d(J)
    q = np.atleast_1d(q)

    S1, S2 = spinmags(q, chi1, chi2)
    L = eval_L(r, q)

    costheta2 = (((J**2-L**2-S**2)*(-q/L)) + (2*q*chieff)/(1+q))/(2*(1-q)*S2)

    return costheta2


def eval_costheta2(deltachi, chieff, q, chi2):
    """
    Cosine of the angle theta1 between the orbital angular momentum and the spin of the primary black hole.

    Call
    ----
    costheta1 = eval_costheta1(S,J,r,chieff,q,chi1,chi2)

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    costheta1: float
        Cosine of the angle between orbital angular momentum and primary spin.
    """

    deltachi = np.atleast_1d(deltachi)
    chieff = np.atleast_1d(chieff)
    q = np.atleast_1d(q)
    chi2 = np.atleast_1d(chi2)

    costheta2 = (1+q)/(2*q*chi2)*(chieff-deltachi)

    return costheta2


def eval_theta2_old(S, J, r, chieff, q, chi1, chi2):
    """
    Angle theta2 between the orbital angular momentum and the spin of the secondary black hole.

    Call
    ----
    theta2 = eval_theta2(S,J,r,chieff,q,chi1,chi2)

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    theta2: float
        Angle between orbital angular momentum and secondary spin.
    """

    costheta2 = eval_costheta2(S, J, r, chieff, q, chi1, chi2)
    theta2 = np.arccos(costheta2)

    return theta2


def eval_theta2(deltachi, chieff, q, chi2):


    costheta2 = eval_costheta1(deltachi, chieff, q, chi2)
    theta2 = np.arccos(costheta2)

    return theta1


def eval_costheta12_old(theta1=None, theta2=None, deltaphi=None, S=None, q=None, chi1=None, chi2=None):
    """
    Cosine of the angle theta12 between the two spins. Valid inputs are either (theta1,theta2,deltaphi) or (S,q,chi1,chi2).

    Call
    ----
    costheta12 = eval_costheta12(theta1=None,theta2=None,deltaphi=None,S=None,q=None,chi1=None,chi2=None)

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
    q: float, optional (default: None)
        Mass ratio: 0<=q<=1.
    chi1: float, optional (default: None)
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float, optional (default: None)
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    costheta12: float
        Cosine of the angle between the two spins.
    """

    if theta1 is not None and theta2 is not None and deltaphi is not None and S is None and q is None and chi1 is None and chi2 is None:

        theta1=np.atleast_1d(theta1)
        theta2=np.atleast_1d(theta2)
        deltaphi=np.atleast_1d(deltaphi)
        costheta12 = np.sin(theta1)*np.sin(theta2)*np.cos(deltaphi) + np.cos(theta1)*np.cos(theta2)

    elif theta1 is None and theta2 is None and deltaphi is None and S is not None and q is not None and chi1 is not None and chi2 is not None:

        S = np.atleast_1d(S)
        S1, S2 = spinmags(q, chi1, chi2)
        costheta12 = (S**2-S1**2-S2**2)/(2*S1*S2)

    else:
        raise TypeError("Provide either (theta1,theta2,deltaphi) or (S,q,chi1,chi2).")

    return costheta12



def eval_costheta12(theta1=None, theta2=None, deltaphi=None, deltachi=None, kappa=None, chieff=None, q=None, chi1=None, chi2=None):
    """
    Cosine of the angle theta12 between the two spins. Valid inputs are either (theta1,theta2,deltaphi) or (S,q,chi1,chi2).

    Call
    ----
    costheta12 = eval_costheta12(theta1=None,theta2=None,deltaphi=None,S=None,q=None,chi1=None,chi2=None)

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
    q: float, optional (default: None)
        Mass ratio: 0<=q<=1.
    chi1: float, optional (default: None)
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float, optional (default: None)
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    costheta12: float
        Cosine of the angle between the two spins.
    """

    if theta1 is not None and theta2 is not None and deltaphi is not None and deltachi is None and kappa is None and chieff is None and q is None and chi1 is None and chi2 is None:

        theta1=np.atleast_1d(theta1)
        theta2=np.atleast_1d(theta2)
        deltaphi=np.atleast_1d(deltaphi)
        costheta12 = np.sin(theta1)*np.sin(theta2)*np.cos(deltaphi) + np.cos(theta1)*np.cos(theta2)

    elif theta1 is None and theta2 is None and deltaphi is None and deltachi is not None and kappa is not None and chieff is not None and q is not None and chi1 is not None and chi2 is not None:

        deltachi = np.atleast_1d(deltachi)
        kappa = np.atleast_1d(kappa)
        chieff = np.atleast_1d(chieff)
        q = np.atleast_1d(q)
        chi1 = np.atleast_1d(chi1)
        chi2 = np.atleast_1d(chi2)


        # Machine generated with eq_generator.nb
        costheta12 = 1/2 * q**(-2) * (chi1)**(-1) * (chi2)**(-1) * (-1 * \
        (chi1)**2 + (-1 * q**4 * (chi2)**2 + q * (1 + q) * (r)**(1/2) * (-1 * \
        (1 + -1 * q) * deltachi + (2 * (1 + q) * kappa + -1 * (1 + q) * \
        chieff))))


    else:
        raise TypeError("Provide either (theta1,theta2,deltaphi) or (S,q,chi1,chi2).")

    return costheta12




# TODO docstrings (actually, all docstrings need to be checked now!)
def eval_theta12_old(theta1=None, theta2=None, deltaphi=None, S=None, q=None, chi1=None, chi2=None):
    """
    Angle theta12 between the two spins. Valid inputs are either (theta1,theta2,deltaphi) or (S,q,chi1,chi2).

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
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    theta12: float
        Angle between the two spins.
    """

    costheta12 = eval_costheta12(theta1=theta1, theta2=theta2, deltaphi=deltaphi, S=S, q=q, chi1=chi1, chi2=chi2)
    theta12 = np.arccos(costheta12)

    return theta12



def eval_theta12(theta1=None, theta2=None, deltaphi=None, deltachi=None, kappa=None, chieff=None, q=None, chi1=None, chi2=None):


    costheta12 = eval_costheta1(theta1=theta1, theta2=theta2, deltaphi=deltaphi, deltachi=deltachi, kappa=kappa, chieff=chieff, q=q, chi1=chi1, chi2=chi2)
    theta12 = np.arccos(costheta12)

    return theta12


def eval_cosdeltaphi_old(S, J, r, chieff, q, chi1, chi2):
    """
    Cosine of the angle deltaphi between the projections of the two spins onto the orbital plane.

    Call
    ----
    cosdeltaphi = eval_cosdeltaphi(S,J,r,chieff,q,chi1,chi2)

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    cosdeltaphi: float
        Cosine of the angle between the projections of the two spins onto the orbital plane.
    """

    q = np.atleast_1d(q)

    S1, S2 = spinmags(q, chi1, chi2)
    costheta1 = eval_costheta1_old(S, J, r, chieff, q, chi1, chi2)
    costheta2 = eval_costheta2_old(S, J, r, chieff, q, chi1, chi2)
    costheta12 = eval_costheta12_old(S=S, q=q, chi1=chi1, chi2=chi2)

    cosdeltaphi = (costheta12 - costheta1*costheta2)/((1-costheta1**2)*(1-costheta2**2))**0.5

    return cosdeltaphi



def eval_cosdeltaphi_old(deltachi, kappa, chieff, q, chi1, chi2):
    """
    Cosine of the angle deltaphi between the projections of the two spins onto the orbital plane.

    Call
    ----
    cosdeltaphi = eval_cosdeltaphi(S,J,r,chieff,q,chi1,chi2)

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    cosdeltaphi: float
        Cosine of the angle between the projections of the two spins onto the orbital plane.
    """

    deltachi = np.atleast_1d(deltachi)
    kappa = np.atleast_1d(kappa)
    chieff = np.atleast_1d(chieff)
    q = np.atleast_1d(q)
    chi1 = np.atleast_1d(chi1)
    chi2 = np.atleast_1d(chi2)

    # Machine generated with eq_generator.nb
    cosdeltaphi = q**(-1) * ((4 * q**2 * (chi2)**2 + -1 * ((1 + q))**2 * \
    ((-1 * deltachi + chieff))**2) * (4 * (chi1)**2 + -1 * ((1 + q))**2 * \
    ((deltachi + chieff))**2))**(-1/2) * (-2 * ((chi1)**2 + q**4 * \
    (chi2)**2) + (2 * q * (1 + q) * (r)**(1/2) * (-1 * (1 + -1 * q) * \
    deltachi + (2 * (1 + q) * kappa + -1 * (1 + q) * chieff)) + -1 * q * \
    ((1 + q))**2 * (-1 * (deltachi)**2 + chieff**2)))

    return cosdeltaphi



def eval_deltaphi_old(S, J, r, chieff, q, chi1, chi2, cyclesign=-1):
    """
    Angle deltaphi between the projections of the two spins onto the orbital plane. By default this is returned in [0,pi]. Setting cyclesign=1 returns the other half of the  precession cycle [-pi,0].

    Call
    ----
    deltaphi = eval_deltaphi(S,J,r,chieff,q,chi1,chi2,cyclesign=-1)

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    cyclesign: integer, optional (default: -1)
        Sign (either +1 or -1) to cover the two halves of a precesion cycle.

    Returns
    -------
    deltaphi: float
        Angle between the projections of the two spins onto the orbital plane.
    """

    cyclesign = np.atleast_1d(cyclesign)
    cosdeltaphi = eval_cosdeltaphi(S, J, r, chieff, q, chi1, chi2)
    deltaphi = -np.sign(cyclesign)*np.arccos(cosdeltaphi)

    return deltaphi


# TODO cyclesign needs to be checked again in the entire code
def eval_deltaphi(deltachi, kappa, chieff, q, chi1, chi2, cyclesign=-1):
    """
    Angle deltaphi between the projections of the two spins onto the orbital plane. By default this is returned in [0,pi]. Setting cyclesign=1 returns the other half of the  precession cycle [-pi,0].

    Call
    ----
    deltaphi = eval_deltaphi(S,J,r,chieff,q,chi1,chi2,cyclesign=-1)

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    cyclesign: integer, optional (default: -1)
        Sign (either +1 or -1) to cover the two halves of a precesion cycle.

    Returns
    -------
    deltaphi: float
        Angle between the projections of the two spins onto the orbital plane.
    """

    cyclesign = np.atleast_1d(cyclesign)
    cosdeltaphi = eval_cosdeltaphi(deltachi, kappa, chieff, q, chi1, chi2)
    deltaphi = -np.sign(cyclesign)*np.arccos(cosdeltaphi)

    return deltaphi



def eval_costhetaL(S, J, r, q, chi1, chi2):
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
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    costhetaL: float
        Cosine of the angle betwen orbital angular momentum and total angular momentum.
    """

    S = np.atleast_1d(S)
    J = np.atleast_1d(J)

    S1, S2 = spinmags(q, chi1, chi2)
    L = eval_L(r, q)
    costhetaL = (J**2+L**2-S**2)/(2*J*L)

    return costhetaL


def eval_thetaL(S, J, r, q, chi1, chi2):
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
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    thetaL: float
        Angle betwen orbital angular momentum and total angular momentum.
    """

    costhetaL = eval_costhetaL(S, J, r, q, chi1, chi2)
    thetaL = np.arccos(costhetaL)

    return thetaL


def eval_J(theta1=None, theta2=None, deltaphi=None, kappa=None, r=None, q=None, chi1=None, chi2=None):
    """
    Magnitude of the total angular momentum. Provide either (theta1,theta,deltaphi,r,q,chi1,chhi2) or (kappa,r,q).

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
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float, optional (default: None)
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    J: float
        Magnitude of the total angular momentum.
    """

    if theta1 is not None and theta2 is not None and deltaphi is not None and kappa is None and r is not None and q is not None and chi1 is not None and chi2 is not None:

        theta1 = np.atleast_1d(theta1)
        theta2 = np.atleast_1d(theta2)
        deltaphi = np.atleast_1d(deltaphi)
        q = np.atleast_1d(q)

        S1, S2 = spinmags(q, chi1, chi2)
        L = eval_L(r, q)
        S = eval_S(theta1, theta2, deltaphi, q, chi1, chi2)

        J = (L**2+S**2+2*L*(S1*np.cos(theta1)+S2*np.cos(theta2)))**0.5

    elif theta1 is None and theta2 is None and deltaphi is None and kappa is not None and r is not None and q is not None and chi1 is None and chi2 is None:

        kappa = np.atleast_1d(kappa)

        L = eval_L(r, q)
        J = (2*L*kappa + L**2)**0.5

    else:
        raise TypeError("Provide either (theta1,theta2,deltaphi,r,q,chi1,chi2) or (kappa,r,q,chi1,chi2).")

    return J


# TODO: This function and the next one needs to be merged together
def eval_S_from_deltachi(deltachi, kappa, r, chieff, q):

    deltachi = np.atleast_1d(deltachi)
    kappa = np.atleast_1d(kappa)
    r = np.atleast_1d(r)
    chieff = np.atleast_1d(chieff)
    q = np.atleast_1d(q)

    S = ( q /(1+q)**2 * r**(1/2) * (2*kappa - chieff - deltachi * (1 - q)/(1 + q)) )**(1/2)

    return S


def eval_S(theta1, theta2, deltaphi, q, chi1, chi2):
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
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    S: float
        Magnitude of the total spin.
    """

    theta1 = np.atleast_1d(theta1)
    theta2 = np.atleast_1d(theta2)
    deltaphi = np.atleast_1d(deltaphi)

    S1, S2 = spinmags(q, chi1, chi2)

    S = (S1**2 + S2**2 + 2*S1*S2*(np.sin(theta1)*np.sin(theta2)*np.cos(deltaphi)+np.cos(theta1)*np.cos(theta2)))**0.5

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

    J = np.atleast_1d(J)

    L = eval_L(r, q)
    kappa = (J**2 - L**2) / (2*L)

    return kappa




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
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    kappainf: float
        Asymptotic value of the regularized momentum kappa.
    """

    theta1inf = np.atleast_1d(theta1inf)
    theta2inf = np.atleast_1d(theta2inf)

    S1, S2 = spinmags(q, chi1, chi2)
    kappainf = S1*np.cos(theta1inf) + S2*np.cos(theta2inf)

    return kappainf


def eval_costheta1inf(kappainf, chieff, q, chi1, chi2):
    """
    Infinite orbital separation limit of the cosine of the angle between the
    orbital angular momentum and the primary spin.

    Call
    ----
    costheta1inf = eval_costheta1inf(kappainf,chieff,q,chi1,chi2)

    Parameters
    ----------
    kappainf: float
        Asymptotic value of the regularized momentum kappa.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    costheta1inf: float
        Cosine of the asymptotic angle between orbital angular momentum and primary spin.
    """

    kappainf = np.atleast_1d(kappainf)
    chieff = np.atleast_1d(chieff)
    q = np.atleast_1d(q)

    S1, S2 = spinmags(q, chi1, chi2)
    costheta1inf = (-chieff + kappainf*(1+1/q)) / (S1*(1/q-q))

    return costheta1inf


def eval_theta1inf(kappainf, chieff, q, chi1, chi2):
    """
    Infinite orbital separation limit of the angle between the orbital angular
    momentum and the primary spin.

    Call
    ----
    theta1inf = eval_theta1inf(kappainf,chieff,q,chi1,chi2)

    Parameters
    ----------
    kappainf: float
        Asymptotic value of the regularized momentum kappa.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    theta1inf: float
        Asymptotic value of the angle between orbital angular momentum and primary spin.
    """

    costheta1inf = eval_costheta1inf(kappainf, chieff, q, chi1, chi2)
    theta1inf = np.arccos(costheta1inf)

    return theta1inf


def eval_costheta2inf(kappainf, chieff, q, chi1, chi2):
    """
    Infinite orbital separation limit of the cosine of the angle between the
    orbital angular momentum and the secondary spin.

    Call
    ----
    theta1inf = eval_costheta2inf(kappainf,chieff,q,chi1,chi2)

    Parameters
    ----------
    kappainf: float
        Asymptotic value of the regularized momentum kappa.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    theta1inf: float
        Asymptotic value of the angle between orbital angular momentum and primary spin.
    """

    kappainf = np.atleast_1d(kappainf)
    chieff = np.atleast_1d(chieff)
    q = np.atleast_1d(q)

    S1, S2 = spinmags(q, chi1, chi2)
    costheta2inf = (chieff - kappainf*(1+q)) / (S2*(1/q-q))

    return costheta2inf


def eval_theta2inf(kappainf, chieff, q, chi1, chi2):
    """
    Infinite orbital separation limit of the angle between the orbital angular
    momentum and the secondary spin.

    Call
    ----
    theta2inf = eval_theta2inf(kappainf,chieff,q,chi1,chi2)

    Parameters
    ----------
    kappainf: float
        Asymptotic value of the regularized momentum kappa.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    theta2inf: float
        Asymptotic value of the angle between orbital angular momentum and secondary spin.
    """

    costheta2inf = eval_costheta2inf(kappainf, chieff, q, chi1, chi2)
    theta2inf = np.arccos(costheta2inf)

    return theta2inf


#TODO regen docstrings
def morphology(J, r, chieff, q, chi1, chi2, simpler=False, precomputedroots=None):
    """
    Evaluate the spin morphology and return `L0` for librating about deltaphi=0, `Lpi` for librating about deltaphi=pi, `C-` for circulating from deltaphi=pi to deltaphi=0, and `C+` for circulating from deltaphi=0 to deltaphi=pi. If simpler=True, do not distinguish between the two circulating morphologies and return `C` for both.

    Call
    ----
    morph = morphology(J,r,chieff,q,chi1,chi2,simpler = False)

    Parameters
    ----------
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    simpler: boolean, optional (default: False)
        If True simplifies output.

    Returns
    -------
    morph: string
        Spin morphology.
    """

    Smin, Smax = Slimits_plusminus(J, r, chieff, q, chi1, chi2, precomputedroots=precomputedroots)
    # Pairs of booleans based on the values of deltaphi at S- and S+
    status = np.transpose([eval_cosdeltaphi(Smin, J, r, chieff, q, chi1, chi2) > 0, eval_cosdeltaphi(Smax, J, r, chieff, q, chi1, chi2) > 0])
    # Map to labels
    dictlabel = {(False, False): "Lpi", (True, True): "L0", (False, True): "C-", (True, False): "C+"}
    # Subsitute pairs with labels
    morphs = np.zeros(Smin.shape)
    for k, v in dictlabel.items():
        morphs = np.where((status == k).all(axis=1), v, morphs)
    # Simplifies output, only one circulating morphology
    if simpler:
        morphs = np.where(np.logical_or(morphs == 'C+', morphs == 'C-'), 'C', morphs)

    return morphs


def eval_cyclesign(dSdt=None, deltaphi=None, varphi=None, Lvec=None, S1vec=None, S2vec=None):
    """
    Evaluate if the input parameters are in the first of the second half of a precession cycle. We refer to this as the 'sign' of a precession cycle, defined as +1 if S is increasing and -1 S is decreasing. Valid inputs are one and not more of the following:
    - dSdt
    - deltaphi
    - varphi
    - Lvec, S1vec, S2vec.

    Call
    ----
    cyclesign = eval_cyclesign(dSdt=None,deltaphi=None,varphi=None,Lvec=None,S1vec=None,S2vec=None)

    Parameters
    ----------
    dSdt: float, optional (default: None)
        Time derivative of the total spin.
    deltaphi: float, optional (default: None)
        Angle between the projections of the two spins onto the orbital plane.
    varphi: float, optional (default: None)
        Generalized nutation coordinate (Eq 9 in arxiv:1506.03492).
    Lvec: array, optional (default: None)
        Cartesian vector of the orbital angular momentum.
    S1vec: array, optional (default: None)
        Cartesian vector of the primary spin.
    S2vec: array, optional (default: None)
        Cartesian vector of the secondary spin.

    Returns
    -------
    cyclesign: integer
        Sign (either +1 or -1) to cover the two halves of a precesion cycle.
    """

    if dSdt is not None and deltaphi is None and varphi is None and Lvec is None and S1vec is None and S2vec is None:
        dSdt = np.atleast_1d(dSdt)
        cyclesign = np.sign(dSdt)

    elif dSdt is None and deltaphi is not None and varphi is None and Lvec is None and S1vec is None and S2vec is None:
        deltaphi = np.atleast_1d(deltaphi)
        cyclesign = -np.sign(deltaphi)

    elif dSdt is None and deltaphi is None and varphi is not None and Lvec is None and S1vec is None and S2vec is None:
        varphi = np.atleast_1d(varphi)
        cyclesign = -np.sign(varphi)

    elif dSdt is None and deltaphi is None and varphi is None and Lvec is not None and S1vec is not None and S2vec is not None:
        Lvec = np.atleast_2d(Lvec)
        S1vec = np.atleast_2d(S1vec)
        S2vec = np.atleast_2d(S2vec)
        cyclesign = -np.sign(dot_nested(S1vec, np.cross(S2vec, Lvec)))

    else:
        TypeError("Please provide one and not more of the following: dSdt, deltaphi, (Lvec, S1vec, S2vec).")

    return cyclesign


def conserved_to_angles(S, J, r, chieff, q, chi1, chi2, cyclesign=+1):
    """
    Convert conserved quantities (S,J,chieff) into angles (theta1,theta2,deltaphi).

    Call
    ----
    theta1,theta2,deltaphi = conserved_to_angles(S,J,r,chieff,q,chi1,chi2,cyclesign=+1)

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    cyclesign: integer, optional (default: +1)
        Sign (either +1 or -1) to cover the two halves of a precesion cycle.

    Returns
    -------
    theta1: float
        Angle between orbital angular momentum and primary spin.
    theta2: float
        Angle between orbital angular momentum and secondary spin.
    deltaphi: float
        Angle between the projections of the two spins onto the orbital plane.
    """

    theta1 = eval_theta1(S, J, r, chieff, q, chi1, chi2)
    theta2 = eval_theta2(S, J, r, chieff, q, chi1, chi2)
    deltaphi = eval_deltaphi(S, J, r, chieff, q, chi1, chi2, cyclesign=cyclesign)

    return np.stack([theta1, theta2, deltaphi])


def angles_to_conserved(theta1, theta2, deltaphi, r, q, chi1, chi2, full_output=False):
    """
    Convert angles (theta1,theta2,deltaphi) into conserved quantities (S,J,chieff).

    Call
    ----
    S,J,chieff = angles_to_conserved(theta1,theta2,deltaphi,r,q,chi1,chi2,full_output=False)
    S,J,chieff,cyclesign = angles_to_conserved(theta1,theta2,deltaphi,r,q,chi1,chi2,full_output=True)

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
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    full_output: boolean, optional (default: False)
        Return additional outputs.

    Returns
    -------
    S: float
        Magnitude of the total spin.
    J: float
        Magnitude of the total angular momentum.
    chieff: float
        Effective spin.

    Other parameters
    -------
    cyclesign: integer
        Sign (either +1 or -1) to cover the two halves of a precesion cycle.
    """

    S = eval_S(theta1, theta2, deltaphi, q, chi1, chi2)
    J = eval_J(theta1=theta1, theta2=theta2, deltaphi=deltaphi, r=r, q=q, chi1=chi1, chi2=chi2)
    chieff = eval_chieff(theta1=theta1, theta2=theta2, q=q, chi1=chi1, chi2=chi2)

    if full_output:
        cyclesign = eval_cyclesign(deltaphi=deltaphi)

        return np.stack([S, J, chieff, cyclesign])

    else:
        return np.stack([S, J, chieff])


def angles_to_asymptotic(theta1inf, theta2inf, q, chi1, chi2):
    """
    Convert asymptotic angles (theta1, theta2) into regularized momentum and effective spin (kappa, chieff).

    Call
    ----
    kappainf,chieff = angles_to_asymptotic(theta1inf,theta2inf,q,chi1,chi2)

    Parameters
    ----------
    theta1inf: float
        Asymptotic value of the angle between orbital angular momentum and primary spin.
    theta2inf: float
        Asymptotic value of the angle between orbital angular momentum and secondary spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    kappainf: float
        Asymptotic value of the regularized momentum kappa.
    chieff: float
        Effective spin.
    """

    kappainf = eval_kappainf(theta1inf, theta2inf, q, chi1, chi2)
    chieff = eval_chieff(theta1=theta1inf, theta2=theta2inf, q=q, chi1=chi1, chi2=chi2)

    return np.stack([kappainf, chieff])


def asymptotic_to_angles(kappainf, chieff, q, chi1, chi2):
    """
    Convert regularized momentum and effective spin (kappa, chieff) into asymptotic angles (theta1, theta2).

    Call
    ----
    theta1inf,theta2inf = asymptotic_to_angles(kappainf,chieff,q,chi1,chi2)

    Parameters
    ----------
    kappainf: float
        Asymptotic value of the regularized momentum kappa.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    theta1inf: float
        Asymptotic value of the angle between orbital angular momentum and primary spin.
    theta2inf: float
        Asymptotic value of the angle between orbital angular momentum and secondary spin.
    """

    theta1inf = eval_theta1inf(kappainf, chieff, q, chi1, chi2)
    theta2inf = eval_theta2inf(kappainf, chieff, q, chi1, chi2)

    return np.stack([theta1inf, theta2inf])


def vectors_to_conserved(Lvec, S1vec, S2vec, q, full_output=False):
    """
    Convert cartesian vectors (L,S1,S2) into conserved quantities (S,J,chieff).

    Call
    ----
    S,J,chieff = vectors_to_conserved(Lvec,S1vec,S2vec,q,full_output=False)
    S,J,chieff,cyclesign = vectors_to_conserved(Lvec,S1vec,S2vec,q,full_output=True)

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
    full_output: boolean, optional (default: False)
        Return additional outputs.

    Returns
    -------
    S: float
        Magnitude of the total spin.
    J: float
        Magnitude of the total angular momentum.
    chieff: float
        Effective spin.

    Other parameters
    -------
    cyclesign: integer
        Sign (either +1 or -1) to cover the two halves of a precesion cycle.
    """

    Lvec = np.atleast_2d(Lvec)
    S1vec = np.atleast_2d(S1vec)
    S2vec = np.atleast_2d(S2vec)

    S = norm_nested(S1vec+S2vec)
    J = norm_nested(S1vec+S2vec+Lvec)
    L = norm_nested(Lvec)
    m1, m2 = masses(q)

    chieff = dot_nested(S1vec, Lvec)/(m1*L) + dot_nested(S2vec, Lvec)/(m2*L)

    if full_output:
        cyclesign = eval_cyclesign(Lvec=Lvec, S1vec=S1vec, S2vec=S2vec)

        return np.stack([S, J, chieff, cyclesign])

    else:
        return np.stack([S, J, chieff])

# TODO: write function to get theta12 from theta1, theta2 and deltaphi


def vectors_to_angles(Lvec, S1vec, S2vec):
    """
    Convert cartesian vectors (L,S1,S2) into angles (theta1,theta2,deltaphi). The convention for the sign of deltaphi is given in Eq. (2d) of arxiv:1506.03492.

    Call
    ----
    theta1,theta2,deltaphi = vectors_to_angles(Lvec,S1vec,S2vec)

    Parameters
    ----------
    Lvec: array
        Cartesian vector of the orbital angular momentum.
    S1vec: array
        Cartesian vector of the primary spin.
    S2vec: array
        Cartesian vector of the secondary spin.

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

    theta1 = np.arccos(dot_nested(S1vec, Lvec))
    theta2 = np.arccos(dot_nested(S2vec, Lvec))
    S1crL = np.cross(S1vec, Lvec)
    S2crL = np.cross(S2vec, Lvec)

    absdeltaphi = np.arccos(dot_nested(normalize_nested(S1crL), normalize_nested(S2crL)))
    cyclesign = eval_cyclesign(Lvec=Lvec, S1vec=S1vec, S2vec=S2vec)
    deltaphi = -absdeltaphi*cyclesign

    return np.stack([theta1, theta2, deltaphi])


def conserved_to_Jframe(S, J, r, chieff, q, chi1, chi2, cyclesign=1):
    """
    Convert the conserved quantities (S,J,chieff) to angular momentum vectors (L,S1,S2) in the frame
    aligned with the total angular momentum. In particular, we set Jx=Jy=Ly=0.

    Call
    ----
    Lvec,S1vec,S2vec = conserved_to_Jframe(S,J,r,chieff,q,chi1,chi2)

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
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

    S = np.atleast_1d(S)
    J = np.atleast_1d(J)

    L = eval_L(r, q)
    S1, S2 = spinmags(q, chi1, chi2)
    varphi = eval_varphi(S, J, r, chieff, q, chi1, chi2, cyclesign=cyclesign)
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
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
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

    S, J, chieff, cyclesign = angles_to_conserved(theta1, theta2, deltaphi, r, q, chi1, chi2, full_output=True)
    Lvec, S1vec, S2vec = conserved_to_Jframe(S, J, r, chieff, q, chi1, chi2, cyclesign=cyclesign)

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
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
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


def conserved_to_Lframe(S, J, r, chieff, q, chi1, chi2, cyclesign=1):
    """
    Convert the conserved quantities (S,J,chieff) to angular momentum vectors (L,S1,S2) in the frame aligned with the orbital angular momentum. In particular, we set Lx=Ly=S1y=0.

    Call
    ----
    Lvec,S1vec,S2vec = conserved_to_Lframe(S,J,r,chieff,q,chi1,chi2,cyclesign=1)

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    cyclesign: integer, optional (default: 1)
        Sign (either +1 or -1) to cover the two halves of a precesion cycle.

    Returns
    -------
    Lvec: array
        Cartesian vector of the orbital angular momentum.
    S1vec: array
        Cartesian vector of the primary spin.
    S2vec: array
        Cartesian vector of the secondary spin.
    """

    theta1, theta2, deltaphi = conserved_to_angles(S, J, r, chieff, q, chi1, chi2, cyclesign=cyclesign)
    Lvec, S1vec, S2vec = angles_to_Lframe(theta1, theta2, deltaphi, r, q, chi1, chi2)

    return np.stack([Lvec, S1vec, S2vec])


def conserved_to_inertial(S, J, r, chieff, q, chi1, chi2, cyclesign=1):
    """
    Convert the conserved quantities (S,J,chieff) to angular momentum vectors (L,S1,S2) in an inertial frame that aligned is were Lx=Ly=S1y=0 as S=S- but, unlike the Jframe, does not co-precesses with L.

    Call
    ----
    Lvec,S1vec,S2vec = conserved_to_inertial(S,J,r,chieff,q,chi1,chi2,cyclesign=1)

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    cyclesign: integer, optional (default: 1)
        Sign (either +1 or -1) to cover the two halves of a precesion cycle.

    Returns
    -------
    Lvec: array
        Cartesian vector of the orbital angular momentum.
    S1vec: array
        Cartesian vector of the primary spin.
    S2vec: array
        Cartesian vector of the secondary spin.
    """

    Lvec, S1vec, S2vec = conserved_to_Jframe(S, J, r, chieff, q, chi1, chi2, cyclesign=cyclesign)
    phiL = eval_phiL(S, J, r, chieff, q, chi1, chi2, cyclesign=cyclesign)

    Lvec = rotate_zaxis(Lvec, phiL)
    S1vec = rotate_zaxis(S1vec, phiL)
    S2vec = rotate_zaxis(S2vec, phiL)

    return np.stack([Lvec, S1vec, S2vec])


def angles_to_inertial(theta1, theta2, deltaphi, r, q, chi1, chi2):
    """
    Convert the angles (theta1, theta2, deltaphi) to angular momentum vectors (L, S1, S2) in an inertial frame that aligned is were Lx=Ly=S1y=0 as S=S- but, unlike the Jframe, does not co-precesses with L.

    Call
    ----
    Lvec,S1vec,S2vec = angles_to_inertial(theta1,theta2,deltaphi,r,q,chi1,chi2)

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
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
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

    deltaphi = np.atleast_1d(deltaphi)
    S, J, chieff, cyclesign = angles_to_conserved(theta1, theta2, deltaphi, r, q, chi1, chi2, full_output=True)
    Lvec, S1vec, S2vec = conserved_to_inertial(S, J, r, chieff, q, chi1, chi2, cyclesign=cyclesign)

    return np.stack([Lvec, S1vec, S2vec])


# Precessional timescale dynamics

def derS_prefactor(r, chieff, q):
    """
    Numerical prefactor to the S derivative.

    Call
    ----
    mathcalA = derS_prefactor(r,chieff,q)

    Parameters
    ----------
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.

    Returns
    -------
    mathcalA: float
        Prefactor in the dSdt equation.
    """

    r = np.atleast_1d(r)
    chieff = np.atleast_1d(chieff)

    eta = eval_eta(q)
    mathcalA = (3/2)*(1/(r**3*eta**0.5))*(1-(chieff/r**0.5))

    return mathcalA


def dSsdtsquared(S, J, r, chieff, q, chi1, chi2):
    """
    Squared first time derivative of the squared total spin, on the precession timescale.

    Call
    ----
    dSsdts = dSsdtsquared(S,J,r,chieff,q,chi1,chi2)

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    dSsdts: float
        Squared first derivative of the squared total spin.
    """

    mathcalA = derS_prefactor(r, chieff, q)
    Sminuss, Spluss, S3s = Ssroots(J, r, chieff, q, chi1, chi2)
    dSsdts = - mathcalA**2 * (S**2-Spluss) * (S**2-Sminuss) * (S**2-S3s)

    return dSsdts


# Change name to this function, otherwise is identical to the returned variable.
def dSsdt(S, J, r, chieff, q, chi1, chi2, cyclesign=1):
    """
    Time derivative of the squared total spin, on the precession timescale.

    Call
    ----
    dSsdt = dSsdt(S,J,r,chieff,q,chi1,chi2,cyclesign=1)

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    cyclesign: integer, optional (default: 1)
        Sign (either +1 or -1) to cover the two halves of a precesion cycle. One has sign(deltaphi)=sign(varphi)=-sign(dS/dt).

    Returns
    -------
    dSsdt: float
        Time derivative of the squared total spin.
    """

    cyclesign = np.atleast_1d(cyclesign)

    return cyclesign*dSsdtsquared(S, J, r, chieff, q, chi1, chi2)**0.5


# Change name to this function, otherwise is identical to the returned variable.
def dSdt(S, J, r, chieff, q, chi1, chi2):
    """
    Time derivative of the total spin, on the precession timescale.

    Call
    ----
    dSdt = dSdt(S,J,r,chieff,q,chi1,chi2)

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    dSdt: float
        Time derivative of the total spin.
    """

    return dSsdt(S, J, r, chieff, q, chi1, chi2) / (2*S)



def elliptic_parameter_old(Sminuss, Spluss, S3s):
    """
    Parameter m entering elliptic functions for the evolution of S.

    Call
    ----
    m = elliptic_parameter(Sminuss,Spluss,S3s)

    Parameters
    ----------
    Sminuss: float
        Lowest physical root, if present, of the effective potential equation.
    Spluss: float
        Largest physical root, if present, of the effective potential equation.
    S3s: float
        Spurious root of the effective potential equation.

    Returns
    -------
    m: float
        Parameter of elliptic function(s).
    """

    Sminuss = np.atleast_1d(Sminuss)
    Spluss = np.atleast_1d(Spluss)
    S3s = np.atleast_1d(S3s)

    m = (Spluss-Sminuss)/(Spluss-S3s)

    return m


def elliptic_amplitude(S, Sminuss, Spluss):
    """
    Amplitdue phi entering elliptic functions for the evolution of S.

    Call
    ----
    phi = elliptic_amplitude(S,Sminuss,Spluss)

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    Sminuss: float
        Lowest physical root, if present, of the effective potential equation.
    Spluss: float
        Largest physical root, if present, of the effective potential equation.

    Returns
    -------
    phi: float
        Amplitude of elliptic function(s).
    """

    S = np.atleast_1d(S)
    Sminuss = np.atleast_1d(Sminuss)
    Spluss = np.atleast_1d(Spluss)

    phi = np.arccos(((S**2 - Sminuss) / (Spluss - Sminuss))**0.5)

    return phi


def elliptic_characheristic(Sminuss, Spluss, J, L, sign):
    """
    Characheristic m entering elliptic functions for the evolution of S.

    Call
    ----
    n = elliptic_characheristic(Sminuss,Spluss,J,L,sign)

    Parameters
    ----------
    Sminuss: float
        Lowest physical root, if present, of the effective potential equation.
    Spluss: float
        Largest physical root, if present, of the effective potential equation.
    J: float
        Magnitude of the total angular momentum.
    L: float
        Magnitude of the Newtonian orbital angular momentum.
    sign: integer
        Sign, either +1 or -1.

    Returns
    -------
    n: float
        Characheristic of elliptic function(s).
    """

    Sminuss = np.atleast_1d(Sminuss)
    Spluss = np.atleast_1d(Spluss)
    J = np.atleast_1d(J)
    L = np.atleast_1d(L)

    # Note: sign here is not cyclesign!
    n = (Spluss - Sminuss)/(Spluss - (J + np.sign(sign)*L)**2)

    return n


# remove this
def time_normalization(Spluss, S3s, r, chieff, q):
    """
    Numerical prefactors entering the precession period.

    Call
    ----
    mathcalT = time_normalization(Spluss,S3s,r,chieff,q)

    Parameters
    ----------
    Spluss: float
        Largest physical root, if present, of the effective potential equation.
    S3s: float
        Spurious root of the effective potential equation.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.

    Returns
    -------
    mathcalT: float
        Prefactor in the tau equation.
    """

    Spluss = np.atleast_1d(Spluss)
    S3s = np.atleast_1d(S3s)

    mathcalA = derS_prefactor(r, chieff, q)
    mathcalT = 2/(mathcalA*(Spluss-S3s)**0.5)

    return mathcalT


# remove this
def eval_tau_old(J, r, chieff, q, chi1, chi2, precomputedroots=None):
    """
    Period of S as it oscillates from S- to S+ and back to S-.

    Call
    ----
    tau = eval_tau(J,r,chieff,q,chi1,chi2,precomputedroots=None)

    Parameters
    ----------
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    precomputedroots: array, optional (default: None)
        Pre-computed output of Ssroots for computational efficiency.

    Returns
    -------
    tau: float
        Nutation period.
    """

    Sminuss, Spluss, S3s = Ssroots(J, r, chieff, q, chi1, chi2, precomputedroots=precomputedroots)
    mathcalT = time_normalization(Spluss, S3s, r, chieff, q)
    m = elliptic_parameter_old(Sminuss, Spluss, S3s)
    tau = 2*mathcalT*scipy.special.ellipk(m)

    return tau


# remove this
def Soft(t, J, r, chieff, q, chi1, chi2, precomputedroots=None):
    """
    Evolution of S on the precessional timescale (without radiation reaction).
    The broadcasting rules for this function are more general than those of the rest of the code. The variable t is allowed to have shapes (N,M) while all the other variables have shape (N,). This is useful to sample M precession configuration for each of the N binaries specified as inputs.

    Call
    ----
    S = Soft(t,J,r,chieff,q,chi1,chi2,precomputedroots=None)

    Parameters
    ----------
    t: float
        Time.
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    precomputedroots: array, optional (default: None)
        Pre-computed output of Ssroots for computational efficiency.

    Returns
    -------
    S: float
        Magnitude of the total spin.
    """

    t = np.atleast_1d(t)
    Sminuss, Spluss, S3s = Ssroots(J, r, chieff, q, chi1, chi2, precomputedroots=precomputedroots)
    mathcalT = time_normalization(Spluss, S3s, r, chieff, q)
    m = elliptic_parameter_old(Sminuss, Spluss, S3s)

    sn, _, dn, _ = scipy.special.ellipj(t.T/mathcalT, m)
    Ssq = Sminuss + (Spluss-Sminuss)*((Sminuss-S3s)/(Spluss-S3s)) * (sn/dn)**2
    S = Ssq.T**0.5

    return S



#remove this
def tofS(S, J, r, chieff, q, chi1, chi2, cyclesign=1, precomputedroots=None):
    """
    Time t as a function of S (without radiation reaction). Only covers half of a precession cycle, assuming t=0 at S=S- and t=tau/2 at S=S+. Set sign=-1 to cover the second half, i.e. from t=tau/2 at S=S+ to t=tau at S=S-.

    Call
    ----
    t = tofS(S,J,r,chieff,q,chi1,chi2,cyclesign=1,precomputedroots=None)

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    cyclesign: integer, optional (default: 1)
        Sign (either +1 or -1) to cover the two halves of a precesion cycle.
    precomputedroots: array, optional (default: None)
        Pre-computed output of Ssroots for computational efficiency.

    Returns
    -------
    t: float
        Time.
    """

    S = np.atleast_1d(S)

    Sminuss, Spluss, S3s = Ssroots(J, r, chieff, q, chi1, chi2, precomputedroots=precomputedroots)

    m = elliptic_parameter_old(Sminuss, Spluss, S3s)
    mathcalT = time_normalization(Spluss, S3s, r, chieff, q)
    phi = elliptic_amplitude(S, Sminuss, Spluss)
    tau = eval_tau_old(J, r, chieff, q, chi1, chi2, precomputedroots=np.stack([Sminuss, Spluss, S3s]))
    t = tau/2 - np.sign(cyclesign)*mathcalT*scipy.special.ellipkinc(phi, m)

    return t



# remove this
def Ssampling(J, r, chieff, q, chi1, chi2, N=1):
    """
    Sample N values of S at fixed separation accoring to its PN-weighted distribution function.
    Can only be used to sample the *same* number of configuration for each binary. If the inputs J,r,chieff,q,chi1, and chi2 have shape (M,) the output will have shape
    - (M,N) if M>1 and N>1;
    - (M,) if N=1;
    - (N,) if M=1.

    Call
    ----
    S = Ssampling(J,r,chieff,q,chi1,chi2,N = 1)

    Parameters
    ----------
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    N: integer, optional (default: 1)
        Number of samples.

    Returns
    -------
    S: float
        Magnitude of the total spin.
    """

    # Compute the S roots only once and pass them to both functions
    Sminuss, Spluss, S3s = Ssroots(J, r, chieff, q, chi1, chi2)

    tau = eval_tau_old(J, r, chieff, q, chi1, chi2, precomputedroots=np.stack([Sminuss, Spluss, S3s]))
    # For each binary, generate N samples between 0 and tau.
    t = np.random.uniform(size=tau.size*N).reshape((tau.size, N)) * tau[:, None]
    # Note the special broadcasting rules of Soft, see Soft.__docs__
    # S has shape (M, N).
    S = Soft(t, J, r, chieff, q, chi1, chi2, precomputedroots=np.stack([Sminuss, Spluss, S3s]))

    # np.squeeze is necessary to return shape (M,) instead of (M,1) if N=1
    # np.atleast_1d is necessary to retun shape (1,) instead of (,) if M=N=1
    return np.atleast_1d(np.squeeze(S))


    # remove this



# re do docstrings
def deltachitildeav(m):
    """
    Factor depending on the elliptic parameter in the precession averaged squared total spin. This is (1 - E(m)/K(m)) / m.

    Call
    ----
    coeff = deltachitildeav(m)

    Parameters
    ----------
    m: float
        Parameter of elliptic function(s).

    Returns
    -------
    coeff: float
        Coefficient.
    """

    m = np.atleast_1d(m)
    # The limit of the Ssav coefficient as m->0 is finite and equal to 1/2.
    # This is implementation is numerically stable up to m~1e-10.
    # For m=1e-7, the analytic m=0 limit is returned with a precision of 1e-9, which is enough.
    m = np.maximum(1e-7, m)
    coeff = (1-scipy.special.ellipe(m)/scipy.special.ellipk(m))/m

    return coeff


# TODO: change name to this function
def Ssav(J, r, chieff, q, chi1, chi2):
    """
    Analytic precession averaged expression for the squared total spin.

    Call
    ----
    Ssq = Ssav(J,r,chieff,q,chi1,chi2)

    Parameters
    ----------
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    Ssq: float
        Squared magnitude of the total spin.
    """

    Sminuss, Spluss, S3s = Ssroots(J, r, chieff, q, chi1, chi2)
    m = elliptic_parameter_old(Sminuss, Spluss, S3s)
    Ssq = Spluss - (Spluss-Sminuss)*deltachitildeav(m)

    return Ssq


def Ssrootsinf(theta1inf, theta2inf, q, chi1, chi2):
    """
    Infinite orbital separation limit of the roots of the cubic equation in S^2.

    Call
    ----
    Sminussinf,Splussinf,S3sinf = Ssrootsinf(theta1inf,theta2inf,q,chi1,chi2)

    Parameters
    ----------
    theta1inf: float
        Asymptotic value of the angle between orbital angular momentum and primary spin.
    theta2inf: float
        Asymptotic value of the angle between orbital angular momentum and secondary spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    Sminussinf: float
        Asymptotic value of the lowest physical root, if present, of the effective potential equation.
    Splussinf: float
        Asymptotic value of the largest physical root, if present, of the effective potential equation.
    S3sinf: float
        Asymptotic value of the spurious root of the effective potential equation.
    """

    S1, S2 = spinmags(q, chi1, chi2)
    coscos = np.cos(theta1inf)*np.cos(theta2inf)
    sinsin = np.sin(theta1inf)*np.sin(theta2inf)
    Sminussinf = S1**2 + S2**2 + 2*S1*S2*(coscos - sinsin)
    Splussinf = S1**2 + S2**2 + 2*S1*S2*(coscos + sinsin)
    S3sinf = -np.inf

    return np.stack([Sminussinf, Splussinf, S3sinf])


def Ssavinf(theta1inf, theta2inf, q, chi1, chi2):
    """
    Infinite orbital separation limit of the precession averaged values of S^2
    from the asymptotic angles (theta1, theta2).

    Call
    ----
    Ssq = Ssavinf(theta1inf,theta2inf,q,chi1,chi2)

    Parameters
    ----------
    theta1inf: float
        Asymptotic value of the angle between orbital angular momentum and primary spin.
    theta2inf: float
        Asymptotic value of the angle between orbital angular momentum and secondary spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
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
    Ssavinf = S1**2 + S2**2 + 2*S1*S2*np.cos(theta1inf)*np.cos(theta2inf)

    return Ssavinf




def ddchidt_prefactor(r, chieff, q):
    """
    Numerical prefactor to the S derivative.

    Call
    ----
    mathcalA = derS_prefactor(r,chieff,q)

    Parameters
    ----------
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.

    Returns
    -------
    mathcalA: float
        Prefactor in the dSdt equation.
    """


    r = np.atleast_1d(r)
    chieff = np.atleast_1d(chieff)
    q = np.atleast_1d(q)

    mathcalA = (3/2)*((1+q)**(-1/2))*(r**(-11/4))*(1-(chieff/r**0.5))

    return mathcalA



def dchidt2_RHS(deltachi, kappa, r, chieff, q, chi1, chi2, precomputedroots=None):

    u= eval_u(r=r,q=q)
    deltachiminus,deltachiplus,deltachi3 = deltachiroots(kappa, u, chieff, q, chi1, chi2, precomputedroots=precomputedroots)

    mathcalA = ddchidt_prefactor(r, chieff, q)

    dchidt2 = mathcalA**2 * ( (deltachi-deltachiminus)*(deltachiplus-deltachi)*(deltachi3-(1-q)*deltachi))

    return dchidt2



def elliptic_parameter(kappa, u, chieff, q, chi1, chi2, precomputedroots=None):
    """
    Parameter m entering elliptic functions for the evolution of S.

    Call
    ----
    m = elliptic_parameter(Sminuss,Spluss,S3s)

    Parameters
    ----------
    Sminuss: float
        Lowest physical root, if present, of the effective potential equation.
    Spluss: float
        Largest physical root, if present, of the effective potential equation.
    S3s: float
        Spurious root of the effective potential equation.

    Returns
    -------
    m: float
        Parameter of elliptic function(s).
    """

    q=np.atleast_1d(q)

    deltachiminus,deltachiplus,deltachi3 = deltachiroots(kappa, u, chieff, q, chi1, chi2, precomputedroots=precomputedroots)

    m = (1-q)*(deltachiplus-deltachiminus)/(deltachi3-(1-q)*deltachiminus)

    return m




def eval_tau(kappa, r, chieff, q, chi1, chi2, precomputedroots=None, return_psiperiod=False):

    q=np.atleast_1d(q)


    # if psiperiod=True return tau/2K(m). Useful to avoid the evaluation of an elliptic integral when it's not needed
    u = eval_u(r,q)
    mathcalA = ddchidt_prefactor(r, chieff, q)
    deltachiminus,deltachiplus,deltachi3 = deltachiroots(kappa, u, chieff, q, chi1, chi2, precomputedroots=precomputedroots)
    m = elliptic_parameter(kappa, u, chieff, q, chi1, chi2, precomputedroots=np.stack([deltachiminus,deltachiplus,deltachi3]))

    psiperiod =  2 / ( mathcalA * (deltachi3 - (1-q)*deltachiminus)**(1/2) )
    if return_psiperiod:
        tau = psiperiod
    else:
        tau = 2*scipy.special.ellipk(m) * psiperiod

    return tau




#I changed the broadcasting rule compared Soft!
def deltachioft(t, kappa , r, chieff, q, chi1, chi2, precomputedroots=None):
    """
    Evolution of S on the precessional timescale (without radiation reaction).
    The broadcasting rules for this function are more general than those of the rest of the code. The variable t is allowed to have shapes (N,M) while all the other variables have shape (N,). This is useful to sample M precession configuration for each of the N binaries specified as inputs.

    Call
    ----
    S = Soft(t,J,r,chieff,q,chi1,chi2,precomputedroots=None)

    Parameters
    ----------
    t: float
        Time.
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    precomputedroots: array, optional (default: None)
        Pre-computed output of Ssroots for computational efficiency.

    Returns
    -------
    S: float
        Magnitude of the total spin.
    """

    t = np.atleast_1d(t)
    u = eval_u(r,q)

    deltachiminus,deltachiplus,deltachi3 = deltachiroots(kappa, u, chieff, q, chi1, chi2, precomputedroots=precomputedroots)
    psiperiod = eval_tau(kappa, r, chieff, q, chi1, chi2, precomputedroots=np.stack([deltachiminus,deltachiplus,deltachi3]), return_psiperiod=True)


    m = elliptic_parameter(kappa, u, chieff, q, chi1, chi2, precomputedroots=np.stack([deltachiminus,deltachiplus,deltachi3]))
 

    sn, _, _, _ = scipy.special.ellipj(t / psiperiod, m)
    deltachitilde = sn**2



    deltachi = deltachirescaling(deltachitilde, kappa, r, chieff, q, chi1, chi2,precomputedroots=np.stack([deltachiminus,deltachiplus,deltachi3]))

    return deltachi



def tofS(S, J, r, chieff, q, chi1, chi2, cyclesign=1, precomputedroots=None):
    """
    Time t as a function of S (without radiation reaction). Only covers half of a precession cycle, assuming t=0 at S=S- and t=tau/2 at S=S+. Set sign=-1 to cover the second half, i.e. from t=tau/2 at S=S+ to t=tau at S=S-.

    Call
    ----
    t = tofS(S,J,r,chieff,q,chi1,chi2,cyclesign=1,precomputedroots=None)

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    cyclesign: integer, optional (default: 1)
        Sign (either +1 or -1) to cover the two halves of a precesion cycle.
    precomputedroots: array, optional (default: None)
        Pre-computed output of Ssroots for computational efficiency.

    Returns
    -------
    t: float
        Time.
    """

    S = np.atleast_1d(S)

    Sminuss, Spluss, S3s = Ssroots(J, r, chieff, q, chi1, chi2, precomputedroots=precomputedroots)

    m = elliptic_parameter_old(Sminuss, Spluss, S3s)
    mathcalT = time_normalization(Spluss, S3s, r, chieff, q)
    phi = elliptic_amplitude(S, Sminuss, Spluss)
    tau = eval_tau_old(J, r, chieff, q, chi1, chi2, precomputedroots=np.stack([Sminuss, Spluss, S3s]))
    t = tau/2 - np.sign(cyclesign)*mathcalT*scipy.special.ellipkinc(phi, m)

    return t

def tofdeltachi(deltachi, kappa , r, chieff, q, chi1, chi2, cyclesign=1, precomputedroots=None):


    deltachiminus,deltachiplus,deltachi3 = deltachiroots(kappa, u, chieff, q, chi1, chi2, precomputedroots=precomputedroots)

    psiperiod = eval_tau(kappa, r, chieff, q, chi1, chi2, precomputedroots=np.stack([deltachiminus,deltachiplus,deltachi3]), return_psiperiod=True)
    deltachitilde = affine(deltachi,deltachiminus,deltachiplus)
    m = elliptic_parameter(kappa, u, chieff, q, chi1, chi2, precomputedroots=np.stack([deltachiminus,deltachiplus,deltachi3]))
    t = np.sign(cyclesign) * psiperiod * scipy.special.ellipkinc(np.arcsin(deltachitilde**(1/2)), m)

    return t 


def deltachisampling(kappa, r, chieff, q, chi1, chi2, N=1, precomputedroots=None):
    """
    Sample N values of S at fixed separation accoring to its PN-weighted distribution function.
    Can only be used to sample the *same* number of configuration for each binary. If the inputs J,r,chieff,q,chi1, and chi2 have shape (M,) the output will have shape
    - (M,N) if M>1 and N>1;
    - (M,) if N=1;
    - (N,) if M=1.

    Call
    ----
    S = Ssampling(J,r,chieff,q,chi1,chi2,N = 1)

    Parameters
    ----------
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    N: integer, optional (default: 1)
        Number of samples.

    Returns
    -------
    S: float
        Magnitude of the total spin.
    """

    u= eval_u(r=r,q=q)

    # Compute the deltachi roots only once and pass them to both functions
    deltachiminus,deltachiplus,deltachi3 = deltachiroots(kappa, u, chieff, q, chi1, chi2, precomputedroots=precomputedroots)

    tau = eval_tau(kappa, r, chieff, q, chi1, chi2, precomputedroots=np.stack([deltachiminus,deltachiplus,deltachi3]))

    # For each binary, generate N samples between 0 and tau.
    t = np.random.uniform(np.zeros(len(tau)),tau,size=(N,len(tau)))

    # np.squeeze is necessary to return shape (M,) instead of (M,1) if N=1
    # np.atleast_1d is necessary to retun shape (1,) instead of (,) if M=N=1
    t= np.atleast_1d(np.squeeze(t))

    # Note the special broadcasting rules of deltachioft, see Soft.__docs__
    # deltachi has shape (M, N).
    deltachi = deltachioft(t, kappa , r, chieff, q, chi1, chi2, precomputedroots=np.stack([deltachiminus,deltachiplus,deltachi3]))
    return deltachi.T


    

###############


# Precession-averaged evolution
def rhs_precav_old(kappa, u, chieff, q, chi1, chi2):
    """
    Right-hand side of the dkappa/du ODE describing precession-averaged inspiral. This is an internal function used by the ODE integrator and is not array-compatible. It is equivalent to Ssav and Ssavinf and it has been re-written for optimization purposes.

    Call
    ----
    RHS = rhs_precav(kappa,u,chieff,q,chi1,chi2)

    Parameters
    ----------
    kappa: float
        Regularized angular momentum (J^2-L^2)/(2L).
    u: float
        Compactified separation 1/(2L).
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    RHS: float
        Right-hand side.
    """

    if u == 0:
        # In this case use analytic result
        theta1inf, theta2inf = asymptotic_to_angles(kappa, chieff, q, chi1, chi2)
        Ssav = Ssavinf(theta1inf, theta2inf, q, chi1, chi2)
    else:

        coeffs = Scubic_coefficients(kappa, u, chieff, q, chi1, chi2)
        coeffs = np.squeeze(coeffs)

        # The first coefficient is tiny, 10^-100 small than the others. This is in practice a 2nd order polynomial
        if np.abs(coeffs[0]) < 10**-100 * np.abs(np.mean(coeffs[1:])):
            warnings.warn("Sanitizing RHS output; solving quadratic. [rhs_precav].", Warning)
            sols = np.roots(coeffs[1:])
            Sminuss, Spluss = sols
            Ssav = np.mean(np.real([Sminuss, Spluss]))

        else:
            sols = np.roots(coeffs)
            S3s, Sminuss, Spluss = np.squeeze(np.sort_complex(sols))

            # Sminus and Splus are complex. This can happen if the binary is very close to a spin-orbit resonance
            if np.iscomplex([Sminuss, Spluss]).any():
                warnings.warn("Sanitizing RHS output; too close to resonance. [rhs_precav].", Warning)
                Ssav = np.mean(np.real([Sminuss, Spluss]))

            # Normal case
            else:
                S3s, Sminuss, Spluss = np.real([S3s, Sminuss, Spluss])

                #print('hhh old')
                #print(S3s, Sminuss, Spluss)

                m = elliptic_parameter_old(Sminuss, Spluss, S3s)
                Ssav = Spluss - (Spluss-Sminuss)*deltachitildeav(m)

    return Ssav




# Precession-averaged evolution
def rhs_precav(kappa, u, chieff, q, chi1, chi2):
    """
    Right-hand side of the dkappa/du ODE describing precession-averaged inspiral. This is an internal function used by the ODE integrator and is not array-compatible. It is equivalent to Ssav and Ssavinf and it has been re-written for optimization purposes.

    Call
    ----
    RHS = rhs_precav(kappa,u,chieff,q,chi1,chi2)

    Parameters
    ----------
    kappa: float
        Regularized angular momentum (J^2-L^2)/(2L).
    u: float
        Compactified separation 1/(2L).
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    RHS: float
        Right-hand side.
    """

    if u == 0:
        # In this case use analytic result
        theta1inf, theta2inf = asymptotic_to_angles(kappa, chieff, q, chi1, chi2)
        Ssav = Ssavinf(theta1inf, theta2inf, q, chi1, chi2)
    else:

        deltachiminus, deltachiplus, deltachi3 = deltachiroots(kappa, u, chieff, q, chi1, chi2)

        # deltachiminus, deltachiplus are complex. This can happen if the binary is very close to a spin-orbit resonance
        if np.iscomplex([deltachiminus, deltachiplus]).any():
            warnings.warn("Sanitizing RHS output; too close to resonance. [rhs_precav].", Warning)
            deltachiav = np.mean(np.real([deltachiminus, deltachiplus]))

        # Normal case
        else:
            deltachiminus, deltachiplus, deltachi3 = np.real([deltachiminus, deltachiplus, deltachi3])
            deltachi3ss = deltachi3/(1-q)
            #print('hhh new')

            #r=eval_r(u=u,q=q)
            #print( q /(1+q)**2 * r**(1/2) * (2*kappa - chieff - deltachiminus * (1 - q)/(1 + q)) )
            #print( q /(1+q)**2 * r**(1/2) * (2*kappa - chieff - deltachiplus * (1 - q)/(1 + q)) )
            #print( q /(1+q)**2 * r**(1/2) * (2*kappa - chieff - deltachi3/(1 + q)) )



            m = elliptic_parameter(kappa, u, chieff, q, chi1, chi2, precomputedroots=np.stack([deltachiminus, deltachiplus, deltachi3]))
            deltachiav = inverseaffine( deltachitildeav(m),  deltachiminus, deltachiplus)
            #print('dchiav' ,deltachiav)
        Ssav = (2*kappa - chieff - (1-q)/(1+q)*deltachiav)/(2*u)

    return Ssav



# Update docstrings
#Careful that here u needs to be an array
def integrator_precav(kappainitial, u, chieff, q, chi1, chi2, **odeint_kwargs):
    """
    Integration of ODE dkappa/du describing precession-averaged inspirals.

    Call
    ----
    kappa = integrator_precav(kappainitial,uinitial,ufinal,chieff,q,chi1,chi2)

    Parameters
    ----------
    kappainitial: float
        Initial value of the regularized momentum kappa.
    uinitial: float
        Initial value of the compactified separation 1/(2L).
    ufinal: float
        Final value of the compactified separation 1/(2L).
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    kappa: float
        Regularized angular momentum (J^2-L^2)/(2L).
    """

    kappainitial = np.atleast_1d(kappainitial)
    u = np.atleast_2d(u)
    chieff = np.atleast_1d(chieff)
    q = np.atleast_1d(q)
    chi1 = np.atleast_1d(chi1)
    chi2 = np.atleast_1d(chi2)

    # Defaults for the integrators, can be changed by the user
    if 'mxstep' not in odeint_kwargs: odeint_kwargs['mxstep']=5000000
    if 'rol' not in odeint_kwargs: odeint_kwargs['rtol']=1e-10
    if 'aol' not in odeint_kwargs: odeint_kwargs['atol']=1e-10
    # I'm sorry but this needs to be forced for compatibility with the rest of the code
    odeint_kwargs['full_output'] = 0 


    def _compute(kappainitial, u, chieff, q, chi1, chi2, odeint_kwargs):

        # h0 controls the first stepsize attempted. If integrating from finite separation, let the solver decide (h0=0). If integrating from infinity, prevent it from being too small.
        # h0= 1e-3 if u[0]==0 else 0

        # Make sure the first step is large enough. This is to avoid LSODA to propose a tiny step which causes the integration to stall
        # if 'h0' not in odeint_kwargs: odeint_kwargs['h0']=min(u[0])/1e6

        # Update: This does not seem to be necessary after all.


        ODEsolution = scipy.integrate.odeint(rhs_precav, kappainitial, u, args=(chieff, q, chi1, chi2), **odeint_kwargs)#, printmessg=0,rtol=1e-10,atol=1e-10)#,tcrit=sing)

        return np.squeeze(ODEsolution)



        #ODEsolution = scipy.integrate.solve_ivp(rhs_precav, (uinitial, ufinal), np.atleast_1d(kappainitial), method='RK45', t_eval=(uinitial, ufinal), dense_output=True, args=(chieff, q, chi1, chi2), atol=1e-8, rtol=1e-8)  # ,events=event)

        # TODO: let user pick rtol and atol

        # Return ODE object. The key methods is .sol --callable, sol(t).
        #return ODEsolution

    ODEsolution = np.array(list(map(_compute, kappainitial, u, chieff, q, chi1, chi2, repeat(odeint_kwargs))))

    return ODEsolution


# TODO: return Sminus and Splus along the solution. Right now these are computed inside Ssampling but not stored
def inspiral_precav(theta1=None, theta2=None, deltaphi=None, S=None, J=None, kappa=None, r=None, u=None, chieff=None, q=None, chi1=None, chi2=None, requested_outputs=None):
    """
    Perform precession-averaged inspirals. The variables q, chi1, and chi2 must always be provided. The integration range must be specified using either r or u (and not both). The initial conditions correspond to the binary at either r[0] or u[0]. The vector r or u needs to monotonic increasing or decreasing, allowing to integrate forwards and backwards in time. In addition, integration can be done between finite separations, forwards from infinite to finite separation, or backwards from finite to infinite separation. For infinity, use r=np.inf or u=0.
    The initial conditions must be specified in terms of one an only one of the following:
    - theta1,theta2, and deltaphi (but note that deltaphi is not necessary if integrating from infinite separation).
    - J, chieff (only if integrating from finite separations because J otherwise diverges).
    - kappa, chieff.
    The desired outputs can be specified with a list e.g. requested_outputs=['theta1','theta2','deltaphi']. All the available variables are returned by default. These are: ['theta1', 'theta2', 'deltaphi', 'S', 'J', 'kappa', 'r', 'u', 'chieff', 'q', 'chi1', 'chi2'].

    Call
    ----
    outputs = inspiral_precav(theta1=None,theta2=None,deltaphi=None,S=None,J=None,kappa=None,r=None,u=None,chieff=None,q=None,chi1=None,chi2=None,requested_outputs=None)

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
    chieff: float, optional (default: None)
        Effective spin.
    q: float, optional (default: None)
        Mass ratio: 0<=q<=1.
    chi1: float, optional (default: None)
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float, optional (default: None)
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    requested_outputs: list, optional (default: None)
        Set of outputs.

    Returns
    -------
    outputs: dictionary
        Set of outputs.
    """

    # Substitute None inputs with arrays of Nones
    inputs = [theta1, theta2, deltaphi, S, J, kappa, r, u, chieff, q, chi1, chi2]
    for k, v in enumerate(inputs):
        if v is None:
            inputs[k] = np.atleast_1d(np.squeeze(tiler(None, np.atleast_1d(q))))
        else:
            if k == 6 or k == 7:  # Either u or r
                inputs[k] = np.atleast_2d(inputs[k])
            else:  # Any of the others
                inputs[k] = np.atleast_1d(inputs[k])
    theta1, theta2, deltaphi, S, J, kappa, r, u, chieff, q, chi1, chi2 = inputs

    def _compute(theta1, theta2, deltaphi, S, J, kappa, r, u, chieff, q, chi1, chi2):

        if q is None or chi1 is None or chi2 is None:
            raise TypeError("Please provide q, chi1, and chi2.")

        if r is not None and u is None:
            assert np.logical_or(ismonotonic(r, '<='), ismonotonic(r, '>=')), 'r must be monotonic'
            u = eval_u(r, tiler(q, r))
        elif r is None and u is not None:
            assert np.logical_or(ismonotonic(u, '<='), ismonotonic(u, '>=')), 'u must be monotonic'
            r = eval_r(u=u, q=tiler(q, u))
        else:
            raise TypeError("Please provide either r or u. Use np.inf for infinity.")

        assert np.sum(u == 0) <= 1 and np.sum(u[1:-1] == 0) == 0, "There can only be one r=np.inf location, either at the beginning or at the end."

        # Start from r=infinity
        if u[0] == 0:

            if theta1 is not None and theta2 is not None and S is None and J is None and kappa is None and chieff is None:
                kappa, chieff = angles_to_asymptotic(theta1, theta2, q, chi1, chi2)
                theta1inf, theta2inf = theta1, theta2

            elif theta1 is None and theta2 is None and deltaphi is None and J is None and kappa is not None and chieff is not None:
                theta1inf, theta2inf = asymptotic_to_angles(kappa, chieff, q, chi1, chi2)

            else:
                raise TypeError("Integrating from infinite separation. Please provide either (theta1,theta2) or (kappa,chieff).")

            # Enforce limits
            kappainfmin, kappainfmax = kappainflimits(chieff=chieff, q=q, chi1=chi1, chi2=chi2, enforce=True)
            assert kappa > kappainfmin and kappa < kappainfmax, "Unphysical initial conditions [inspiral_precav]."

        # Start from finite separations
        else:

            # User provides theta1,theta2, and deltaphi.
            if theta1 is not None and theta2 is not None and deltaphi is not None and S is None and J is None and kappa is None and chieff is None:
                S, J, chieff = angles_to_conserved(theta1, theta2, deltaphi, r[0], q, chi1, chi2)
                kappa = eval_kappa(J, r[0], q)

            # User provides J, chieff, and maybe S.
            elif theta1 is None and theta2 is None and deltaphi is None and J is not None and kappa is None and chieff is not None:
                kappa = eval_kappa(J, r[0], q)

            # User provides kappa, chieff, and maybe S.
            elif theta1 is None and theta2 is None and deltaphi is None and J is None and kappa is not None and chieff is not None:
                J = eval_J(kappa=kappa, r=r[0], q=q)

            else:
                TypeError("Integrating from finite separations. Please provide one and not more of the following: (theta1,theta2,deltaphi), (J,chieff), (S,J,chieff), (kappa,chieff), (S,kappa,chieff).")

            # Enforce limits
            Jmin, Jmax = Jlimits(r=r[0], chieff=chieff, q=q, chi1=chi1, chi2=chi2, enforce=True)
            assert J > Jmin and J < Jmax, "Unphysical initial conditions [inspiral_precav]."

        # TODO: pass rtol and atol to integrator_precav

        # Integration. Return interpolant along the solution
        kappa = integrator_precav(kappa, u, chieff, q, chi1, chi2)[0]
        # Evaluate the interpolant at the requested values of u
        #kappa = np.squeeze(ODEsolution.item().sol(u))
        # Select finite separations
        rok = r[u != 0]
        kappaok = kappa[u != 0]

        # Resample S and assign random sign to deltaphi
        J = eval_J(kappa=kappaok, r=rok, q=tiler(q, rok))
        S = Ssampling(J, rok, tiler(chieff, rok), tilere(q, rok),
        tiler(chi1, rok), tiler(chi2, rok), N=1)
        theta1, theta2, deltaphi = conserved_to_angles(S, J, rok, chieff, tiler(q, rok),
        tiler(chi1, rok), tiler(chi2, rok))
        deltaphi = deltaphi * np.random.choice([-1, 1], deltaphi.shape)

        # Integrating from infinite separation.
        if u[0] == 0:
            J = np.concatenate(([np.inf], J))
            S = np.concatenate(([np.nan], S))
            theta1 = np.concatenate((np.atleast_1d(theta1inf), theta1))
            theta2 = np.concatenate((np.atleast_1d(theta2inf), theta2))
            deltaphi = np.concatenate(([np.nan], deltaphi))
        # Integrating backwards to infinity
        elif u[-1] == 0:
            J = np.concatenate((J, [np.inf]))
            S = np.concatenate((S, [np.nan]))
            theta1inf, theta2inf = asymptotic_to_angles(kappa[-1], chieff, q, chi1, chi2)
            theta1 = np.concatenate((theta1, theta1inf))
            theta2 = np.concatenate((theta2, theta2inf))
            deltaphi = np.concatenate((deltaphi, [np.nan]))
        else:
            pass

        return theta1, theta2, deltaphi, S, J, kappa, r, u, chieff, q, chi1, chi2

    # This array has to match the outputs of _compute (in the right order!)
    alloutputs = np.array(['theta1', 'theta2', 'deltaphi', 'S', 'J', 'kappa', 'r', 'u', 'chieff', 'q', 'chi1', 'chi2'])

    # Here I force dtype=object because the outputs have different shapes
    allresults = np.array(list(map(_compute, theta1, theta2, deltaphi, S, J, kappa, r, u, chieff, q, chi1, chi2)), dtype=object).T

    # Handle the outputs.
    # If in doubt, return everything
    if requested_outputs is None:
        requested_outputs = alloutputs
    # Return only those requested (in1d return boolean array)
    wantoutputs = np.in1d(alloutputs, requested_outputs)

    # Store into a dictionary
    outcome = {}

    for k, v in zip(alloutputs[wantoutputs], allresults[wantoutputs]):
        outcome[k] = np.squeeze(np.stack(v))

        if k == 'chieff' or k == 'q' or k == 'chi1' or k == 'chi2':  # Constants of motion
            outcome[k] = np.atleast_1d(outcome[k])
        else:
            outcome[k] = np.atleast_2d(outcome[k])

    return outcome

# TODO: Add an exmple to the docstrings
def precession_average(J, r, chieff, q, chi1, chi2, func, *args, method='quadrature', Nsamples=1e4):
    """
    Average a generic function over a precession cycle. The function needs to have call: func(S, *args). Keywords arguments are not supported.

    There are integration methods implemented:
    - method='quadrature' uses scipy.integrate.quad. This is set by default and should be preferred.
    - method='montecarlo' samples t(S) and approximate the integral with a Monte Carlo sum. The number of samples can be specifed by Nsamples.

    Call
    ----
    func_av = precession_average(J,r,chieff,q,chi1,chi2,func,*args,method='quadrature',Nsamples=1e4)

    Parameters
    ----------
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    func: function
        Function to precession-average.
    *args: tuple
        Extra arguments to pass to func.
    method: string (default: 'quadrature')
        Either 'quadrature' or 'montecarlo'
    Nsamples: integer (default: 1e4)
        Number of Monte Carlo samples.

    Returns
    -------
    func_av: float
        Precession averaged value of func.
    """

    if method == 'quadrature':

        Sminuss, Spluss, S3s = Ssroots(J, r, chieff, q, chi1, chi2)
        m = elliptic_parameter_old(Sminuss, Spluss, S3s)
        # This is proportional to tau, takes care of the denominator
        tau_prop = scipy.special.ellipk(m) / ((Spluss-S3s)**0.5)

        # Each args needs to be iterable
        args = [np.atleast_1d(a) for a in args]

        # Compute the numerator explicitely
        def _integrand(S, Sminuss, Spluss, S3s, *sargs):
            # This is proportional to dSdt
            dSdt_prop = (-(S**2-Spluss) * (S**2-Sminuss) * (S**2-S3s))**0.5 / S
            return func(S, *sargs) / dSdt_prop

        def _compute(Sminuss, Spluss, S3s, *sargs):
            return scipy.integrate.quad(_integrand, Sminuss**0.5, Spluss**0.5, args=(Sminuss, Spluss, S3s, *sargs))[0]

        func_av = np.array(list(map(_compute, Sminuss, Spluss, S3s, *args))) / tau_prop

    elif method == 'montecarlo':

        S = np.transpose(Ssampling(J, r, chieff, q, chi1, chi2, N=int(Nsamples)))
        evals = np.transpose(func(S, *args))
        func_av = np.sum(evals, axis=-1)/Nsamples
        func_av = np.atleast_1d(func_av)

    else:
        raise ValueError("Available methods are 'quadrature' and 'montecarlo'.")

    return func_av

# TODO Add updown endpoint.
# TODO Add limits of the resonances at small separations from the endpoint paper


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
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    rudp: float
        Outer orbital separation in the up-down instability.
    rudm: float
        Inner orbital separation in the up-down instability.
    """

    q = np.atleast_1d(q)
    chi1 = np.atleast_1d(chi1)
    chi2 = np.atleast_1d(chi2)

    #Ignore q=1 "divide by zero" warning here
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
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
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    which: string
        Select function behavior.

    Returns
    -------
    omegasq: float
        Squared frequency.
    """

    q = np.atleast_1d(q)

    # These are all the valid input flags
    uulabels = np.array(['uu', 'up-up', 'upup', '++'])
    udlabels = np.array(['ud', 'up-down', 'updown', '+-'])
    dulabels = np.array(['du', 'down-up', 'downup', '-+'])
    ddlabels = np.array(['dd', 'down-down', 'downdown', '--'])

    assert np.isin(which, np.concatenate([uulabels, udlabels, dulabels, ddlabels])).all(), "Set `which` flag to either uu, ud, du, or dd."

    # +1 if primary is co-aligned, -1 if primary is counter-aligned
    alpha1 = np.where(np.isin(which, np.concatenate([uulabels, udlabels])), 1, -1)
    # +1 if secondary is co-aligned, -1 if secondary is counter-aligned
    alpha2 = np.where(np.isin(which, np.concatenate([uulabels, dulabels])), 1, -1)

    L = eval_L(r, q)
    S1, S2 = spinmags(q, chi1, chi2)
    # Slightly rewritten from Eq. 18 in arXiv:2003.02281, regularized for q=1
    omegasq = (3 * q**5 / (2 * (1 + q)**11 * L**7))**2 * (L - (q *
        alpha1 * S1 + alpha2 * S2) / (1 + q))**2 * (L**2 * (1 - q)**2 -
        2 * L * (q * alpha1 * S1 - alpha2 * S2) * (1 - q) + (q * alpha1 *
        S1 + alpha2 * S2)**2)

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
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    r_wide: float
        Orbital separation where wide nutations becomes possible.
    """

    q = np.atleast_1d(q)
    chi1 = np.atleast_1d(chi1)
    chi2 = np.atleast_1d(chi2)

    rwide = ((q*chi2 - chi1) / (1-q))**2

    return rwide

# TODO: write function with values of J and chieff where wide nutation happens

# Orbit averaged things

# TODO: replace quadrupole_formula flag with parameter to select a given PN order. Update docstrings when you do it


def rhs_orbav(allvars, v, q, m1, m2, eta, chi1, chi2, S1, S2, quadrupole_formula=False):
    """
    Right-hand side of the systems of ODEs describing orbit-averaged inspiral. The equations are reported in Sec 4A of Gerosa and Kesden, arXiv:1605.01067. The format is d[allvars]/dv=RHS where allvars=[Lhx,Lhy,Lhz,S1hx,S1hy,S1hz,S2hx,S2hy,S2hz,t], h indicates unite vectors, v is the orbital velocity, and t is time. This is an internal function used by the ODE integrator and is not array-compatible.

    Call
    ----
    RHS = rhs_orbav(v,allvars,q,m1,m2,eta,chi1,chi2,S1,S2,quadrupole_formula=False)

    Parameters
    ----------
    v: float
        Newtonian orbital velocity.
    allvars: array
        Packed ODE input variables.
    q: float
        Mass ratio: 0<=q<=1.
    m1: float
        Mass of the primary (heavier) black hole.
    m2: float
        Mass of the secondary (lighter) black hole.
    eta: float
        Symmetric mass ratio 0<=eta<=1/4.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    S1: float
        Magnitude of the primary spin.
    S2: float
        Magnitude of the secondary spin.
    MISSING: COULD NOT BUILD, optional (default: False)
        FILL MANUALLY.

    Returns
    -------
    RHS: float
        Right-hand side.
    """

    # Unpack inputs
    print(v)
    Lh = allvars[0:3]
    S1h = allvars[3:6]
    S2h = allvars[6:9]
    t = allvars[9]

    # Angles
    ct1 = np.dot(S1h, Lh)
    ct2 = np.dot(S2h, Lh)
    ct12 = np.dot(S1h, S2h)

    # Spin precession for S1
    Omega1 = eta*v**5*(2+3*q/2)*Lh + v**6*(S2*S2h-3*S2*ct2*Lh-3*q*S1*ct1*Lh)/2
    dS1hdt = np.cross(Omega1, S1h)

    # Spin precession for S2
    Omega2 = eta*v**5*(2+3/(2*q))*Lh + v**6*(S1*S1h-3*S1*ct1*Lh-3*S2*ct2*Lh/q)/2
    dS2hdt = np.cross(Omega2, S2h)

    # Conservation of angular momentum
    dLhdt = -v*(S1*dS1hdt+S2*dS2hdt)/eta

    # Radiation reaction
    if quadrupole_formula:  # Use to switch off higher-order terms
        dvdt = (32*eta*v**9/5)
    else:
        dvdt = (32*eta*v**9/5) * (1
            - v**2 * (743+924*eta)/336
            + v**3 * (4*np.pi
                     - chi1*ct1*(113*m1**2/12 + 25*eta/4)
                     - chi2*ct2*(113*m2**2/12 + 25*eta/4))
            + v**4 * (34103/18144 + 13661*eta/2016 + 59*eta**2/18
                     + eta*chi1*chi2 * (721*ct1*ct2 - 247*ct12)/48
                     + ((m1*chi1)**2 * (719*ct1**2-233))/96
                     + ((m2*chi2)**2 * (719*ct2**2-233))/96)
            - v**5 * np.pi*(4159+15876*eta)/672
            + v**6 * (16447322263/139708800 + 16*np.pi**2/3
                     - 1712*(0.5772156649+np.log(4*v))/105
                     + (451*np.pi**2/48 - 56198689/217728)*eta
                     + 541*eta**2/896 - 5605*eta**3/2592)
            + v**7 * np.pi*(-4415/4032 + 358675*eta/6048
                     + 91495*eta**2/1512))

    # Integrate in v, not in time
    dtdv = 1./dvdt
    dLhdv = dLhdt*dtdv
    dS1hdv = dS1hdt*dtdv
    dS2hdv = dS2hdt*dtdv

    # Pack outputs
    return np.concatenate([dLhdv, dS1hdv, dS2hdv, [dtdv]])


# TODO: update docstrings when you fix the quadrupole_formula flag
#Fix v instead of v initial v final
# Pass rtol and atol
def integrator_orbav(Lhinitial, S1hinitial, S2hinitial, v, q, chi1, chi2, quadrupole_formula=False, **odeint_kwargs):
    """
    Integration of the systems of ODEs describing orbit-averaged inspirals. Integration is performed in a reference frame
    where the z axis is along J and L lies in the x-z plane at the initial separation.

    Call
    ----
    ODEsolution = integrator_orbav(Lhinitial,S1hinitial,S2hinitial,vinitial,vfinal,q,chi1,chi2,quadrupole_formula=False)

    Parameters
    ----------
    Lhinitial: array
        Initial direction of the orbital angular momentum, unit vector.
    S1hinitial: array
        Initial direction of the primary spin, unit vector.
    S2hinitial: array
        Initial direction of the secondary spin, unit vector.
    vinitial: float
        Initial value of the newtonian orbital velocity.
    vfinal: float
        Final value of the newtonian orbital velocity.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    MISSING: COULD NOT BUILD, optional (default: False)
        FILL MANUALLY.

    Returns
    -------
    ODEsolution: array of scipy OdeSolution objects
        Solution of the ODE. Key method is .sol(t).
    """

    Lhinitial = np.atleast_2d(Lhinitial)
    S1hinitial = np.atleast_2d(S1hinitial)
    S2hinitial = np.atleast_2d(S2hinitial)
    v = np.atleast_2d(v)
    q = np.atleast_1d(q)
    chi1 = np.atleast_1d(chi1)
    chi2 = np.atleast_1d(chi2)

    # Defaults for the integrators, can be changed by the user
    if 'mxstep' not in odeint_kwargs: odeint_kwargs['mxstep']=5000000
    if 'rol' not in odeint_kwargs: odeint_kwargs['rtol']=1e-10
    if 'aol' not in odeint_kwargs: odeint_kwargs['atol']=1e-10
    odeint_kwargs['full_output']=0 # This needs to be forced for compatibility with the rest of the code

    def _compute(Lhinitial, S1hinitial, S2hinitial, v, q, chi1, chi2):

        # I need unit vectors
        assert np.isclose(np.linalg.norm(Lhinitial), 1)
        assert np.isclose(np.linalg.norm(S1hinitial), 1)
        assert np.isclose(np.linalg.norm(S2hinitial), 1)

        # Pack inputs
        ic = np.concatenate([Lhinitial, S1hinitial, S2hinitial, [0]])

        # Compute these quantities here instead of inside the RHS for speed
        m1 = eval_m1(q).item()
        m2 = eval_m2(q).item()
        S1 = eval_S1(q, chi1).item()
        S2 = eval_S2(q, chi2).item()
        eta = eval_eta(q).item()

        # Integration
        # t0=time.time()
        # res =scipy.integrate.odeint(rhs_orbav, ic, v, args=(q, m1, m2, eta, chi1, chi2, S1, S2, tracktime, quadrupole_formula), mxstep=5000000, full_output=0, printmessg=0, rtol=1e-12, atol=1e-12)
        # print(time.time()-t0)
        #print(ic)
        #ODEsolution = scipy.integrate.solve_ivp(rhs_orbav, (vinitial, vfinal), ic, method='LSODA', t_eval=(vinitial, vfinal), dense_output=True, args=(q, m1, m2, eta, chi1, chi2, S1, S2, quadrupole_formula),rtol=1e-12,atol=1e-12)
        #ODEsolution = scipy.integrate.solve_ivp(rhs_orbav, (vinitial, vfinal), ic, t_eval=(vinitial, vfinal), dense_output=True, args=(q, m1, m2, eta, chi1, chi2, S1, S2, quadrupole_formula))

        #print(odeint_kwargs)
        #sys.exit()

        # Make sure the first step is large enough. This is to avoid LSODA to propose a tiny step which causes the integration to stall
        if 'h0' not in odeint_kwargs: odeint_kwargs['h0']=v[0]/1e6

        ODEsolution = scipy.integrate.odeint(rhs_orbav, ic, v, args=(q, m1, m2, eta, chi1, chi2, S1, S2, quadrupole_formula), **odeint_kwargs)#, printmessg=0,rtol=1e-10,atol=1e-10)#,tcrit=sing)
        return ODEsolution

    ODEsolution = np.array(list(map(_compute, Lhinitial, S1hinitial, S2hinitial, v, q, chi1, chi2)))

    return ODEsolution


# TODO: update docstrings when you fix the quadrupole_formula flag
# Docstrings odeing_kwargs
def inspiral_orbav(theta1=None, theta2=None, deltaphi=None, S=None, Lh=None, S1h=None, S2h=None, J=None, kappa=None, r=None, u=None, chieff=None, q=None, chi1=None, chi2=None, quadrupole_formula=False, requested_outputs=None, **odeint_kwargs):
    """
    Perform orbit-averaged inspirals. The variables q, chi1, and chi2 must always be provided. The integration range must be specified using either r or u (and not both). The initial conditions correspond to the binary at either r[0] or u[0]. The vector r or u needs to monotonic increasing or decreasing, allowing to integrate forwards and backwards in time. Orbit-averaged integration can only be done between finite separations.
    The initial conditions must be specified in terms of one an only one of the following:
    - Lh, S1h, and S2h
    - theta1,theta2, and deltaphi.
    - J, chieff, and S.
    - kappa, chieff, and S.
    The desired outputs can be specified with a list e.g. requested_outputs=['theta1','theta2','deltaphi']. All the available variables are returned by default. These are: ['t', 'theta1', 'theta2', 'deltaphi', 'S', 'Lh', 'S1h', 'S2h', 'J', 'kappa', 'r', 'u', 'chieff', 'q', 'chi1', 'chi2']

    Call
    ----
    outputs = inspiral_orbav(theta1=None,theta2=None,deltaphi=None,S=None,Lh=None,S1h=None,S2h=None,J=None,kappa=None,r=None,u=None,chieff=None,q=None,chi1=None,chi2=None,quadrupole_formula=False,requested_outputs=None)

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
    Lh: array, optional (default: None)
        Direction of the orbital angular momentum, unit vector.
    S1h: array, optional (default: None)
        Direction of the primary spin, unit vector.
    S2h: array, optional (default: None)
        Direction of the secondary spin, unit vector.
    J: float, optional (default: None)
        Magnitude of the total angular momentum.
    kappa: float, optional (default: None)
        Regularized angular momentum (J^2-L^2)/(2L).
    r: float, optional (default: None)
        Binary separation.
    u: float, optional (default: None)
        Compactified separation 1/(2L).
    chieff: float, optional (default: None)
        Effective spin.
    q: float, optional (default: None)
        Mass ratio: 0<=q<=1.
    chi1: float, optional (default: None)
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float, optional (default: None)
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    MISSING: COULD NOT BUILD, optional (default: False)
        FILL MANUALLY.
    requested_outputs: list, optional (default: None)
        Set of outputs.

    Returns
    -------
    outputs: dictionary
        Set of outputs.
    """

    # Substitute None inputs with arrays of Nones
    inputs = [theta1, theta2, deltaphi, S, Lh, S1h, S2h, J, kappa, r, u, chieff, q, chi1, chi2]
    for k, v in enumerate(inputs):
        if v is None:
            inputs[k] = np.atleast_1d(np.squeeze(tiler(None, np.atleast_1d(q))))
        else:
            if k == 4 or k == 5 or k == 6 or k == 9 or k == 10:  # Lh, S1h, S2h, u, or r
                inputs[k] = np.atleast_2d(inputs[k])
            else:  # Any of the others
                inputs[k] = np.atleast_1d(inputs[k])
    theta1, theta2, deltaphi, S, Lh, S1h, S2h, J, kappa, r, u, chieff, q, chi1, chi2 = inputs

    def _compute(theta1, theta2, deltaphi, S, Lh, S1h, S2h, J, kappa, r, u, chieff, q, chi1, chi2):

        if q is None or chi1 is None or chi2 is None:
            raise TypeError("Please provide q, chi1, and chi2.")

        if r is not None and u is None:
            assert np.logical_or(ismonotonic(r, '<='), ismonotonic(r, '>=')), 'r must be monotonic'
            u = eval_u(r, tiler(q, r))
        elif r is None and u is not None:
            assert np.logical_or(ismonotonic(u, '<='), ismonotonic(u, '>=')), 'u must be monotonic'
            r = eval_r(u=u, q=tiler(q, u))
        else:
            raise TypeError("Please provide either r or u.")

        # User provides Lh, S1h, and S2h
        if Lh is not None and S1h is not None and S2h is not None and theta1 is None and theta2 is None and deltaphi is None and S is None and J is None and kappa is None and chieff is None:
            pass

        # User provides theta1, theta2, and deltaphi.
        elif Lh is None and S1h is None and S2h is None and theta1 is not None and theta2 is not None and deltaphi is not None and S is None and J is None and kappa is None and chieff is None:
            Lh, S1h, S2h = angles_to_Jframe(theta1, theta2, deltaphi, r[0], q, chi1, chi2)

        # User provides J, chieff, and S.
        elif Lh is None and S1h is None and S2h is None and theta1 is None and theta2 is None and deltaphi is None and S is not None and J is not None and kappa is None and chieff is not None:
            # TODO: how do I set cyclesign here?
            Lh, S1h, S2h = conserved_to_Jframe(S, J, r[0], chieff, q, chi1, chi2)

        # User provides kappa, chieff, and S.
        elif Lh is None and S1h is None and S2h is None and theta1 is None and theta2 is None and deltaphi is None and S is not None and J is None and kappa is not None and chieff is not None:
            J = eval_J(kappa=kappa, r=r[0], q=q)
            # TODO: how do I set cyclesign here?
            Lh, S1h, S2h = conserved_to_Jframe(S, J, r[0], chieff, q, chi1, chi2)

        else:
            TypeError("Please provide one and not more of the following: (Lh,S1h,S2h), (theta1,theta2,deltaphi), (S,J,chieff), (S,kappa,chieff).")

        # Make sure vectors are normalized
        Lh = Lh/np.linalg.norm(Lh)
        S1h = S1h/np.linalg.norm(S1h)
        S2h = S2h/np.linalg.norm(S2h)

        v = eval_v(r)

        # Integration
        evaluations = integrator_orbav(Lh, S1h, S2h, v, q, chi1, chi2, quadrupole_formula=quadrupole_formula,**odeint_kwargs)[0].T

        #evaluations = np.squeeze(ODEsolution.item().sol(v))
        # Returned output is
        # Lx, Ly, Lz, S1x, S1y, S1z, S2x, S2y, S2z, (t)
        Lh = evaluations[0:3, :].T
        S1h = evaluations[3:6, :].T
        S2h = evaluations[6:9, :].T
        t = evaluations[9, :]
        # TODO: Should I renormalize here? The normalization is not enforced by the integrator, it is only maintaied within numerical accuracy.

        S1, S2 = spinmags(q, chi1, chi2)
        L = eval_L(r, tiler(q, r))
        Lvec = (L*Lh.T).T
        S1vec = S1*S1h
        S2vec = S2*S2h

        theta1, theta2, deltaphi = vectors_to_angles(Lvec, S1vec, S2vec)
        S, J, chieff = vectors_to_conserved(Lvec, S1vec, S2vec, q)
        kappa = eval_kappa(J, r, q)

        return t, theta1, theta2, deltaphi, S, Lh, S1h, S2h, J, kappa, r, u, chieff, q, chi1, chi2

    # This array has to match the outputs of _compute (in the right order!)
    alloutputs = np.array(['t', 'theta1', 'theta2', 'deltaphi', 'S', 'Lh', 'S1h', 'S2h', 'J', 'kappa', 'r', 'u', 'chieff', 'q', 'chi1', 'chi2'])

    # Here I force dtype=object because the outputs have different shapes
    allresults = np.array(list(map(_compute, theta1, theta2, deltaphi, S, Lh, S1h, S2h, J, kappa, r, u, chieff, q, chi1, chi2)), dtype=object).T

    # Handle the outputs.
    # Return all
    if requested_outputs is None:
        requested_outputs = alloutputs
    # Return only those requested (in1d return boolean array)
    wantoutputs = np.in1d(alloutputs, requested_outputs)

    # Store into a dictionary
    outcome = {}
    for k, v in zip(alloutputs[wantoutputs], allresults[wantoutputs]):
        outcome[k] = np.squeeze(np.stack(v))

        if k == 'q' or k == 'chi1' or k == 'chi2':  # Constants of motion
            outcome[k] = np.atleast_1d(outcome[k])
        else:
            outcome[k] = np.atleast_2d(outcome[k])

    return outcome


def inspiral_hybrid(theta1=None, theta2=None, deltaphi=None, S=None, J=None, kappa=None, r=None, rswitch=None, u=None, uswitch=None, chieff=None, q=None, chi1=None, chi2=None, requested_outputs=None):
    """
    Perform hybrid inspirals, i.e. evolve the binary at large separation with a pression-averaged evolution and at small separation with an orbit-averaged evolution, properly matching the two. The variables q, chi1, and chi2 must always be provided. The integration range must be specified using either r or u (and not both); provide also uswitch and rswitch consistently. The initial conditions correspond to the binary at either r[0] or u[0]. The vector r or u needs to monotonic increasing or decreasing, allowing to integrate forwards and backwards in time. If integrating forwards in time, perform the precession-average evolution first and then swith to orbit averaging.  If integrating backwards in time, perform the orbit-average evolution first and then swith to precession averaging. For infinitely large separation in the precession-averaged case, use r=np.inf or u=0. The switch value will not part of the output unless it is also present in the r/u array.
    The initial conditions must be specified in terms of one an only one of the following:
    - theta1,theta2, and deltaphi (but note that deltaphi is not necessary if integrating from infinite separation).
    - J, chieff (only if integrating from finite separations because J otherwise diverges).
    - kappa, chieff.
    The desired outputs can be specified with a list e.g. requested_outputs=['theta1','theta2','deltaphi']. All the available variables are returned by default. These are: ['theta1', 'theta2', 'deltaphi', 'S', 'J', 'kappa', 'r', 'u', 'chieff', 'q', 'chi1', 'chi2'].

    Call
    ----
    outputs = inspiral_hybrid(theta1=None,theta2=None,deltaphi=None,S=None,J=None,kappa=None,r=None,rswitch=None,u=None,uswitch=None,chieff=None,q=None,chi1=None,chi2=None,requested_outputs=None)

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
    rswitch: float, optional (default: None)
        Matching separation between the precession- and orbit-averaged chunks.
    u: float, optional (default: None)
        Compactified separation 1/(2L).
    uswitch: float, optional (default: None)
        Matching compactified separation between the precession- and orbit-averaged chunks.
    chieff: float, optional (default: None)
        Effective spin.
    q: float, optional (default: None)
        Mass ratio: 0<=q<=1.
    chi1: float, optional (default: None)
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float, optional (default: None)
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    requested_outputs: list, optional (default: None)
        Set of outputs.

    Returns
    -------
    outputs: dictionary
        Set of outputs.
    """

    # Outputs available in both orbit-averaged and precession-averaged evolutions
    alloutputs = np.array(['theta1', 'theta2', 'deltaphi', 'S', 'J', 'kappa', 'r', 'u', 'chieff', 'q', 'chi1', 'chi2'])
    if requested_outputs is None:
        requested_outputs = alloutputs
        # Return only those requested (in1d return boolean array)
    wantoutputs = np.intersect1d(alloutputs, requested_outputs)

    # Substitute None inputs with arrays of Nones
    inputs = [theta1, theta2, deltaphi, S, J, kappa, r, rswitch, u, uswitch, chieff, q, chi1, chi2]
    for k, v in enumerate(inputs):
        if v is None:
            inputs[k] = np.atleast_1d(np.squeeze(tiler(None, np.atleast_1d(q))))
        else:
            if k == 6 or k == 8:  # Either u or r
                inputs[k] = np.atleast_2d(inputs[k])
            else:  # Any of the others
                inputs[k] = np.atleast_1d(inputs[k])
    theta1, theta2, deltaphi, S, J, kappa, r, rswitch, u, uswitch, chieff, q, chi1, chi2 = inputs

    def _compute(theta1, theta2, deltaphi, S, J, kappa, r, rswitch, u, uswitch, chieff, q, chi1, chi2):

        if r is None and rswitch is None and u is not None and uswitch is not None:
            r = eval_r(u=u, q=tiler(q, u))
            rswitch = eval_r(u=uswitch, q=tiler(q, uswitch))

        forwards = ismonotonic(r, ">=")
        backwards = ismonotonic(r, "<=")

        assert np.logical_or(forwards, backwards), "r must be monotonic"
        assert rswitch > np.min(r) and rswitch < np.max(r), "The switching condition must to be within the range spanned by r or u."

        rlarge = r[r >= rswitch]
        rsmall = r[r < rswitch]

        # Integrating forwards: precession average first, then orbit average
        if forwards:
            inspiral_first = inspiral_precav
            rfirst = np.append(rlarge, rswitch)
            inspiral_second = inspiral_orbav
            rsecond = np.append(rswitch, rsmall)

        # Integrating backwards: orbit average first, then precession average
        elif backwards:
            inspiral_first = inspiral_orbav
            rfirst = np.append(rsmall, rswitch)
            inspiral_second = inspiral_precav
            rsecond = np.append(rswitch, rlarge)

        # First chunk of the evolution
        evolution_first = inspiral_first(theta1=theta1, theta2=theta2, deltaphi=deltaphi, S=S, J=J, kappa=kappa, r=rfirst, chieff=chieff, q=q, chi1=chi1, chi2=chi2, requested_outputs=alloutputs)

        # Second chunk of the evolution
        evolution_second = inspiral_second(theta1=evolution_first['theta1'][-1], theta2=evolution_first['theta2'][-1], deltaphi=evolution_first['deltaphi'][-1], r=rsecond, q=q, chi1=chi1, chi2=chi2, requested_outputs=alloutputs)

        # Store outputs
        evolution_full = {}
        for k in wantoutputs:
            # Quantities that vary in both the precession-averaged and the orbit-averaged evolution
            if k in ['theta1', 'theta2', 'deltaphi', 'S', 'J', 'kappa', 'r', 'u']:
                evolution_full[k] = np.atleast_2d(np.append(evolution_first[k][:, :-1], evolution_second[k][:, 1:]))
            # Quantities that vary only on the orbit-averaged evolution
            if k in ['chieff']:
                if forwards:
                    evolution_full[k] = np.atleast_2d(np.append(tiler(evolution_first[k][:], rfirst[:-1]), evolution_second[k][:, 1:]))
                elif backwards:
                    evolution_full[k] = np.atleast_2d(np.append(evolution_first[k][:, :-1], tiler(evolution_second[k][:], rsecond[1:])))
            # Quanties that do not vary
            if k in ['q', 'chi1', 'chi2']:
                evolution_full[k] = evolution_second[k]

        return evolution_full

    allresults = list(map(_compute, theta1, theta2, deltaphi, S, J, kappa, r, rswitch, u, uswitch, chieff, q, chi1, chi2))
    evolution_full = {}
    for k in allresults[0].keys():
        evolution_full[k] = np.concatenate(list(evolution_full[k] for evolution_full in allresults))

    return evolution_full


def inspiral(*args, which=None, **kwargs):
    """
    TODO write docstings. This is the ultimate wrapper the user should call.
    """

    # Precession-averaged integrations
    if which in ['precession', 'precav', 'precessionaveraged', 'precessionaverage', 'precession-averaged', 'precession-average', 'precessionav']:
        return inspiral_precav(*args, **kwargs)

    elif which in ['orbit', 'orbav', 'orbitaveraged', 'orbitaverage', 'orbit-averaged', 'orbit-average', 'orbitav']:
        return inspiral_orbav(*args, **kwargs)

    elif which in ['hybrid']:
        return inspiral_hybrid(*args, **kwargs)

    else:
        raise ValueError("`which` needs to be `precav`, `orbav` or `hybrid`.")


def frequency_prefactor(J, r, chieff, q, chi1, chi2):
    """
    Numerical prefactors entering the precession frequency.

    Call
    ----
    mathcalC0,mathcalCplus,mathcalCminus = frequency_prefactor(J,r,chieff,q,chi1,chi2)

    Parameters
    ----------
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    mathcalC0: float
        Prefactor in the OmegaL equation.
    mathcalCplus: float
        Prefactor in the OmegaL equation.
    mathcalCminus: float
        Prefactor in the OmegaL equation.
    """

    J = np.atleast_1d(J)
    chieff = np.atleast_1d(chieff)
    q = np.atleast_1d(q)
    S1, S2 = spinmags(q, chi1, chi2)
    L = eval_L(r, q)
    eta = eval_eta(q)

    mathcalC0 = (J/2)*(eta/L)**6
    mathcalCplus = 3/2 * (L*(1+q)**2 - q*chieff)/(J*q*(1+q)**2) * ((1+q)*((1+q)*(J+L)**2 - (1-q)*(S1**2-S2**2)) + 2*q*chieff*(L+J))
    mathcalCminus = - 3/2 * (L*(1+q)**2 - q*chieff)/(J*q*(1+q)**2) * ((1+q)*((1+q)*(J-L)**2 - (1-q)*(S1**2-S2**2)) + 2*q*chieff*(L-J))

    return np.stack([mathcalC0, mathcalCplus, mathcalCminus])


def azimuthalangle_prefactor(J, r, chieff, q, chi1, chi2, precomputedroots=None):
    """
    Numerical prefactors entering the precession frequency.

    Call
    ----
    mathcalC0prime,mathcalCplusprime,mathcalCminusprime = azimuthalangle_prefactor(J,r,chieff,q,chi1,chi2,precomputedroots=None)

    Parameters
    ----------
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    precomputedroots: array, optional (default: None)
        Pre-computed output of Ssroots for computational efficiency.

    Returns
    -------
    mathcalC0prime: float
        Prefactor in the PhiL equation.
    mathcalCplusprime: float
        Prefactor in the PhiL equation.
    mathcalCminusprime: float
        Prefactor in the PhiL equation.
    """

    J = np.atleast_1d(J)
    L = eval_L(r, q)

    Sminuss, Spluss, S3s = Ssroots(J, r, chieff, q, chi1, chi2, precomputedroots=precomputedroots)

    mathcalC0, mathcalCplus, mathcalCminus = frequency_prefactor(J, r, chieff, q, chi1, chi2)
    mathcalT = time_normalization(Spluss, S3s, r, chieff, q)

    mathcalC0prime = mathcalT*mathcalC0
    mathcalCplusprime = -mathcalT*mathcalC0*mathcalCplus/(Spluss - (J+L)**2)
    mathcalCminusprime = -mathcalT*mathcalC0*mathcalCminus/(Spluss - (J-L)**2)

    return np.stack([mathcalC0prime, mathcalCplusprime, mathcalCminusprime])


def eval_OmegaL(S, J, r, chieff, q, chi1, chi2):
    """
    Compute the precession frequency OmegaL along the precession cycle.

    Call
    ----
    OmegaL = eval_OmegaL(S,J,r,chieff,q,chi1,chi2)

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    OmegaL: float
        Precession frequency of L about J.
    """

    S = np.atleast_1d(S)
    J = np.atleast_1d(J)
    L = eval_L(r, q)

    mathcalC0, mathcalCplus, mathcalCminus = frequency_prefactor(J, r, chieff, q, chi1, chi2)

    OmegaL = mathcalC0 * (1 + mathcalCplus/((J+L)**2 - S**2) + mathcalCminus/((J-L)**2 - S**2))

    return OmegaL


def eval_alpha(J, r, chieff, q, chi1, chi2, precomputedroots=None):
    """
    Compute the azimuthal angle spanned by L about J during an entire nutation cycle.

    Call
    ----
    alpha = eval_alpha(J,r,chieff,q,chi1,chi2,precomputedroots=None)

    Parameters
    ----------
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    precomputedroots: array, optional (default: None)
        Pre-computed output of Ssroots for computational efficiency.

    Returns
    -------
    alpha: float
        Azimuthal angle spanned by L about J during an entire cycle.
    """

    L = eval_L(r, q)
    Sminuss, Spluss, S3s = Ssroots(J, r, chieff, q, chi1, chi2, precomputedroots=precomputedroots)
    m = elliptic_parameter_old(Sminuss, Spluss, S3s)
    nplus = elliptic_characheristic(Sminuss, Spluss, J, L, +1)
    nminus = elliptic_characheristic(Sminuss, Spluss, J, L, -1)
    mathcalC0prime, mathcalCplusprime, mathcalCminusprime = azimuthalangle_prefactor(J, r, chieff, q, chi1, chi2, precomputedroots=np.stack([Sminuss, Spluss, S3s]))

    alpha = 2*(mathcalC0prime*scipy.special.ellipk(m) + mathcalCplusprime*ellippi(nplus, np.pi/2, m) + mathcalCminusprime*ellippi(nminus, np.pi/2, m))

    return alpha


def eval_phiL(S, J, r, chieff, q, chi1, chi2, cyclesign=1, precomputedroots=None):
    """
    Compute the azimuthal angle spanned by L about J. This is the integral of the frequency OmegaL.

    Call
    ----
    phiL = eval_phiL(S,J,r,chieff,q,chi1,chi2,cyclesign=1,precomputedroots=None)

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    J: float
        Magnitude of the total angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    cyclesign: integer, optional (default: 1)
        Sign (either +1 or -1) to cover the two halves of a precesion cycle.
    precomputedroots: array, optional (default: None)
        Pre-computed output of Ssroots for computational efficiency.

    Returns
    -------
    phiL: float
        Azimuthal angle spanned by L about J.
    """

    L = eval_L(r, q)
    Sminuss, Spluss, S3s = Ssroots(J, r, chieff, q, chi1, chi2, precomputedroots=precomputedroots)
    alpha = eval_alpha(J, r, chieff, q, chi1, chi2, precomputedroots=np.stack([Sminuss, Spluss, S3s]))
    m = elliptic_parameter_old(Sminuss, Spluss, S3s)
    phi = elliptic_amplitude(S, Sminuss, Spluss)
    nplus = elliptic_characheristic(Sminuss, Spluss, J, L, +1)
    nminus = elliptic_characheristic(Sminuss, Spluss, J, L, -1)
    mathcalC0prime, mathcalCplusprime, mathcalCminusprime = azimuthalangle_prefactor(J, r, chieff, q, chi1, chi2, precomputedroots=np.stack([Sminuss, Spluss, S3s]))

    phiL = alpha/2 - np.sign(cyclesign)*(mathcalC0prime*scipy.special.ellipkinc(phi, m) + mathcalCplusprime*ellippi(nplus, phi, m) + mathcalCminusprime*ellippi(nminus, phi, m))

    return phiL


def chip_terms(theta1, theta2, q, chi1, chi2):
    """
    Compute the two terms entering the effective precessing spin chip.

    Call
    ----
    chipterm1,chipterm2 = chip_terms(theta1,theta2,q,chi1,chi2)

    Parameters
    ----------
    theta1: float
        Angle between orbital angular momentum and primary spin.
    theta2: float
        Angle between orbital angular momentum and secondary spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    chipterm1: float
        Term in effective precessing spin chip.
    chipterm2: float
        Term in effective precessing spin chip.
    """

    theta1 = np.atleast_1d(theta1)
    theta2 = np.atleast_1d(theta2)
    q = np.atleast_1d(q)

    chipterm1 = chi1*np.sin(theta1)
    omegatilde = q*(4*q+3)/(4+3*q)
    chipterm2 = omegatilde * chi2*np.sin(theta2)

    return np.stack([chipterm1, chipterm2])


def eval_chip_heuristic(theta1, theta2, q, chi1, chi2):
    """
    Heuristic definition of the effective precessing spin chip (Schmidt et al 2015), see arxiv:2011.11948. This definition inconsistently averages over some, but not all, variations on the precession timescale.

    Call
    ----
    chip = eval_chip_heuristic(theta1,theta2,q,chi1,chi2)

    Parameters
    ----------
    theta1: float
        Angle between orbital angular momentum and primary spin.
    theta2: float
        Angle between orbital angular momentum and secondary spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    chip: float
        Effective precessing spin chip.
    """

    term1, term2 = chip_terms(theta1, theta2, q, chi1, chi2)
    chip = np.maximum(term1, term2)
    return chip


def eval_chip_generalized(theta1, theta2, deltaphi, q, chi1, chi2):
    """
    Generalized definition of the effective precessing spin chip, see arxiv:2011.11948. This definition retains all variations on the precession timescale.

    Call
    ----
    chip = eval_chip_generalized(theta1,theta2,deltaphi,q,chi1,chi2)

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
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    chip: float
        Effective precessing spin chip.
    """

    term1, term2 = chip_terms(theta1, theta2, q, chi1, chi2)
    chip = (term1**2 + term2**2 + 2*term1*term2*np.cos(deltaphi))**0.5
    return chip


def eval_chip_asymptotic(theta1, theta2, q, chi1, chi2):
    """
    Asymptotic definition of the effective precessing spin chip, see arxiv:2011.11948. This definition is valid when spin-spin couplings can be neglected, notably at infinitely large separations.

    Call
    ----
    chip = eval_chip_asymptotic(theta1,theta2,q,chi1,chi2)

    Parameters
    ----------
    theta1: float
        Angle between orbital angular momentum and primary spin.
    theta2: float
        Angle between orbital angular momentum and secondary spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    chip: float
        Effective precessing spin chip.
    """

    term1, term2 = chip_terms(theta1, theta2, q, chi1, chi2)
    chip = (np.abs(term1-term2) * scipy.special.ellipe(-4*term1*term2/(term1-term2)**2) + np.abs(term1+term2) * scipy.special.ellipe(4*term1*term2/(term1+term2)**2))/np.pi
    return chip


def eval_chip_averaged(theta1=None, theta2=None, deltaphi=None, J=None, r=None, chieff=None, q=None, chi1=None, chi2=None, method='quadrature', Nsamples=1e4):
    """
    Averaged definition of the effective precessing spin chip, see arxiv:2011.11948. This definition consistently averages over all variations on the precession timescale. Valid inputs are one of the following (but not both)
    - theta1, theta2, deltaphi
    - J, chieff
    The parameters r, q, chi1, and chi2 should always be provided. The keywords arguments method and Nsamples are passed directly to `precession_average`.

    Call
    ----
    chip = eval_chip_averaged(theta1=None,theta2=None,deltaphi=None,J=None,r=None,chieff=None,q=None,chi1=None,chi2=None,method='quadrature',Nsamples=1e4)

    Parameters
    ----------
    theta1: float, optional (default: None)
        Angle between orbital angular momentum and primary spin.
    theta2: float, optional (default: None)
        Angle between orbital angular momentum and secondary spin.
    deltaphi: float, optional (default: None)
        Angle between the projections of the two spins onto the orbital plane.
    J: float, optional (default: None)
        Magnitude of the total angular momentum.
    r: float, optional (default: None)
        Binary separation.
    chieff: float, optional (default: None)
        Effective spin.
    q: float, optional (default: None)
        Mass ratio: 0<=q<=1.
    chi1: float, optional (default: None)
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float, optional (default: None)
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    method: string (default: 'quadrature')
        Either 'quadrature' or 'montecarlo'
    Nsamples: integer (default: 1e4)
        Number of Monte Carlo samples.

    Returns
    -------
    chip: float
        Effective precessing spin chip.
    """

    if r is None or q is None or chi1 is None or chi2 is None:
        raise TypeError("Please provide r, q, chi1, and chi2.")

    if theta1 is not None and theta2 is not None and deltaphi is not None and J is None and chieff is None:
        # cyclesign doesn't matter here. Outout S is not needed
        _, J, chieff = angles_to_conserved(theta1, theta2, deltaphi, r, q, chi1, chi2)

    elif theta1 is None and theta2 is None and deltaphi is None and J is not None and chieff is not None:
        pass

    else:
        raise TypeError("Please provide either (theta1,theta2,deltaphi) or (J,chieff).")

    def _integrand(S, J, r, chieff, q, chi1, chi2):
        theta1, theta2, deltaphi = conserved_to_angles(S, J, r, chieff, q, chi1, chi2)
        chip_integrand = eval_chip_generalized(theta1, theta2, deltaphi, q, chi1, chi2)
        return chip_integrand

    chip = precession_average(J, r, chieff, q, chi1, chi2, _integrand, J, r, chieff, q, chi1, chi2, method=method, Nsamples=Nsamples)

    return chip


def eval_chip(theta1=None, theta2=None, deltaphi=None, J=None, r=None, chieff=None, q=None, chi1=None, chi2=None, which="averaged", method='quadrature', Nsamples=1e4):
    """
    Compute the effective precessing spin chip, see arxiv:2011.11948. The keyword `which` one of the following definitions:
    - `heuristic`, as in Schmidt et al 2015. Required inputs: theta1,theta2,q,chi1,chi2
    - `generalized`, retail all precession-timescale variations. Required inputs: theta1,theta2,deltaphi,q,chi1,chi2
    - `asymptotic`, large-separation limit. Required inputs: theta1,theta2,q,chi1,chi2
    - `averaged` (default), averages over all precession-timescale variations. Required inputs are either (theta1,theta2,deltaphi,r,q,chi1,chi2) or (J,r,chieff,q,chi1,chi2). The additional keywords `methods` and `Nsamples` are passed to `precession_average`.

    Call
    ----
    chip = eval_chip(theta1=None,theta2=None,deltaphi=None,J=None,r=None,chieff=None,q=None,chi1=None,chi2=None,which="averaged",method='quadrature',Nsamples=1e4)

    Parameters
    ----------
    theta1: float, optional (default: None)
        Angle between orbital angular momentum and primary spin.
    theta2: float, optional (default: None)
        Angle between orbital angular momentum and secondary spin.
    deltaphi: float, optional (default: None)
        Angle between the projections of the two spins onto the orbital plane.
    J: float, optional (default: None)
        Magnitude of the total angular momentum.
    r: float, optional (default: None)
        Binary separation.
    chieff: float, optional (default: None)
        Effective spin.
    q: float, optional (default: None)
        Mass ratio: 0<=q<=1.
    chi1: float, optional (default: None)
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float, optional (default: None)
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    which: string, optional (default: "averaged")
        Select function behavior.
    method: string (default: 'quadrature')
        Either 'quadrature' or 'montecarlo'
    Nsamples: integer (default: 1e4)
        Number of Monte Carlo samples.

    Returns
    -------
    chip: float
        Effective precessing spin chip.
    """

    if which == 'heuristic':
        chip = eval_chip_heuristic(theta1, theta2, q, chi1, chi2)

    elif which == 'generalized':
        chip = eval_chip_generalized(theta1, theta2, deltaphi, q, chi1, chi2)

    elif which == 'asymptotic':
        chip = eval_chip_asymptotic(theta1, theta2, q, chi1, chi2)

    elif which == 'averaged':
        chip = eval_chip_averaged(theta1=theta1, theta2=theta2, deltaphi=deltaphi, J=J, r=r, chieff=chieff, q=q, chi1=chi1, chi2=chi2, method='quadrature', Nsamples=1e4)

    else:
        raise ValueError("`which` needs to be one of the following: `heuristic`, `generalized`, `asymptotic`, `averaged`.")

    return chip


# TODO: insert flag to select PN order
def gwfrequency_to_pnseparation(theta1, theta2, deltaphi, f, q, chi1, chi2, M_msun):
    """
    Convert GW frequency in Hz to PN orbital separation in natural units (c=G=M=1). We use the 2PN expression reported in Eq. 4.13 of Kidder 1995, arxiv:gr-qc/9506022.

    Call
    ----
    r = gwfrequency_to_pnseparation(theta1,theta2,deltaphi,f,q,chi1,chi2,M_msun)

    Parameters
    ----------
    theta1: float
        Angle between orbital angular momentum and primary spin.
    theta2: float
        Angle between orbital angular momentum and secondary spin.
    deltaphi: float
        Angle between the projections of the two spins onto the orbital plane.
    f: float
        Gravitational-wave frequency in Hz.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    M_msun: float
        Total mass of the binary in solar masses.

    Returns
    -------
    r: float
        Binary separation.
    """

    theta1 = np.atleast_1d(theta1)
    theta2 = np.atleast_1d(theta2)
    f = np.atleast_1d(f)
    chi1 = np.atleast_1d(chi1)
    chi2 = np.atleast_1d(chi2)
    M_msun = np.atleast_1d(M_msun)

    # Convert GW frequency in hz to orbital velocity in natural units.
    # Prefactor is Msun*G/c^3/s with values of the constants as given by Mathematica:
    # https://www.wolframalpha.com/input/?i=Msun+*+G+%2Fc%5E3
    # Factor of pi and not 2pi because of f is the GW frequency while omega is the orbital angular velocity
    omega = 4.93e-6 * M_msun * np.pi * f

    m1, m2 = masses(q)
    eta = eval_eta(q)
    ct1 = np.cos(theta1)
    ct2 = np.cos(theta2)
    ct12 = eval_costheta12(theta1=theta1, theta2=theta2, deltaphi=deltaphi)
    # Eq. 4.13, Kidder 1995. gr-qc/9506022
    r = omega**(-2/3)*(1
            - (1/3)*(3-eta)*omega**(2/3)
            - (1/3)*(chi1*ct1*(2*m1**2+3*eta) + chi2*ct2*(2*m2**2+3*eta))*omega
            + (eta*(19/4 + eta/9) - eta*chi1*chi2/2 * (ct12 - 3*ct1*ct2))*omega**(4/3))
    return r


# TODO: insert flag to select PN order
def pnseparation_to_gwfrequency(theta1, theta2, deltaphi, r, q, chi1, chi2, M_msun):
    """
    Convert PN orbital separation in natural units (c=G=M=1) to GW frequency in Hz. We use the 2PN expression reported in Eq. 4.5 of Kidder 1995, arxiv:gr-qc/9506022.

    Call
    ----
    r = pnseparation_to_gwfrequency(theta1,theta2,deltaphi,f,q,chi1,chi2,M_msun)

    Parameters
    ----------
    theta1: float
        Angle between orbital angular momentum and primary spin.
    theta2: float
        Angle between orbital angular momentum and secondary spin.
    deltaphi: float
        Angle between the projections of the two spins onto the orbital plane.
    f: float
        Gravitational-wave frequency in Hz.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    M_msun: float
        Total mass of the binary in solar masses.

    Returns
    -------
    r: float
        Binary separation.
    """

    theta1 = np.atleast_1d(theta1)
    theta2 = np.atleast_1d(theta2)
    r = np.atleast_1d(r)
    chi1 = np.atleast_1d(chi1)
    chi2 = np.atleast_1d(chi2)
    M_msun = np.atleast_1d(M_msun)

    m1, m2 = masses(q)
    eta = eval_eta(q)
    ct1 = np.cos(theta1)
    ct2 = np.cos(theta2)
    ct12 = eval_costheta12(theta1=theta1, theta2=theta2, deltaphi=deltaphi)

    omegasquared = (1/r**3)*(1
        - (3-eta)/r
        - (chi1*ct1*(2*m1**2+3*eta) + chi2*ct2*(2*m2**2+3*eta))/r**(3/2)
        + (6 + 41/4*eta + eta**2 - 3/2*eta*chi1*chi2*(ct12-3*ct1*ct2))/r**(2))

    # Convert orbital velocity in natural units to GW frequency in Hz.
    # Prefactor is Msun*G/c^3/s with values of the constants as given by Mathematica:
    # https://www.wolframalpha.com/input/?i=Msun+*+G+%2Fc%5E3
    # Factor of pi and not 2pi because of f is the GW frequency while omega is the orbital angular velocity
    f = np.sqrt(omegasquared) / (4.93e-6 * M_msun * np.pi)

    return f


def remnantmass(theta1, theta2, q, chi1, chi2):
    """
    Estimate the final mass of the post-merger renmant. We implement the fitting
    formula to numerical relativity simulations by Barausse Morozova Rezzolla
    2012. This formula has to be applied *close to merger*, where numerical
    relativity simulations are available. You should do a PN evolution to
    transfer binaries to r~10M.

    Call
    ----
    mfin = remnantmass(theta1,theta2,q,chi1,chi2)

    Parameters
    ----------
    theta1: float
        Angle between orbital angular momentum and primary spin.
    theta2: float
        Angle between orbital angular momentum and secondary spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.

    Returns
    -------
    mfin: float
        Mass of the black-hole remnant.
    """

    q = np.atleast_1d(q)
    chi1 = np.atleast_1d(chi1)
    chi2 = np.atleast_1d(chi2)
    eta = eval_eta(q)

    chit_par =  ( chi2*q**2 * np.cos(theta2) + chi1*np.cos(theta1) ) / (1+q)**2

    #Final mass. Barausse Morozova Rezzolla 2012
    p0 = 0.04827
    p1 = 0.01707
    Z1 = 1 + (1-chit_par**2)**(1/3)* ((1+chit_par)**(1/3)+(1-chit_par)**(1/3))
    Z2 = (3* chit_par**2 + Z1**2)**(1/2)
    risco = 3 + Z2 - np.sign(chit_par) * ((3-Z1)*(3+Z1+2*Z2))**(1/2)
    Eisco = (1-2/(3*risco))**(1/2)
    #Radiated energy, in units of the initial total mass of the binary
    Erad = eta*(1-Eisco) + 4* eta**2 * (4*p0+16*p1*chit_par*(chit_par+1)+Eisco-1)
    Mfin = 1- Erad # Final mass

    return Mfin


def remnantspin(theta1, theta2, deltaphi, q, chi1, chi2, which='HBR16_34corr'):
    """
    Estimate the final spin of the post-merger renmant. We implement the fitting
    formula to numerical relativity simulations by  Barausse and Rezzolla 2009
    and Hofmann, Barausse and Rezzolla 2016. This can be selected by the keywork `
    `which`, see those references for details. By default this returns the
    Hofmann+ expression with nM=3, nJ=4 and corrections for the effective
    angles (HBR16_34corr). This formula has to be applied *close to merger*,
    where numerical relativity simulations are available. You should do a PN
    evolution to transfer binaries at r~10M.

    Call
    ----
    chifin = remnantspin(theta1,theta2,deltaphi,q,chi1,chi2,which='HBR16_34corr')

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
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    which: string, optional (default: 'HBR16_34corr')
        Select function behavior.

    Returns
    -------
    chifin: float
        Spin of the black-hole remnant.
    """


    q = np.atleast_1d(q)
    chi1 = np.atleast_1d(chi1)
    chi2 = np.atleast_1d(chi2)
    eta = eval_eta(q)

    if which in ['HBR16_12', 'HBR16_12corr', 'HBR16_33', 'HBR16_33corr', 'HBR16_34', 'HBR16_34corr']:

        kfit = {}

        if 'HBR16_12' in which:
            kfit = np.array( [[np.nan, -1.2019, -1.20764] ,
                              [3.79245, 1.18385, 4.90494] ]  )
            xifit = 0.41616

        if 'HBR16_33' in which:
            kfit = np.array( [[np.nan, 2.87025, -1.53315, -3.78893] ,
                              [32.9127, -62.9901, 10.0068, 56.1926],
                              [-136.832, 329.32, -13.2034, -252.27],
                              [210.075, -545.35, -3.97509, 368.405]]  )
            xifit = 0.463926

        if 'HBR16_34' in which:
            kfit = np.array( [[np.nan, 3.39221, 4.48865, -5.77101, -13.0459] ,
                              [35.1278, -72.9336, -86.0036, 93.7371, 200.975],
                              [-146.822, 387.184, 447.009, -467.383, -884.339],
                              [223.911, -648.502, -697.177, 753.738, 1166.89]])
            xifit = 0.474046

        # Calculate K00 from Eq 11
        kfit[0,0] = 4**2 * ( 0.68646 - np.sum( kfit[1:,0] /(4**(3+np.arange(kfit.shape[0]-1)))) - (3**0.5)/2)

        theta12 = eval_theta12(theta1=theta1, theta2=theta2, deltaphi=deltaphi)

        # Eq. 18
        if 'corr' in which:
            eps1 = 0.024
            eps2 = 0.024
            eps12 = 0
            theta1 = theta1 + eps1 * np.sin(theta1)
            theta2 = theta2 + eps2 * np.sin(theta2)
            theta12 = theta12 + eps12 * np.sin(theta12)

        # Eq. 14 - 15
        atot = ( chi1*np.cos(theta1) + chi2*np.cos(theta2)*q**2 ) / (1+q)**2
        aeff = atot + xifit*eta* ( chi1*np.cos(theta1) + chi2*np.cos(theta2) )

        # Eq. 2 - 6 evaluated at aeff, as specified in Eq. 11
        Z1= 1 + (1-(aeff**2))**(1/3) * ( (1+aeff)**(1/3) + (1-aeff)**(1/3) )
        Z2= ( (3*aeff**2) + (Z1**2) )**(1/2)
        risco= 3 + Z2 - np.sign(aeff) * ( (3-Z1)*(3+Z1+2*Z2) )**(1/2)
        Eisco=(1-2/(3*risco))**(1/2)
        Lisco = (2/(3*(3**(1/2)))) * ( 1 + 2*(3*risco - 2 )**(1/2) )

        # Eq. 13
        etatoi = eta[:,np.newaxis]**(1+np.arange(kfit.shape[0]))
        innersum = np.sum(kfit.T * etatoi[:,np.newaxis],axis=2)
        aefftoj = aeff[:,np.newaxis]**(np.arange(kfit.shape[1]))
        sumell = (np.sum(innersum  * aefftoj,axis=1))
        ell = np.abs( Lisco  - 2*atot*(Eisco-1)  + sumell )

        # Eq. 16
        chifin = (1/(1+q)**2) * ( chi1**2 + (chi2**2)*(q**4)  + 2*chi1*chi2*(q**2)*np.cos(theta12)
                + 2*(chi1*np.cos(theta1) + chi2*(q**2)*np.cos(theta2))*ell*q + ((ell*q)**2)  )**(1/2)

    else:
        raise ValueError("`which` needs to be one of the following: `HBR16_12`, `HBR16_12corr`, `HBR16_33`, `HBR16_33corr`, `HBR16_34`, `HBR16_34corr`.")

    return np.minimum(chifin,1)


def remnantkick(theta1, theta2, deltaphi, q, chi1, chi2, kms=False, maxphase=False, superkick=True, hangupkick=True, crosskick=True, full_output=False):
    """
    Estimate the kick of the merger remnant. We collect various numerical-relativity
    results, as described in Gerosa and Kesden 2016. Flags let you switch the
    various contributions on and off (all on by default): superkicks (Gonzalez et al. 2007a;
    Campanelli et al. 2007), hang-up kicks (Lousto & Zlochower 2011),
    cross-kicks (Lousto & Zlochower 2013). The orbital-plane kick components are
    implemented as described in Kesden et al. 2010a.  The final kick depends on
    the orbital phase at merger. By default, this is assumed to be uniformly
    distributed in [0,2pi]. The maximum kick is realized for Theta=0 and can be
    computed with the optional argument maxphase. The final kick is returned in
    geometrical units (i.e. vkick/c) by default, and converted to km/s if
    kms=True. This formula has to be applied *close to merger*, where
    numerical relativity simulations are available. You should do a PN evolution
    to transfer binaries at r~10M.

    Call
    ----
    vk = remnantkick(theta1,theta2,deltaphi,q,chi1,chi2,kms=False,maxphase=False,superkick=True,hangupkick=True,crosskick=True,full_output=False)
    vk,vk_array = remnantkick(theta1,theta2,deltaphi,q,chi1,chi2,kms=False,maxphase=False,superkick=True,hangupkick=True,crosskick=True,full_output=True)

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
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    kms: boolean, optional (default: False)
        Return velocities in km/s.
    maxphase: boolean, optional (default: False)
        Maximize over orbital phase at merger.
    superkick: boolean, optional (default: True)
        Switch kick terms on and off.
    hangupkick: boolean, optional (default: True)
        Switch kick terms on and off.
    crosskick: boolean, optional (default: True)
        Switch kick terms on and off.
    full_output: boolean, optional (default: False)
        Return additional outputs.

    Returns
    -------
    vk: float
        Kick of the black-hole remnant (magnitude).

    Other parameters
    -------
    vk_array: array
        Kick of the black-hole remnant (in a frame aligned with L).
    """


    q = np.atleast_1d(q)
    chi1 = np.atleast_1d(chi1)
    chi2 = np.atleast_1d(chi2)
    eta = eval_eta(q)

    Lvec,S1vec,S2vec = angles_to_Lframe(theta1, theta2, deltaphi, 1, q, chi1, chi2)
    hatL = normalize_nested(Lvec)
    hatS1 = normalize_nested(S1vec)
    hatS2 = normalize_nested(S2vec)

    #More spin parameters.
    Delta = - scalar_nested(1/(1+q), (scalar_nested(q*chi2,hatS2)-scalar_nested(chi1,hatS1)) )
    Delta_par = dot_nested(Delta,hatL)
    Delta_perp = norm_nested(np.cross(Delta,hatL))
    chit = scalar_nested(1/(1+q)**2, (scalar_nested(chi2*q**2,hatS2)+scalar_nested(chi1,hatS1)) )
    chit_par = dot_nested(chit,hatL)
    chit_perp = norm_nested(np.cross(chit,hatL))

    #Coefficients are quoted in km/s
    #vm and vperp from Kesden at 2010a. vpar from Lousto Zlochower 2013
    zeta=np.radians(145)
    A=1.2e4
    B=-0.93
    H=6.9e3

    #Multiply by 0/1 boolean flags to select terms
    V11 = 3677.76 * superkick
    VA = 2481.21 * hangupkick
    VB = 1792.45 * hangupkick
    VC = 1506.52 * hangupkick
    C2 = 1140 * crosskick
    C3 = 2481 * crosskick

    #maxkick
    bigTheta=np.random.uniform(0, 2*np.pi,q.shape) * (not maxphase)

    vm = A * eta**2 * (1+B*eta) * (1-q)/(1+q)
    vperp = H * eta**2 * Delta_par
    vpar = 16*eta**2 * (Delta_perp * (V11 + 2*VA*chit_par + 4*VB*chit_par**2 + 8*VC*chit_par**3) + chit_perp * Delta_par * (2*C2 + 4*C3*chit_par)) * np.cos(bigTheta)
    kick = np.array([vm+vperp*np.cos(zeta),vperp*np.sin(zeta),vpar]).T

    if not kms:
        kick = kick/299792.458 # speed of light in km/s

    vk = norm_nested(kick)

    if full_output:
        return vk, kick
    else:
        return vk

# Check...
def ftor_PN(f, M_msun, q, chi1, chi2, theta1, theta2, deltaphi):
    '''Convert GW frequency to PN orbital separation conversion'''

    c_cgs = 2.99e10
    G_cgs = 6.67e-8
    om = np.pi * f
    M_sec = M_msun * 2e33 * G_cgs / c_cgs**3
    mom = M_sec * om
    m1 = 1 / (1+q)
    m2 = q / (1+q)
    eta = m1*m2
    ct1 = np.cos(theta1)
    ct2 = np.cos(theta2)
    ct12 = np.sin(theta1) * np.sin(theta2) * np.cos(deltaphi) + ct1 * ct2
    # Eq. 4.13, Kidder 1995. gr-qc/9506022
    r = (mom)**(-2./3.)*(1. \
                    - (1./3.)*(3.-eta)*mom**(2./3.)  \
                    - (1./3.)* ( chi1*ct1*(2.*m1**2.+3.*eta) + chi2*ct2*(2.*m2**2.+3.*eta))*mom \
                    + ( eta*(19./4. + eta/9.) -eta*chi1*chi2/2. * (ct12 - 3.*ct1*ct2 ))*mom**(4./3.)\
                    )
    return r

if __name__ == '__main__':

    import sys
    import os
    import time
    np.set_printoptions(threshold=sys.maxsize)

    # q=0.9
    # chi1=1
    # chi2=1
    # r=10
    # J=1
    # chieff=0.3
    # kappa = eval_kappa(J, r, q)
    # u = eval_u(r, q)

    # print(Ssroots(J, r, chieff, q, chi1, chi2)**(1/2))
    # #print(Slimits_plusminus(J, r, chieff, q, chi1, chi2))
    #
    # dchi = deltachiroots(kappa, u, chieff, q, chi1, chi2)
    # dchi[2]= dchi[2]/(1-q)
    # Sconv = eval_S_from_deltachi(dchi, np.tile(kappa, dchi.shape), np.tile(r, dchi.shape), np.tile(chieff, dchi.shape), np.tile(q, dchi.shape))
    # print(Sconv)

    #print(kapparesonances(u, chieff, q, chi1, chi2))
    #kappa = wraproots(kappadiscriminant_coefficients, u, chieff, q, chi1, chi2)
    #J = eval_J(kappa=kappa, r=np.tile(r, kappa.shape), q=np.tile(q, kappa.shape))
    #print(kappa,J)

    #kappa = wraproots(kappadiscriminant_coefficients_new, u, chieff, q, chi1, chi2)
    #J = eval_J(kappa=kappa, r=np.tile(r, kappa.shape), q=np.tile(q, kappa.shape))
    #print(kappa,J)
    # print(kapparesonances(u, chieff, q, chi1, chi2))
    #
    #
    # print(kapparesonances_new(r, chieff, q, chi1, chi2))

    q=0.45
    chi1=0.8
    chi2=0.9
    chieff=0.30
    r=10
    kappatilde = 0.9
    deltachitilde = 0.7
    kappa = float(kapparescaling(kappatilde, r, chieff, q, chi1, chi2))
    #print(kappa)
    #kappa=0.19702426300035386
    u=eval_u(r=r,q=q)
    #u = eval_u([r,1000,100,10], [q,q,q,q])
    #print(integrator_precav(kappa, u, chieff, q, chi1, chi2))


    #print(rhs_precav(kappa, u[0], chieff, q, chi1, chi2))
    #print(rhs_precav_old(kappa, u[0], chieff, q, chi1, chi2))


    # print(deltachisampling(kappa, r, chieff, q, chi1, chi2))

    # print(deltachisampling(kappa, r, chieff, q, chi1, chi2,N=5))


    # q=[0.7,0.1]
    # chi1=[0.8,0.8]
    # chi2=[0.9,0.9]
    # chieff=[0.3,0.3]
    # r=[10,100000]
    # kappatilde = [0.5,0.5]
    # u = eval_u(r, q)
    # kappa = kapparescaling(kappatilde, r, chieff, q, chi1, chi2)
    # #print(kappa)


    # print(deltachisampling(kappa, r, chieff, q, chi1, chi2))

    # print(deltachisampling(kappa, r, chieff, q, chi1, chi2,N=10))

    #kappa = kapparescaling([kappatilde,kappatilde], [r,r], [chieff,chieff], [q,q], [chi1,chi1], [chi2,chi2])
    #print(kappa)

    #dchim,dchip = deltachilimits_plusminus(kappa, r, chieff, q, chi1, chi2)
    #print(dchim,dchip)
    #kappamin,kappamax = kapparesonances(r, chieff, q, chi1,chi2)

    #print(kappamin,kappamax)

    #kappamin,kappamax = kapparesonances(np.inf, chieff, q, chi1,chi2)

    #print(kappamin,kappamax)

    #print((chi1 + q**2 * chi2) / (1+q)**2)

    #kappa = kapparescaling(kappatilde, r, chieff, q, chi1, chi2)
    J=eval_J(kappa=kappa, r=r, q=q)


    #deltachi = deltachirescaling(deltachitilde, kappa, r, chieff, q, chi1, chi2)

    # S = eval_S_from_deltachi(deltachi, kappa, r, chieff, q)

    # print(eval_costheta1(deltachi, chieff, q, chi1))
    # print(eval_theta1(deltachi, chieff, q, chi1))

    #print(eval_cosdeltaphi_old(S=S, J=J, r=r, chieff=chieff, q=q, chi1=chi1,chi2=chi2))
    #print(eval_cosdeltaphi(deltachi=deltachi, kappa=kappa, chieff=chieff, q=q, chi1=chi1,chi2=chi2))

    #print(eval_costheta1(deltachi=deltachi, kappa=kappa, chieff=chieff, q=q, chi1=chi1,chi2=chi2))

    #tnew = eval_tau(kappa, r, chieff, q, chi1, chi2)
    #print('%.15f' % tnew)
    #t = np.squeeze(np.linspace(0,tnew/2,10))
    #print(t)
    #print(Soft(t, J, r, chieff, q, chi1, chi2))

    #tan = 4*np.pi*r**(11/4) / (3* (2*kappa-chieff)**(1/2) * (1 -chieff/ r**(1/2)))

    #print(tan)

    #dchim,dchip = deltachilimits_plusminus(kappa, r, chieff, q, chi1, chi2)
    #print(dchip-dchim)
    #aman = (chieff/2)*np.abs(chi1**2 - chi2**2)*(2*kappa-chieff)**(-1) *r**(-1/2)
    #print(aman)
    #print('FROM HERE')
    #dchi = deltachioft(t, kappa , r, chieff, q, chi1, chi2)
    #print()
    #dchi = deltachioft(np.repeat(t,2), np.repeat(kappa,2) , np.repeat(r,2), np.repeat(chieff,2), np.repeat(q,2), np.repeat(chi1,2), np.repeat(chi2,2))
    #dchi=np.squeeze(dchi)
    #print(dchi)


    #print(tofdeltachi(dchi, kappa , r, chieff, q, chi1, chi2) -t )

    # Snew = eval_S_from_deltachi(dchi, tiler(kappa,dchi), tiler(r,dchi), tiler(chieff,dchi), tiler(q,dchi))

    #told = eval_tau_old(J, r, chieff, q, chi1, chi2)
    #print('%.15f' % told)
    # t = np.linspace(0,told/2,5)

    # Sold = np.squeeze(Soft(t, J, r, chieff, q, chi1, chi2))

    # print(Snew)

    # print(Sold)

    # print(Snew[::-1]-Sold)



    # print(tnew,told)


    # print(kappa)
    # kappamin,kappamax = kapparesonances(r, chieff, q, chi1, chi2)
    # print(kappamin,kappamax)

    # kappamin,kappamax = kapparesonances_old(u, chieff, q, chi1, chi2)
    # print(kappamin,kappamax)

# TODO: Do we need this?
    # Jmin,Jmax = Jlimits_LS1S2(r, q, chi1, chi2)
    # print(Jmin,Jmax)

    # kmin,kmax = kappalimits_geometrical(r , q, chi1, chi2)
    # print(eval_J(kappa=np.squeeze([kmin,kmax]), r=[r,r], q=[q,q]))

