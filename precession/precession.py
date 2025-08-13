import warnings
import numpy as np
import scipy.special
import scipy.integrate
import scipy.spatial.transform
from itertools import repeat

################ Utilities ################


def roots_vec(p): #, enforce=False):
    """
    Locate roots of polynomial using a vectorized version of numpy.roots.
    Credit to stackoverflow user 'pv' (https://stackoverflow.com/a/35853977) for some of this code, which we generalize to accomodate equations of different degrees.
    
    Parameters
    ----------
    p: float
        Polynomial coefficients.
    
    Returns
    -------
    roots: float
        Polynomial roots.
    
    Examples
    --------
    ``roots = precession.roots_vec(p)``
    """

    p = np.atleast_1d(p).astype(float)
    non_zeros = np.count_nonzero(p, axis=1)
    
    # Mask arrays with all zeros with a dummy equation
    p[non_zeros==0,0]=1

    #if not non_zeros.all()!=0:
    #    if enforce:
    #        raise ValueError("There is at least one coefficients line with all zeros [roots_vec_zeros].")
    #    else:
    #        warnings.warn("There is at least one coefficients line with all zeros [roots_vec_zeros].", Warning)

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

    resind = np.where(resind<nansol, np.nan, results)

    # Replace nans for arrays where it's all zeros
    resind = np.where(non_zeros==0, np.ones(resind.shape).T*np.nan, resind.T).T

    return resind


def norm_nested(x):
    """
    Norm of 2D array of shape (N,3) along last axis.

    Parameters
    ----------
    x : array
        Input array.

    Returns
    -------
    n : array
        Norm of the input arrays.

    Examples
    --------
    ``n = norm_nested(x)``
    """

    return np.linalg.norm(x, axis=1)


def normalize_nested(x):
    """
    Normalize 2D array of shape (N,3) along last axis.

    Parameters
    ----------
    x : array
        Input array.

    Returns
    -------
    y : array
        Normalized array.

    Examples
    --------
    ``y = normalize_nested(x)``
    """

    return x/norm_nested(x)[:, None]


def dot_nested(x, y):
    """
    Dot product between 2D arrays along last axis.

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

    Examples
    --------
    ``z = dot_nested(x, y)``
    """

    return np.einsum('ij, ij->i', x, y)


def scalar_nested(k, x):
    """
    Nested scalar product between a 1D and a 2D array.

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

    Examples
    --------
    ``y = scalar_nested(k, x)``
    """

    return k[:,np.newaxis]*x


def rotate_nested(vec, align_zaxis, align_xzplane):
    """
    Rotate a given vector vec to a frame such that the vector align_zaxis lies along z and the vector align_xzplane lies in the xz plane.

    Parameters
    ----------
    vec : array
        Input array.
    align_zaxis : array
        Reference array.
    align_xzplane : array
        Reference array.

    Returns
    -------
    vecrot : array
        Rotated array.
 
    Examples
    --------
    ``vecrot = rotate_nested(vec, align_zaxis, align_xzplane)``
    """

    vec = np.atleast_2d(vec)
    align_zaxis = np.atleast_2d(align_zaxis)
    align_xzplane = np.atleast_2d(align_xzplane)
    
    align_zaxis = normalize_nested(align_zaxis)
    
    angle1 = np.arccos(align_zaxis[:,2])
    vec1 = np.cross(align_zaxis,[0,0,1])
    vec1 = normalize_nested(vec1)
    r1 = scipy.spatial.transform.Rotation.from_rotvec(angle1[:,None] * vec1)

    align_xzplane = r1.apply(align_xzplane)    
    align_xzplane[:,2]=0
    align_xzplane = normalize_nested(align_xzplane)

    angle2= -np.sign(align_xzplane[:,1])*np.arccos(align_xzplane[:,0])

    vec2 = np.array([0,0,1])
    r2 = scipy.spatial.transform.Rotation.from_rotvec(angle2[:,None] * vec2)
    
    vecrot = r2.apply(r1.apply(vec))

    return vecrot


def sample_unitsphere(N=1):
    """
    Sample points uniformly on a sphere of unit radius. Returns array of shape (N,3).

    Parameters
    ----------
    N: integer, optional (default: 1)
        Number of samples.

    Returns
    -------
    vec: array
        Vector in Cartesian coomponents.

    Examples
    --------
    ``vec = sample_unitsphere(N=1)``
    """

    vec = np.random.randn(3, N)
    vec /= np.linalg.norm(vec, axis=0)
    return vec.T


def isotropic_angles(N=1):
    """
    Sample the angles theta1, theta2, and deltaphi from an isotropic distribution.
    
    Parameters
    ----------
    N: integer, optional (default: 1)
        Number of samples.
    
    Returns
    -------
    vec: array
        Vector in Cartesian coomponents.
    
    Examples
    --------
    ``vec = precession.isotropic_angles(N=1)``
    """

    theta1=np.arccos(np.random.uniform(-1,1,N))
    theta2=np.arccos(np.random.uniform(-1,1,N))
    deltaphi=np.random.uniform(-np.pi,np.pi,N)

    return np.stack([theta1,theta2,deltaphi])


def tiler(thing,shaper):
    """
    Repeat the quantity thing a numer of times give by the shape of shaper.
    
    Parameters
    ----------
    thing: float  
        Quantity to be repeated
    shaper: array
        Quantity providing the shape.
    
    Returns
    -------
    thing: float  
        Quantity to be repeated
    
    Examples
    --------
    ``thing = precession.tiler(thing,shaper)``
    """

    thing =np.atleast_1d(thing)
    shaper =np.atleast_1d(shaper)
    assert thing.ndim == 1 and shaper.ndim==1

    return np.squeeze(np.tile(thing, np.shape(shaper)).reshape(len(shaper),len(thing)))


def affine(vec, low, up):
    """
    Perform an affine trasformation to scale a vector given upper and lower limits: rescaled = ( vec - low ) / (up - low). If these are outside the domain of the vector, the result will be in [0,1].
    
    Parameters
    ----------
    vec: array
        Vector in Cartesian coomponents.
    low: float
        Lower limit.
    up: float
        Upper limit.
    
    Returns
    -------
    rescaled: float, optional
        Rescaled vector.
    
    Examples
    --------
    ``rescaled = precession.affine(vec,low,up)``
    """

    vec = np.atleast_1d(vec).astype(float)
    up = np.atleast_1d(up).astype(float)
    low = np.atleast_1d(low).astype(float)

    rescaled = ( vec - low ) / (up - low)

    return rescaled

def inverseaffine(rescaled, low, up):
    """
    Perform an inferse affine trasformation to scale a vector from some upper and lower limits: vec = low + rescaled*(up-low). If rescaled is defined in [0,1], the result will be defined in [low,up].
    
    Parameters
    ----------
    rescaled: float
        Rescaled vector.
    low: float
        Lower limit.
    up: float
        Upper limit.
    
    Returns
    -------
    vec: array
        Vector in Cartesian coomponents.
    
    Examples
    --------
    ``vec = precession.inverseaffine(rescaled,low,up)``
    """

    rescaled = np.atleast_1d(rescaled).astype(float)
    up = np.atleast_1d(up).astype(float)
    low = np.atleast_1d(low).astype(float)

    vec = low + rescaled*(up-low)

    return vec


def wraproots(coefficientfunction, *args, **kwargs):
    """
    Find roots of a polynomial given coefficients, ordered according to their real part. Complex roots are masked with nans. This is essentially a wrapper of numpy.roots.

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

    Examples
    --------
    ``sols = precession.wraproots(coefficientfunction, *args, **kwargs)``
    """

    coeffs = coefficientfunction(*args, **kwargs)
    sols = np.sort_complex(roots_vec(coeffs.T))
    sols = np.real(np.where(np.isreal(sols), sols, np.nan))

    return sols


def ellippi(n, phi, m):
    """
    Incomplete elliptic integral of the third kind. This is reconstructed using scipy's implementation of Carlson's R integrals (arxiv:math/9409227).
 
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

    Examples
    --------
    ``piintegral = precession.ellippi(n, phi, m)``
    """

    # Important: this requires scipy>=1.8.0
    # https://docs.scipy.org/doc/scipy/release.1.8.0.html

    # Notation used here:
    # https://reference.wolfram.com/language/ref/EllipticPi.html

    # A much slower implementation using simpy
    # from sympy import elliptic_pi
    #return float(elliptic_pi(float(n), float(phi), float(m)))

    n = np.array(n)
    phi = np.array(phi)
    m = np.array(m)

    if ~np.all(phi>=0) or ~np.all(phi<=np.pi/2) or ~np.all(m>=0) or ~np.all(m<=1):
        warnings.warn("Elliptic intergal of the third kind evaluated outside of the expected domain. Our implementation has not been tested in this regime!", Warning)

    # Eq (61) in Carlson 1994 (arxiv:math/9409227v1). Careful with the notation: one has k^2 --> m and n --> -n.
    c = (1/np.sin(phi))**2
    return scipy.special.elliprf(c-1,c-m,c) +(np.array(n)/3)*scipy.special.elliprj(c-1,c-m,c,c-n)


def ismonotonic(vec, which):
    """
    Check if an array is monotonic. The parameter `which` can takes the following values:
        - `<` check array is strictly increasing.
        - `<=` check array is increasing.
        - `>` check array is strictly decreasing.
        - `>=` check array is decreasing.

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

    Examples
    --------
    ``check = ismonotonic(vec, which)``
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


################ Some definitions ################


def eval_m1(q):
    """
    Mass of the heavier black hole in units of the total mass.
    
    Parameters
    ----------
    q: float
        Mass ratio: 0<=q<=1.
    
    Returns
    -------
    m1: float
        Mass of the primary (heavier) black hole.
    
    Examples
    --------
    ``m1 = precession.eval_m1(q)``
    """

    q = np.atleast_1d(q).astype(float)
    m1 = 1/(1+q)

    return m1


def eval_m2(q):
    """
    Mass of the lighter black hole in units of the total mass.
    
    Parameters
    ----------
    q: float
        Mass ratio: 0<=q<=1.
    
    Returns
    -------
    m2: float
        Mass of the secondary (lighter) black hole.
    
    Examples
    --------
    ``m2 = precession.eval_m2(q)``
    """

    q = np.atleast_1d(q).astype(float)
    m2 = q/(1+q)

    return m2


def eval_q(m1, m2):
    """
    Mass ratio, 0 < q = m2/m1 < 1.
    
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
    
    Examples
    --------
    ``q = precession.eval_q(m1,m2)``
    """

    m1 = np.atleast_1d(m1).astype(float)
    m2 = np.atleast_1d(m2).astype(float)
    q = m2/m1
    assert (q < 1).all(), "The convention used in this code is q=m2/m1<1."

    return q


def eval_eta(q):
    """
    Symmetric mass ratio eta = m1*m2/(m1+m2)^2 = q/(1+q)^2.
    
    Parameters
    ----------
    q: float
        Mass ratio: 0<=q<=1.
    
    Returns
    -------
    eta: float
        Symmetric mass ratio 0<=eta<=1/4.
    
    Examples
    --------
    ``eta = precession.eval_eta(q)``
    """

    q = np.atleast_1d(q).astype(float)
    eta = q/(1+q)**2

    return eta


def eval_S1(q, chi1):
    """
    Spin angular momentum of the heavier black hole.
    
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
    
    Examples
    --------
    ``S1 = precession.eval_S1(q,chi1)``
    """

    chi1 = np.atleast_1d(chi1).astype(float)
    S1 = chi1*(eval_m1(q))**2

    return S1


def eval_S2(q, chi2):
    """
    Spin angular momentum of the lighter black hole.
    
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
    
    Examples
    --------
    ``S2 = precession.eval_S2(q,chi2)``
    """

    chi2 = np.atleast_1d(chi2).astype(float)
    S2 = chi2*(eval_m2(q))**2

    return S2


def eval_chi1(q, S1):
    """
    Dimensionless spin of the heavier black hole.
    
    Parameters
    ----------
    q: float
        Mass ratio: 0<=q<=1.
    S1: float
        Magnitude of the primary spin.
    
    Returns
    -------
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    
    Examples
    --------
    ``chi1 = precession.eval_chi1(q,S1)``
    """

    S1 = np.atleast_1d(S1).astype(float)
    chi1 = S1/(eval_m1(q))**2

    return chi1


def eval_chi2(q, S2):
    """
    Dimensionless spin of the lighter black hole.
    
    Parameters
    ----------
    q: float
        Mass ratio: 0<=q<=1.
    S2: float
        Magnitude of the secondary spin.
    
    Returns
    -------
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    
    Examples
    --------
    ``chi2 = precession.eval_chi2(q,S2)``
    """

    S2 = np.atleast_1d(S2).astype(float)
    chi2 = S2/(eval_m2(q))**2

    return chi2


def eval_L(r, q):
    """
    Newtonian angular momentum of the binary.
    
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
    
    Examples
    --------
    ``L = precession.eval_L(r,q)``
    """

    r = np.atleast_1d(r).astype(float)
    q = np.atleast_1d(q).astype(float)

    L = (q/(1+q)**2)*r**0.5

    return L


def eval_v(r):
    """
    Newtonian orbital velocity of the binary.
    
    Parameters
    ----------
    r: float
        Binary separation.
    
    Returns
    -------
    v: float
        Newtonian orbital velocity.
    
    Examples
    --------
    ``v = precession.eval_v(r)``
    """

    r = np.atleast_1d(r).astype(float)
    v = 1/r**0.5

    return v


def eval_r(L=None,u=None,q=None):
    """
    Orbital separation of the binary. Valid inputs are either (L,q) or (u,q).
    
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
    
    Examples
    --------
    ``r = precession.eval_r(L=L,q=q)``
    ``r = precession.eval_r(u=u,q=q)``
    """

    q = np.atleast_1d(q).astype(float)

    if L is not None and u is None and q is not None:

        L = np.atleast_1d(L).astype(float)
        r = (L * (1+q)**2 / q )**2

    elif L is None and u is not None and q is not None:

        u = np.atleast_1d(u).astype(float)
        r = (2*u*q/(1+q)**2)**(-2)

    else:
        raise TypeError("Provide either (L,q) or (u,q).")

    return r


def eval_u(r, q):
    """
    Compactified orbital separation.
    
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
    
    Examples
    --------
    ``u = precession.eval_u(r,q)``
    """

    L = eval_L(r, q)
    u = 1 / (2*L)

    return u


def eval_chieff(theta1, theta2, q, chi1, chi2):
    """
    Eftective spin.
    
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
    chieff: float
        Effective spin.
    
    Examples
    --------
    ``chieff = precession.eval_chieff(theta1,theta2,q,chi1,chi2)``
    """

    theta1 = np.atleast_1d(theta1).astype(float)
    theta2 = np.atleast_1d(theta2).astype(float)
    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)

    chieff = (chi1*np.cos(theta1) + q*chi2*np.cos(theta2))/(1+q)

    return chieff


def eval_deltachi(theta1, theta2, q, chi1, chi2):
    """
    Weighted spin difference.
    
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
    deltachi: float
        Weighted spin difference.
    
    Examples
    --------
    ``deltachi = precession.eval_deltachi(theta1,theta2,q,chi1,chi2)``
    """

    theta1 = np.atleast_1d(theta1).astype(float)
    theta2 = np.atleast_1d(theta2).astype(float)
    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)

    deltachi = (chi1*np.cos(theta1) -q*chi2*np.cos(theta2))/(1+q)

    return deltachi


def eval_deltachiinf(kappa, chieff, q, chi1, chi2):
    """
    Large-separation limit of the weighted spin difference.
    
    Parameters
    ----------
    kappa: float
        Asymptotic angular momentum.
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
    deltachi: float
        Weighted spin difference.
    
    Examples
    --------
    ``deltachi = precession.eval_deltachiinf(kappa,chieff,q,chi1,chi2)``
    """

    kappa = np.atleast_1d(kappa).astype(float)
    chieff = np.atleast_1d(chieff).astype(float)
    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)

    deltachi = (1+q)/(1-q)*(2*kappa-chieff)

    return deltachi


def eval_costheta1(deltachi, chieff, q, chi1):
    """
    Cosine of the angle between the orbital angular momentum and the spin of the primary black hole.
    
    Parameters
    ----------
    deltachi: float
        Weighted spin difference.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    
    Returns
    -------
    costheta1: float
        Cosine of the angle between orbital angular momentum and primary spin.
    
    Examples
    --------
    ``costheta1 = precession.eval_costheta1(deltachi,chieff,q,chi1)``
    """

    deltachi = np.atleast_1d(deltachi).astype(float)
    chieff = np.atleast_1d(chieff).astype(float)
    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)

    costheta1 = (1+q)/(2*chi1)*(chieff+deltachi)

    return costheta1


def eval_theta1(deltachi, chieff, q, chi1):
    """
    Angle between the orbital angular momentum and the spin of the primary black hole.
    
    Parameters
    ----------
    deltachi: float
        Weighted spin difference.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    
    Returns
    -------
    theta1: float
        Angle between orbital angular momentum and primary spin.
    
    Examples
    --------
    ``theta1 = precession.eval_theta1(deltachi,chieff,q,chi1)``
    """

    costheta1 = eval_costheta1(deltachi, chieff, q, chi1)
    theta1 = np.arccos(costheta1)

    return theta1


def eval_costheta2(deltachi, chieff, q, chi2):
    """
    Cosine of the angle between the orbital angular momentum and the spin of the secondary black hole.
    
    Parameters
    ----------
    deltachi: float
        Weighted spin difference.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    
    Returns
    -------
    costheta2: float
        Cosine of the angle between orbital angular momentum and secondary spin.
    
    Examples
    --------
    ``costheta2 = precession.eval_costheta2(deltachi,chieff,q,chi2)``
    """

    deltachi = np.atleast_1d(deltachi).astype(float)
    chieff = np.atleast_1d(chieff).astype(float)
    q = np.atleast_1d(q).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)

    costheta2 = (1+q)/(2*q*chi2)*(chieff-deltachi)

    return costheta2


def eval_theta2(deltachi, chieff, q, chi2):
    """
    Angle between the orbital angular momentum and the spin of the secondary black hole.
    
    Parameters
    ----------
    deltachi: float
        Weighted spin difference.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    
    Returns
    -------
    theta2: float
        Angle between orbital angular momentum and secondary spin.
    
    Examples
    --------
    ``theta2 = precession.eval_theta2(deltachi,chieff,q,chi2)``
    """

    costheta2 = eval_costheta2(deltachi, chieff, q, chi2)
    theta2 = np.arccos(costheta2)

    return theta2


def eval_costheta12(theta1=None, theta2=None, deltaphi=None, deltachi=None, kappa=None, chieff=None, q=None, chi1=None, chi2=None):
    """
    Cosine of the angle between the two spins. Valid inputs are either (theta1,theta2,deltaphi) or (deltachi,kappa,chieff,q,chi1,chi2).
    
    Parameters
    ----------
    theta1: float, optional (default: None)
        Angle between orbital angular momentum and primary spin.
    theta2: float, optional (default: None)
        Angle between orbital angular momentum and secondary spin.
    deltaphi: float, optional (default: None)
        Angle between the projections of the two spins onto the orbital plane.
    deltachi: float, optional (default: None)
        Weighted spin difference.
    kappa: float, optional (default: None)
        Asymptotic angular momentum.
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
    costheta12: float
        Cosine of the angle between the two spins.
    
    Examples
    --------
    ``costheta12 = precession.eval_costheta12(theta1=theta1,theta2=theta2,deltaphi=deltaphi)``
    ``costheta12 = precession.eval_costheta12(deltachi=deltachi,kappa=kappa,chieff=chieff,q=q,chi1=chi1,chi2=chi2)``
    """

    if theta1 is not None and theta2 is not None and deltaphi is not None and deltachi is None and kappa is None and chieff is None and q is None and chi1 is None and chi2 is None:

        theta1=np.atleast_1d(theta1).astype(float)
        theta2=np.atleast_1d(theta2).astype(float)
        deltaphi=np.atleast_1d(deltaphi).astype(float)
        costheta12 = np.sin(theta1)*np.sin(theta2)*np.cos(deltaphi) + np.cos(theta1)*np.cos(theta2)

    elif theta1 is None and theta2 is None and deltaphi is None and deltachi is not None and kappa is not None and chieff is not None and q is not None and chi1 is not None and chi2 is not None:

        deltachi = np.atleast_1d(deltachi).astype(float)
        kappa = np.atleast_1d(kappa).astype(float)
        chieff = np.atleast_1d(chieff).astype(float)
        q = np.atleast_1d(q).astype(float)
        chi1 = np.atleast_1d(chi1).astype(float)
        chi2 = np.atleast_1d(chi2).astype(float)

        # Machine generated with eq_generator.nb
        costheta12 = 1/2 * q**(-2) * (chi1)**(-1) * (chi2)**(-1) * (-1 * \
        (chi1)**2 + (-1 * q**4 * (chi2)**2 + q * (1 + q) * (r)**(1/2) * (-1 * \
        (1 + -1 * q) * deltachi + (2 * (1 + q) * kappa + -1 * (1 + q) * \
        chieff))))

    else:
        raise TypeError("Provide either (theta1,theta2,deltaphi) or (deltachi,kappa,chieff,q,chi1,chi2).")

    return costheta12


def eval_theta12(theta1=None, theta2=None, deltaphi=None, deltachi=None, kappa=None, chieff=None, q=None, chi1=None, chi2=None):
    """
    Angle between the two spins. Valid inputs are either (theta1,theta2,deltaphi) or (deltachi,kappa,chieff,q,chi1,chi2).
    
    Parameters
    ----------
    theta1: float, optional (default: None)
        Angle between orbital angular momentum and primary spin.
    theta2: float, optional (default: None)
        Angle between orbital angular momentum and secondary spin.
    deltaphi: float, optional (default: None)
        Angle between the projections of the two spins onto the orbital plane.
    deltachi: float, optional (default: None)
        Weighted spin difference.
    kappa: float, optional (default: None)
        Asymptotic angular momentum.
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
    theta12: float
        Angle between the two spins.
    
    Examples
    --------
    ``theta12 = precession.eval_theta12(theta1=theta1,theta2=theta2,deltaphi=deltaphi)``
    ``theta12 = precession.eval_theta12(deltachi=deltachi,kappa=kappa,chieff=chieff,q=q,chi1=chi1,chi2=chi2)``
    """

    costheta12 = eval_costheta12(theta1=theta1, theta2=theta2, deltaphi=deltaphi, deltachi=deltachi, kappa=kappa, chieff=chieff, q=q, chi1=chi1, chi2=chi2)
    theta12 = np.arccos(costheta12)

    return theta12


def eval_cosdeltaphi(deltachi, kappa, r, chieff, q, chi1, chi2):
    """
    Cosine of the angle between the projections of the two spins onto the orbital plane.
    
    Parameters
    ----------
    deltachi: float
        Weighted spin difference.
    kappa: float
        Asymptotic angular momentum.
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
    
    Examples
    --------
    ``cosdeltaphi = precession.eval_cosdeltaphi(deltachi,kappa,r,chieff,q,chi1,chi2)``
    """

    deltachi = np.atleast_1d(deltachi).astype(float)
    kappa = np.atleast_1d(kappa).astype(float)
    r = np.atleast_1d(r).astype(float)
    chieff = np.atleast_1d(chieff).astype(float)
    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)


    with warnings.catch_warnings():
        
        # If there are infinitely large separation in the array the following will throw a warning. You can safely ignore it because that value is not used, see below  
        if np.inf in r:
            warnings.filterwarnings("ignore", category=RuntimeWarning)
 
        # Machine generated with eq_generator.nb
        cosdeltaphi = q**(-1) * ((4 * q**2 * (chi2)**2 + -1 * ((1 + q))**2 * \
        ((-1 * deltachi + chieff))**2) * (4 * (chi1)**2 + -1 * ((1 + q))**2 * \
        ((deltachi + chieff))**2))**(-1/2) * (-2 * ((chi1)**2 + q**4 * \
        (chi2)**2) + (2 * q * (1 + q) * (r)**(1/2) * (-1 * (1 + -1 * q) * \
        deltachi + (2 * (1 + q) * kappa + -1 * (1 + q) * chieff)) + -1 * q * \
        ((1 + q))**2 * (-1 * (deltachi)**2 + chieff**2)))

    # At infinity, the only thing I can do is putting a random number for deltaphi, unformly distributed
    cosdeltaphi = np.where(r!=np.inf, cosdeltaphi, np.cos(np.random.uniform(0,np.pi, len(cosdeltaphi))))

    return cosdeltaphi


def eval_deltaphi(deltachi, kappa, r, chieff, q, chi1, chi2, cyclesign=1):
    """
    Angle between the projections of the two spins onto the orbital plane.
    
    Parameters
    ----------
    deltachi: float
        Weighted spin difference.
    kappa: float
        Asymptotic angular momentum.
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
    deltaphi: float
        Angle between the projections of the two spins onto the orbital plane.
    
    Examples
    --------
    ``deltaphi = precession.eval_deltaphi(deltachi,kappa,r,chieff,q,chi1,chi2,cyclesign=1)``
    """

    cyclesign = np.atleast_1d(cyclesign)
    cosdeltaphi = eval_cosdeltaphi(deltachi, kappa, r, chieff, q, chi1, chi2)
    deltaphi = np.sign(cyclesign)*np.arccos(cosdeltaphi)

    return deltaphi


def eval_costhetaL(deltachi, kappa, r, chieff, q):
    """
    Cosine of the angle betwen the orbital angular momentum and the total angular momentum.
    
    Parameters
    ----------
    deltachi: float
        Weighted spin difference.
    kappa: float
        Asymptotic angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    
    Returns
    -------
    costhetaL: float
        Cosine of the angle betwen orbital angular momentum and total angular momentum.
    
    Examples
    --------
    ``costhetaL = precession.eval_costhetaL(deltachi,kappa,r,chieff,q)``
    """

    deltachi = np.atleast_1d(deltachi).astype(float)
    kappa = np.atleast_1d(kappa).astype(float)
    chieff = np.atleast_1d(chieff).astype(float)
    q = np.atleast_1d(q).astype(float)

    # Machine generated with eq_generator.nb
    costhetaL = ((1 + 2 * q**(-1) * ((1 + q))**2 * (r)**(-1/2) * \
    kappa))**(-1/2) * (1 + 1/2 * q**(-1) * (1 + q) * (r)**(-1/2) * ((1 + \
    -1 * q) * deltachi + (1 + q) * chieff))

    return costhetaL


def eval_thetaL(deltachi, kappa, r, chieff, q):
    """
    Angle betwen the orbital angular momentum and the total angular momentum.
    
    Parameters
    ----------
    deltachi: float
        Weighted spin difference.
    kappa: float
        Asymptotic angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    
    Returns
    -------
    thetaL: float
        Angle betwen orbital angular momentum and total angular momentum.
    
    Examples
    --------
    ``thetaL = precession.eval_thetaL(deltachi,kappa,r,chieff,q)``
    """

    costhetaL = eval_costhetaL(deltachi, kappa, r,chieff, q)
    thetaL = np.arccos(costhetaL)

    return thetaL


def eval_J(theta1=None, theta2=None, deltaphi=None, kappa=None, r=None, q=None, chi1=None, chi2=None):
    """
    Magnitude of the total angular momentum. Provide either (theta1,theta2,deltaphi,r,q,chi1,chhi2) or (kappa,r,q).
    
    Parameters
    ----------
    theta1: float, optional (default: None)
        Angle between orbital angular momentum and primary spin.
    theta2: float, optional (default: None)
        Angle between orbital angular momentum and secondary spin.
    deltaphi: float, optional (default: None)
        Angle between the projections of the two spins onto the orbital plane.
    kappa: float, optional (default: None)
        Asymptotic angular momentum.
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
    
    Examples
    --------
    ``J = precession.eval_J(theta1=theta1,theta2=theta2,deltaphi=deltaphi,r=r,q=q,chi1=chi1,chi2=chi2)``
    ``J = precession.eval_J(kappa=kappa,r=r,q=q,chi1=chi1,chi2=chi2)``
    """

    if theta1 is not None and theta2 is not None and deltaphi is not None and kappa is None and r is not None and q is not None and chi1 is not None and chi2 is not None:

        theta1 = np.atleast_1d(theta1).astype(float)
        theta2 = np.atleast_1d(theta2).astype(float)
        deltaphi = np.atleast_1d(deltaphi).astype(float)
        q = np.atleast_1d(q).astype(float)

        S1 = eval_S1(q, chi1)
        S2 = eval_S2(q, chi2)
        L = eval_L(r, q)
        S = eval_S(theta1=theta1, theta2=theta2, deltaphi=deltaphi, q=q, chi1=chi1, chi2=chi2)
        J = (L**2+S**2+2*L*(S1*np.cos(theta1)+S2*np.cos(theta2)))**0.5

    elif theta1 is None and theta2 is None and deltaphi is None and kappa is not None and r is not None and q is not None and chi1 is None and chi2 is None:

        kappa = np.atleast_1d(kappa).astype(float)

        L = eval_L(r, q)
        J = (2*L*kappa + L**2)**0.5

    else:
        raise TypeError("Provide either (theta1,theta2,deltaphi,r,q,chi1,chhi2) or (kappa,r,q).")

    return J


def eval_kappa(theta1=None, theta2=None, deltaphi=None, J=None, r=None, q=None, chi1=None, chi2=None):
    """
    Asymptotic angular momentum. Provide either (theta1,theta2,deltaphi,r,q,chi1,chhi2) or (J,r,q).
    
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
    q: float, optional (default: None)
        Mass ratio: 0<=q<=1.
    chi1: float, optional (default: None)
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float, optional (default: None)
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    
    Returns
    -------
    kappa: float
        Asymptotic angular momentum.
    
    Examples
    --------
    ``kappa = precession.eval_kappa(theta1=theta1,theta2=theta2,deltaphi=deltaphi,r=r,q=q,chi1=chi1,chi2=chi2)``
    ``kappa = precession.eval_kappa(J=J,r=r,q=q)``
    """

    if theta1 is None and theta2 is None and deltaphi is None and J is not None and r is not None and q is not None and chi1 is None and chi2 is None:

        J = np.atleast_1d(J).astype(float)
        L = eval_L(r, q)
        kappa = (J**2 - L**2) / (2*L)

    elif theta1 is not None and theta2 is not None and deltaphi is not None and J is None and r is not None and q is not None and chi1 is not None and chi2 is not None:

        theta1 = np.atleast_1d(theta1).astype(float)
        theta2 = np.atleast_1d(theta2).astype(float)
        deltaphi = np.atleast_1d(deltaphi).astype(float)
        r = np.atleast_1d(r).astype(float)
        q = np.atleast_1d(q).astype(float)
        chi1 = np.atleast_1d(chi1).astype(float)
        chi2 = np.atleast_1d(chi2).astype(float)

        kappa = (chi1 * np.cos(theta1) + q**2 * chi2 * np.cos(theta2) )/(1+q)**2 + \
                (chi1**2 + q**4 *chi2**2 + 2*chi1*chi2*q**2 * (np.cos(theta1)*np.cos(theta2) + np.cos(deltaphi)*np.sin(theta1)*np.sin(theta2))) / (2*q*(1+q)**2*r**(1/2))

    else:
        TypeError("Please provide provide either (theta1,theta2,deltaphi,r,q,chi1,chhi2) or (J,r,q).")

    return kappa


def eval_S(theta1=None, theta2=None, deltaphi=None, deltachi=None, kappa=None, r=None, chieff=None, q=None, chi1=None, chi2=None):
    """
    Magnitude of the total spin. Valid inputs are either (theta1, theta2, deltaphi, q, chi1, chi2) or (deltachi, kappa, r, chieff, q).
    
    Parameters
    ----------
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    
    Returns
    -------
    S: float
        Magnitude of the total spin.
    
    Examples
    --------
    ``S = precession.eval_S(theta1=theta1,theta2=theta2,deltaphi=deltaphi,q=q,chi1=chi1,chi2=chi2)``
    ``S = precession.eval_S(deltachi=deltachi,kappa=kappa,r=r,chieff=chieff,q=q)``
    """

    if theta1 is not None and theta2 is not None and deltaphi is not None and deltachi is None and kappa is None and r is None and chieff is None and q is not None and chi1 is not None and chi2 is not None:

        theta1 = np.atleast_1d(theta1).astype(float)
        theta2 = np.atleast_1d(theta2).astype(float)
        deltaphi = np.atleast_1d(deltaphi).astype(float)

        S1 = eval_S1(q, chi1)
        S2 = eval_S2(q, chi2)

        S = (S1**2 + S2**2 + 2*S1*S2*(np.sin(theta1)*np.sin(theta2)*np.cos(deltaphi)+np.cos(theta1)*np.cos(theta2)))**0.5

    if theta1 is None and theta2 is None and deltaphi is None and deltachi is not None and kappa is not None and r is not None and chieff is not None and q is not None and chi1 is None and chi2 is None:

        deltachi = np.atleast_1d(deltachi).astype(float)
        kappa = np.atleast_1d(kappa).astype(float)
        r = np.atleast_1d(r).astype(float)
        chieff = np.atleast_1d(chieff).astype(float)
        q = np.atleast_1d(q).astype(float)

        S = ( q /(1+q)**2 * r**(1/2) * (2*kappa - chieff - deltachi * (1 - q)/(1 + q)) )**(1/2)

    else:
        TypeError("Please provide provide either (theta1,theta2,deltaphi,r,q,chi1,chhi2) or (J,r,q).")


    return S


################ Conversions ################


def eval_cyclesign(ddeltachidt=None, deltaphi=None, Lvec=None, S1vec=None, S2vec=None):
    """
    Evaluate if the input parameters are in the first of the second half of a precession cycle. We refer to this as the 'sign' of a precession cycle, defined as +1 if deltachi is increasing and -1 deltachi is decreasing. Valid inputs are one and not more of the following:
        - dSdt
        - deltaphi
        - Lvec, S1vec, S2vec.
    
    Parameters
    ----------
    ddeltachidt: float, optional (default: None)
        Time derivative of the total spin.
    deltaphi: float, optional (default: None)
        Angle between the projections of the two spins onto the orbital plane.
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
    
    Examples
    --------
    ``cyclesign = precession.eval_cyclesign(ddeltachidt=ddeltachidt)``
    ``cyclesign = precession.eval_cyclesign(deltaphi=deltaphi)``
    ``cyclesign = precession.eval_cyclesign(Lvec=Lvec,S1vec=S1vec,S2vec=S2vec)``
    """



    if ddeltachidt is not None and deltaphi is None and Lvec is None and S1vec is None and S2vec is None:
        ddeltachidt = np.atleast_1d(ddeltachidt).astype(float)
        cyclesign = np.sign(ddeltachidt)

    elif ddeltachidt is None and deltaphi is not None and Lvec is None and S1vec is None and S2vec is None:
        deltaphi = np.atleast_1d(deltaphi).astype(float)
        cyclesign = np.sign(deltaphi)

    elif ddeltachidt is None and deltaphi is None and Lvec is not None and S1vec is not None and S2vec is not None:
        Lvec = np.atleast_2d(Lvec).astype(float)
        S1vec = np.atleast_2d(S1vec).astype(float)
        S2vec = np.atleast_2d(S2vec).astype(float)
        cyclesign = np.sign(dot_nested(S1vec, np.cross(S2vec, Lvec)))

    else:
        raise TypeError("Please provide one and not more of the following: ddeltachidt, deltaphi, (Lvec, S1vec, S2vec).")

    return cyclesign


def conserved_to_angles(deltachi, kappa, r, chieff, q, chi1, chi2, cyclesign=+1):
    """
    Convert conserved quantities (deltachi,kappa,chieff) into angles (theta1,theta2,deltaphi).
    
    Parameters
    ----------
    deltachi: float
        Weighted spin difference.
    kappa: float
        Asymptotic angular momentum.
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
    deltaphi: float
        Angle between the projections of the two spins onto the orbital plane.
    theta1: float
        Angle between orbital angular momentum and primary spin.
    theta2: float
        Angle between orbital angular momentum and secondary spin.
    
    Examples
    --------
    ``theta1,theta2,deltaphi = precession.conserved_to_angles(deltachi,kappa,r,chieff,q,chi1,chi2,cyclesign=+1)``
    """


    theta1= eval_theta1(deltachi, chieff, q, chi1)
    theta2 = eval_theta2(deltachi, chieff, q, chi2)
    deltaphi = eval_deltaphi(deltachi, kappa, r, chieff, q, chi1, chi2, cyclesign=cyclesign)

    return np.stack([theta1, theta2, deltaphi])


def angles_to_conserved(theta1, theta2, deltaphi, r, q, chi1, chi2, full_output=False):
    """
    Convert angles (theta1,theta2,deltaphi) into conserved quantities (deltachi,kappa,chieff).
    
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
    chieff: float
        Effective spin.
    cyclesign: integer, optional
        Sign (either +1 or -1) to cover the two halves of a precesion cycle.
    deltachi: float
        Weighted spin difference.
    kappa: float
        Asymptotic angular momentum.
    
    Examples
    --------
    ``deltachi,kappa,chieff = precession.angles_to_conserved(theta1,theta2,deltaphi,r,q,chi1,chi2)``
    ``deltachi,kappa,chieff,cyclesign = precession.angles_to_conserved(theta1,theta2,deltaphi,r,q,chi1,chi2,full_output=True)``
    """


    deltachi = eval_deltachi(theta1, theta2, q, chi1, chi2)
    kappa = eval_kappa(theta1=theta1, theta2=theta2, deltaphi=deltaphi, r=r, q=q, chi1=chi1, chi2=chi2)
    chieff = eval_chieff(theta1, theta2, q, chi1, chi2)

    if full_output:
        cyclesign = np.where(r==np.inf,np.nan,eval_cyclesign(deltaphi=deltaphi))
        return np.stack([deltachi, kappa, chieff, cyclesign])

    else:
        return np.stack([deltachi, kappa, chieff])


def vectors_to_angles(Lvec, S1vec, S2vec):
    """
    Convert cartesian vectors (L,S1,S2) into angles (theta1,theta2,deltaphi).
    
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
    deltaphi: float
        Angle between the projections of the two spins onto the orbital plane.
    theta1: float
        Angle between orbital angular momentum and primary spin.
    theta2: float
        Angle between orbital angular momentum and secondary spin.
    
    Examples
    --------
    ``theta1,theta2,deltaphi = precession.vectors_to_angles(Lvec,S1vec,S2vec)``
    """

    Lvec = np.atleast_2d(Lvec).astype(float)
    S1vec = np.atleast_2d(S1vec).astype(float)
    S2vec = np.atleast_2d(S2vec).astype(float)

    S1vec = normalize_nested(S1vec)
    S2vec = normalize_nested(S2vec)
    Lvec = normalize_nested(Lvec)

    theta1 = np.arccos(dot_nested(S1vec, Lvec))
    theta2 = np.arccos(dot_nested(S2vec, Lvec))
    S1crL = np.cross(S1vec, Lvec)
    S2crL = np.cross(S2vec, Lvec)

    absdeltaphi = np.arccos(dot_nested(normalize_nested(S1crL), normalize_nested(S2crL)))
    cyclesign = eval_cyclesign(Lvec=Lvec, S1vec=S1vec, S2vec=S2vec)
    deltaphi = absdeltaphi*cyclesign

    return np.stack([theta1, theta2, deltaphi])


def vectors_to_Jframe(Lvec, S1vec, S2vec):
    """
    Rotate vectors of the three momenta onto a frame where J is along z and L lies in the x-z plane.
    
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
    Lvec: array
        Cartesian vector of the orbital angular momentum.
    S1vec: array
        Cartesian vector of the primary spin.
    S2vec: array
        Cartesian vector of the secondary spin.
    
    Examples
    --------
    ``Lvec,S1vec,S2vec = precession.vectors_to_Jframe(Lvec,S1vec,S2vec)``
    """

    Jvec = Lvec + S1vec + S2vec

    rotation = lambda vec: rotate_nested(vec, Jvec, Lvec)

    Lvecrot = rotation(Lvec)
    S1vecrot = rotation(S1vec)
    S2vecrot = rotation(S2vec)

    return np.stack([Lvecrot, S1vecrot, S2vecrot])


def vectors_to_Lframe(Lvec, S1vec, S2vec):
    """
    Rotate vectors of the three momenta onto a frame where L is along z and S1 lies in the x-z plane.
    
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
    Lvec: array
        Cartesian vector of the orbital angular momentum.
    S1vec: array
        Cartesian vector of the primary spin.
    S2vec: array
        Cartesian vector of the secondary spin.
    
    Examples
    --------
    ``Lvec,S1vec,S2vec = precession.vectors_to_Lframe(Lvec,S1vec,S2vec)``
    """

    Jvec = Lvec + S1vec + S2vec

    rotation = lambda vec: rotate_nested(vec, Lvec, S1vec)

    Lvecrot = rotation(Lvec)
    S1vecrot = rotation(S1vec)
    S2vecrot = rotation(S2vec)

    return np.stack([Lvecrot, S1vecrot, S2vecrot])


def angles_to_Lframe(theta1, theta2, deltaphi, r, q, chi1, chi2):
    """
    Convert the angles (theta1,theta2,deltaphi) to angular momentum vectors (L,S1,S2) in the frame
    aligned with the orbital angular momentum. In particular, we set Lx=Ly=S1y=0.
    
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
    
    Examples
    --------
    ``Lvec,S1vec,S2vec = precession.angles_to_Lframe(theta1,theta2,deltaphi,r,q,chi1,chi2)``
    """

    L = eval_L(r, q)
    S1 = eval_S1(q, chi1)
    S2 = eval_S2(q, chi2)

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


def angles_to_Jframe(theta1, theta2, deltaphi, r, q, chi1, chi2):
    """
    Convert the angles (theta1,theta2,deltaphi) to angular momentum vectors (L,S1,S2) in the frame
    aligned with the total angular momentum. In particular, we set Jx=Jy=Ly=0.
    
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
    
    Examples
    --------
    ``Lvec,S1vec,S2vec = precession.angles_to_Jframe(theta1,theta2,deltaphi,r,q,chi1,chi2)``
    """

    Lvec, S1vec, S2vec = angles_to_Lframe(theta1, theta2, deltaphi, r, q, chi1, chi2)
    Lvec, S1vec, S2vec = vectors_to_Jframe(Lvec, S1vec, S2vec)

    return np.stack([Lvec, S1vec, S2vec])


def conserved_to_Lframe(deltachi, kappa, r, chieff, q, chi1, chi2, cyclesign=+1):
    """
    Convert the conserved quanties (deltachi,kappa,chieff) to angular momentum vectors (L,S1,S2) in the frame
    aligned with the orbital angular momentum. In particular, we set Lx=Ly=S1y=0.
    
    Parameters
    ----------
    deltachi: float
        Weighted spin difference.
    kappa: float
        Asymptotic angular momentum.
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
    Lvec: array
        Cartesian vector of the orbital angular momentum.
    S1vec: array
        Cartesian vector of the primary spin.
    S2vec: array
        Cartesian vector of the secondary spin.
    
    Examples
    --------
    ``Lvec,S1vec,S2vec = precession.conserved_to_Lframe(deltachi,kappa,r,chieff,q,chi1,chi2,cyclesign=+1)``
    """

    theta1,theta2,deltaphi = conserved_to_angles(deltachi, kappa, r, chieff, q, chi1, chi2, cyclesign=cyclesign)
    Lvec, S1vec, S2vec = angles_to_Lframe(theta1, theta2, deltaphi, r, q, chi1, chi2)

    return np.stack([Lvec, S1vec, S2vec])


def conserved_to_Jframe(deltachi, kappa, r, chieff, q, chi1, chi2, cyclesign=+1):
    """
    Convert the conserved quanties (deltachi,kappa,chieff) to angular momentum vectors (L,S1,S2) in the frame
    aligned with the total angular momentum. In particular, we set Jx=Jy=Ly=0.
    
    Parameters
    ----------
    deltachi: float
        Weighted spin difference.
    kappa: float
        Asymptotic angular momentum.
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
    Lvec: array
        Cartesian vector of the orbital angular momentum.
    S1vec: array
        Cartesian vector of the primary spin.
    S2vec: array
        Cartesian vector of the secondary spin.
    
    Examples
    --------
    ``Lvec,S1vec,S2vec = precession.conserved_to_Jframe(deltachi,kappa,r,chieff,q,chi1,chi2,cyclesign=+1)``
    """

    theta1,theta2,deltaphi = conserved_to_angles(deltachi, kappa, r, chieff, q, chi1, chi2, cyclesign=cyclesign)
    Lvec, S1vec, S2vec = angles_to_Jframe(theta1, theta2, deltaphi, r, q, chi1, chi2)

    return np.stack([Lvec, S1vec, S2vec])


def vectors_to_conserved(Lvec, S1vec, S2vec, q,full_output=False):
    """
    Convert vectors (L,S1,S2) to conserved quanties (deltachi,kappa,chieff).
    
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
    chieff: float
        Effective spin.
    cyclesign: integer, optional
        Sign (either +1 or -1) to cover the two halves of a precesion cycle.
    deltachi: float
        Weighted spin difference.
    kappa: float
        Asymptotic angular momentum.
    
    Examples
    --------
    ``deltachi,kappa,chieff = precession.vectors_to_conserved(Lvec,S1vec,S2vec,q)``
    ``deltachi,kappa,chieff,cyclesign = precession.vectors_to_conserved(Lvec,S1vec,S2vec,q,full_output=True)``
    """

    L = norm_nested(Lvec)
    S1 = norm_nested(S1vec)
    S2 = norm_nested(S2vec)

    r = eval_r(L=L,q=q)
    chi1 = eval_chi1(q,S1)
    chi2 = eval_chi2(q,S2)

    theta1,theta2,deltaphi = vectors_to_angles(Lvec, S1vec, S2vec)

    deltachi, kappa, chieff, cyclesign= angles_to_conserved(theta1, theta2, deltaphi, r, q, chi1, chi2, full_output=True)

    if full_output:
        return np.stack([deltachi, kappa, chieff, cyclesign])

    else:
        return np.stack([deltachi, kappa, chieff])


################ Spin-orbit resonances ################


def kappadiscriminant_coefficients(u, chieff, q, chi1, chi2):
    """
    Coefficients of the quintic equation in kappa that defines the spin-orbit resonances.
    
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
    coeff0: float
        Coefficient to the x^0 term in polynomial.
    coeff1: float
        Coefficient to the x^1 term in polynomial.
    coeff2: float
        Coefficient to the x^2 term in polynomial.
    coeff3: float
        Coefficient to the x^3 term in polynomial.
    coeff4: float
        Coefficient to the x^4 term in polynomial.
    coeff5: float
        Coefficient to the x^5 term in polynomial.
    
    Examples
    --------
    ``coeff5,coeff4,coeff3,coeff2,coeff1,coeff0 = precession.kappadiscriminant_coefficients(u,chieff,q,chi1,chi2)``
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
    """
    Limits in kappa conditioned on r, q, chi1, chi2.
    
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
    kappamax: float
        Maximum value of the asymptotic angular momentum kappa.
    kappamin: float
        Minimum value of the asymptotic angular momentum kappa.
    
    Examples
    --------
    ``kappamin,kappamax = precession.kappalimits_geometrical(r,q,chi1,chi2)``
    """

    r = np.atleast_1d(r).astype(float)
    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)
    
    k1 = -q*r**(1/2)/(2*(1+q)**2)
    k2 = np.where(q*r**(1/2)>=chi1+chi2*q**2,
                (chi1+chi2*q**2)/(1+q)**2 *(-1+ (chi1+chi2*q**2)/(2*q*r**(1/2))),
                1/(1+q)**2 * (-q*r**(1/2) + (chi1+chi2*q**2) - (chi1+chi2*q**2)**2 /(2*q*r**(1/2)))
                )
    k3 = np.where(np.abs(chi1-chi2*q**2)>= q*r**(1/2),
                np.abs(chi1-chi2*q**2)/(1+q)**2 *(-1+ np.abs(chi1-chi2*q**2)/(2*q*r**(1/2))),
                1/(1+q)**2 * (-q*r**(1/2) + np.abs(chi1-chi2*q**2) - (chi1-chi2*q**2)**2 /(2*q*r**(1/2)))
                )

    kappamin = np.maximum.reduce([k1,k2,k3])

    # An alternative implementation that breaks down for r=inf
    # def squarewithsign(x):
    #     return x*np.abs(x)
    # kappamin_old= q*r**(1/2)/(2*(1+q)**2)*(
    #      np.maximum.reduce([np.zeros(q.shape), 
    #         squarewithsign( 1- (chi1+chi2*q**2) / (q*r**(1/2))),
    #         squarewithsign( np.abs(chi1-chi2*q**2) / (q*r**(1/2)) - 1 )] )
    #      -1)

    kappamax = (chi1+chi2*q**2) / (1+q)**2 * ( (chi1+chi2*q**2) / (2*q*r**(1/2)) +1 )


    return np.stack([kappamin,kappamax])


def kapparesonances(r, chieff, q, chi1, chi2,tol=1e-4):
    """
    asymptotic angular momentum of the two spin-orbit resonances. The resonances minimizes and maximizes kappa for given values of r, chieff, q, chi1, chi2. The minimum corresponds to deltaphi=pi and the maximum corresponds to deltaphi=0.
    
    Parameters
    ----------
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
    tol: float, optional (default: 1e-4)
        Numerical tolerance, see source code for details.
    
    Returns
    -------
    kappamax: float
        Maximum value of the asymptotic angular momentum kappa.
    kappamin: float
        Minimum value of the asymptotic angular momentum kappa.
    
    Examples
    --------
    ``kappamin,kappamax = precession.kapparesonances(u,chieff,q,chi1,chi2,tol=1e-4)``
    """

    chieff = np.atleast_1d(chieff).astype(float)
    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)

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

        upup,updown,downup,downdown=eval_chieff([0,0,np.pi,np.pi], [0,np.pi,0,np.pi], np.repeat(q,4), np.repeat(chi1,4), np.repeat(chi2,4))


        # If too close to perfect alignment, return the analytical result.
        if np.isclose(np.repeat(chieff,2),np.squeeze([upup,downdown])).any():
            warnings.warn("Close to either up-up or down-down configuration. Using analytical results.", Warning)

            S1 = eval_S1(q, chi1)
            S2 = eval_S2(q, chi2)
            L=1/(2*u)
            kappar = ((L+np.sign(chieff)*(S1+S2))**2 - L**2) / (2*L)
            kappares=np.squeeze([kappar,kappar])



        # In this case, the spurious solution is always the smaller one. Just leave it out.
        elif len(kapparoots)==3:
            kappares = kapparoots[1:]

        # Here we have two candidate pairs of resonances...
        elif len(kapparoots)==5:

            # Edge case with two coincident roots that are exactly zeros. This happens for q=chi1=chi2=1
            if np.count_nonzero(kapparoots)==3:
                kappares = kapparoots[kapparoots != 0][1:]
            elif np.count_nonzero(kapparoots)==1:
                kappares = np.sort(np.concatenate([[0],kapparoots[kapparoots != 0]]))
            else:
                # Compute the corresponding values of deltachi at the resonances
                deltachires = deltachiresonance(kappa=kapparoots, u=tiler(u,kapparoots), chieff=tiler(chieff,kapparoots), q=tiler(q,kapparoots), chi1=tiler(chi1,kapparoots), chi2=tiler(chi2,kapparoots))
                # Check which of those values is within the allowed region
                deltachimin,deltachimax = deltachilimits_rectangle(chieff, q, chi1, chi2)
                check = np.squeeze(np.logical_and(deltachires>deltachimin,deltachires<deltachimax))

                # The first root cannot possibly be right
                if check[0] and not np.isclose(kapparoots[1],kapparoots[0]):
                    raise ValueError("Input values are not compatible [kapparesonances].")
                elif check[1] and check[2] and not check[3] and not check[4]:
                    kappares = kapparoots[1:3]
                elif not check[1] and not check[2] and check[3] and check[4]:
                    kappares = kapparoots[3:5]
                elif check[1] and check[2] and check[3] and check[4]:
                    
                    warnings.warn("Unphysical resonances detected and removed", Warning)
                    
                    # Root 1 is a spurious copy of root 0
                    if np.isclose(kapparoots[1],kapparoots[0]):



                        kappares = np.array([np.mean(kapparoots[2:4]),kapparoots[4]])
                    #err = np.abs( (kapparoots[2]-kapparoots[3])/np.mean(kapparoots[2:4]))
                    #warnings.warn("Unphysical resonances detected and removed. Relative accuracy Delta_kappa/kappa="+str(err)+", [kapparesonances].", Warning)
                    else:
                        kappares=np.array([kapparoots[1],kapparoots[4]])
                    
                    # # Root 1 is a spurious copy of root 0:
                    # if kapparoots[1]-kapparoots[0]<kapparoots[3]-kapparoots[2]:
                    #     kappares = kapparoots[3:5]
                    # # Root 2 and 3 are actually complex but appear real because of numerical errors 
                    # else:
                    #     kappares=np.array([kapparoots[1],kapparoots[4]])

        # This is an edge (and hopefully rare) case where the resonances are missed because of numerical errors
        elif len(kapparoots)==1:

            # 5 complex solutions correspond to 3 real parts (i.e. thare are two conjugate pairs)
            kapparoots = np.unique(np.real(kapparootscomplex))

            deltachires = deltachiresonance(kappa=kapparoots, u=tiler(u,kapparoots), chieff=tiler(chieff,kapparoots), q=tiler(q,kapparoots), chi1=tiler(chi1,kapparoots), chi2=tiler(chi2,kapparoots))

            deltachimin,deltachimax = deltachilimits_rectangle(chieff, q, chi1, chi2)
            check = np.squeeze(np.logical_and(deltachires>deltachimin,deltachires<deltachimax))

            # Two of these are compatible
            if np.sum(check)==2:
                kappares=kapparoots[check]
                warnings.warn("Resonances not detected, best guess returned (soft sanitizing)", Warning)

            # In case that also fails, returns the closest two
            else:


                diffmin = np.abs(deltachires-deltachimin)
                diffmax = np.abs(deltachires-deltachimax)

                try:
                    kappares = np.sort(np.squeeze([kapparoots[diffmin==min(diffmin)], kapparoots[diffmax==min(diffmax)]]))
            
                except:
                    # If that also fails and you're close to q=1, return the analytic result
                    if np.isclose(q,1):

                        r=float(eval_r(u=u,q=q))
                        kappamin = np.maximum((chi1-chi2)**2 / 8 , chieff**2 /2) /r**0.5 + chieff/2
                        kappamax = (chi1+chi2)**2 / 8 /r**0.5 + chieff/2

                        kappares=np.array([kappamin,kappamax])
                        warnings.warn("Resonances not detected, best guess returned (using analytic result for q=1)", Warning)

                else:
                    warnings.warn("Resonances not detected, best guess returned (aggressive sanitizing)", Warning)

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
            S1 = eval_S1(q, chi1)
            S2 = eval_S2(q, chi2)
            L=1/(2*u)
            kappares[1]= ((L+S1-S2)**2 - L**2) / (2*L)
        if np.isclose(chieff,downup):
            S1 = eval_S1(q, chi1)
            S2 = eval_S2(q, chi2)
            L=1/(2*u)
            kappares[0]= ((L-S1+S2)**2 - L**2) / (2*L)

        if kappares is None:
            raise ValueError("Input values are not compatible [kapparesonances].")

        # If you didn't find enough solutions, append nans
        #kappares = np.concatenate([kappares, np.repeat(np.nan, 2-len(kappares))])
        
        return kappares

    kappamin, kappamax = np.array(list(map(_compute, kapparootscomplex, u, chieff, q, chi1, chi2))).T

    return np.stack([kappamin, kappamax])


def kapparescaling(kappatilde, r, chieff, q, chi1, chi2):
    """
    Compute kappa from the rescaled parameter 0<=kappatilde<=1.
    
    Parameters
    ----------
    kappatilde: float
        Rescaled version of the asymptotic angular momentum.
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
    kappa: float
        Asymptotic angular momentum.
    
    Examples
    --------
    ``kappa = precession.kapparescaling(kappatilde,r,chieff,q,chi1,chi2)``
    """

    kappatilde = np.atleast_1d(kappatilde)
    kappaminus, kappaplus = kapparesonances(r, chieff, q, chi1, chi2)
    kappa = inverseaffine(kappatilde,kappaminus,kappaplus)
    return kappa


def kappalimits(r=None, chieff=None, q=None, chi1=None, chi2=None, enforce=False, **kwargs):
    """
    Limits on the asymptotic angular momentum. The contraints considered depend on the inputs provided.
        - If r, q, chi1, chi2 are provided, returns the geometrical limits.
        - If r, chieff, q, chi1, and chi2 are provided, returns the spin-orbit resonances.
    The boolean flag enforce raises an error in case the inputs are not compatible. Additional kwargs are passed to kapparesonances.
    
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
        Perform additional checks.
    **kwargs: unpacked dictionary, optional
        Additional keyword arguments.
    
    Returns
    -------
    kappamax: float
        Maximum value of the asymptotic angular momentum kappa.
    kappamin: float
        Minimum value of the asymptotic angular momentum kappa.
    
    Examples
    --------
    ``kappamin,kappamax = precession.kappalimits(r=r,q=q,chi1=chi1,chi2=chi2)``
    ``kappamin,kappamax = precession.kappalimits(r=r,chieff=chieff,q=q,chi1=chi1,chi2=chi2)``
    ``kappamin,kappamax = precession.kappalimits(r=r,chieff=chieff,q=q,chi1=chi1,chi2=chi2,enforce=True)``
    """

    if r is not None and chieff is None and q is not None and chi1 is not None and chi2 is not None:
        kappamin, kappamax = kappalimits_geometrical(r, q, chi1, chi2)

    elif r is not None and chieff is not None and q is not None and chi1 is not None and chi2 is not None:
        kappamin, kappamax = kapparesonances(r, chieff, q, chi1, chi2, **kwargs)
        # Check precondition
        kappamin_cond, kappamax_cond = kappalimits_geometrical(r , q, chi1, chi2)

        if (kappamin >= kappamin_cond).all() and (kappamax <= kappamax_cond).all():
            pass
        else:
            if enforce:
                raise ValueError("Input values are not compatible [kappalimits].")
            else:
                warnings.warn("Input values are not compatible [kappalimits].", Warning)

    else:
        raise TypeError("Provide either (r,q,chi1,chi2) or (r,chieff,q,chi1,chi2).")

    return np.stack([kappamin, kappamax])


def chiefflimits(q, chi1, chi2):
    """
    Limits on the effective spin based only on its definition.
    
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
    chieffmax: float
        Maximum value of the effective spin.
    chieffmin: float
        Minimum value of the effective spin.
    
    Examples
    --------
    ``chieffmin,chieffmax = precession.chiefflimits(q,chi1,chi2)``
    """


    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)
    chiefflim = (chi1+q*chi2)/(1+q)

    return np.stack([-chiefflim, chiefflim])


def deltachilimits_definition(q, chi1, chi2):
    """
    Limits on the weighted spin difference based only on its definition.
    
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
    deltachimax: float
        Maximum value of the effective spin chieff.
    deltachimin: float
        Minimum value of the effective spin chieff.
    
    Examples
    --------
    ``deltachimin,deltachimax = precession.deltachilimits_definition(q,chi1,chi2)``
    """

    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)
    deltachilim = np.abs((chi1-q*chi2)/(1+q))

    return np.stack([-deltachilim, deltachilim])


def anglesresonances(r, chieff, q, chi1, chi2):
    """
    Compute the values of the angles corresponding to the two spin-orbit resonances.
    
    Parameters
    ----------
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
    deltaphiatmax: float
        Value of the angle deltaphi at the resonance that maximizes kappa.
    deltaphiatmin: float
        Value of the angle deltaphi at the resonance that minimizes kappa.
    theta1atmax: float
        Value of the angle theta1 at the resonance that maximizes kappa.
    theta1atmin: float
        Value of the angle theta1 at the resonance that minimizes kappa.
    theta2atmax: float
        Value of the angle theta2 at the resonance that maximizes kappa.
    theta2atmin: float
        Value of the angle theta2 at the resonance that minimizes kappa.
    
    Examples
    --------
    ``theta1atmin,theta2atmin,deltaphiatmin,theta1atmax,theta2atmax,deltaphiatmax = precession.anglesresonances(r,chieff,q,chi1,chi2)``
    """

    q = np.atleast_1d(q).astype(float)

    kappamin, kappamax = kapparesonances(r, chieff, q, chi1, chi2)

    deltachiatmin = deltachiresonance(kappa=kappamin, r=r, chieff=chieff, q=q, chi1=chi1, chi2=chi2)
    theta1atmin = eval_theta1(deltachiatmin, chieff, q, chi1)
    theta2atmin = eval_theta2(deltachiatmin, chieff, q, chi2)
    deltaphiatmin = np.atleast_1d(tiler(np.pi, q))

    deltachiatmax = deltachiresonance(kappa=kappamax, r=r, chieff=chieff, q=q, chi1=chi1, chi2=chi2)
    theta1atmax = eval_theta1(deltachiatmax, chieff, q, chi1)
    theta2atmax = eval_theta2(deltachiatmax, chieff, q, chi2)
    deltaphiatmax = np.atleast_1d(tiler(0, q))

    return np.stack([theta1atmin, theta2atmin, deltaphiatmin, theta1atmax, theta2atmax, deltaphiatmax])


################ Precession parametrization ################


def deltachicubic_coefficients(kappa, u, chieff, q, chi1, chi2):
    """
    Coefficients of the cubic equation in deltachi for its time evolution.
    
    Parameters
    ----------
    kappa: float
        Asymptotic angular momentum.
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
    coeff0: float
        Coefficient to the x^0 term in polynomial.
    coeff1: float
        Coefficient to the x^1 term in polynomial.
    coeff2: float
        Coefficient to the x^2 term in polynomial.
    coeff3: float
        Coefficient to the x^3 term in polynomial.
    
    Examples
    --------
    ``coeff3,coeff2,coeff1,coeff0 = precession.deltachicubic_coefficients(kappa,u,chieff,q,chi1,chi2)``
    """

    kappa = np.atleast_1d(kappa).astype(float)
    u = np.atleast_1d(u).astype(float)
    chieff = np.atleast_1d(chieff).astype(float)
    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)

    coeff3 = u*(1-q)

    # Machine generated with eq_generator.nb
    coeff2 = (-1/2 * ((1 + -1 * q))**2 * q**(-1) * (1 + q) + (2 * (1 + -1 \
    * q) * ((1 + q))**(-3) * u**2 * (chi1**2 + -1 * q**3 * chi2**2) + -1 \
    * (1 + q) * u * (2 * kappa + -1 * chieff)))

    # Machine generated with eq_generator.nb
    coeff1 = ((1 + -1 * q) * q**(-1) * ((1 + q))**2 * (2 * kappa + -1 * \
    chieff) + (4 * q * ((1 + q))**(-3) * u**2 * (chi1**2 + -1 * q**2 * \
    chi2**2) * chieff + -1 * (1 + -1 * q) * q**(-1) * ((1 + q))**(-2) * u \
    * (2 * (chi1**2 + q**4 * chi2**2) + q * ((1 + q))**2 * chieff**2)))


    # Machine generated with eq_generator.nb
    coeff0 = (-1/2 * q**(-1) * ((1 + q))**3 * ((2 * kappa + -1 * \
    chieff))**2 + (q**(-1) * ((1 + q))**(-1) * u * (2 * kappa + -1 * \
    chieff) * (2 * (chi1**2 + q**4 * chi2**2) + q * ((1 + q))**2 * \
    chieff**2) + -2 * q**(-1) * ((1 + q))**(-5) * u**2 * (((chi1**2 + -1 \
    * q**4 * chi2**2))**2 + q * ((1 + q))**3 * (chi1**2 + q**3 * chi2**2) \
    * chieff**2)))

    return np.stack([coeff3, coeff2, coeff1, coeff0])


def deltachicubic_rescaled_coefficients(kappa, u, chieff, q, chi1, chi2, precomputedcoefficients=None):
    """
    Rescaled coefficients of the cubic equation in deltachi for its time evolution. This is necessary to avoid dividing by (1-q).
    
    Parameters
    ----------
    kappa: float
        Asymptotic angular momentum.
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
    precomputedcoefficients: array, optional (default: None)
        Pre-computed output of deltachicubic_coefficients for computational efficiency.
    
    Returns
    -------
    coeff0: float
        Coefficient to the x^0 term in polynomial.
    coeff1: float
        Coefficient to the x^1 term in polynomial.
    coeff2: float
        Coefficient to the x^2 term in polynomial.
    coeff3: float
        Coefficient to the x^3 term in polynomial.
    
    Examples
    --------
    ``coeff3,coeff2,coeff1,coeff0 = precession.deltachicubic_rescaled_coefficients(kappa,u,chieff,q,chi1,chi2)``
    ``coeff3,coeff2,coeff1,coeff0 = precession.deltachicubic_rescaled_coefficients(kappa,u,chieff,q,chi1,chi2,precomputedcoefficients=coeffs)``
    """


    u = np.atleast_1d(u).astype(float)
    q = np.atleast_1d(q).astype(float)

    if precomputedcoefficients is None:
        _, coeff2, coeff1, coeff0 = deltachicubic_coefficients(kappa, u, chieff, q, chi1, chi2)
    else:
        assert precomputedcoefficients.shape[0] == 4, "Shape of precomputedroots must be (3,N), i.e. deltachiminus, deltachiplus, deltachi3. [deltachiroots]"
        _, coeff2, coeff1, coeff0=np.array(precomputedcoefficients)

    # Careful! Do not divide coeff3 by (1-q) but recompute explicitely
    coeff3r = u 
    coeff2r = coeff2
    coeff1r = (1-q) * coeff1
    coeff0r = (1-q)**2 * coeff0

    return np.stack([coeff3r, coeff2r, coeff1r, coeff0r])


def deltachiroots(kappa, u, chieff, q, chi1, chi2, full_output=True, precomputedroots=None):
    """
    Roots of the cubic equation in deltachi that describes the dynamics on the precession timescale.
    
    Parameters
    ----------
    kappa: float
        Asymptotic angular momentum.
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
    full_output: boolean, optional (default: True)
        Return additional outputs.
    precomputedroots: array, optional (default: None)
        Pre-computed output of deltachiroots for computational efficiency.
    
    Returns
    -------
    deltachi3: float, optional
        Spurious root of the deltachi evolution.
    deltachiminus: float
        Lowest physical root of the deltachi evolution.
    deltachiplus: float
        Lowest physical root of the deltachi evolution.
    
    Examples
    --------
    ``deltachiminus,deltachiplus,deltachi3 = precession.deltachiroots(kappa,u,chieff,q,chi1,chi2)``
    ``deltachiminus,deltachiplus,deltachi3 = precession.deltachiroots(kappa,u,chieff,q,chi1,chi2,precomputedroots=roots)``
    ``deltachiminus,deltachiplus = precession.deltachiroots(kappa,u,chieff,q,chi1,chi2,full_output=False)``
    """

    if precomputedroots is None:
        deltachiminus, deltachiplus, _ = wraproots(deltachicubic_coefficients, kappa, u, chieff, q, chi1, chi2).T

        # If you need the spurious root as well.
        if full_output:
            _, _, deltachi3 = wraproots(deltachicubic_rescaled_coefficients, kappa, u, chieff, q, chi1, chi2).T
        # Otherwise avoid (for computational efficiency)
        else:
            deltachi3 = np.atleast_1d(tiler(np.nan,deltachiminus))

        return np.stack([deltachiminus, deltachiplus, deltachi3])

    else:
        precomputedroots=np.array(precomputedroots)
        assert precomputedroots.shape[0] == 3, "Shape of precomputedroots must be (3,N), i.e. deltachiminus, deltachiplus, deltachi3. [deltachiroots]"
        return precomputedroots


def deltachilimits_rectangle(chieff, q, chi1, chi2):
    """
    Limits on the weighted spin difference for given (chieff, q, chi1, chi2). This is a rectangle in the (chieff,deltachi) plane.
    
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
    deltachimax: float
        Maximum value of the weighted spin difference.
    deltachimin: float
        Minimum value of the weighted spin difference.
    
    Examples
    --------
    ``deltachimin,deltachimax = precession.deltachilimits_rectangle(chieff,q,chi1,chi2)``
    """

    chieff = np.atleast_1d(chieff).astype(float)
    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)


    deltachimin = np.maximum( -chieff - 2*chi1/(1+q), chieff - 2*q*chi2/(1+q))
    deltachimax = np.minimum( -chieff + 2*chi1/(1+q), chieff + 2*q*chi2/(1+q))

    return np.stack([deltachimin, deltachimax])


def deltachilimits_plusminus(kappa, r, chieff, q, chi1, chi2, precomputedroots=None):
    """
    Limits on the weighted spin difference for given (kappa, r, chieff, q, chi1, chi2). These are two of the solutions of the underlying cubic polynomial.
    
    Parameters
    ----------
    kappa: float
        Asymptotic angular momentum.
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
        Pre-computed output of deltachiroots for computational efficiency.
    
    Returns
    -------
    deltachimax: float
        Maximum value of the weighted spin difference.
    deltachimin: float
        Minimum value of the weighted spin difference.
    
    Examples
    --------
    ``deltachimin,deltachimax = precession.deltachilimits_plusminus(kappa,r,chieff,q,chi1,chi2)``
    """

    u = eval_u(r,q)
    deltachiminus, deltachiplus, _ = deltachiroots(kappa, u, chieff, q, chi1, chi2, full_output=False, precomputedroots=precomputedroots)

    # Correct when too close to perfect alignment
    angleup=tiler(0,q)
    angledown=tiler(np.pi,q)

    chieffupup = eval_chieff(angleup, angleup, q, chi1, chi2)
    deltachiupup = eval_deltachi(angleup, angleup, q, chi1, chi2)
    deltachiminus = np.where(np.isclose(chieff,chieffupup), deltachiupup,deltachiminus)
    deltachiplus = np.where(np.isclose(chieff,chieffupup), deltachiupup,deltachiplus)

    chieffdowndown = eval_chieff(angledown, angledown, q, chi1, chi2)
    deltachidowndown = eval_deltachi(angledown, angledown, q, chi1, chi2)
    deltachiminus = np.where(np.isclose(chieff,chieffdowndown), deltachidowndown,deltachiminus)
    deltachiplus = np.where(np.isclose(chieff,chieffdowndown), deltachidowndown,deltachiplus)

    return np.stack([deltachiminus, deltachiplus])


def deltachilimits(kappa=None, r=None, chieff=None, q=None, chi1=None, chi2=None,precomputedroots=None):
    """
    Limits on the weighted spin difference. The contraints considered depend on the inputs provided.
        - If q, chi1, chi2 are provided, returns the geometrical limits.
        - If chieff, q, chi1, and chi2 are provided, returns the  joint 'rectagle' limits.
        - If kappa, r, chieff, q, chi1, and chi2 are provided, returns the  solution of the cubic that sets the precession-timescale evolution.
    
    Parameters
    ----------
    kappa: float, optional (default: None)
        Asymptotic angular momentum.
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
    precomputedroots: array, optional (default: None)
        Pre-computed output of deltachiroots for computational efficiency.
    
    Returns
    -------
    deltachimax: float
        Maximum value of the weighted spin difference.
    deltachimin: float
        Minimum value of the weighted spin difference.
    
    Examples
    --------
    ``deltachimin,deltachimax = precession.deltachilimits(q=q,chi1=chi1,chi2=chi2)``
    ``deltachimin,deltachimax = precession.deltachilimits(chieff=chieff,q=q,chi1=chi1,chi2=chi2)``
    ``deltachimin,deltachimax = precession.deltachilimits(kappa=kappa,r=r,chieff=chieff,q=q,chi1=chi1,chi2=chi2)``
    """

    if kappa is None and r is None and chieff is None and q is not None and chi1 is not None and chi2 is not None:
        deltachimin, deltachimax = deltachilimits_definition(q, chi1, chi2)


    elif kappa is None and r is None and chieff is not None and q is not None and chi1 is not None and chi2 is not None:
        deltachimin, deltachimax = deltachilimits_rectangle(chieff, q, chi1, chi2)

    elif kappa is not None and r is not None and chieff is not None and q is not None and chi1 is not None and chi2 is not None:
        deltachimin, deltachimax = deltachilimits_plusminus(kappa, r, chieff, q, chi1, chi2, precomputedroots=precomputedroots)

    else:
        raise TypeError("Provide either (q, chi1, chi2), (chieff, q, chi1, chi2), or (kappa, r, chieff, q, chi1, chi2).")

    return np.stack([deltachimin, deltachimax])


def deltachirescaling(deltachitilde, kappa, r, chieff, q, chi1, chi2,precomputedroots=None):
    """
    Compute deltachi from the rescaled parameter 0<=deltachitilde<=1.
    
    Parameters
    ----------
    deltachitilde: float
        Rescaled version of the weighted spin difference.
    kappa: float
        Asymptotic angular momentum.
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
        Pre-computed output of deltachiroots for computational efficiency.
    
    Returns
    -------
    deltachi: float
        Weighted spin difference.
    
    Examples
    --------
    ``deltachi = precession.deltachirescaling(deltachitilde,kappa,r,chieff,q,chi1,chi2)``
    """

    deltachiminus, deltachiplus = deltachilimits_plusminus(kappa, r, chieff, q, chi1, chi2,precomputedroots=precomputedroots)
    deltachi =  inverseaffine(deltachitilde, deltachiminus, deltachiplus)

    return deltachi


def deltachiresonance(kappa=None, r=None, u=None, chieff=None, q=None, chi1=None, chi2=None):
    """
    Assuming that the inputs correspond to a spin-orbit resonance, find the corresponding value of deltachi. There will be two roots that are coincident if not for numerical errors: for concreteness, return the mean of the real part.
    This function is dangerous: we do not check that the input is a resonance, that's up to the user (so use at your own risk).
    Valid inputs are either (kappa,r,chieff,q,chi1,chi2) or (kappa,u,chieff,q,chi1,chi2).
    
    Parameters
    ----------
    kappa: float, optional (default: None)
        Asymptotic angular momentum.
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
    deltachi: float
        Weighted spin difference.
    
    Examples
    --------
    ``deltachi = precession.deltachiresonance(kappa=kappa,r=r,chieff=chieff,q=q,chi1=chi1,chi2=chi2)``
    ``deltachi = precession.deltachiresonance(kappa=kappa,u=u,chieff=chieff,q=q,chi1=chi1,chi2=chi2)``
    """

    if q is None or chi1 is None or chi2 is None or kappa is None:
        raise TypeError("Please provide q, chi1, and chi2.")

    if r is None and u is None:
        raise TypeError("Please provide either r or u.")
    elif r is not None and u is None:
        u = eval_u(r=r, q=q)

    coeffs = deltachicubic_coefficients(kappa, u, chieff, q, chi1, chi2)

    with np.errstate(invalid='ignore'):  # nan is ok here
        deltachires = np.mean(np.real(np.sort_complex(roots_vec(coeffs.T))[:,:-1]),axis=1)

    return deltachires


def elliptic_parameter(kappa, u, chieff, q, chi1, chi2, precomputedroots=None):
    """
    Parameter m entering elliptic functions for the evolution of deltachi.
    
    Parameters
    ----------
    kappa: float
        Asymptotic angular momentum.
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
    precomputedroots: array, optional (default: None)
        Pre-computed output of deltachiroots for computational efficiency.
    
    Returns
    -------
    m: float
        Parameter of elliptic function(s).
    
    Examples
    --------
    ``m = precession.elliptic_parameter(kappa,u,chieff,q,chi1,chi2)``
    """

    q=np.atleast_1d(q).astype(float)

    deltachiminus,deltachiplus,deltachi3 = deltachiroots(kappa, u, chieff, q, chi1, chi2, precomputedroots=precomputedroots)
    m = (1-q)*(deltachiplus-deltachiminus)/(deltachi3-(1-q)*deltachiminus)

    return m


def deltachitildeav(m,tol=1e-7):
    """
    Precession average of the rescaled weighted spin difference.
    
    Parameters
    ----------
    m: float
        Parameter of elliptic function(s).
    tol: float, optional (default: 1e-7)
        Numerical tolerance, see source code for details.
    
    Returns
    -------
    coeff: float
        Coefficient.
    
    Examples
    --------
    ``coeff = precession.deltachitildeav(m,tol=1e-7)``
    """

    m = np.atleast_1d(m).astype(float)
    # The limit of the Ssav coefficient as m->0 is finite and equal to 1/2.
    # This is implementation is numerically stable up to m~1e-10.
    # For m=1e-7, the analytic m=0 limit is returned with a precision of 1e-9, which is enough.
    m = np.minimum(np.maximum(tol, m),1-tol)
    coeff = (1-scipy.special.ellipe(m)/scipy.special.ellipk(m))/m

    return coeff


def deltachitildeav2(m,tol=1e-7):
    """
    Precession average of the rescaled weighted spin difference squared.
    
    Parameters
    ----------
    m: float
        Parameter of elliptic function(s).
    tol: float, optional (default: 1e-7)
        Numerical tolerance, see source code for details.
    
    Returns
    -------
    coeff: float
        Coefficient.
    
    Examples
    --------
    ``coeff = precession.deltachitildeav2(m,tol=1e-7)``
    """

    m = np.atleast_1d(m).astype(float)
    # The limit of the Ssav coefficient as m->0 is finite and equal to 1/2.
    # This is implementation is numerically stable up to m~1e-10.
    # For m=1e-7, the analytic m=0 limit is returned with a precision of 1e-9, which is enough.
    m = np.minimum(np.maximum(tol, m),1-tol)

    coeff = (2+m-2*(1+m)*scipy.special.ellipe(m)/scipy.special.ellipk(m)) / (3*m**2)

    return coeff


def ddchidt_prefactor(r, chieff, q):
    """
    Numerical prefactor to the ddeltachi/dt derivative.
    
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
        Prefactor in the ddeltachi/dt equation.
    
    Examples
    --------
    ``mathcalA = precession.ddchidt_prefactor(r,chieff,q)``
    """

    r = np.atleast_1d(r).astype(float)
    chieff = np.atleast_1d(chieff).astype(float)
    q = np.atleast_1d(q).astype(float)

    mathcalA = (3/2)*((1+q)**(-1/2))*(r**(-11/4))*(1-(chieff/r**0.5))

    return mathcalA


def dchidt2_RHS(deltachi, kappa, r, chieff, q, chi1, chi2, precomputedroots=None, donotnormalize=False):
    """
    Right-hand side of the (ddeltachi/dt)^2 equation. 
    Setting donotnormalize=True is equivalent to mathcalA=1.
    
    Parameters
    ----------
    deltachi: float
        Weighted spin difference.
    kappa: float
        Asymptotic angular momentum.
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
        Pre-computed output of deltachiroots for computational efficiency.
    donotnormalize: boolean, optional (default: False)
        If True omit the numerical prefactor.
    
    Returns
    -------
    dchidt2: float
        Squared time derivative of the weighted spin difference.
    
    Examples
    --------
    ``dchidt2 = precession.dchidt2_RHS(r,chieff,q)``
    """

    q=np.atleast_1d(q).astype(float)

    u= eval_u(r=r,q=q)
    deltachiminus,deltachiplus,deltachi3 = deltachiroots(kappa, u, chieff, q, chi1, chi2, precomputedroots=precomputedroots)

    if donotnormalize:
        mathcalA = 1
    else:
        mathcalA = ddchidt_prefactor(r, chieff, q)
    
    dchidt2 = mathcalA**2 * ( (deltachi-deltachiminus)*(deltachiplus-deltachi)*(deltachi3-(1-q)*deltachi))

    return dchidt2


def eval_tau(kappa, r, chieff, q, chi1, chi2, precomputedroots=None, return_psiperiod=False, donotnormalize=False):
    """
    Evaluate the nutation period.
    Setting return_psiperiod=True return the phase-period instead of the time period (i.e. returns tau/2K(m) insteaf of tau). This is useful to avoid the evaluation of an elliptic integral when it's not needed.
    Setting donotnormalize=True is equivalent to mathcalA=1.
    
    Parameters
    ----------
    kappa: float
        Asymptotic angular momentum.
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
        Pre-computed output of deltachiroots for computational efficiency.
    MISSING: COULD NOT BUILD, optional (default: False)
        FILL MANUALLY.
    donotnormalize: boolean, optional (default: False)
        If True omit the numerical prefactor.
    
    Returns
    -------
    tau: float
        Nutation period.
    
    Examples
    --------
    ``tau = precession.eval_tau(kappa,r,chieff,q,chi1,chi2)``
    ``tau = precession.eval_tau(kappa,r,chieff,q,chi1,chi2,return_psiperiod=True)``
    """

    q=np.atleast_1d(q).astype(float)

    u = eval_u(r,q)

    if donotnormalize:
        mathcalA = 1
    else:
        mathcalA = ddchidt_prefactor(r, chieff, q)

    deltachiminus,deltachiplus,deltachi3 = deltachiroots(kappa, u, chieff, q, chi1, chi2, precomputedroots=precomputedroots)
    m = elliptic_parameter(kappa, u, chieff, q, chi1, chi2, precomputedroots=np.stack([deltachiminus,deltachiplus,deltachi3]))

    psiperiod =  2 / ( mathcalA * (deltachi3 - (1-q)*deltachiminus)**(1/2) )
    if return_psiperiod:
        tau = psiperiod
    else:
        tau = 2*scipy.special.ellipk(m) * psiperiod

    return tau


def deltachioft(t, kappa , r, chieff, q, chi1, chi2, precomputedroots=None):
    """
    Evolution of deltachi on the precessional timescale (without radiation reaction).
    The broadcasting rules for this function are more general than those of the rest of the code. The variable t is allowed to have shapes (N,M) while all the other variables have shape (N,). This is useful to sample M precession configuration for each of the N binaries specified as inputs.
    
    Parameters
    ----------
    t: float
        Time.
    kappa: float
        Asymptotic angular momentum.
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
        Pre-computed output of deltachiroots for computational efficiency.
    
    Returns
    -------
    deltachi: float
        Weighted spin difference.
    
    Examples
    --------
    ``deltachi = precession.deltachioft(t,kappa,r,chieff,q,chi1,chi2)``
    """

    t = np.atleast_1d(t).astype(float)
    u = eval_u(r,q)

    deltachiminus,deltachiplus,deltachi3 = deltachiroots(kappa, u, chieff, q, chi1, chi2, precomputedroots=precomputedroots)
    psiperiod = eval_tau(kappa, r, chieff, q, chi1, chi2, precomputedroots=np.stack([deltachiminus,deltachiplus,deltachi3]), return_psiperiod=True)


    m = elliptic_parameter(kappa, u, chieff, q, chi1, chi2, precomputedroots=np.stack([deltachiminus,deltachiplus,deltachi3]))
 

    sn, _, _, _ = scipy.special.ellipj(t / psiperiod, m)
    deltachitilde = sn**2



    deltachi = deltachirescaling(deltachitilde, kappa, r, chieff, q, chi1, chi2,precomputedroots=np.stack([deltachiminus,deltachiplus,deltachi3]))

    return deltachi


def tofdeltachi(deltachi, kappa , r, chieff, q, chi1, chi2, cyclesign=1, precomputedroots=None):
    """
    Time as a function of deltachi on the precessional timescale (without radiation reaction).
    
    Parameters
    ----------
    deltachi: float
        Weighted spin difference.
    kappa: float
        Asymptotic angular momentum.
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
        Pre-computed output of deltachiroots for computational efficiency.
    
    Returns
    -------
    t: float
        Time.
    
    Examples
    --------
    ``t = precession.tofdeltachi(deltachi,kappa,r,chieff,q,chi1,chi2,cyclesign=1)``
    """

    u= eval_u(r=r,q=q)
    deltachiminus,deltachiplus,deltachi3 = deltachiroots(kappa, u, chieff, q, chi1, chi2, precomputedroots=precomputedroots)

    psiperiod = eval_tau(kappa, r, chieff, q, chi1, chi2, precomputedroots=np.stack([deltachiminus,deltachiplus,deltachi3]), return_psiperiod=True)
    deltachitilde = affine(deltachi,deltachiminus,deltachiplus)
    m = elliptic_parameter(kappa, u, chieff, q, chi1, chi2, precomputedroots=np.stack([deltachiminus,deltachiplus,deltachi3]))
    t = np.sign(cyclesign) * psiperiod * scipy.special.ellipkinc(np.arcsin(deltachitilde**(1/2)), m)

    return t 


def deltachisampling(kappa, r, chieff, q, chi1, chi2, N=1, precomputedroots=None):
    """
    Sample N values of deltachi at fixed separation accoring to its PN-weighted distribution function.
    Can only be used to sample the *same* number of configuration for each binary. If the inputs have shape (M,) the output will have shape
        - (M,N) if M>1 and N>1;
        - (M,) if N=1;
        - (N,) if M=1.
    
    Parameters
    ----------
    kappa: float
        Asymptotic angular momentum.
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
    precomputedroots: array, optional (default: None)
        Pre-computed output of deltachiroots for computational efficiency.
    
    Returns
    -------
    deltachi: float
        Weighted spin difference.
    
    Examples
    --------
    ``deltachi = precession.deltachisampling(kappa,r,chieff,q,chi1,chi2,N=1)``
    """

    u= eval_u(r=r,q=q)

    # Compute the deltachi roots only once and pass them to both functions
    deltachiminus,deltachiplus,deltachi3 = deltachiroots(kappa, u, chieff, q, chi1, chi2, precomputedroots=precomputedroots)

    tau = eval_tau(kappa, r, chieff, q, chi1, chi2, precomputedroots=np.stack([deltachiminus,deltachiplus,deltachi3]))

    # For each binary, generate N samples between 0 and tau.
    # For r=infinity use a simple placeholder
    t = np.random.uniform(np.zeros(len(tau)),np.where(u!=0, tau, 0),size=(N,len(tau)))


    # np.squeeze is necessary to return shape (M,) instead of (M,1) if N=1
    # np.atleast_1d is necessary to return shape (1,) instead of (,) if M=N=1
    t= np.atleast_1d(np.squeeze(t))

    # Note the special broadcasting rules of deltachioft, see deltachioft.__docs__
    # deltachi has shape (M, N).
    deltachi = deltachioft(t, kappa , r, chieff, q, chi1, chi2, precomputedroots=np.stack([deltachiminus,deltachiplus,deltachi3]))

    # For infinity use the analytic result. Ignore q=1 "divide by zero" warning:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        deltachiinf = np.squeeze(np.tile( eval_deltachiinf(kappa, chieff, q, chi1, chi2), (N,1) ))

    deltachi=np.where(u!=0, deltachi,deltachiinf)

    return np.squeeze(deltachi.T)


################ Dynamics in an intertial frame ################


def intertial_ingredients(kappa, r, chieff, q, chi1, chi2):
    """
    Numerical prefactors entering the precession frequency.
    
    Parameters
    ----------
    kappa: float
        Asymptotic angular momentum.
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
    bigC0: float
        Prefactor in the OmegaL equation.
    bigCminus: float
        Prefactor in the OmegaL equation.
    bigCplus: float
        Prefactor in the OmegaL equation.
    bigRminus: float
        Prefactor in the PhiL equation.
    bigRplus: float
        Prefactor in the PhiL equation.
    
    Examples
    --------
    ``bigC0,bigCplus,bigCminus,bigRplus,bigRminus = precession.intertial_ingredients(kappa,r,chieff,q,chi1,chi2)``
    """

    kappa = np.atleast_1d(kappa).astype(float)
    r = np.atleast_1d(r).astype(float)
    chieff = np.atleast_1d(chieff).astype(float)
    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)


    # Machine generated with eq_generator.nb
    bigC0 = 1/2 * q * ((1 + q))**(-2) * (r)**(-5/2) * ((1 + 2 * q**(-1) * \
    ((1 + q))**2 * (r)**(-1/2) * kappa))**(1/2)

    # Machine generated with eq_generator.nb
    bigCplus = 3 * ((1 + 2 * q**(-1) * ((1 + q))**2 * (r)**(-1/2) * \
    kappa))**(-1/2) * (1 + -1 * (r)**(-1/2) * chieff) * (q**(-1) * ((1 + \
    q))**3 * (r)**(-1/2) * kappa + (-1/2 * (1 + -1 * q) * q**(-2) * \
    (r)**(-1) * (chi1**2 + -1 * q**4 * chi2**2) + (1 + q) * (1 + ((1 + 2 \
    * q**(-1) * ((1 + q))**2 * (r)**(-1/2) * kappa))**(1/2)) * (1 + \
    (r)**(-1/2) * chieff)))

    # Machine generated with eq_generator.nb
    bigCminus = -3 * ((1 + 2 * q**(-1) * ((1 + q))**2 * (r)**(-1/2) * \
    kappa))**(-1/2) * (1 + -1 * (r)**(-1/2) * chieff) * (q**(-1) * ((1 + \
    q))**3 * (r)**(-1/2) * kappa + (-1/2 * (1 + -1 * q) * q**(-2) * \
    (r)**(-1) * (chi1**2 + -1 * q**4 * chi2**2) + (1 + q) * (1 + -1 * ((1 \
    + 2 * q**(-1) * ((1 + q))**2 * (r)**(-1/2) * kappa))**(1/2)) * (1 + \
    (r)**(-1/2) * chieff)))

    # Machine generated with eq_generator.nb
    bigRplus = (-2 * q * ((1 + q))**(-1) * (1 + ((1 + 2 * q**(-1) * ((1 + \
    q))**2 * (r)**(-1/2) * kappa))**(1/2)) + -1 * (1 + q) * (r)**(-1/2) * \
    chieff)

    # Machine generated with eq_generator.nb
    bigRminus = (-2 * q * ((1 + q))**(-1) * (1 + -1 * ((1 + 2 * q**(-1) * \
    ((1 + q))**2 * (r)**(-1/2) * kappa))**(1/2)) + -1 * (1 + q) * \
    (r)**(-1/2) * chieff)

    return np.stack([bigC0, bigCplus, bigCminus,bigRplus,bigRminus])


def eval_OmegaL(deltachi, kappa, r, chieff, q, chi1, chi2):
    """
    Evaluate the precession frequency OmegaL along the precession cycle.
    
    Parameters
    ----------
    deltachi: float
        Weighted spin difference.
    kappa: float
        Asymptotic angular momentum.
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
    
    Examples
    --------
    ``OmegaL = precession.eval_OmegaL(deltachi,kappa,r,chieff,q,chi1,chi2)``
    """

    deltachi = np.atleast_1d(deltachi).astype(float)
    q = np.atleast_1d(q).astype(float)
    r = np.atleast_1d(r).astype(float)

    bigC0, bigCplus, bigCminus,bigRplus,bigRminus = intertial_ingredients(kappa, r, chieff, q, chi1, chi2)

    OmegaL =  bigC0 * (1 - bigCplus/(bigRplus - deltachi * (1-q)*r**(-1/2)) -  bigCminus/(bigRminus - deltachi * (1-q)*r**(-1/2)) )

    return OmegaL



def eval_phiL(deltachi, kappa, r, chieff, q, chi1, chi2, cyclesign=1, precomputedroots=None):
    """
    Evaluate the azimuthal angle of L in a plane orthogonal to J along the precession cycle.
    
    Parameters
    ----------
    deltachi: float
        Weighted spin difference.
    kappa: float
        Asymptotic angular momentum.
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
        Pre-computed output of deltachiroots for computational efficiency.
    
    Returns
    -------
    phiL: float
        Azimuthal angle spanned by L about J.
    
    Examples
    --------
    ``phiL = precession.eval_phiL(deltachi,kappa,r,chieff,q,chi1,chi2,cyclesign=1)``
    """

    q = np.atleast_1d(q).astype(float)
    r = np.atleast_1d(r).astype(float)
    
    u= eval_u(r=r,q=q)
    deltachiminus,deltachiplus,deltachi3 = deltachiroots(kappa, u, chieff, q, chi1, chi2, precomputedroots=precomputedroots)

    bigC0, bigCplus, bigCminus,bigRplus,bigRminus = intertial_ingredients(kappa, r, chieff, q, chi1, chi2)

    psiperiod = eval_tau(kappa, r, chieff, q, chi1, chi2, precomputedroots=np.stack([deltachiminus,deltachiplus,deltachi3]), return_psiperiod=True)
    deltachitilde = affine(deltachi,deltachiminus,deltachiplus)
    m = elliptic_parameter(kappa, u, chieff, q, chi1, chi2, precomputedroots=np.stack([deltachiminus,deltachiplus,deltachi3]))

    phiL = np.sign(cyclesign) * bigC0 * psiperiod * ( scipy.special.ellipkinc(np.arcsin(deltachitilde**(1/2)), m)
        - bigCplus / (bigRplus - deltachiminus*(1-q)*r**(-1/2))
        * ellippi( (1-q)*r**(-1/2)*(deltachiplus-deltachiminus) /  (bigRplus - deltachiminus*(1-q)*r**(-1/2)), np.arcsin(deltachitilde**(1/2)), m)
        - bigCminus / (bigRminus - deltachiminus*(1-q)*r**(-1/2))
        * ellippi( (1-q)*r**(-1/2)*(deltachiplus-deltachiminus) /  (bigRminus - deltachiminus*(1-q)*r**(-1/2)), np.arcsin(deltachitilde**(1/2)), m) )
    return phiL 



def eval_alpha(kappa, r, chieff, q, chi1, chi2, precomputedroots=None):
    """
    Evaluate the precession angle spanned by L during an entire cycle.
    
    Parameters
    ----------
    kappa: float
        Asymptotic angular momentum.
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
        Pre-computed output of deltachiroots for computational efficiency.
    
    Returns
    -------
    alpha: float
        Azimuthal angle spanned by L about J during an entire cycle.
    
    Examples
    --------
    ``alpha = precession.eval_alpha(kappa,r,chieff,q,chi1,chi2)``
    """

    q = np.atleast_1d(q).astype(float)
    r = np.atleast_1d(r).astype(float)

    u= eval_u(r=r,q=q)

    with warnings.catch_warnings():
        
        # If there are infinitely large separation in the array the following will throw a warning. You can safely ignore it because that value is not used, see below  
        if 0 in u:
            warnings.filterwarnings("ignore", category=Warning)
 
        deltachiminus,deltachiplus,deltachi3 = deltachiroots(kappa, u, chieff, q, chi1, chi2, precomputedroots=precomputedroots)
        bigC0, bigCplus, bigCminus,bigRplus,bigRminus = intertial_ingredients(kappa, r, chieff, q, chi1, chi2)
        psiperiod = eval_tau(kappa, r, chieff, q, chi1, chi2, precomputedroots=np.stack([deltachiminus,deltachiplus,deltachi3]),return_psiperiod=True)
        m = elliptic_parameter(kappa, u, chieff, q, chi1, chi2, precomputedroots=np.stack([deltachiminus,deltachiplus,deltachi3]))

        alpha = 2 * bigC0 * psiperiod * ( scipy.special.ellipk(m)
            - bigCplus / (bigRplus - deltachiminus*(1-q)*r**(-1/2))
            * ellippi( (1-q)*r**(-1/2)*(deltachiplus-deltachiminus) /  (bigRplus - deltachiminus*(1-q)*r**(-1/2)), np.pi/2, m)
            - bigCminus / (bigRminus - deltachiminus*(1-q)*r**(-1/2))
            * ellippi( (1-q)*r**(-1/2)*(deltachiplus-deltachiminus) /  (bigRminus - deltachiminus*(1-q)*r**(-1/2)), np.pi/2, m) )

    # At infinitely large separation use the analytic result
    if 0 in u:
        
        mathcalY =  2 * q * (1+q)**3 * kappa * chieff - (1+q)**5 * kappa**2 +(1-q) *(chi1**2 -q**4 * chi2**2)
        alphainf1= 2*np.pi*(4+3*q)*q/3/(1-q**2)
        alphainf2 = 2*np.pi*(4*q+3)/3/(1-q**2)

        alphainf = np.where(mathcalY>=0, alphainf1, alphainf2)
        alpha =np.where(u>0,alpha,alphainf)
    
    return alpha 


################ More phenomenology ################

def morphology(kappa, r, chieff, q, chi1, chi2, simpler=False, precomputedroots=None):
    """
    Evaluate the spin morphology. Returns:
        - L0 for librating about deltaphi=0,
        - Lpi for librating about deltaphi=pi
        - C- for circulating from deltaphi=pi to deltaphi=0,
        - C+ for circulating from deltaphi=0 to deltaphi=pi.
    If simpler=True, do not distinguish between the two circulating morphologies and return C for both.
    
    Parameters
    ----------
    kappa: float
        Asymptotic angular momentum.
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
    precomputedroots: array, optional (default: None)
        Pre-computed output of deltachiroots for computational efficiency.
    
    Returns
    -------
    morph: string
        Spin morphology.
    
    Examples
    --------
    ``morph = precession.morphology(kappa,r,chieff,q,chi1,chi2,simpler=False)``
    """


    deltachiminus,deltachiplus = deltachilimits_plusminus(kappa, r, chieff, q, chi1, chi2)
    # Pairs of booleans based on the values of deltaphi at S- and S+
    status = np.transpose([eval_cosdeltaphi(deltachiminus, kappa, r, chieff, q, chi1, chi2) > 0, eval_cosdeltaphi(deltachiplus, kappa, r, chieff, q, chi1, chi2) > 0])
    # Map to labels
    dictlabel = {(False, False): "Lpi", (True, True): "L0", (False, True): "C-", (True, False): "C+"}
    # Subsitute pairs with labels
    morphs = np.zeros(deltachiminus.shape)
    for k, v in dictlabel.items():
        morphs = np.where((status == k).all(axis=1), v, morphs)
    # Simplifies output, only one circulating morphology
    if simpler:
        morphs = np.where(np.logical_or(morphs == 'C+', morphs == 'C-'), 'C', morphs)

    return morphs


def chip_terms(theta1, theta2, q, chi1, chi2):
    """
    Compute the two terms entering the effective precessing spin chip.
    
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
        Term in effective precessing spin.
    chipterm2: float
        Term in effective precessing spin.
    
    Examples
    --------
    ``chipterm1,chipterm2 = precession.chip_terms(theta1,theta2,q,chi1,chi2)``
    """

    theta1 = np.atleast_1d(theta1).astype(float)
    theta2 = np.atleast_1d(theta2).astype(float)
    q = np.atleast_1d(q).astype(float)

    chipterm1 = chi1*np.sin(theta1)
    omegatilde = q*(4*q+3)/(4+3*q)
    chipterm2 = omegatilde * chi2*np.sin(theta2)

    return np.stack([chipterm1, chipterm2])


def eval_chip_heuristic(theta1, theta2, q, chi1, chi2):
    """
    Heuristic definition of the effective precessing spin chip (Schmidt et al 2015), see arxiv:2011.11948. This definition inconsistently averages over some, but not all, variations on the precession timescale.
    
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
        Effective precessing spin.
    
    Examples
    --------
    ``chip = precession.eval_chip_heuristic(theta1,theta2,q,chi1,chi2)``
    """

    term1, term2 = chip_terms(theta1, theta2, q, chi1, chi2)
    chip = np.maximum(term1, term2)
    return chip


def eval_chip_generalized(theta1, theta2, deltaphi, q, chi1, chi2):
    """
    Generalized definition of the effective precessing spin chip, see arxiv:2011.11948. This definition retains all variations on the precession timescale.
    
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
        Effective precessing spin.
    
    Examples
    --------
    ``chip = precession.eval_chip_generalized(theta1,theta2,deltaphi,q,chi1,chi2)``
    """

    term1, term2 = chip_terms(theta1, theta2, q, chi1, chi2)
    chip = (term1**2 + term2**2 + 2*term1*term2*np.cos(deltaphi))**0.5
    return chip


def eval_chip_averaged(kappa, r, chieff, q, chi1, chi2, **kwargs):
    """
    Averaged definition of the effective precessing spin, see arxiv:2011.11948.
    Additional keywords arguments are passed to `precession_average`.
    
    Parameters
    ----------
    kappa: float
        Asymptotic angular momentum.
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
    **kwargs: unpacked dictionary, optional
        Additional keyword arguments.
    
    Returns
    -------
    chip: float
        Effective precessing spin.
    
    Examples
    --------
    ``chip = precession.eval_chip_averaged(kappa,r,chieff,q,chi1,chi2)``
    """

    def _integrand(deltachi, kappa, r, chieff, q, chi1, chi2):
        theta1, theta2, deltaphi = conserved_to_angles(deltachi, kappa, r, chieff, q, chi1, chi2)
        chip_integrand = eval_chip_generalized(theta1, theta2, deltaphi, q, chi1, chi2)
        return chip_integrand

    chip = precession_average(kappa, r, chieff, q, chi1, chi2, _integrand, kappa, r, chieff, q, chi1, chi2, **kwargs)

    return chip


def eval_chip_rms(kappa, r, chieff, q, chi1, chi2):
    """
    Root-mean-square definition of the effective precessing spin. This is much faster to evaluate than the plain average.
    
    Parameters
    ----------
    kappa: float
        Asymptotic angular momentum.
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
    chip: float
        Effective precessing spin.
    
    Examples
    --------
    ``chip = precession.eval_chip_rms(kappa,r,chieff,q,chi1,chi2)``
    """

    kappa = np.atleast_1d(kappa).astype(float)
    r = np.atleast_1d(r).astype(float)
    chieff = np.atleast_1d(chieff).astype(float)
    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)

    u=eval_u(r=r,q=q)

    # Machine generated with eq_generator.nb
    lambdabar = 1/4 * q**(-1) * (1 + q) * ((4 + 3 * q))**(-2) * u**(-1)

    # Machine generated with eq_generator.nb
    lambda2 = -1 * ((1 + -1 * q))**2 * q * (1 + q) * u

    # Machine generated with eq_generator.nb
    lambda1 = -2 * (1 + -1 * q) * ((1 + q))**2 * ((4 + 3 * q) * (3 + 4 * \
    q) + 7 * q * u * chieff)

    # Machine generated with eq_generator.nb
    lambda0 = (2 * ((1 + q))**3 * (4 + 3 * q) * (3 + 4 * q) * (2 * kappa \
    + -1 * chieff) + -1 * u * (12 * (1 + -1 * q) * ((4 + 3 * q) * chi1**2 \
    + -1 * q**3 * (3 + 4 * q) * chi2**2) + 49 * q * ((1 + q))**3 * \
    chieff**2))

    deltachiminus,deltachiplus,deltachi3 = deltachiroots(kappa, u, chieff, q, chi1, chi2)

    m = elliptic_parameter(kappa, u, chieff, q, chi1, chi2, precomputedroots=np.stack([deltachiminus, deltachiplus, deltachi3]))
    
    chip = lambdabar**(1/2) * ( (deltachiplus-deltachiminus)**2 * lambda2 * deltachitildeav2(m)  +
        (deltachiplus-deltachiminus) * (lambda1 + 2*deltachiminus*lambda2) * deltachitildeav(m) +
        (deltachiminus*lambda1 + deltachiminus**2*lambda2 + lambda0 ))**(1/2)

    # Debug:
    # def _integrand(deltachi, kappa, r, chieff, q, chi1, chi2):
    #     theta1, theta2, deltaphi = conserved_to_angles(deltachi, kappa, r, chieff, q, chi1, chi2)
    #     chip_integrand = eval_chip_generalized(theta1, theta2, deltaphi, q, chi1, chi2)**2
    #     return chip_integrand
    # chip = precession_average(kappa, r, chieff, q, chi1, chi2, _integrand, kappa, r, chieff, q, chi1, chi2, **kwargs)**0.5

    return chip


def eval_chip(theta1=None, theta2=None, deltaphi=None, deltachi=None, kappa=None, r=None, chieff=None, q=None, chi1=None, chi2=None, which="averaged", **kwargs):
    """
    Compute the effective precessing spin. The keyword `which` one of the following definitions:
        - `heuristic`, as in Schmidt et al 2015. Required inputs are either
        - `generalized`, retail all precession-timescale variations,
        - `averaged` (default), averages over precession-timescale variations,
        - `rms`, compute the root-mean-square average, which is faster to evaluate.
    The required inputs are either (theta1,theta2,deltaphi,r,q,chi1,chi2) or (deltachi,kappa,chieff,r,q,chi1,chi2). In the latter case, deltachi can be omitted and will be resampled from its PN-weighted distribution.
    Some inputs will be ignored for some of the chip definitions (for instance the heuristic chip does not depend on either r or deltaphi), but dummy values are still requested for code consistency.
    The keywords arguments are only used for the `averaged` formulation and are passed to `precession_average`.
    
    Parameters
    ----------
    theta1: float, optional (default: None)
        Angle between orbital angular momentum and primary spin.
    theta2: float, optional (default: None)
        Angle between orbital angular momentum and secondary spin.
    deltaphi: float, optional (default: None)
        Angle between the projections of the two spins onto the orbital plane.
    deltachi: float, optional (default: None)
        Weighted spin difference.
    kappa: float, optional (default: None)
        Asymptotic angular momentum.
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
    **kwargs: unpacked dictionary, optional
        Additional keyword arguments.
    
    Returns
    -------
    chip: float
        Effective precessing spin.
    
    Examples
    --------
    ``chip = precession.eval_chip(theta1=theta1,theta2=theta2,deltaphi=deltaphi,r=r,q=q,chi1=chi1,chi2=chi2,which="heuristic")``
    ``chip = precession.eval_chip(theta1=theta1,theta2=theta2,deltaphi=deltaphi,r=r,q=q,chi1=chi1,chi2=chi2,which="generalized")``
    ``chip = precession.eval_chip(theta1=theta1,theta2=theta2,deltaphi=deltaphi,r=r,q=q,chi1=chi1,chi2=chi2,which="averaged")``
    ``chip = precession.eval_chip(theta1=theta1,theta2=theta2,deltaphi=deltaphi,r=r,q=q,chi1=chi1,chi2=chi2,which="rms")``
    ``chip = precession.eval_chip(deltachi=deltachi,kappa=kappa,r=r,chieff=chieff,q=q,chi1=chi1,chi2=chi2,which="heuristic")``
    ``chip = precession.eval_chip(deltachi=deltachi,kappa=kappa,r=r,chieff=chieff,q=q,chi1=chi1,chi2=chi2,which="generalized")``
    ``chip = precession.eval_chip(deltachi=deltachi,kappa=kappa,r=r,chieff=chieff,q=q,chi1=chi1,chi2=chi2,which="averaged")``
    ``chip = precession.eval_chip(deltachi=deltachi,kappa=kappa,r=r,chieff=chieff,q=q,chi1=chi1,chi2=chi2,which="rms")``
    """


    if r is None or q is None or chi1 is None or chi2 is None:
        raise ValueError("Provide r, q, chi1, and chi2.")

    if theta1 is not None and theta2 is not None and deltaphi is not None and deltachi is None and kappa is None and chieff is None:
        deltachi, kappa, chieff = angles_to_conserved(theta1, theta2, deltaphi, r, q, chi1, chi2, full_output=False)
    
    elif theta1 is None and theta2 is None and deltaphi is None and kappa is not None and chieff is not None:
        if deltachi is None: 
            if which == 'heuristic' or which == 'generalized':
                deltachi = deltachisampling(kappa, r, chieff, q, chi1, chi2)
        theta1, theta2, deltaphi = conserved_to_angles(deltachi, kappa, r, chieff, q, chi1, chi2)
     
    else:
        raise ValueError("Provide either (theta1,theta2,deltaphi), (deltachi,kappa,chieff), or (kappa,chieff).")


    if which == 'heuristic':
        chip = eval_chip_heuristic(theta1, theta2, q, chi1, chi2)

    elif which == 'generalized':
        chip = eval_chip_generalized(theta1, theta2, deltaphi, q, chi1, chi2)

    elif which == 'averaged':
        chip_finite = eval_chip_averaged(kappa, r, chieff, q, chi1, chi2, **kwargs)

        term1, term2 = chip_terms(theta1, theta2, q, chi1, chi2)
        chip_infinity = 2* np.abs(term1+term2) * scipy.special.ellipe(4*term1*term2/(term1+term2)**2)/np.pi         
        chip = np.where(r!=np.inf, chip_finite, chip_infinity)

    elif which == 'rms':
        chip_finite = eval_chip_rms(kappa, r, chieff, q, chi1, chi2)

        term1, term2 = chip_terms(theta1, theta2, q, chi1, chi2)
        chip_infinity = (term1**2 + term2**2)**(1/2)         
        chip = np.where(r!=np.inf, chip_finite, chip_infinity)

    else:
        raise ValueError("`which` needs to be one of the following: `heuristic`, `generalized`, `averaged`, 'rms`.")

    return chip


def eval_nutation_freq(kappa, r, chieff, q, chi1, chi2, precomputedroots=None):
    """
    Nutation frequency of deltachi as it oscillates from deltachi- to deltachi+ back to deltachi-. 
    This of the five phenomenological parameters from arxiv:2103.03894.
    
    Parameters
    ----------
    kappa: float
        Asymptotic angular momentum.
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
        Pre-computed output of deltachiroots for computational efficiency.
    
    Returns
    -------
    littleomega: float
        Squared time derivative of the weighted spin difference.
    
    Examples
    --------
    ``littleomega = precession.eval_nutation_freq(kappa,r,chieff,q,chi1,chi2)``
    """

    tau = eval_tau(kappa, r, chieff, q, chi1, chi2, precomputedroots=precomputedroots)
    littleomega = (2 * np.pi) / tau
    return littleomega


def eval_bracket_omega(kappa, r, chieff, q, chi1, chi2, precomputedroots=None):
    """
    Precession average of the precession frequency deltachi as it oscillates from deltachi- to deltachi+ back to deltachi-.
    This of the five phenomenological parameters from arxiv:2103.03894.

    Parameters
    ----------
    kappa: float
        Asymptotic angular momentum.
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
        Pre-computed output of deltachiroots for computational efficiency.
    
    Returns
    -------
    bracket_omega: float
        Precession-averaged precession frequency.
    
    Examples
    --------
    ``bracket_omega = precession.eval_bracket_omega(kappa,r,chieff,q,chi1,chi2)``
    """

    alpha = eval_alpha(kappa, r, chieff, q, chi1, chi2, precomputedroots=precomputedroots)
    tau = eval_tau(kappa, r, chieff, q, chi1, chi2, precomputedroots=precomputedroots)
    bracket_omega = alpha / tau
    return bracket_omega


def eval_delta_omega(kappa, r, chieff, q, chi1, chi2, precomputedroots=None):
    """
    Variation of the precession frequency of deltachi as it oscillates from deltachi- to deltachi+ back to deltachi-.
    This of the five phenomenological parameters from arxiv:2103.03894.

    Parameters
    ----------
    kappa: float
        Asymptotic angular momentum.
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
        Pre-computed output of deltachiroots for computational efficiency.
    
    Returns
    -------
    delta_omega: float
        Precession frequency variation due to nutation.
    
    Examples
    --------
    ``delta_omega = precession.eval_delta_omega(kappa,r,chieff,q,chi1,chi2)``
    """

    if precomputedroots is None:
        deltachimin, deltachiplus = deltachilimits_plusminus(kappa, r, chieff, q, chi1, chi2)
    else:
        deltachimin, deltachiplus, _ = precomputedroots[:-1]
    Omega_minus = eval_OmegaL(deltachiplus, kappa, r, chieff, q, chi1, chi2)
    Omega_plus = eval_OmegaL(deltachimin, kappa, r, chieff, q, chi1, chi2)
    delta_omega = (Omega_plus - Omega_minus)/2
    return delta_omega


def eval_delta_theta(kappa, r, chieff, q, chi1, chi2, precomputedroots=None):
    """
    Nutation amplitude of deltachi as it oscillates from deltachi- to deltachi+ back to deltachi-.
    This of the five phenomenological parameters from arxiv:2103.03894.
    
    Parameters
    ----------
    kappa: float
        Asymptotic angular momentum.
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
        Pre-computed output of deltachiroots for computational efficiency.
    
    Returns
    -------
    delta_theta: float
        Nutation amplitude.
    
    Examples
    --------
    ``delta_theta = precession.eval_delta_theta(kappa,r,chieff,q,chi1,chi2)``
    """

    if precomputedroots is None:
        deltachimin, deltachiplus = deltachilimits_plusminus(kappa, r, chieff, q, chi1, chi2)
    else:
        deltachimin, deltachiplus, _ = precomputedroots[:-1]
    theta_minus = eval_thetaL(deltachiplus, kappa, r, chieff, q)
    theta_plus = eval_thetaL(deltachimin, kappa, r, chieff, q)
    delta_theta = (theta_plus - theta_minus)/2
    return delta_theta


def eval_bracket_theta(kappa, r, chieff, q, chi1, chi2, **kwargs):
    """
    Precession average of precession amplitude of deltachi as it oscillates from deltachi- to deltachi+ back to deltachi-.
    This of the five phenomenological parameters from arxiv:2103.03894. Additional keywords arguments are passed to `precession_average`.

    
    Parameters
    ----------
    kappa: float
        Asymptotic angular momentum.
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
    **kwargs: unpacked dictionary, optional
        Additional keyword arguments.
    
    Returns
    -------
    bracket_theta: float
        Precession-averaged precession amplitude.
    
    Examples
    --------
    ``bracket_theta = precession.eval_bracket_theta(kappa,r,chieff,q,chi1,chi2)``
    """

    def _integrand(deltachi, kappa, r, chieff, q):
        bracket_theta_integrand = eval_thetaL(deltachi, kappa, r, chieff, q)
        return bracket_theta_integrand

    bracket_theta = precession_average(kappa, r, chieff, q, chi1, chi2, _integrand, kappa, r, chieff, q, **kwargs)
    return bracket_theta


def rupdown(q, chi1, chi2):
    """
    The critical separations r_ud+/- marking the region of the up-down precessional instability.
    
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
    rudm: float
        Inner orbital separation in the up-down instability.
    rudp: float
        Outer orbital separation in the up-down instability.
    
    Examples
    --------
    ``rudp,rudm = precession.rupdown(q,chi1,chi2)``
    """

    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)

    #Ignore q=1 "divide by zero" warning here
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        rudp = (chi1**0.5+(q*chi2)**0.5)**4/(1-q)**2
        rudm = (chi1**0.5-(q*chi2)**0.5)**4/(1-q)**2

    return np.stack([rudp, rudm])


def updown_endpoint(q, chi1, chi2):
    """
    Endpoint of the up-down precessional instability.
    
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
    deltaphi: float
        Angle between the projections of the two spins onto the orbital plane.
    theta1: float
        Angle between orbital angular momentum and primary spin.
    theta2: float
        Angle between orbital angular momentum and secondary spin.
    
    Examples
    --------
    ``theta1,theta2,deltaphi = precession.updown_endpoint(q,chi1,chi2)``
    """

    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)
    
    costhetaupdown = (chi1 - q * chi2) / (chi1 + q * chi2)

    theta1 = np.arccos(costhetaupdown)
    theta2 = np.arccos(costhetaupdown)
    deltaphi = np.zeros(len(theta1))

    return np.stack([theta1, theta2, deltaphi])


def angleresonances_endpoint(q, chi1, chi2, chieff):
    """
    Small-separation limit of the spin-orbit resonances.
    
    Parameters
    ----------
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    chieff: float
        Effective spin.
    
    Returns
    -------
    deltaphiatmax: float
        Value of the angle deltaphi at the resonance that maximizes kappa.
    deltaphiatmin: float
        Value of the angle deltaphi at the resonance that minimizes kappa.
    theta1atmax: float
        Value of the angle theta1 at the resonance that maximizes kappa.
    theta1atmin: float
        Value of the angle theta1 at the resonance that minimizes kappa.
    theta2atmax: float
        Value of the angle theta2 at the resonance that maximizes kappa.
    theta2atmin: float
        Value of the angle theta2 at the resonance that minimizes kappa.
    
    Examples
    --------
    ``theta1atmin,theta2atmin,deltaphiatmin,theta1atmax,theta2atmax,deltaphiatmax = precession.angleresonances_endpoint(q,chi1,chi2,chieff)``
    """

    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)
    chieff = np.atleast_1d(chieff).astype(float)
    
    chieff_uu = eval_chieff(0, 0, q, chi1, chi2)
    chieff_ud = eval_chieff(0, -np.pi, q, chi1, chi2)
    
    thetakappaplus = np.arccos(chieff / chieff_uu)
    deltaphikappaplus = np.zeros_like(thetakappaplus)
    
    
    condition = np.abs(chieff) <= np.abs(chieff_ud)
    costheta1kappaminus_less = chieff / chieff_ud
    costheta1kappaminus_gtr = (1+q) * (chieff**2+chieff_ud*chieff_uu) / (2*chi1*chieff)
    costheta1kappaminus = np.where(condition, costheta1kappaminus_less, costheta1kappaminus_gtr)
    theta1kappaminus = np.arccos(costheta1kappaminus)
    costheta2kappaminus_less = -costheta1kappaminus_less
    costheta2kappaminus_gtr = (1+q) * (chieff**2-chieff_ud*chieff_uu) / (2*q*chi2*chieff)
    costheta2kappaminus = np.where(condition, costheta2kappaminus_less, costheta2kappaminus_gtr)
    theta2kappaminus = np.arccos(costheta2kappaminus)
    deltaphikappaminus = np.ones_like(condition) * np.pi
    
    return np.stack([theta1kappaminus, theta2kappaminus, deltaphikappaminus, thetakappaplus, thetakappaplus, deltaphikappaplus])


def omegasq_aligned(r, q, chi1, chi2, which):
    """
    Squared oscillation frequency of a given perturbed aligned-spin binary.
    The flag which needs to be set to `uu` for up-up, `ud` for up-down, `du` for down-up or `dd` for down-down where the term before (after) the hyphen refers to the spin of the heavier (lighter) black hole.
    
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
    
    Examples
    --------
    ``omegasq = precession.omegasq_aligned(r,q,chi1,chi2,which)``
    """

    r = np.atleast_1d(r).astype(float)
    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)

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

    theta1 = np.arccos(alpha1)
    theta2 = np.arccos(alpha2)
    deltachi = eval_deltachi(theta1, theta2, q, chi1, chi2)
    chieff = eval_chieff(theta1, theta2, q, chi1, chi2)
    
    omegasq = (9/4) * (1/r)**7 * (r**0.5-chieff)**2 * (
        ((1-q)/(1+q))**2*r - 2*((1-q)/(1+q))*deltachi*r**0.5 + chieff**2
        )

    return omegasq


def widenutation_separation(q, chi1, chi2):
    """
    The critical separation r_wide below which the binary component with
    smaller dimensionless spin may undergo wide nutations.
    
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
    
    Examples
    --------
    ``r_wide = precession.widenutation_separation(q,chi1,chi2)``
    """

    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)

    rwide = ((chi1 - q*chi2) / (1-q))**2

    return rwide


def widenutation_condition(r, q, chi1, chi2):
    """
    Conditions for wide nutation to take place. The returned flag which is
        - `wide1` if wide nutation is allowed for the primary BH.
        - `wide2` if wide nutation is allowed for the secondary BH.
        - `nowide` if wide nutation is not allowed.
    
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
    chieff: float
        Effective spin.
    kappa: float
        Asymptotic angular momentum.
    which: string
        Select function behavior.
    
    Examples
    --------
    ``which,kappa,chieff = precession.widenutation_condition(r,q,chi1,chi2)``
    """

    r = np.atleast_1d(r).astype(float)
    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)

    rwide = widenutation_separation(q, chi1, chi2)

    kappawide1 = (chi1**2 - 2*q*chi1**2 + q**4*chi2**2 - 2*q**2*(1-q)*r)/(2*q*(1+q)**2 * r**0.5)
    chieffwide1 = -(1-q)*r**0.5/(1+q)

    kappawide2 = (chi1**2 - 2*q**3*chi1**2 + q**4*chi2**2 + 2*q*(1-q)*r)/(2*q*(1+q)**2 * r**0.5)
    chieffwide2 = (1-q)*r**0.5/(1+q)

    which = np.where(r<=rwide, np.where(chi1<=chi2,"wide1","wide2"), "nowide")
    kappa = np.where(r<=rwide, np.where(chi1<=chi2,kappawide1,kappawide2), np.nan)
    chieff = np.where(r<=rwide, np.where(chi1<=chi2,chieffwide1,chieffwide2), np.nan)

    return np.stack[(which, kappa, chieff)]


################ Precession-averaged evolution ################


def rhs_precav(kappa, u, chieff, q, chi1, chi2):
    """
    Right-hand side of the dkappa/du ODE describing precession-averaged inspiral.
    This is an internal function used by the ODE integrator and is not array-compatible.
    
    Parameters
    ----------
    kappa: float
        Asymptotic angular momentum.
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
    
    Examples
    --------
    ``RHS = precession.rhs_precav(kappa,u,chieff,q,chi1,chi2)``
    """

    if u <= 0:
       # In this case use analytic result
        if q==1: # TODO: think about this again
            Ssav = (chi1**2+q**4 * chi2**2)/(1 + q)**4  #- ( 2*q*(kappa*(1+q) -chieff)*(kappa*(1+q) -q*chieff)/((-1 + q)**2 *(1 + q)**2))
        else:
            Ssav = (chi1**2+q**4 * chi2**2)/(1 + q)**4  - ( 2*q*(kappa*(1+q) -chieff)*(kappa*(1+q) -q*chieff)/((1-q)**2 *(1 + q)**2))

    else:
        # I don't use deltachiroots here because I want to keep complex numbers. This is needed to sanitize the output in some tricky cases
        coeffs = deltachicubic_coefficients(kappa, u, chieff, q, chi1, chi2)        
        coeffsr = deltachicubic_rescaled_coefficients(kappa, u, chieff, q, chi1, chi2, precomputedcoefficients=coeffs)
        deltachiminus, deltachiplus, _ = np.squeeze(np.sort_complex(roots_vec(coeffs.T)))
        _, _, deltachi3 = np.squeeze(np.sort_complex(roots_vec(coeffsr.T)))

        # deltachiminus, deltachiplus are complex. This can happen if the binary is very close to a spin-orbit resonance
        if np.iscomplex(deltachiminus) and np.iscomplex(deltachiplus):
            warnings.warn("Sanitizing RHS output; too close to resonance. [rhs_precav].", Warning)
            deltachiav = np.mean(np.real([deltachiminus, deltachiplus]))

        # Normal case
        else:
            deltachiminus, deltachiplus, deltachi3 = np.real([deltachiminus, deltachiplus, deltachi3])      
            m = elliptic_parameter(kappa, u, chieff, q, chi1, chi2, precomputedroots=np.stack([deltachiminus, deltachiplus, deltachi3]))
            deltachiav = inverseaffine( deltachitildeav(m),  deltachiminus, deltachiplus)

        Ssav = (2*kappa - chieff - (1-q)/(1+q)*deltachiav)/(2*u)

    return float(Ssav)


def integrator_precav(kappainitial, u, chieff, q, chi1, chi2, **odeint_kwargs):
    """
    Integration of ODE dkappa/du describing precession-averaged inspirals. Here u needs to be an array with lenght >=1, where u[0] corresponds to the initial condition and u[1:] corresponds to the location where outputs are returned. For past time infinity use u=0.
    The function is vectorized: evolving N multiple binaries with M outputs requires kappainitial, chieff, q, chi1, chi2 to be of shape (N,) and u of shape (M,N).
    Additional keywords arguments are passed to `scipy.integrate.odeint` after some custom-made default settings.
    
    Parameters
    ----------
    kappainitial: float
        Initial value of the regularized momentum kappa.
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
    **odeint_kwargs: unpacked dictionary, optional
        Additional keyword arguments.
    
    Returns
    -------
    kappa: float
        Asymptotic angular momentum.
    
    Examples
    --------
    ``kappa = precession.integrator_precav(kappainitial,u,chieff,q,chi1,chi2)``
    """

    kappainitial = np.atleast_1d(kappainitial).astype(float)
    u = np.atleast_2d(u).astype(float)
    chieff = np.atleast_1d(chieff).astype(float)
    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)

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

        # solve_ivp implementation. Didn't really work.
        #ODEsolution = scipy.integrate.solve_ivp(rhs_precav, (uinitial, ufinal), np.atleast_1d(kappainitial), method='RK45', t_eval=(uinitial, ufinal), dense_output=True, args=(chieff, q, chi1, chi2), atol=1e-8, rtol=1e-8)  # ,events=event)
        # Return ODE object. The key methods is .sol --callable, sol(t).
        #return ODEsolution

    ODEsolution = np.array(list(map(_compute, kappainitial, u, chieff, q, chi1, chi2, repeat(odeint_kwargs))))

    return ODEsolution


def inspiral_precav(theta1=None, theta2=None, deltaphi=None, kappa=None, r=None, u=None, chieff=None, q=None, chi1=None, chi2=None, requested_outputs=None, enforce=False, **odeint_kwargs):
    """
    Perform precession-averaged inspirals. The variables q, chi1, and chi2 must always be provided.
    The integration range must be specified using either r or u (and not both). These need to be arrays with lenght >=1, where e.g. r[0] corresponds to the initial condition and r[1:] corresponds to the location where outputs are returned. For past time infinity use either u=0 or r=np.inf.
    The function is vectorized: evolving N multiple binaries with M outputs requires kappainitial, chieff, q, chi1, chi2 to be of shape (N,) and u of shape (M,N).
    The initial conditions must be specified in terms of one an only one of the following:
        - theta1,theta2, and deltaphi (but note that deltaphi is not necessary if integrating from infinite separation).
        - kappa, chieff.
    The desired outputs can be specified with a list e.g. requested_outputs=['theta1','theta2','deltaphi']. All the available variables are returned by default. These are: ['theta1', 'theta2', 'deltaphi', 'deltachi', 'kappa', 'r', 'u', 'deltachiminus', 'deltachiplus', 'deltachi3', 'chieff', 'q', 'chi1', 'chi2'].
    The flag enforce allows checking the consistency of the input variables.
    Additional keywords arguments are passed to `scipy.integrate.odeint` after some custom-made default settings.
    
    Parameters
    ----------
    theta1: float, optional (default: None)
        Angle between orbital angular momentum and primary spin.
    theta2: float, optional (default: None)
        Angle between orbital angular momentum and secondary spin.
    deltaphi: float, optional (default: None)
        Angle between the projections of the two spins onto the orbital plane.
    kappa: float, optional (default: None)
        Asymptotic angular momentum.
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
    enforce: boolean, optional (default: False)
        If True raise errors, if False raise warnings.
    **odeint_kwargs: unpacked dictionary, optional
        Additional keyword arguments.
    
    Returns
    -------
    outputs: dictionary
        Set of outputs.
    
    Examples
    --------
    ``outputs = precession.inspiral_precav(theta1=theta1,theta2=theta2,deltaphi=deltaphi,r=r,q=q,chi1=chi1,chi2=chi2)``
    ``outputs = precession.inspiral_precav(theta1=theta1,theta2=theta2,deltaphi=deltaphi,u=u,q=q,chi1=chi1,chi2=chi2)``
    ``outputs = precession.inspiral_precav(kappa,r=r,chieff=chieff,q=q,chi1=chi1,chi2=chi2)``
    ``outputs = precession.inspiral_precav(kappa,u=u,chieff=chieff,q=q,chi1=chi1,chi2=chi2)``
    """

    # Substitute None inputs with arrays of Nones
    inputs = [theta1, theta2, deltaphi, kappa, r, u, chieff, q, chi1, chi2]
    for k, v in enumerate(inputs):
        if v is None:
            inputs[k] = np.atleast_1d(np.squeeze(tiler(None, np.atleast_1d(q))))
        else:
            if k == 4 or k == 5:  # Either u or r
                inputs[k] = np.atleast_2d(inputs[k])
            else:  # Any of the others
                inputs[k] = np.atleast_1d(inputs[k])
    theta1, theta2, deltaphi, kappa, r, u, chieff, q, chi1, chi2 = inputs

    # This array has to match the outputs of _compute (in the right order!)
    alloutputs = np.array(['theta1', 'theta2', 'deltaphi', 'deltachi', 'kappa', 'r', 'u', 'deltachiminus', 'deltachiplus', 'deltachi3', 'chieff', 'q', 'chi1', 'chi2'])
    # If in doubt, return everything
    if requested_outputs is None:
        requested_outputs = alloutputs

    def _compute(theta1, theta2, deltaphi, kappa, r, u, chieff, q, chi1, chi2):

        # Make sure you have q, chi1, and chi2.
        if q is None or chi1 is None or chi2 is None:
            raise TypeError("Please provide q, chi1, and chi2.")

        # Make sure you have either r or u.
        if r is not None and u is None:
            assert np.logical_or(ismonotonic(r, '<='), ismonotonic(r, '>=')), 'r must be monotonic'
            u = eval_u(r, tiler(q, r))
        elif r is None and u is not None:
            assert np.logical_or(ismonotonic(u, '<='), ismonotonic(u, '>=')), 'u must be monotonic'
            r = eval_r(u=u, q=tiler(q, u))
        else:
            raise TypeError("Please provide either r or u. Use np.inf for infinity.")

        assert np.sum(u == 0) <= 1 and np.sum(u[1:-1] == 0) == 0, "There can only be one r=np.inf location, either at the beginning or at the end."

        # User provided theta1,theta2, and deltaphi. Get chieff and kappa.
        if theta1 is not None and theta2 is not None and deltaphi is not None and kappa is None and chieff is None:
            deltachi, kappa, chieff = angles_to_conserved(theta1, theta2, deltaphi, r[0], q, chi1, chi2)

        # User provides kappa, chieff
        elif theta1 is None and theta2 is None and deltaphi is None and kappa is not None and chieff is not None:
            pass

        else:
            raise TypeError("Please provide one and not more of the following: (theta1,theta2,deltaphi), (kappa,chieff).")

        # Enforce limits. Uncomment if you want to be more restrictive (though some integrations are going to fail for roundoff errors in the resonant finder)
        if enforce:
            chieffmin, chieffmax = chiefflimits(q, chi1, chi2)
            assert chieff >= chieffmin and chieff <= chieffmax, "Unphysical initial conditions [inspiral_precav]."+str(theta1)+" "+str(theta2)+" "+str(deltaphi)+" "+str(kappa)+" "+str(r)+" "+str(u)+" "+str(chieff)+" "+str(q)+" "+str(chi1)+" "+str(chi2)
            kappamin,kappamax = kappalimits(r=r[0], chieff=chieff, q=q, chi1=chi1, chi2=chi2)
            assert kappa >= kappamin and kappa <= kappamax, "Unphysical initial conditions [inspiral_precav]."+str(theta1)+" "+str(theta2)+" "+str(deltaphi)+" "+str(kappa)+" "+str(r)+" "+str(u)+" "+str(chieff)+" "+str(q)+" "+str(chi1)+" "+str(chi2)

        # Actual integration.
        kappa = np.squeeze(integrator_precav(kappa, u, chieff, q, chi1, chi2,**odeint_kwargs))

        deltachiminus = None
        deltachiplus = None
        deltachi3 = None
        deltachi=None
        theta1=None
        theta2=None
        deltaphi=None

        # Roots along the evolution
        if any(x in requested_outputs for x in ['theta1', 'theta2', 'deltaphi', 'deltachi', 'deltachiminus', 'deltachiplus', 'deltachi3']):
            deltachiminus,deltachiplus,deltachi3 = deltachiroots(kappa, u, tiler(chieff,r), tiler(q,r),tiler(chi1,r),tiler(chi2,r))

            # Resample deltachi
            if any(x in requested_outputs for x in ['theta1', 'theta2', 'deltaphi', 'deltachi']):
                deltachi = deltachisampling(kappa, r, tiler(chieff,r), tiler(q,r),tiler(chi1,r),tiler(chi2,r), precomputedroots=np.stack([deltachiminus,deltachiplus,deltachi3]))

                # Compute the angles. Assign random cyclesign
                if any(x in requested_outputs for x in ['theta1', 'theta2', 'deltaphi']):
                    theta1,theta2,deltaphi = conserved_to_angles(deltachi, kappa, r, tiler(chieff,r), tiler(q,r),tiler(chi1,r),tiler(chi2,r), cyclesign = np.random.choice([-1, 1], r.shape))

        return theta1, theta2, deltaphi, deltachi, kappa, r, u, deltachiminus, deltachiplus, deltachi3, chieff, q, chi1, chi2

    # Here I force dtype=object because the outputs have different shapes
    allresults = np.array(list(map(_compute, theta1, theta2, deltaphi, kappa, r, u, chieff, q, chi1, chi2)), dtype=object).T

    # Return only requested outputs (in1d return boolean array)
    wantoutputs = np.in1d(alloutputs, requested_outputs)

    # Store into a dictionary
    outcome = {}

    for k, v in zip(alloutputs[wantoutputs], allresults[wantoutputs]):
        outcome[k] = np.squeeze(np.stack(v))

        # For the constants of motion...
        if k == 'chieff' or k == 'q' or k == 'chi1' or k == 'chi2':  # Constants of motion
            outcome[k] = np.atleast_1d(outcome[k])
        #... and everything else
        else:
            outcome[k] = np.atleast_2d(outcome[k])

    return outcome


def precession_average(kappa, r, chieff, q, chi1, chi2, func, *args, method='quadrature', Nsamples=1e4):
    """
    Average a generic function over a precession cycle. The function needs to have call: func(deltachi, *args). Keywords arguments are not supported.
    There are integration methods implemented:
        - method='quadrature' uses scipy.integrate.quad. This is set by default and should be preferred.
        - method='montecarlo' samples t(deltachi) and approximate the integral with a Monte Carlo sum. The number of samples can be specifed by Nsamples.
    Additional keyword arguments are passed to scipy.integrate.quad.
    
    Parameters
    ----------
    kappa: float
        Asymptotic angular momentum.
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
    Nsamples: integer, optional (default: 1e4)
        Number of Monte Carlo samples.
    
    Returns
    -------
    func_av: float
        Precession average of func.
    
    Examples
    --------
    ``func_av = precession.precession_average(kappa,r,chieff,q,chi1,chi2,func,*args,method='quadrature',Nsamples=1e4)``
    """

    kappa=np.atleast_1d(kappa).astype(float)
    r=np.atleast_1d(r).astype(float)
    chieff=np.atleast_1d(chieff).astype(float)
    q=np.atleast_1d(q).astype(float)
    chi1=np.atleast_1d(chi1).astype(float)
    chi2=np.atleast_1d(chi2).astype(float)

    u = eval_u(r=r,q=q)
    deltachiminus,deltachiplus,deltachi3 = deltachiroots(kappa, u, chieff, q, chi1, chi2)

    if method == 'quadrature':

        tau = eval_tau(kappa, r, chieff, q, chi1, chi2, precomputedroots=np.stack([deltachiminus,deltachiplus,deltachi3]), donotnormalize=True)

        # Each args needs to be iterable
        args = [np.atleast_1d(a) for a in args]

        # Compute the numerator explicitely
        def _integrand(deltachi, deltachiminus,deltachiplus,deltachi3, kappa, r, chieff, q, chi1, chi2, *args):
            dchidt2 = dchidt2_RHS(deltachi, kappa, r, chieff, q, chi1, chi2, precomputedroots=np.stack([deltachiminus,deltachiplus,deltachi3]),donotnormalize=True)

            return func(deltachi, *args) / dchidt2**(1/2)

        def _compute(deltachiminus,deltachiplus,deltachi3, kappa, r, chieff, q, chi1, chi2, *args):
            return scipy.integrate.quad(_integrand, deltachiminus, deltachiplus, args=(deltachiminus,deltachiplus,deltachi3, kappa, r, chieff, q, chi1, chi2, *args))[0]

        func_av = np.array(list(map(_compute, deltachiminus,deltachiplus,deltachi3, kappa, r, chieff, q, chi1, chi2, *args))) / tau * 2 

    elif method == 'montecarlo':

        deltachi = deltachisampling(kappa, r, chieff, q, chi1, chi2, N=int(Nsamples), precomputedroots=np.stack([deltachiminus,deltachiplus,deltachi3]))
        evals = func(deltachi, *args)
        func_av = np.sum(evals, axis=-1)/Nsamples
        func_av = np.atleast_1d(func_av)

    else:
        raise ValueError("Available methods are 'quadrature' and 'montecarlo'.")

    return func_av


################ Orbit-averaged evolution ################

def rhs_orbav(allvars, v, q, m1, m2, eta, chi1, chi2, S1, S2, PNorderpre=[0,0.5], PNorderrad=[0,1,1.5,2,2.5,3,3.5]):
    """
    Right-hand side of the systems of ODEs describing orbit-averaged inspiral. The equations are reported in Sec 4A of Gerosa and Kesden, arXiv:1605.01067. The format is d[allvars]/dv=RHS where allvars=[Lhx,Lhy,Lhz,S1hx,S1hy,S1hz,S2hx,S2hy,S2hz,t], h indicates unit vectors, v is the orbital velocity, and t is time.
    This is an internal function used by the ODE integrator and is not array-compatible.
    
    Parameters
    ----------
    allvars: array
        Packed ODE input variables.
    v: float
        Newtonian orbital velocity.
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
    PNorderpre: array (default: [0,0.5])
        PN orders considered in the spin-precession equations.
    PNorderrad: array (default: [0,0.5])
        PN orders considered in the radiation-reaction equation.
    
    Returns
    -------
    RHS: float
        Right-hand side.
    
    Examples
    --------
    ``RHS = precession.rhs_orbav(allvars,v,q,m1,m2,eta,chi1,chi2,S1,S2,PNorderpre=[0,0.5],PNorderrad=[0,1,1.5,2,2.5,3,3.5])``
    """

    # Unpack inputs
    Lh = allvars[0:3]
    S1h = allvars[3:6]
    S2h = allvars[6:9]
    t = allvars[9]

    # Angles
    ct1 = np.dot(S1h, Lh)
    ct2 = np.dot(S2h, Lh)
    ct12 = np.dot(S1h, S2h)

    # Spin precession for S1
    Omega1 = (0 in PNorderpre) * eta*v**5*(2+3*q/2)*Lh + (0.5 in PNorderpre) * v**6*(S2*S2h-3*S2*ct2*Lh-3*q*S1*ct1*Lh)/2
    dS1hdt = np.cross(Omega1, S1h)

    # Spin precession for S2
    Omega2 = (0 in PNorderpre) * eta*v**5*(2+3/(2*q))*Lh + (0.5 in PNorderpre) * v**6*(S1*S1h-3*S1*ct1*Lh-3*S2*ct2*Lh/q)/2
    dS2hdt = np.cross(Omega2, S2h)

    # Conservation of angular momentum
    dLhdt = -v*(S1*dS1hdt+S2*dS2hdt)/eta

    dvdt = (32*eta*v**9/5) * (
        + (0 in PNorderrad) * 1
        - (1 in PNorderrad)*v**2 
                 * (743+924*eta)/336
        + (1.5 in PNorderrad) * v**3 
                 * (4*np.pi
                 - chi1*ct1*(113*m1**2/12 + 25*eta/4)
                 - chi2*ct2*(113*m2**2/12 + 25*eta/4))
        + (2 in PNorderrad) * v**4 
                 * (34103/18144 + 13661*eta/2016 + 59*eta**2/18
                 + eta*chi1*chi2 * (721*ct1*ct2 - 247*ct12)/48
                 + ((m1*chi1)**2 * (719*ct1**2-233))/96
                 + ((m2*chi2)**2 * (719*ct2**2-233))/96)
        - (2.5 in PNorderrad) * v**5 
                 * np.pi*(4159+15876*eta)/672
        + (3 in PNorderrad)*v**6 
                 * (16447322263/139708800 + 16*np.pi**2/3
                 - 1712*(0.5772156649+np.log(4*v))/105
                 + (451*np.pi**2/48 - 56198689/217728)*eta
                 + 541*eta**2/896 - 5605*eta**3/2592)
        + (3.5 in PNorderrad) * v**7 
                 * np.pi*(-4415/4032 + 358675*eta/6048
                 + 91495*eta**2/1512))

    # Integrate in v, not in time
    dtdv = 1./dvdt
    dLhdv = dLhdt*dtdv
    dS1hdv = dS1hdt*dtdv
    dS2hdv = dS2hdt*dtdv

    # Pack outputs
    return np.concatenate([dLhdv, dS1hdv, dS2hdv, [dtdv]])


def integrator_orbav(Lhinitial, S1hinitial, S2hinitial, v, q, chi1, chi2, PNorderpre=[0,0.5], PNorderrad=[0,1,1.5,2,2.5,3,3.5], **odeint_kwargs):
    """
    Integration of the systems of ODEs describing orbit-averaged inspirals.
    Additional keywords arguments are passed to `scipy.integrate.odeint` after some custom-made default settings.
    
    Parameters
    ----------
    Lhinitial: array
        Initial direction of the orbital angular momentum, unit vector.
    S1hinitial: array
        Initial direction of the primary spin, unit vector.
    S2hinitial: array
        Initial direction of the secondary spin, unit vector.
    v: float
        Newtonian orbital velocity.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    PNorderpre: array (default: [0,0.5])
        PN orders considered in the spin-precession equations.
    PNorderrad: array (default: [0,0.5])
        PN orders considered in the radiation-reaction equation.
    **odeint_kwargs: unpacked dictionary, optional
        Additional keyword arguments.
    
    Returns
    -------
    ODEsolution: array
        Solution of the ODE.
    
    Examples
    --------
    ``ODEsolution = precession.integrator_orbavintegrator_orbav(Lhinitial,S1hinitial,S2hinitial,v,q,chi1,chi2)``
    """

    Lhinitial = np.atleast_2d(Lhinitial).astype(float)
    S1hinitial = np.atleast_2d(S1hinitial).astype(float)
    S2hinitial = np.atleast_2d(S2hinitial).astype(float)
    v = np.atleast_2d(v).astype(float)
    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)

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

        # solve_ivp implementation. Didn't really work.
        #ODEsolution = scipy.integrate.solve_ivp(rhs_orbav, (vinitial, vfinal), ic, method='LSODA', t_eval=(vinitial, vfinal), dense_output=True, args=(q, m1, m2, eta, chi1, chi2, S1, S2, quadrupole_formula),rtol=1e-12,atol=1e-12)
        #ODEsolution = scipy.integrate.solve_ivp(rhs_orbav, (vinitial, vfinal), ic, t_eval=(vinitial, vfinal), dense_output=True, args=(q, m1, m2, eta, chi1, chi2, S1, S2, quadrupole_formula))

        # Make sure the first step is large enough. This is to avoid LSODA to propose a tiny step which causes the integration to stall
        if 'h0' not in odeint_kwargs: odeint_kwargs['h0']=  np.sign(v[-1]-v[0]) *  v[0]/1e6

        ODEsolution = scipy.integrate.odeint(rhs_orbav, ic, v, args=(q, m1, m2, eta, chi1, chi2, S1, S2, PNorderpre, PNorderrad), **odeint_kwargs)#, printmessg=0,rtol=1e-10,atol=1e-10)#,tcrit=sing)
        return ODEsolution

    ODEsolution = np.array(list(map(_compute, Lhinitial, S1hinitial, S2hinitial, v, q, chi1, chi2)))

    return ODEsolution


def inspiral_orbav(theta1=None, theta2=None, deltaphi=None, Lh=None, S1h=None, S2h=None, deltachi=None, kappa=None, r=None, u=None, chieff=None, q=None, chi1=None, chi2=None, cyclesign=+1, PNorderpre=[0,0.5], PNorderrad=[0,1,1.5,2,2.5,3,3.5], requested_outputs=None, **odeint_kwargs):
    """
    Perform precession-averaged inspirals. The variables q, chi1, and chi2 must always be provided.
    The integration range must be specified using either r or u (and not both). These need to be arrays with lenght >=1, where e.g. r[0] corresponds to the initial condition and r[1:] corresponds to the location where outputs are returned.
    The function is vectorized: evolving N multiple binaries with M outputs requires kappainitial, chieff, q, chi1, chi2 to be of shape (N,) and u of shape (M,N).
    The initial conditions must be specified in terms of one an only one of the following:
        - Lh, S1h, and S2h
        - theta1,theta2, and deltaphi.
        - deltachi, kappa, chieff, cyclesign.
    The desired outputs can be specified with a list e.g. requested_outputs=['theta1','theta2','deltaphi']. All the available variables are returned by default. These are: ['theta1', 'theta2', 'deltaphi', 'deltachi', 'kappa', 'r', 'u', 'deltachiminus', 'deltachiplus', 'deltachi3', 'chieff', 'q', 'chi1', 'chi2'].
    The flag enforce allows checking the consistency of the input variables.
    Additional keywords arguments are passed to `scipy.integrate.odeint` after some custom-made default settings.
    
    Parameters
    ----------
    theta1: float, optional (default: None)
        Angle between orbital angular momentum and primary spin.
    theta2: float, optional (default: None)
        Angle between orbital angular momentum and secondary spin.
    deltaphi: float, optional (default: None)
        Angle between the projections of the two spins onto the orbital plane.
    Lh: array, optional (default: None)
        Direction of the orbital angular momentum, unit vector.
    S1h: array, optional (default: None)
        Direction of the primary spin, unit vector.
    S2h: array, optional (default: None)
        Direction of the secondary spin, unit vector.
    deltachi: float, optional (default: None)
        Weighted spin difference.
    kappa: float, optional (default: None)
        Asymptotic angular momentum.
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
    cyclesign: integer, optional (default: +1)
        Sign (either +1 or -1) to cover the two halves of a precesion cycle.
    PNorderpre: array (default: [0,0.5])
        PN orders considered in the spin-precession equations.
    PNorderrad: array (default: [0,0.5])
        PN orders considered in the radiation-reaction equation.
    requested_outputs: list, optional (default: None)
        Set of outputs.
    **odeint_kwargs: unpacked dictionary, optional
        Additional keyword arguments.
    
    Returns
    -------
    outputs: dictionary
        Set of outputs.
    
    Examples
    --------
    ``outputs = precession.inspiral_orbav(Lh=Lh,S1h=S1h,S2h=S2h,r=r,q=q,chi1=chi1,chi2=chi2)``
    ``outputs = precession.inspiral_orbav(Lh=Lh,S1h=S1h,S2h=S2h,u=u,q=q,chi1=chi1,chi2=chi2)``
    ``outputs = precession.inspiral_orbav(theta1=theta1,theta2=theta2,deltaphi=deltaphi,r=r,q=q,chi1=chi1,chi2=chi2)``
    ``outputs = precession.inspiral_orbav(theta1=theta1,theta2=theta2,deltaphi=deltaphi,u=u,q=q,chi1=chi1,chi2=chi2)``
    ``outputs = precession.inspiral_orbav(deltachi=deltachi,kappa=kappa,r=r,chieff=chieff,q=q,chi1=chi1,chi2=chi2)``
    """


    # Substitute None inputs with arrays of Nones
    inputs = [theta1, theta2, deltaphi, Lh, S1h, S2h, deltachi, kappa, r, u, chieff, q, chi1, chi2]
    for k, v in enumerate(inputs):
        if v is None:
            inputs[k] = np.atleast_1d(np.squeeze(tiler(None, np.atleast_1d(q))))
        else:
            if k == 3 or k == 4 or k == 5 or k == 8 or k == 9:  # Lh, S1h, S2h, u, or r
                inputs[k] = np.atleast_2d(inputs[k])
            else:  # Any of the others
                inputs[k] = np.atleast_1d(inputs[k])
    theta1, theta2, deltaphi, Lh, S1h, S2h, deltachi, kappa, r, u, chieff, q, chi1, chi2 = inputs

    def _compute(theta1, theta2, deltaphi, Lh, S1h, S2h, deltachi, kappa, r, u, chieff, q, chi1, chi2,cyclesign):

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
        if Lh is not None and S1h is not None and S2h is not None and theta1 is None and theta2 is None and deltaphi is None and deltachi is None and kappa is None and chieff is None:
            pass

        # User provides theta1, theta2, and deltaphi.
        elif Lh is None and S1h is None and S2h is None and theta1 is not None and theta2 is not None and deltaphi is not None and deltachi is None and kappa is None and chieff is None:
            Lh, S1h, S2h = angles_to_Jframe(theta1, theta2, deltaphi, r[0], q, chi1, chi2)


        # User provides deltachi, kappa, and chieff.
        elif Lh is None and S1h is None and S2h is None and theta1 is None and theta2 is None and deltaphi is None and deltachi is not None and kappa is not None and chieff is not None:
            # cyclesign=+1 by default
            Lh, S1h, S2h = conserved_to_Jframe(deltachi, kappa, r[0], chieff, q, chi1, chi2, cyclesign=cyclesign)
        else:
            raise TypeError("Please provide one and not more of the following: (Lh,S1h,S2h), (theta1,theta2,deltaphi), (deltachi,kappa,chieff).")

        # Make sure vectors are normalized
        Lh = Lh/np.linalg.norm(Lh)
        S1h = S1h/np.linalg.norm(S1h)
        S2h = S2h/np.linalg.norm(S2h)

        v = eval_v(r)

        # Integration
        evaluations = integrator_orbav(Lh, S1h, S2h, v, q, chi1, chi2, PNorderpre=PNorderpre, PNorderrad=PNorderrad,**odeint_kwargs)[0].T
        # For solve_ivp implementation
        #evaluations = np.squeeze(ODEsolution.item().sol(v))

        # Returned output is
        # Lx, Ly, Lz, S1x, S1y, S1z, S2x, S2y, S2z, (t)
        Lh = evaluations[0:3, :].T
        S1h = evaluations[3:6, :].T
        S2h = evaluations[6:9, :].T
        t = evaluations[9, :]

        # Renormalize. The normalization is not enforced by the integrator, it is only maintaied within numerical accuracy.
        #Lh = Lh/np.linalg.norm(Lh)
        #S1h = S1h/np.linalg.norm(S1h)
        #S2h = S2h/np.linalg.norm(S2h)

        S1 = eval_S1(q, chi1)
        S2 = eval_S2(q, chi2)
        L = eval_L(r, tiler(q, r))
        Lvec = (L*Lh.T).T
        S1vec = S1*S1h
        S2vec = S2*S2h
        theta1, theta2, deltaphi = vectors_to_angles(Lvec, S1vec, S2vec)
        deltachi, kappa, chieff, cyclesign = vectors_to_conserved(Lvec, S1vec, S2vec, tiler(q,r), full_output=True)

        return t, theta1, theta2, deltaphi, Lh, S1h, S2h, deltachi, kappa, r, u, chieff, q, chi1, chi2, cyclesign

    # This array has to match the outputs of _compute (in the right order!)
    alloutputs = np.array(['t', 'theta1', 'theta2', 'deltaphi', 'Lh', 'S1h', 'S2h', 'deltachi', 'kappa', 'r', 'u', 'chieff', 'q', 'chi1', 'chi2', 'cyclesign'])


    if cyclesign ==+1 or cyclesign==-1:
        cyclesign=np.atleast_1d(tiler(cyclesign,q))
    
    # Here I force dtype=object because the outputs have different shapes
    allresults = np.array(list(map(_compute, theta1, theta2, deltaphi, Lh, S1h, S2h, deltachi, kappa, r, u, chieff, q, chi1, chi2, cyclesign)), dtype=object).T

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

        if k == 'q' or k == 'chi1' or k == 'chi2':  # Constants of motion (chieff is not enforced!)
            outcome[k] = np.atleast_1d(outcome[k])
        else:
            outcome[k] = np.atleast_2d(outcome[k])

    return outcome


def inspiral_hybrid(theta1=None, theta2=None, deltaphi=None, deltachi=None, kappa=None, r=None, rswitch=None, u=None, uswitch=None, chieff=None, q=None, chi1=None, chi2=None, requested_outputs=None,**odeint_kwargs):
    """
    Perform hybrid inspirals, i.e. evolve the binary at large separation with a pression-averaged evolution and at small separation with an orbit-averaged evolution, matching the two. The variables q, chi1, and chi2 must always be provided. The integration range must be specified using either r or u (and not both); provide also uswitch and rswitch consistently.
    Either r of u needs to be arrays with lenght >=1, where e.g. r[0] corresponds to the initial condition and r[1:] corresponds to the location where outputs are returned. For past time infinity use either u=0 or r=np.inf.
    The function is vectorized: evolving N multiple binaries with M outputs requires kappainitial, chieff, q, chi1, chi2 to be of shape (N,) and u of shape (M,N).
    The initial conditions must be specified in terms of one an only one of the following:
        - theta1,theta2, and deltaphi (but note that deltaphi is not necessary if integrating from infinite separation).
        - kappa, chieff.
    The desired outputs can be specified with a list e.g. requested_outputs=['theta1','theta2','deltaphi']. All the available variables are returned by default. These are: ['theta1', 'theta2', 'deltaphi', 'deltachi', 'kappa', 'r', 'u', 'deltachiminus', 'deltachiplus', 'deltachi3', 'chieff', 'q', 'chi1', 'chi2'].
    The flag enforce allows checking the consistency of the input variables.
    Additional keywords arguments are passed to `scipy.integrate.odeint` after some custom-made default settings.
    
    Parameters
    ----------
    theta1: float, optional (default: None)
        Angle between orbital angular momentum and primary spin.
    theta2: float, optional (default: None)
        Angle between orbital angular momentum and secondary spin.
    deltaphi: float, optional (default: None)
        Angle between the projections of the two spins onto the orbital plane.
    deltachi: float, optional (default: None)
        Weighted spin difference.
    kappa: float, optional (default: None)
        Asymptotic angular momentum.
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
    **odeint_kwargs: unpacked dictionary, optional
        Additional keyword arguments.
    
    Returns
    -------
    outputs: dictionary
        Set of outputs.
    
    Examples
    --------
    ``outputs = precession.inspiral_hybrid(theta1=theta1,theta2=theta2,deltaphi=deltaphi,r=r,q=q,chi1=chi1,chi2=chi2)``
    ``outputs = precession.inspiral_hybrid(theta1=theta1,theta2=theta2,deltaphi=deltaphi,u=u,q=q,chi1=chi1,chi2=chi2)``
    ``outputs = precession.inspiral_hybrid(kappa,r=r,chieff=chieff,q=q,chi1=chi1,chi2=chi2)``
    ``outputs = precession.inspiral_hybrid(kappa,u=u,chieff=chieff,q=q,chi1=chi1,chi2=chi2)``
    """

    # Outputs available in both orbit-averaged and precession-averaged evolutions
    alloutputs = np.array(['theta1', 'theta2', 'deltaphi', 'deltachi', 'kappa', 'r', 'u', 'chieff', 'q', 'chi1', 'chi2'])
    if requested_outputs is None:
        requested_outputs = alloutputs
        # Return only those requested (in1d return boolean array)
    wantoutputs = np.intersect1d(alloutputs, requested_outputs)

    # Substitute None inputs with arrays of Nones
    inputs = [theta1, theta2, deltaphi, deltachi, kappa, r, rswitch, u, uswitch, chieff, q, chi1, chi2]
    for k, v in enumerate(inputs):
        if v is None:
            inputs[k] = np.atleast_1d(np.squeeze(tiler(None, np.atleast_1d(q))))
        else:
            if k == 5 or k == 7:  # Either u or r
                inputs[k] = np.atleast_2d(inputs[k])
            else:  # Any of the others
                inputs[k] = np.atleast_1d(inputs[k])
    theta1, theta2, deltaphi, deltachi, kappa, r, rswitch, u, uswitch, chieff, q, chi1, chi2 = inputs

    def _compute(theta1, theta2, deltaphi, deltachi, kappa, r, rswitch, u, uswitch, chieff, q, chi1, chi2):

        if r is None and rswitch is None and u is not None and uswitch is not None:
            r = eval_r(u=u, q=tiler(q, u))
            rswitch = eval_r(u=uswitch, q=tiler(q, uswitch))

        forward = ismonotonic(r, ">=")
        backward = ismonotonic(r, "<=")

        assert np.logical_or(forward, backward), "r must be monotonic"
        assert rswitch > np.min(r) and rswitch < np.max(r), "The switching condition must to be within the range spanned by r or u."

        rlarge = r[r >= rswitch]
        rsmall = r[r < rswitch]

        # Integrating forward: precession-averaged first, then orbit-averaged
        if forward:
            inspiral_first = inspiral_precav
            rfirst = np.append(rlarge, rswitch)
            inspiral_second = inspiral_orbav
            rsecond = np.append(rswitch, rsmall)

        # Integrating backward: orbit-averaged first, then precession-averaged
        elif backward:
            inspiral_first = inspiral_orbav
            rfirst = np.append(rsmall, rswitch)
            inspiral_second = inspiral_precav
            rsecond = np.append(rswitch, rlarge)

        # First chunk of the evolution
        evolution_first = inspiral_first(theta1=theta1, theta2=theta2, deltaphi=deltaphi, deltachi=deltachi, kappa=kappa, r=rfirst, chieff=chieff, q=q, chi1=chi1, chi2=chi2, requested_outputs=alloutputs,**odeint_kwargs)

        # Second chunk of the evolution
        evolution_second = inspiral_second(theta1=np.squeeze(evolution_first['theta1'])[-1], theta2=np.squeeze(evolution_first['theta2'])[-1], deltaphi=np.squeeze(evolution_first['deltaphi'])[-1], r=rsecond, q=q, chi1=chi1, chi2=chi2, requested_outputs=alloutputs,**odeint_kwargs)

        # Store outputs
        evolution_full = {}
        for k in wantoutputs:
            # Quantities that vary in both the precession-averaged and the orbit-averaged evolution
            if k in ['theta1', 'theta2', 'deltaphi', 'deltachi', 'kappa', 'r', 'u']:
                evolution_full[k] = np.atleast_2d(np.append(evolution_first[k][:, :-1], evolution_second[k][:, 1:]))
            # Quantities that vary only on the orbit-averaged evolution
            if k in ['chieff']:
                if forward:
                    evolution_full[k] = np.atleast_2d(np.append(tiler(evolution_first[k][:], rfirst[:-1]), evolution_second[k][:, 1:]))
                elif backward:
                    evolution_full[k] = np.atleast_2d(np.append(evolution_first[k][:, :-1], tiler(evolution_second[k][:], rsecond[1:])))
            # Quanties that do not vary
            if k in ['q', 'chi1', 'chi2']:
                evolution_full[k] = evolution_second[k]

        return evolution_full

    allresults = list(map(_compute, theta1, theta2, deltaphi, deltachi, kappa, r, rswitch, u, uswitch, chieff, q, chi1, chi2))
    evolution_full = {}
    for k in allresults[0].keys():
        evolution_full[k] = np.concatenate(list(evolution_full[k] for evolution_full in allresults))

    return evolution_full


def inspiral(*args, which=None, **kwargs):
    """
    Generic wrapper for binary inspirals. The keyword which selects between the different approximation:
        - If which=precession use precession-averaged inspiral (i.e. run `inspiral_precav`)
        - If which=orbit use orbit-averaged inspiral (i.e. run `inspiral_orbav`)
        - If which=hybrid use hybrid inspiral (i.e. run `inspiral_hybrid`)
    All other aguments and keyword arguments are passed on to the relevant inspiral function; see their docs for details.

    Parameters
    ----------
    **args: unpacked dictionary, optional
        Additional arguments.
    which: string, optional (default: None)
        Select function behavior.
    **kwargs: unpacked dictionary, optional
        Additional keyword arguments.
    
    Returns
    -------
    outputs: dictionary, optional
        Set of outputs.
    
    Examples
    --------
    ``outputs = precession.inspiral(*args,which='precession',**kwargs)``
    ``outputs = precession.inspiral(*args,which='orbit',**kwargs)``
    ``outputs = precession.inspiral(*args,which='hybrid',**kwargs)``
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


def gwfrequency_to_pnseparation(theta1, theta2, deltaphi, fGW, q, chi1, chi2, M_msun, PNorder=[0,1,1.5,2]):
    """
    Convert GW frequency (in Hz) to PN orbital separation (in natural units, c=G=M=1). We use the 2PN expression reported in Eq. 4.13 of Kidder 1995, arxiv:gr-qc/9506022.
    
    Parameters
    ----------
    theta1: float
        Angle between orbital angular momentum and primary spin.
    theta2: float
        Angle between orbital angular momentum and secondary spin.
    deltaphi: float
        Angle between the projections of the two spins onto the orbital plane.
    fGW: float
        Gravitational-wave frequency.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    M_msun: float
        Total mass of the binary in solar masses.
    PNorder: array (default: [0,1,1.5,2])
        PN orders considered.
    
    Returns
    -------
    r: float
        Binary separation.
    
    Examples
    --------
    ``r = precession.gwfrequency_to_pnseparation(theta1,theta2,deltaphi,fGW,q,chi1,chi2,M_msun,PNorder=[0,1,1.5,2])``
    """

    theta1 = np.atleast_1d(theta1).astype(float)
    theta2 = np.atleast_1d(theta2).astype(float)
    deltaphi = np.atleast_1d(deltaphi).astype(float)
    fGW = np.atleast_1d(fGW).astype(float)
    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)
    M_msun = np.atleast_1d(M_msun).astype(float)

    # Prefactor is pi*Msun*G/c^3/s. It's pi and not 2pi because f is the GW frequency while Kidder's omega is the orbital angular velocity
    tildeomega = M_msun * fGW * 1.5473886176432832e-05 

    r =  tildeomega**(-2/3) * (
        (0 in PNorder) * 1 
        + (1 in PNorder) * tildeomega**(2/3) * (-1/3 * ((1 + q))**(-2) * (3 + q * (5 + 3 * q)) ) 
        + (1.5 in PNorder) * tildeomega *(-1/3 * ((1 + q))**(-2) * (np.cos(theta1) * (2 + 3 * q) * chi1 + np.cos(theta2)* q * (3 + 2 * q) * chi2))
        + (2 in PNorder) * tildeomega**(4/3) * (1/36 * ((1 + q))**(-4) * (9 * (-1 + 3 * (np.cos(theta1))**(2)) * (chi1)**(2) + q \
        * (171 + (18 * (-1 + 3 * (np.cos(theta1))**(2)) * (chi1)**(2) + (q * (346 + (171 \
        * q + 9 * (-1 + 3 * (np.cos(theta1))**(2)) * (chi1)**(2))) + (18 * ((1 + q))**(2) \
        * (2 * np.cos(theta1) * np.cos(theta2) + -1 * np.cos(deltaphi) * np.sin(theta1) * np.sin(theta2)) * chi1 * chi2 + 9 * \
        (-1 + 3 * (np.cos(theta2))**(2)) * q * ((1 + q))**(2) * (chi2)**(2))))))) 
        )

    return r


def pnseparation_to_gwfrequency(theta1, theta2, deltaphi, r, q, chi1, chi2, M_msun, PNorder=[0,1,1.5,2]):
    """
    Convert PN orbital separation in natural units (c=G=M=1) to GW frequency in Hz. We use the 2PN expression reported in Eq. 4.5 of Kidder 1995, arxiv:gr-qc/9506022.
    
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
    M_msun: float
        Total mass of the binary in solar masses.
    PNorder: array (default: [0,1,1.5,2])
        PN orders considered.
    
    Returns
    -------
    fGW: float
        Gravitational-wave frequency.
    
    Examples
    --------
    ``fGW = precession.pnseparation_to_gwfrequency(theta1,theta2,deltaphi,r,q,chi1,chi2,M_msun,PNorder=[0,1,1.5,2])``
    """


    theta1 = np.atleast_1d(theta1).astype(float)
    theta2 = np.atleast_1d(theta2).astype(float)
    deltaphi = np.atleast_1d(deltaphi).astype(float)
    r = np.atleast_1d(r).astype(float)
    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)
    M_msun = np.atleast_1d(M_msun).astype(float)
   
    tildeomega = r**(-3/2) * (
        (0 in PNorder) * 1 
        + (1 in PNorder) * r**(-1) *(-1/2 * ((1 + q))**(-2) * (3 + q * (5 + 3 * q)))
        + (1.5 in PNorder) * r**(-3/2) * (-1/2 * ((1 + q))**(-2) * (np.cos(theta1) * (2 + 3 * q) * chi1 + np.cos(theta2) * q * (3 + 2 * q) * chi2))
        + (2 in PNorder) * r**(-2) * (1/8 * ((1 + q))**(-4) * (15 + ((-3 + 9 * (np.cos(theta1))**(2)) * (chi1)**(2) + \
        q * (107 + (6 * (-1 + 3 * (np.cos(theta1))**(2)) * (chi1)**(2) + (q * (187 + (q * \
        (107 + 15 * q) + (-3 + 9 * (np.cos(theta1))**(2)) * (chi1)**(2))) + (6 * ((1 + \
        q))**(2) * (2 * np.cos(theta1) * np.cos(theta2) + -1 * np.cos(deltaphi) * np.sin(theta1) * np.sin(theta2)) * chi1 * \
        chi2 + 3 * (-1 + 3 * (np.cos(theta2))**(2)) * q * ((1 + q))**(2) * (chi2)**(2)))))))))


    # Prefactor is pi*Msun*G/c^3/s. It's pi and not 2pi because f is the GW frequency while Kidder's omega is the orbital angular velocity
    fGW = tildeomega / (1.5473886176432832e-05  * M_msun)

    return fGW




################ Remnant properties ################

def remnantmass(theta1, theta2, q, chi1, chi2):
    """
    Estimate the final mass of the post-merger renmant. We implement the fitting
    formula to numerical relativity simulations by Barausse Morozova Rezzolla
    2012. This formula has to be applied *close to merger*, where numerical
    relativity simulations are available. You should do a PN evolution to
    transfer binaries to r~10M.
    
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
    
    Examples
    --------
    ``mfin = precession.remnantmass(theta1,theta2,q,chi1,chi2)``
    """

    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)
    eta = eval_eta(q).astype(float)

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
    
    Examples
    --------
    ``chifin = precession.remnantspin(theta1,theta2,deltaphi,q,chi1,chi2)``
    """


    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)
    eta = eval_eta(q).astype(float)

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


def remnantspindirection(theta1, theta2, deltaphi, r, q, chi1, chi2):
    """
    Angle between the spin of the remnant and the binary angular momentum, assuming that the spins stays in the direction of the total angular momentum 'at plunge'.
    
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
    thetaL: float
        Angle betwen orbital angular momentum and total angular momentum.
    
    Examples
    --------
    ``thetaL = precession.remnantspindirection(theta1,theta2,deltaphi,r,q,chi1,chi2)``
    """

    Lvec,S1vec,S2vec = angles_to_Lframe(theta1, theta2, deltaphi, r, q, chi1, chi2)
    Jvec = Lvec + S1vec + S2vec
    hatL = normalize_nested(Lvec)
    hatJ = normalize_nested(Jvec)
    thetaL = np.arccos(dot_nested(hatL,hatJ))

    return thetaL

    

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
    vk_array: array, optional
        Kick of the black-hole remnant (in a frame aligned with L).
    
    Examples
    --------
    ``vk = precession.remnantkick(theta1,theta2,deltaphi,q,chi1,chi2,full_output=False)``
    ``vk,vk_array = precession.remnantkick(theta1,theta2,deltaphi,q,chi1,chi2,full_output=True)``
    """

    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)
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
        return np.concatenate([vk[:, None], kick], axis=-1)
    else:
        return vk




################ Main ################

if __name__ == '__main__':
    pass


    print ( remnantkick([1.,1.], [1.,1.], [1.,1.], [1.,1.], [1.,1.], [1.,1.], maxphase = True, full_output=True) )
