"""
precession
"""

import numpy as np
import scipy as sp
# Using precession_v1 functions for now
import precession as pre


__all__ = [] # Why is this necessary?


@np.vectorize
def mass1(q):
    """Mass of the heavier black hole in units of the total mass.

    Parameters
    ----------
    q: float
        Mass ratio: 0 <= q <= 1.

    Returns
    -------
    m1: float
        Mass of the primary black hole.

    """

    return 1/(1+q)


@np.vectorize
def mass2(q):
    """Mass of the lighter black hole in units of the total mass.

    Parameters
    ----------
    q: float or numpy array
        Mass ratio: 0 <= q <= 1.

    Returns
    -------
    m2: float or numpy array
        Mass of the secondary black hole.

    """

    return q/(1+q)


@np.vectorize
def spin1(q,chi1):
    """Spin of the heavier black hole in units of the total mass.

    Parameters
    ----------
    q: float or numpy array
        Mass ratio: 0 <= q <= 1.
    chi1: float or numpy array
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.

    Returns
    -------
    S2: float or numpy array
        Spin of the primary black hole.
    """

    return chi1*(mass1(q))**2


@np.vectorize
def spin2(q,chi2):
    """Spin of the heavier black hole in units of the total mass.

    Parameters
    ----------
    q: float or numpy array
        Mass ratio: 0 <= q <= 1.
    chi2: float or numpy array
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.

    Returns
    -------
    S1: float or numpy array
        Spin of the primary black hole.
    """

    return chi2*(mass2(q))**2


def spinmags(q,chi1,chi2):
    """Spins of the black holes in units of the total mass.

    Parameters
    ----------
    q: float or numpy array
        Mass ratio: 0 <= q <= 1.
    chi1: float or numpy array
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.
    chi2: float or numpy array
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.

    Returns
    -------
    S2: float or numpy array
        Spin of the primary black hole.
    """

    return np.array([spin1(q,chi1),spin2(q,chi2)])





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

    q=[0.3,0.6]
    chi1=[0.4,0.6]
    chi2=[0.4,0.6]
    S1,S2=spinmags(q,chi1,chi2)
    print(S1)

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
