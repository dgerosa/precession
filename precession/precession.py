"""
precession
"""

import numpy as np
import scipy as sp
# Using precession_v1 functions for now
import precession as pre


# How should we represent a single binary?

# The fixed parameters without initialization are
# M, q, chi1, chi2 (or m1, m2, S1, S2)

# To initialize a binary we also require
# r (or L) and xi, J, S (or theta1, theta2 and DeltaPhi)

# Different parameters are required for different evolutions:
# finite/infinite precession-averaged or orbit-averaged
# For the precession-averaged we can include finite and infinite in one
# and allow for r = np.inf with a flag to check for this

# Class for binary population as a parent/child of binary single class?


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


class UpUp:
    pass


class DownDown:
    pass


class DownUp:
    pass


class UpDown:
    pass


def alignedBinaries():
    return UpUp(), DownDown(), DownUp(), UpDown()


class BinaryPopulation:
    pass


class FieldPopulation:
    pass


class ClusterPopulation:
    pass


# Example
#--------
q = .7
chi1 = .8
chi2 = .9
b = Binary(q, chi1, chi2)

ri = 1000.
t1i = np.pi/2.
t2i = np.pi/3.
dpi = np.pi/4.
xii, Ji, Si = b.config_angles(ri, t1i, t2i, dpi)

rf = 10.
nr = 1000
r = np.linspace(ri, rf, nr)
xi, J, S = b.orb_evolve(r)

#--------


__all__ = []
