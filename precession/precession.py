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


def Jlimits(r,q,chi1,chi2):
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
    Jmin = np.maximum(0, L-S1-S2, np.abs(S1-S2)-L)
    Jmax = L+S1+S1

    return np.array([Jmin,Jmax])


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


def Slimits_LJ(r,J,q):
    """
    Limits on the total spin magnitude due to the vector relation S=J-L

    Parameters
    ----------
    r: float
        Binary separation.
    J: float
        Magnitude of the total angular momentum.
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


def _limits_check(testvalue,interval):
    """Check if a value is within a given interval"""
    # Is there a way to exclude functions from the documentation?
    return np.logical_and(testvalue>interval[0],testvalue<interval[1])


def limits_check(function=None, S=None,r=None,J=None,q=None,chi1=None,chi2=None):
    """
    Check if a given variable satisfies the relevant geometrical constraints. The behaviour is set by `function`. For instance, to check if some values of J are compatible with the provides values of r, q, chi1, and chi2 use function=`Jlimits`. Not all the parameters are necessary. For instance, to check the limits in J one does not need to provide values of S.

    Parameters
    ----------
    S: float
        Magnitude of the total spin.
    r: float
        Binary separation.
    J: float
        Magnitude of the total angular momentum.
    chi1: float or numpy array
        Dimensionless spin of the primary black hole: 0 <= chi1 <= 1.
    chi2: float or numpy array
        Dimensionless spin of the secondary black hole: 0 <= chi1 <= 1.

    Returns
    -------
    Boolean flag.
    """

    if function=='Jlimits':
        return _limits_check(J,Jlimits(r,q,chi1,chi2))

    elif function=='Slimits_S1S2':
        return _limits_check(S,Slimits_S1S2(q,chi1,chi2))

    elif function=='Slimits_JL':
        return _limits_check(S,Slimits_LJ(r,J,q))

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

    q=[0.3,0.6]
    chi1=[0.4,0.6]
    chi2=[0.4,0.6]

    #print(spinmags(q,chi1,chi2))

    #sys.exit()
    q=[0.5,1,0.3]
    chi1=[0.5,0.5,0.67]
    chi2=[0.5,0.5,0.8]
    r=[10,10,10]
    #print(Slimits_S1S2(q,chi1,chi2))
    print(_limits_check([0.24,4,6],Slimits_S1S2(q,chi1,chi2)))

    #print(xilimits(q[0],chi1[0],chi2[0]))

    #print(xilimits(q,chi1,chi2))

    #v= limits_check(function="Jlimits",r=r,J=[7,4,5],q=q,chi1=chi1,chi2=chi2)
    #print(v)

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
