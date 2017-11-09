'''
# DYNAMICS OF SPINNING BLACK-HOLE BINARIES WITH PYTHON

`precession` is an open-source Python module to study the dynamics of precessing
black-hole binaries in the post-Newtonian regime.  The code provides a
comprehensive toolbox to (i) study the evolution of the black-hole spins  along
their precession cycles, (ii) perform gravitational-wave driven binary inspirals
using both orbit-averaged and precession-averaged integrations, and (iii)
predict the properties of the merger remnant through fitting formulae obtained
from numerical relativity simulations. `precession` is a ready-to-use tool to
add  the black-hole spin dynamics to larger-scale numerical studies such as
gravitational-wave parameter estimation codes, population synthesis models to
predict gravitational-wave event rates, galaxy merger trees and  cosmological
simulations of structure formation. `precession` provides fast and reliable
integration methods to propagate statistical samples of black-hole binaries
from/to large separations where they form to/from small separations where they
become detectable, thus linking gravitational-wave observations of spinning
black-hole binaries to their astrophysical formation history. The code is also a
useful tool to compute initial parameters for numerical relativity simulations
targeting specific precessing systems.

This code is released to the community under the [Creative Commons Attribution
International license](http://creativecommons.org/licenses/by/4.0).
Essentially, you may use `precession` as you like but must make reference to
our work. When using `precession` in any published work, please cite the paper
describing its implementation:

- *PRECESSION: Dynamics of spinning black-hole binaries with python.*
D. Gerosa, M. Kesden. PRD 93 (2016)
[124066](http://journals.aps.org/prd/abstract/10.1103/PhysRevD.93.124066).
[arXiv:1605.01067](https://arxiv.org/abs/1605.01067)

`precession` is an open-source code distributed under git version-control system on

- [github.com/dgerosa/precession](https://github.com/dgerosa/precession)

API documentation can be generated automatically in html format from the code
docstrings using `pdoc`, and is uplodad to a dedicated branch of the git
repository

- [dgerosa.github.io/precession](https://dgerosa.github.io/precession)

Further information and scientific results are available at:

- [www.tapir.caltech.edu/~dgerosa/precession](http://www.tapir.caltech.edu/~dgerosa/precession)
- [www.davidegerosa.com/precession](http://www.davidegerosa.com/precession)


### INSTALLATION

`precession` works in python 2.x and has been tested on 2.7.10. It can be
installed through [pip](https://pypi.python.org/pypi/precession):

    pip install precession

Prerequisites are `numpy`, `scipy` and `parmap`, which can be all installed
through pip. Information on all code functions are available through Pyhton's
built-in help system

    import precession
    help(precession.function)

Several tests and tutorial are available in the submodule `precession.test`. A
detailed description of the functionalies of the code is provided in the
scientific paper [arXiv:1605.01067](https://arxiv.org/abs/1605.01067), where
examples are also presented.


### RESULTS

`precession` has been used in the following published papers:

- Gerosa and Sesana. MNRAS 446 (2015) 38-55. [arXiv:1405.2072](https://arxiv.org/abs/1405.2072)
- Kesden et al. PRL 114 (2015) 081103. [arXiv:1411.0674](https://arxiv.org/abs/1411.0674)
- Gerosa et al. MNRAS 451 (2015) 3941-3954. [arXiv:1503.06807](https://arxiv.org/abs/1503.06807)
- Gerosa et al. PRD 92 (2015) 064016. [arXiv:1506.03492](https://arxiv.org/abs/1506.03492)
- Gerosa et al. PRL 115 (2015) 141102. [arXiv:1506.09116](https://arxiv.org/abs/1506.09116)
- Trifiro' et al. PRD 93 (2016) 044071. [arXiv:1507.05587](https://arxiv.org/abs/1507.05587)
- Gerosa and Kesden. PRD 93 (2016) 124066. [arXiv:1605.01067](https://arxiv.org/abs/1605.01067)
- Gerosa and Moore. PRL 117 (2016) 011101. [arXiv:1606.04226](https://arxiv.org/abs/1606.04226)
- Rodriguez et al. APJL 832 (2016) L2 [arXiv:1609.05916](https://arxiv.org/abs/1609.05916)
- Gerosa et al. CQG 34 (2017) 6, 064004 [arXiv:1612.05263](https://arxiv.org/abs/1612.05263)
- Gerosa and Berti.  PRD 95 (2017) 124046. [arXiv:1703.06223](https://arxiv.org/abs/1703.06223)
- Zhao et al. PRD 96 (2017) 024007. [arXiv:1705.02369](https://arxiv.org/abs/1705.02369)
- Wysocki et al. [arXiv:1709.01943](https://arxiv.org/abs/1709.01943)


### THINGS TO KEEP IN MIND

1. **Units**. All quantities in the code are specified in units where c=G=1.
Moreover, the binary total mass M=m1+m2 is  set to 1, and everything else is
computed accordingly. In practice, this means that e.g. the binary separation r
is actually r/M. If you are trying to use `precession` from a code with
different units (lal?), you should just pass r/M instead of your cgs or SI r.
Equivalently, the angular momentum L, the spins Si and the total angular
momentum J are actually L/M^2, Si/M^2 and J/M^2.

2. **Don't go too close to the limits (i)**. This is a code to study double-spin
precession and, in general, it won't behave nicely if you are too close to
aligned, hence non-precessing, configurations. These configurations are given by
the limits Jmin and Jmax of the total angular momentum J, and/or by
sin(theta_i)=0. I strongly recommend to always set a tolerance, for instance,

        Jmin,Jmax=precession.J_lim(q,S1,S2,r) for J in
        numpy.linspace(Jmin+1e-6,Jmax-1e-6,100):
            do things...

3. **Don't go too close to the limits (ii)**. For the same reason, some
quantities cannot be computed efficiently for binaries which are very close to a
spin-orbit resonance. For instance, the computation of the angle alpha may be
inaccurate for binaries very close to xi_min and xi_max as returned by
xi_allowed.

4. **Checkpointing**. Checkpointing is implemented in some functions for
computational efficiency. Temporary data are stored in a local directory and
will be read in if available. To delete all previous data run

        precession.empty_temp()

    By default, data are stored in a local directory called
    `precession_checkpoints`, which is created when needed. You can change it
    setting

        precession.storedir=[path]

5. **Parallelization**. Some parts of the code are parallelized using the
`parmap` module. Instructions on code parallelization are set by the global
variable CPUs: (i) `CPUs=1` enforces a serial computation; (ii) `CPUs=integer`
specifies the number of parallel processes; (iii) `CPUs=0` (default) autodetects
the number of core in the current machine.

    You can set this variable using

        precession.CPUs = [integer]

6. **The equal-mass limit**. The equal-mass q=1 limit requires some extra care.
If q=1 the total-spin magnitude S cannot be used to parametrize the precession
cycle and the angle varphi needs to be tracked explicitly. The q=1 case is
implemented in the code: inputs and outputs of some of the functions are
actually specified as cos(varphi), even though for simplicity we still call them
**S**. In case of precession-averaged integrations to/from infinity, kappa_inf
becomes degenerate with xi and an initial value of S is required.
Please, refer to the documentation below for details. The generic unequal-mass
part of the code works fine up to q<0.995. To run higher values of q we
recommend setting q=1 and learn the relevant parts of the code.

7. **Stalling**. When performing precession-averaged evolutions, some binaries
may occasionally stall and take longer to run. This is due to the first step
attempted by the ODE integrator. This is a minor issue and  only happens to
roughly one binary in a million or so. If you really want to fix this, you
should play with the `h0` optional paramenter in scipy's odeint function.

8. **Typos** There are two typos in [arXiv:1605.01067](https://arxiv.org/abs/1605.01067).
(i) The (1+4eta) factor in Eq.(37) should really be (1-4eta), such that p0
correspond to the radiated energy for equal-mass non-spinning BHs. The origin of
this typo is a trivial mistake when rewriting the analogous equation from
Gerosa and Sesana 2015. (ii) A minus sign is missing in the Delta vector of
Eq.(36). This error, which is present in various papers since at least 2010,
originates from the conversion between a notation where m1 (m2) is the heavier
(lighter) BH like the one used here, to the opposite one used by other groups.
This has been corrected since v1.0.2.

### RELEASES

[![DOI](https://zenodo.org/badge/21015/dgerosa/precession.svg)](https://zenodo.org/badge/latestdoi/21015/dgerosa/precession)   v1.0.0 (stable)

### CREDITS
The code is developed and maintained by [Davide Gerosa](www.davidegerosa.com).
Please, report bugs to

    dgerosa@caltech.edu

I am happy to help you out!

**Thanks**: M. Kesden, U. Sperhake, E. Berti, R. O'Shaughnessy, A. Sesana, D.
Trifiro', A. Klein, J. Vosmera and X. Zhao.

'''

__author__ = "Davide Gerosa"
__email__ = "dgerosa@caltech.edu"
__copyright__ = "Copyright (C) 2016 Davide Gerosa"
__license__ = "CC BY 4.0"
__version__ = "1.0.2"


__doc__="**Author** "+__author__+"\n\n"+\
        "**email** "+__email__+"\n\n"+\
        "**Copyright** "+__copyright__+"\n\n"+\
        "**Licence** "+__license__+"\n\n"+\
        "**Version** "+__version__+"\n\n"+\
        __doc__

def why():
    print "\nIt's all about python and gravity. Go to"
    print "http://imgs.xkcd.com/comics/python.png\n"
    sys.exit()


################################
########### STARTUP ############
################################

import os, sys, imp
import numpy as np
import scipy as sp
from scipy import optimize
from scipy import integrate
import random
import math
import parmap # if you don't have parmap, try "pip install parmap".
import __main__

##Unbuffereb stdout. Less efficient but interesting for debugging.
if False:
    class Unbuffered(object):
        def __init__(self, stream):
            self.stream = stream
        def write(self, data):
            self.stream.write(data)
            self.stream.flush()
        def __getattr__(self, attr):
            return getattr(self.stream, attr)
    sys.stdout = Unbuffered(sys.stdout)

# Checkpoint setup. Checkpoints will be stored in the storedir directory.
# storedir=imp.find_module('precession')[1].split("precession.py")[0]+"precession_checkpoints" ## Global directory
storedir="precession_checkpoints" ## Local directory
'''
Directory path to store checkpoints. Deafult is `./precession_checkpoints`.
'''

def make_temp():

    '''
    Make an empty directory to store checkpoints. Calling this function is
    typically not necessary, because the checkpoint directory is created when
    needed.

    **Call:**

        precession.make_temp()
    '''

    global storedir
    print "[make_temp] Creating temp directory: "+storedir
    os.system("mkdir -p "+storedir)


def empty_temp():

    '''
    Remove all checkpoints.

    **Call:**

        precession.empty_temp()
    '''

    global storedir
    print "[empty_temp] Removing temp files from directory: "+storedir
    os.system("rm -rf "+storedir)
#empty_temp(storedir)

M=1.
'''
The total mass is just a free scale, and we set it to 1. Please, don't change
this, because I never checked that the various M factors are all right.
'''

flags_q1=list(np.zeros((20), dtype=bool))
'''
Global flags to reduce warnings in the equal-mass limit q=1.
'''



#################################
############# LIMITS ############
#################################

def get_fixed(q,chi1,chi2):

    '''
    Compute individual masses and spins, from mass ratio (q<1) and
    dimensionless spins (0<chi<1).

    **Call:**

        M,m1,m2,S1,S2=precession.get_fixed(q,chi1,chi2)

    **Parameters:**

    - `q`: binary mass ratio. Must be q<=1.
    - `chi1`: dimensionless spin magnitude of the primary BH. Must be 0<chi1<1.
    - `chi2`: dimensionless spin magnitude of the secondary BH. Must be 0<chi2<1.

    **Returns:**

    - `M`: total mass of the binary (set to 1).
    - `m1`: mass of the primary BH.
    - `m2`: mass of the secondary BH.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    '''

    global M
    m1=M/(1.+q) # Primary mass
    m2=q*M/(1.+q) # Secondary mass
    S1=chi1*m1**2 # Primary spin magnitude
    S2=chi2*m2**2 # Secondary spin magnitude
    return M,m1,m2,S1,S2


def get_L(r,q):

    '''
    Return Newtonian expression for the orbital angular momentum. This function is not called explicitely within the `precession` module to increase efficiency.

    **Call:**

        L=precession.get_L(r,q)

    **Parameters:**

    - `q`: binary mass ratio. Must be q<=1.
    - `r`: binary separation.

    **Returns:**

    - `L`: Magnitude of the orbital angular momentum


    '''

    L=(q/(1.+q)**2)*(r*M**3)**.5
    return L


def J_lim(q,S1,S2,r, verbose=False):

    '''
    Compute the limits on the magnitude of the total angular momentum J, defined
    as J=|L+S1+S2|.

    **Call:**

        Jmin,Jmax=precession.J_lim(q,S1,S2,r, verbose=False)

    **Parameters:**

    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `r`: binary separation.
    - `verbose`: if `True` print additional information.

    **Returns:**

    - `Jmin`: minimum value of J from geometrical constraints.
    - `Jmax`: maximum value of J from geometrical constraints.
    '''

    L=(q/(1.+q)**2)*(r*M**3)**.5
    Jmin=max(1e-20,L-S1-S2,S1-L-S2,S2-S1-L)
    Jmax=L+S1+S2

    if verbose:
        print "[J_lim] L=",L, " S1=",S1," S2=",S2
        print "[J_lim] Jmin=",Jmin, " Jmax=",Jmax
        if Jmin==1e-20:
            print "[J_lim] Jmin=0, Jmax=L+S1+S2"
        elif Jmin==L-S1-S2:
            print "[J_lim] Jmin=L-S1-S2, Jmax=L+S1+S2"
        if Jmin==S1-L-S2:
            print "[J_lim] Jmin=S1-L-S2, Jmax=L+S1+S2"
        if Jmin==S2-L-S1:
            print "[J_lim] Jmin=S2-L-S1, Jmax=L+S1+S2"

    return Jmin,Jmax

def St_limits(J,q,S1,S2,r,verbose=False):

    '''
    Compute the *total* limits on the magnitude of the total spin S. S has to
    satisfy both S=S1+S2 and S=J-L.

    **Call:**

        St_min,St_max=precession.St_limits(J,q,S1,S2,r,verbose=False)

    **Parameters:**

    - `J`: magnitude of the total angular momentum.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `r`: binary separation.
    - `verbose`: if `True` print additional information.

    **Returns:**

    - `St_min`: minimum value of S from geometrical constraints. This is S_min in our papers.
    - `St_max`: maximum value of S from geometrical constraints. This is S_max in our papers.
    '''

    ttol=0.0
    L=(q/(1.+q)**2)*(r*M**3)**.5
    St_min=float(max(np.abs(J-L),np.abs(S1-S2)))
    St_max=min(J+L,S1+S2)
    if verbose:
        print "[S_lim] L=",L, " J",J, " S1=",S1," S2=",S2
        print "[S_lim] St_min=",St_min, " St_max=",St_max
        if St_min==np.abs(J-L) and St_max==J+L:
            print "[S_lim] St_min=|J-L|, St_max=J+L"
        elif St_min==np.abs(J-L) and St_max==S1+S2:
            print "[S_lim] St_min=|J-L|, St_max=S1+S2"
        elif St_min==np.abs(S1-S2) and St_max==J+L:
            print "[S_lim] St_min=|S1-S2|, St_max=J+L"
        elif St_min==np.abs(S1-S2) and St_max==S1+S2:
            print "[S_lim] St_min=|S1-S2|, St_max=S1+S2"

    return St_min+ttol,St_max-ttol


def Sso_limits(S1,S2):

    '''
    Compute the *spin-only* limits on the magnitude of the total spin S,
    considering the single constraint S=S1+S2.  This is needed e.g. to provide
    initial condition to precession-averaged integration from infinity in the
    q=1 limit.

    **Call:**

        Sso_min,Sso_max=precession.Sso_limits(S1,S2)

    **Parameters:**

    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.

    **Returns:**

    - `Sso_min`: minimum value of S from the spin constraint only.
    - `Sso_max`: maximum value of S from the spin constraint only.
    '''

    return np.abs(S1-S2), S1+S2


def xi_lim(q,S1,S2):

    '''
    Compute the absolute limits on xi (i.e. regardless of J). Check
    `precession.xi_allowed` for the limits on xi for a given J. This functions
    is simply checks -1<cos(theta_i)<1.

    **Call:**

        xi_min,xi_max=precession.xi_lim(q,S1,S2)

    **Parameters:**

    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.

    **Returns:**

    - `xi_min`: minimum value of xi from geometrical constraints.
    - `xi_max`: maximum value of xi from geometrical constraints.
    '''

    xi_max= ((1.+q)*S1+(1.+q**-1)*S2)*M**-2
    xi_min=-1.*xi_max
    return xi_min,xi_max


def xi_at_Jlim(q,S1,S2,r,more=False):

    '''
    Find the value of xi (and S, optional) when J is either Jmax or Jmin.

    **Call:**

        xi_Jmin,xi_Jmax=precession.xi_at_Jlim(q,S1,S2,r,more=False)

    **Parameters:**

    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `r`: binary separation.
    - `more`: if `True` returns additional quantities.

    **Returns:**

    - `xi_Jmin`: value of xi when J=Jmin.
    - `xi_Jmax`: value of xi when J=Jmax.
    - `S_Jmin`: (optional) value of S when J=Jmin.
    - `S_Jmax`: (optional) value of S when J=Jmax.
    '''

    # Find Jmin and Jmax
    L=(q/(1.+q)**2)*(r*M**3)**.5
    Jmin=max(1e-20,L-S1-S2,S1-L-S2,S2-S1-L)
    Jmax=L+S1+S2

    # Everything is aligned at Jmax
    ct1=1.
    ct2=1.
    xi_Jmax= ((1.+q)*S1*ct1+(1.+q**-1)*S2*ct2)*M**-2
    S_Jmax= S1+S2

    # Split the Jmin cases
    if Jmin==1e-20: # Force vectors in a plane, closed triangle
        ct1= (-S1**2+S2**2-L**2)/(2.*L*S1)
        ct2= (-S2**2+S1**2-L**2)/(2.*L*S2)
        S_Jmin=0
    elif Jmin==L-S1-S2: # Both antialigned
        ct1=-1.
        ct2=-1.
        S_Jmin=S1+S2
    elif Jmin==S1-L-S2: # One antialigned
        ct1=-1.
        ct2=1.
        S_Jmin=np.abs(S1-S2)
    elif Jmin==S2-L-S1: # One antialigned
        ct1=1.
        ct2=-1.
        S_Jmin=np.abs(S1-S2)
    xi_Jmin= ((1.+q)*S1*ct1+(1.+q**-1)*S2*ct2)*M**-2

    if more:
        return xi_Jmin, xi_Jmax, S_Jmin, S_Jmax
    else:
        return xi_Jmin, xi_Jmax


def kappainf_lim(S1,S2):

    '''
    Absolute limits in kappa_inf (asymptotic value of kappa). At large
    separations, kappa is the projection of the total spin along L.

    **Call:**

        kappainf_min,kappainf_max=precession.kappainf_lim(S1,S2)

    **Parameters:**

    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.

    **Returns:**

    - `kappainf_min`: minimum value of kappa at infinitely large separations.
    - `kappainf_max`: maximum value of kappa at infinitely large separations.
    '''

    return -(S1+S2), S1+S2


def xiinf_allowed(kappa_inf,q,S1,S2):

    '''
    Limits on xi for a given value of kappa_inf, obtained forcing
    -1<cos(theta_i)<1.

    **Call:**

        xi_low,xi_up=precession.xiinf_allowed(kappa_inf,q,S1,S2)

    **Parameters:**

    - `kappa_inf`: asymptotic value of kappa at large separations.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.

    **Returns:**

    - `xi_low`: minimum allowed value of xi, given the input parameters.
    - `xi_up`: maximum allowed value of xi, given the input parameters.
    '''

    xi_low = max( kappa_inf*(1+q) - (q**-1-q)*S2 , kappa_inf*(1+q**-1) - (q**-1-q)*S1 )
    xi_up = min( kappa_inf*(1+q) + (q**-1-q)*S2 , kappa_inf*(1+q**-1) + (q**-1-q)*S1 )
    return xi_low,xi_up


def kappainf_allowed(xi,q,S1,S2):

    '''
    Limits on kappa_inf for a given value of xi, obtained forcing
    -1<cos(theta_i)<1.

    **Call:**

        kappainf_low,kappainf_up=precession.kappainf_allowed(xi,q,S1,S2)

    **Parameters:**

    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.

    **Returns:**

    - `kappainf_low`: minimum allowed value of kappa_inf, given the input parameters.
    - `kappainf_up`: maximum allowed value of kappa_inf, given the input parameters.
    '''

    kappainf_low = max( (xi - (q**-1-q)*S2)/(1+q) , (xi - (q**-1-q)*S1)/(1+q**-1) )
    kappainf_up = min( (xi + (q**-1-q)*S2)/(1+q) , (xi + (q**-1-q)*S1)/(1+q**-1) )
    return kappainf_low,kappainf_up




#################################
##### EFFECTIVE POTENTIALS ######
#################################


def xi_contour(varphi,S,J,q,S1,S2,r):

    '''
    Compute the projection of the effective spin xi as a function of the
    spin-rotation degree of freedom varphi and the total spin magnitude S.

    **Call:**

        xi=precession.xi_contour(varphi,S,J,q,S1,S2,r)

    **Parameters:**

    - `varphi`: angle describing the rotation of S1 and S2 about S, in a frame aligned with J.
    - `S`: magnitude of the total spin.
    - `J`: magnitude of the total angular momentum.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `r`: binary separation.

    **Returns:**

    - `xi`: projection of the effective spin along the orbital angular momentum.
    '''

    L=(q/(1.+q)**2)*(r*M**3)**.5
    t1=(1.+q)/(4.*q*M**2*S**2*L)
    t2=J**2-L**2-S**2
    t3=S**2*(1.+q)-(S1**2-S2**2)*(1.-q)
    t4=(1.-q)*((L+S)**2-J**2)**.5
    t5=(J**2-(L-S)**2)**.5
    t6=((S1+S2)**2-S**2)**.5
    t7=(S**2-(S1-S2)**2)**.5
    return t1*((t2*t3)-(t4*t5*t6*t7*np.cos(varphi)))


def xi_plus(S,J,q,S1,S2,r):

    '''
    Upper effective potential, corresponding to cos(varphi)=-1.

    **Call:**

        xi=precession.xi_plus(S,J,q,S1,S2,r)

    **Parameters:**

    - `S`: magnitude of the total spin.
    - `J`: magnitude of the total angular momentum.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `r`: binary separation.

    **Returns:**

    - `xi`: projection of the effective spin along the orbital angular momentum.
    '''

    # Do it explicitely. Somehow faster than calling xi_contour(np.pi,...)
    L=(q/(1.+q)**2)*(r*M**3)**.5
    t1=(1.+q)/(4.*q*M**2*S**2*L)
    t2=J**2-L**2-S**2
    t3=S**2*(1.+q)-(S1**2-S2**2)*(1.-q)

    if S in St_limits(J,q,S1,S2,r): #if you're on the limits, the second bit must be zero
        t4=t5=t6=t7=0.
    else:
        t4=(1.-q)*((L+S)**2-J**2)**.5
        t5=(J**2-(L-S)**2)**.5
        t6=((S1+S2)**2-S**2)**.5
        t7=(S**2-(S1-S2)**2)**.5

    return t1*((t2*t3)-(t4*t5*t6*t7*(-1.)))


def xi_minus(S,J,q,S1,S2,r):

    '''
    Lower effective potential, corresponding to cos(varphi)=+1.

    **Call:**

        xi=precession.xi_minus(S,J,q,S1,S2,r)

    **Parameters:**

    - `S`: magnitude of the total spin.
    - `J`: magnitude of the total angular momentum.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `r`: binary separation.

    **Returns:**

    - `xi`: projection of the effective spin along the orbital angular momentum.
    '''

    # Do it explicitely. Somehow faster than calling xi_contour(np.pi,...)
    L=(q/(1.+q)**2)*(r*M**3)**.5
    t1=(1.+q)/(4.*q*M**2*S**2*L)
    t2=J**2-L**2-S**2
    t3=S**2*(1.+q)-(S1**2-S2**2)*(1.-q)

    if S in St_limits(J,q,S1,S2,r): #if you're on the limits, the second bit must be zero
        t4=t5=t6=t7=0.
    else:
        t4=(1.-q)*((L+S)**2-J**2)**.5
        t5=(J**2-(L-S)**2)**.5
        t6=((S1+S2)**2-S**2)**.5
        t7=(S**2-(S1-S2)**2)**.5

    return t1*((t2*t3)-(t4*t5*t6*t7*(1.)))


def dxidS_plus(S,J,q,S1,S2,r):

    '''
    Derivative of the effective potential xi_plus with respect to S.

    **Call:**

        dxidS=precession.dxidS_plus(S,J,q,S1,S2,r)

    **Parameters:**

    - `S`: magnitude of the total spin.
    - `J`: magnitude of the total angular momentum.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `r`: binary separation.

    **Returns:**

    - `dxidS`: derivative of effective potential with respect to S.
    '''

    L=(q/(1.+q)**2)*(r*M**3)**.5
    A1=np.sqrt(J**2-(L-S)**2)
    A2=np.sqrt((L+S)**2-J**2)
    A3=np.sqrt(S**2-(S1-S2)**2)
    A4=np.sqrt((S1+S2)**2-S**2)
    Fp = (J**2-L**2-S**2)*(S**2*(1+q)**2-(S1**2-S2**2)*(1-q**2))+(1-q**2)*A1*A2*A3*A4
    G=4*q*M**2*S**2*L
    dFpdS = -2*S*(S**2*(1+q)**2-(S1**2-S2**2)*(1-q**2))+2*S*(1+q)**2*(J**2-L**2-S**2)+(1-q**2)*(((L-S)*A2*A3*A4)/A1+((L+S)*A3*A4*A1)/A2+(S*A4*A1*A2)/A3-(S*A1*A2*A3)/A4)
    dGdS=8*q*M**2*S*L
    dxipdS=(dFpdS*G-dGdS*Fp)/(G**2)
    return dxipdS


def dxidS_minus(S,J,q,S1,S2,r):

    '''
    Derivative of the effective potential xi_minus with respect to S.

    **Call:**

        dxidS=precession.dxidS_minus(S,J,q,S1,S2,r)

    **Parameters:**

    - `S`: magnitude of the total spin.
    - `J`: magnitude of the total angular momentum.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `r`: binary separation.

    **Returns:**

    - `dxidS`: derivative of effective potential with respect to S.
    '''

    L=(q/(1.+q)**2)*(r*M**3)**.5
    A1=np.sqrt(J**2-(L-S)**2)
    A2=np.sqrt((L+S)**2-J**2)
    A3=np.sqrt(S**2-(S1-S2)**2)
    A4=np.sqrt((S1+S2)**2-S**2)
    Fm = (J**2-L**2-S**2)*(S**2*(1+q)**2-(S1**2-S2**2)*(1-q**2))-(1-q**2)*A1*A2*A3*A4
    G=4*q*M**2*S**2*L
    dFmdS = -2*S*(S**2*(1+q)**2-(S1**2-S2**2)*(1-q**2))+2*S*(1+q)**2*(J**2-L**2-S**2)-(1-q**2)*(((L-S)*A2*A3*A4)/A1+((L+S)*A3*A4*A1)/A2+(S*A4*A1*A2)/A3-(S*A1*A2*A3)/A4)
    dGdS=8*q*M**2*S*L
    dximdS=(dFmdS*G-dGdS*Fm)/(G**2)
    return dximdS


def get_varphi(xi,S,J,q,S1,S2,r,sign=1):

    '''
    Compute varphi from a given xi. This can be seen as the inverse of
    xi_contour. If phase==1 (default) return varphi in [0,pi], if sign==-1
    return varphi in [-pi,0].

    WARKNING: Don't run for q=1, as varphi is independent of S in this limit.

    **Call:**

        varphi=precession.get_varphi(xi,S,J,q,S1,S2,r,sign=1)

    **Parameters:**

    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `S`: magnitude of the total spin.
    - `J`: magnitude of the total angular momentum.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `r`: binary separation.
    - `sign`: if 1 return angle in [0,pi], if -1 return angle in [-pi,0].

    **Returns:**

    - `varphi`: angle describing the rotation of S1 and S2 about S, in a frame aligned with J.
    '''

    if q==1:
        assert False, "[get_varphi] Error: I'm sorry, can't run for q=1. S is degenerate with varphi."

    L=(q/(1.+q)**2)*(r*M**3)**.5
    t1=(1.+q)/(4.*q*M**2*S**2*L)
    t2=J**2-L**2-S**2
    t3=S**2*(1.+q)-(S1**2-S2**2)*(1.-q)
    t4=(1.-q)*((L+S)**2-J**2)**.5
    t5=(J**2-(L-S)**2)**.5
    t6=((S1+S2)**2-S**2)**.5
    t7=(S**2-(S1-S2)**2)**.5
    cosvarphi= ((t2*t3)-(xi/t1))/(t4*t5*t6*t7)

    return np.arccos(cosvarphi)*sign


def Sb_limits(xi,J,q,S1,S2,r):

    '''
    Compute the *bounded* limits on S, using xi as a constant of motion. The
    routine first guesses where the extrema are expected to be, then brakets the
    solution, and finally runs root finder. In some cases the braketing may
    fail: this typically happens if the two roots are very close (DeltaS<1e-8)
    and cannot be distinguished numerically. In this case, assume Sb_min=Sb_max.

    WARNING: This function is critical. It is well tested, but is tricky
    numerical issues may still be present.

    **Call:**

        Sb_min,Sb_max=precession.Sb_limits(xi,J,q,S1,S2,r)

    **Parameters:**

    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `J`: magnitude of the total angular momentum.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `r`: binary separation.

    **Returns:**

    - `Sb_min`: minimum value of S from geometrical constraints. This is S- in our papers.
    - `Sb_max`: maximum value of S from geometrical constraints. This is S+ in our papers.
    '''

    St_min,St_max=St_limits(J,q,S1,S2,r)
    if St_max<St_min:
        print "[Sb_limits] Problem in the absolute limits at r="+str(r)+". Assume Sb_min=Sb_max=mean(St_min,St_max)"
        return np.mean([St_min,St_max]), np.mean([St_min,St_max])

    if q==1: # if q=1 the effective potential loop shrinks to a lime (Sb_min=Sb_max) and the equations xi_/pm(S)=xi can be solved analytically.
        L=(q/(1.+q)**2)*(r*M**3)**.5
        Sb_both=np.sqrt(J**2-L**2-xi*L*M**2)
        return Sb_both,Sb_both

    # Remember: xi_minus=xi_plus at St_min and St_max
    xi_low=xi_minus(St_min,J,q,S1,S2,r)
    xi_up=xi_minus(St_max,J,q,S1,S2,r)

    #Debug option: print the parameter-space region of the initial guess
    whereareyou=False

    # Both roots on xi_plus. Split the interval first
    if xi > xi_low and xi > xi_up:
        if whereareyou:
            print "[Sb_limits] Both roots on xi_plus"

        resmax= sp.optimize.fminbound(lambda S: -1.*xi_plus(S,J,q,S1,S2,r), St_min, St_max,full_output=1)
        S_up=resmax[0]
        xi_max=-1.*resmax[1]

        if xi_max<xi: #Braket failed!
            print "[Sb_limits] Braket failed on xi_plus at r="+str(r)+". Assume Sb_min=Sb_max"
            #print  xi_plus(S_up,J,q,S1,S2,r), xi
            Sb_min=S_up
            Sb_max=S_up
        else: #Braket succeeded!
            Sb_min= sp.optimize.brentq(lambda S: xi_plus(S,J,q,S1,S2,r)-xi, St_min, S_up)
            Sb_max= sp.optimize.brentq(lambda S: xi_plus(S,J,q,S1,S2,r)-xi, S_up, St_max)

    # Both roots on xi_minus. Split the interval first
    elif xi < xi_low and xi < xi_up:
        if whereareyou:
            print "[Sb_limits] Both roots on xi_minus"

        resmin= sp.optimize.fminbound(lambda S: xi_minus(S,J,q,S1,S2,r), St_min, St_max,full_output=1)
        S_low=resmin[0]
        xi_min=resmin[1]

        if xi_min>xi: #Braket failed!
            print "[Sb_limits] Braket failed on xi_minus at r="+str(r)+". Assume Sb_min=Sb_max"
            Sb_min=S_low
            Sb_max=S_low
        else: #Braket succeeded!
            Sb_min= sp.optimize.brentq(lambda S: xi_minus(S,J,q,S1,S2,r)-xi, St_min, S_low)
            Sb_max= sp.optimize.brentq(lambda S: xi_minus(S,J,q,S1,S2,r)-xi, S_low, St_max)

    # One root on xi_plus and the other one on xi_plus. No additional maximization is neeeded
    elif xi >= xi_low and xi <= xi_up:
        if whereareyou:
            print "[Sb_limits] Sb_min on xi_plus, Sb_max on xi_minus"

        Sb_min= sp.optimize.brentq(lambda S: xi_plus(S,J,q,S1,S2,r)-xi, St_min, St_max)
        Sb_max= sp.optimize.brentq(lambda S: xi_minus(S,J,q,S1,S2,r)-xi, St_min, St_max)

    elif xi <= xi_low and xi >= xi_up:
        if whereareyou:
            print "[Sb_limits] Sb_min on xi_minus, Sb_max on xi_plus"

        Sb_min= sp.optimize.brentq(lambda S: xi_minus(S,J,q,S1,S2,r)-xi, St_min, St_max)
        Sb_max= sp.optimize.brentq(lambda S: xi_plus(S,J,q,S1,S2,r)-xi, St_min, St_max)

    else:
        print "[Sb_limits] Erorr in case selection"
        print "xi=", xi
        print "xi(Stmin)=", xi_low
        print "xi(Stmax)=", xi_up
        print "Stmin=", St_min
        print "Stmax", St_max
        print "J=", J
        print "L=", (q/(1.+q)**2)*(r*M**3)**.5
        print "r=", r
        assert False, "[Sb_limits] Erorr in case selection"

    btol=1e-8 # Never go to close to the actual limits, because of numerical stabilty
    Sb_min+=btol
    Sb_max-=btol

    if whereareyou:
        print "[Sb_limits] Results:", Sb_min,Sb_max

    if Sb_min>Sb_max: # This may happen (numerically) if they're too close to each other. Assume they're the same.
        return np.mean([Sb_min,Sb_max]), np.mean([Sb_min,Sb_max])
    else:
        return Sb_min, Sb_max


def parametric_angles(S,J,xi,q,S1,S2,r):

    '''
    Compute the angles theta1,theta2,deltaphi and theta12, given S, J and xi.
    Roundoff errors are fixed forcing cosines to be in [-1,1]. The thetas are
    polar angles in [0,pi]. Deltaphi is an azimuthal angle, in principle lies in
    [-pi,pi]. Here we assumed DeltaPhi to be in [0,pi] as returned by arcccos:
    one may need to add a sign, depending on the actual application of this
    function (see e.g. `precession.orbit_angles` below). This function can be
    seen as the inverse of `precession.from_the_angles`. In the equal-mass limit
    q=1, S doesn't parametrize the precessional motion; we track the binary
    precession using varphi explicitly.

    **Call:**

        theta1,theta2,deltaphi,theta12=precession.parametric_angles(S,J,xi,q,S1,S2,r)

    **Parameters:**

    - `S`: magnitude of the total spin.
    - `J`: magnitude of the total angular momentum.
    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `r`: binary separation.

    **Returns:**

    - `theta1`: angle between the spin of the primary and the orbital angular momentum.
    - `theta2`: angle between the spin of the secondary and the orbital angular momentum.
    - `deltaphi`: angle between the projection of the two spins on the orbital plane.
    - `theta12`: angle between the two spins.
    '''

    L=(q/(1.+q)**2)*(r*M**3)**.5

    global flags_q1
    if q==1:
        if flags_q1[0]==False:
            print "[parametric_angles] Warning q=1: input here is cos(varphi), not S."
            flags_q1[0]=True # Suppress future warnings
        cosvarphi=S # The input variable is actually cos(varphi)
        S=(J**2-L**2-xi*L*M**2)**.5
        t4=J**2-(L-S)**2
        t5=(L+S)**2-J**2
        t6=(S1+S2)**2-S**2
        t7=S**2-(S1-S2)**2
        B=max(0,t4*t5*t6*t7)
        costheta1= (1/(4*S1*S**2*L))*((J**2-L**2-S**2)*(S**2+S1**2-S2**2)+np.sqrt(B)*cosvarphi)
        costheta2= (1/(4*S2*S**2*L))*((J**2-L**2-S**2)*(S**2+S2**2-S1**2)-np.sqrt(B)*cosvarphi)

    else:
        costheta1= ( ((J**2-L**2-S**2)/L) - (2.*q*M**2*xi)/(1.+q) )/(2.*(1.-q)*S1)
        costheta2= ( ((J**2-L**2-S**2)*(-q/L)) + (2.*q*M**2*xi)/(1.+q) )/(2.*(1.-q)*S2)

    # Force all cosines in [-1,1].
    costheta1=max(-1,min(costheta1,1.))
    theta1=np.arccos(costheta1)
    costheta2=max(-1,min(costheta2,1.))
    theta2=np.arccos(costheta2)
    costheta12=(S**2-S1**2-S2**2)/(2.*S1*S2)
    costheta12=max(-1,min(costheta12,1.))
    theta12=np.arccos(costheta12)
    cosdeltaphi= (costheta12 - costheta1*costheta2)/(np.sin(theta1)*np.sin(theta2))
    cosdeltaphi=max(-1,min(cosdeltaphi,1.))
    deltaphi=np.arccos(cosdeltaphi)

    return theta1,theta2,deltaphi,theta12


def from_the_angles(theta1,theta2,deltaphi,q,S1,S2,r):

    '''
    Convert a set of angles theta1,theta2,deltaphi into values of J,xi,S. This
    function can be seen as the inverse of `precession.parametric_angles`. In
    the equal-mass limit q=1, S doesn't parametrize the precessional motion; we
    track the binary precession using varphi explicitly.

    **Call:**

        xi,J,S=precession.from_the_angles(theta1,theta2,deltaphi,q,S1,S2,r)

    **Parameters:**

    - `theta1`: angle between the spin of the primary and the orbital angular momentum.
    - `theta2`: angle between the spin of the secondary and the orbital angular momentum.
    - `deltaphi`: angle between the projection of the two spins on the orbital plane.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `r`: binary separation.

    **Returns:**

    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `J`: magnitude of the total angular momentum.
    - `S`: magnitude of the total spin.
    '''

    L=(q/(1.+q)**2)*(r*M**3)**.5

    global flags_q1
    if q==1:
        if flags_q1[1]==False:
            print "[from_the_angles] Warning q=1: output here is cos(varphi), not S."
            flags_q1[1]=True # Suppress future warnings
        xi = (2/M**2)*(S1 *np.cos(theta1)+S2*np.cos(theta2))
        Ssq = S1**2 + S2**2 +2*S1*S2*(np.cos(theta1)*np.cos(theta2)+np.sin(theta1)*np.sin(theta2)*np.cos(deltaphi))
        J=np.sqrt(Ssq+L**2+xi*L*M**2)
        t6=((S1+S2)**2-Ssq)
        t7=(Ssq-(S1-S2)**2)
        B=(4*Ssq-xi**2*M**4)*t6*t7
        cosvarphi= (4*S1*Ssq*np.cos(theta1)-xi*M**2*(Ssq+S1**2-S2**2))/(np.sqrt(B))
        return xi,J,cosvarphi # The output variable is actually cos(varphi)

    else:
        xi= ((1.+q)*S1*np.cos(theta1)+(1.+q**-1)*S2*np.cos(theta2))*M**-2
        S= (S1**2+S2**2+2.*S1*S2*(np.sin(theta1)*np.sin(theta2)*np.cos(deltaphi)+np.cos(theta1)*np.cos(theta2)))**.5
        J= (L**2+S**2+2.*L*(S1*np.cos(theta1)+S2*np.cos(theta2)))**.5
        return xi,J,S

def build_angles(Lvec,S1vec,S2vec):

    '''
    Compute the angles theta1, theta2, deltaphi and theta12 from the xyz
    components of L, S1 and S2.

    **Call:**

        theta1,theta2,deltaphi,theta12=precession.build_angles(Lvec,S1vec,S2vec)

    **Parameters:**

    - `Lvec`: components of L in a reference frame (3 values for x,y,z).
    - `S1vec`: components of S1 in a reference frame (3 values for x,y,z).
    - `S2vec`: components of S2 in a reference frame (3 values for x,y,z).


    **Returns:**

    - `theta1`: angle between the spin of the primary and the orbital angular momentum.
    - `theta2`: angle between the spin of the secondary and the orbital angular momentum.
    - `deltaphi`: angle between the projection of the two spins on the orbital plane.
    - `theta12`: angle between the two spins.
    '''

    L=np.linalg.norm(Lvec)
    S1=np.linalg.norm(S1vec)
    S2=np.linalg.norm(S2vec)

    theta1 = np.arccos(np.dot(Lvec,S1vec)/(L*S1))
    theta2 = np.arccos(np.dot(Lvec,S2vec)/(L*S2))
    theta12= np.arccos(np.dot(S1vec,S2vec)/(S1*S2))
    deltaphi= np.arccos((np.cos(theta12) - np.cos(theta1)*np.cos(theta2))/(np.sin(theta1)*np.sin(theta2))   )

    deltaphi*=math.copysign(1., np.dot(Lvec,np.cross(np.cross(Lvec,S1vec),np.cross(Lvec,S2vec))))
    return theta1,theta2,deltaphi,theta12




def xi_allowed(J,q,S1,S2,r, more=False,verbose=False):

    '''
    Find the allowed range of xi for fixed J, corresponding to the extrema of
    the effective potential. Two implementations are presented, and are
    controlled by the inner flag use_derivative. If False (default, suggested),
    scipy's fminbound minimization algorithm is applied to the effective
    potentials `precession.xi_minus` and `precession.xi_plus`. If True, we
    explicitly look for the zeroes of the derivative of the effective potentials
    with respect to S. J. Vosmera found that the bisect root finder behaves
    better than brentq for low mass ratio. We believe both implementation are
    correct: the former has been tested more extensively, the latter has been
    found to be more reliable in the q->1 limit.

    WARNING: This function is critical. It's tested, but is tricky numerical
    issues may still be present.

    **Call:**

        xi_low,xi_up=precession.xi_allowed(J,q,S1,S2,r,more=False,verbose=False)

        xi_low,xi_up,S_xilow,S_xiup=precession.xi_allowed(J,q,S1,S2,r,more=True,verbose=False)


    **Parameters:**

    - `J`: magnitude of the total angular momentum.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `r`: binary separation.
    - `more`: if `True` returns additional quantities.
    - `verbose`: if `True` print additional information.

    **Returns:**

    - `xi_low`: minimum allowed value of xi, given the input parameters.
    - `xi_up`: maximum allowed value of xi, given the input parameters.
    - `S_xilow`: (optional) value of S when xi=xi_low.
    - `S_xiup`: (optional) value of S when xi=xi_up.
    '''

    # Take care of possible pathologies at the edges of the parameter space
    Jmin,Jmax= J_lim(q,S1,S2,r)
    if J==Jmin or J==Jmax:
        if J==Jmin:
            xiboth, dummy, Sboth, dummy = xi_at_Jlim(q,S1,S2,r,more=True)
        elif J==Jmax:
            dummy, xiboth, dummy, Sboth = xi_at_Jlim(q,S1,S2,r,more=True)
        if more:
            return xiboth, xiboth, Sboth, Sboth
        else:
            return xiboth, xiboth

    St_min,St_max=St_limits(J,q,S1,S2,r)

    # The extrema are at S=St_min, St_max
    if q==1:
        L=(q/(1.+q)**2)*(r*M**3)**.5
        xi_low=(J**2-L**2-St_max**2)/(L*M**2)
        xi_up=(J**2-L**2-St_min**2)/(L*M**2)
        S_xilow=St_max
        S_xiup=St_min

    # Extremize the two effective potentials.
    else:

        use_derivative=False

        #Run a minimization algorithms on the effective potentials.
        if use_derivative==False:

            # Minimum of xi_minus
            resmin= sp.optimize.fminbound(lambda S: xi_minus(S,J,q,S1,S2,r), St_min, St_max,xtol=1e-12,full_output=1)
            S_xilow=resmin[0]
            xi_low=resmin[1]

            # Maximum of xi_plus. Scipy provides minimization algorithms: minimize -xi_plus and change sign at the end.
            resmax= sp.optimize.fminbound(lambda S: -1.*xi_plus(S,J,q,S1,S2,r), St_min, St_max,xtol=1e-12,full_output=1)
            S_xiup=resmax[0]
            xi_up=-1.*resmax[1]

        # Run a root finders on the derivative of the effective potentials.
        elif use_derivative==True:

            if q<0.1: # bisect behaves better for extreme mass ratios...
                S_xilow=sp.optimize.bisect(lambda S: dxidS_plus(S,J,q,S1,S2,r), St_min, St_max,xtol=1e-12)
                S_xiup=sp.optimize.bisect(lambda S: dxidS_minus(S,J,q,S1,S2,r), St_min, St_max,xtol=1e-12)
            else: # ... but brentq is faster
                S_xilow=sp.optimize.brentq(lambda S: dxidS_plus(S,J,q,S1,S2,r), St_min, St_max,xtol=1e-12)
                S_xiup=sp.optimize.brentq(lambda S: dxidS_minus(S,J,q,S1,S2,r), St_min, St_max,xtol=1e-12)
            xi_low=xi_minus(S_xilow,J,q,S1,S2,r)
            xi_up =xi_plus(S_xiup,J,q,S1,S2,r)

    if verbose:
        print "[xi_allowed] xi_low", xi_low, " xi_up=", xi_up
    if more: # Return the S values as well
        return xi_low, xi_up, S_xilow, S_xiup
    else:
        return xi_low, xi_up


def resonant_finder(xi,q,S1,S2,r, more=False):

    '''
    Find the spin-orbit resonances, for given xi, as extrema of the allowed
    region in the parameter space. Two resonances are present for DeltaPhi=0 and
    DeltaPhi=pi. They maximize (0) and minimize (pi) J for fixed xi. This is an
    alternative (and more powerful) approach to solving the Schnittman equation,
    given in Eq.(35) of [PRD
    70,124020(2004)](http://journals.aps.org/prd/abstract/10.1103/PhysRevD.70.
    124020).

    **Call:**

        theta1_dp0,theta2_dp0,theta1_dp180,theta2_dp180=precession.resonant_finder(xi,q,S1,S2,r,more=False)

        J_dp0,S_dp0,theta1_dp0,theta2_dp0,J_dp180,S_dp180,theta1_dp180,theta2_dp180=precession.resonant_finder(xi,q,S1,S2,r,more=True)

    **Parameters:**

    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `r`: binary separation.
    - `more`: if `True` returns additional quantities.

    **Returns:**

    - J_dp0: (optional) value of J for the DeltaPhi=0 resonance.
    - S_dp0: (optional) value of S for the DeltaPhi=0 resonance.
    - theta1_dp0: value of theta1 for the DeltaPhi=0 resonance.
    - theta2_dp0: value of theta2 for the DeltaPhi=pi resonance.
    - J_dp180: (optional) value of J for the DeltaPhi=pi resonance.
    - S_dp180: (optional) value of S for the DeltaPhi=pi resonance.
    - theta1_dp180: value of theta1 for the DeltaPhi=pi resonance.
    - theta2_dp180: value of theta2 for the DeltaPhi=pi resonance.
    '''

    Jmin,Jmax=J_lim(q,S1,S2,r)
    L=(q/(1.+q)**2)*(r*M**3)**.5

    #DeltaPhi=0 resonance.
    J_dp0=sp.optimize.brentq(lambda J: xi_allowed(J,q,S1,S2,r)[0] -xi , abs(L-S1-S2), Jmax,xtol=1e-12)

    if q==1:
        # Find S. St_max
        S_dp0=St_limits(J_dp0,q,S1,S2,r)[1]
        # Find angles
        theta1_dp0,theta2_dp0,deltaphi_dp0,dummy = parametric_angles(0,J_dp0,xi,q,S1,S2,r)

    else:
        # Find S. Minimum of effective potential
        dummy,dummy,S_dp0,dummy= xi_allowed(J_dp0,q,S1,S2,r,more=True)
        # Find angles
        theta1_dp0,theta2_dp0,deltaphi_dp0,dummy = parametric_angles(S_dp0,J_dp0,xi,q,S1,S2,r)

    #DeltaPhi=180 resonance.
    xi_Jmin,xi_Jmax= xi_at_Jlim(q,S1,S2,r)

    if xi>xi_Jmin:
        # Find J. Solution always on xi_max, between Jmin and Jmax
        J_dp180=sp.optimize.brentq(lambda J: xi_allowed(J,q,S1,S2,r)[1] -xi , Jmin, Jmax,xtol=1e-12)
        if q==1:
            # Find S. St_min
            S_dp180=St_limits(J_dp180,q,S1,S2,r)[0]
            # Find angles
            theta1_dp180,theta2_dp180,deltaphi_dp0,dummy = parametric_angles(0,J_dp180,xi,q,S1,S2,r)
        else:
            # Find S. Maximum of effective potential
            dummy,dummy,dummy,S_dp180= xi_allowed(J_dp180,q,S1,S2,r,more=True)
            # Find angles
            theta1_dp180,theta2_dp180,deltaphi_dp180,dummy = parametric_angles(S_dp180,J_dp180,xi,q,S1,S2,r)
    else:
        # Find J. Solution still on xi_min, between Jmin and L-S1-S2. You're not here if Jmin=L-S1-S2, because xi_Jmin is the lower allowed value for xi in that case.
        J_dp180=sp.optimize.brentq(lambda J: xi_allowed(J,q,S1,S2,r)[0] -xi ,Jmin, abs(L-S1-S2),xtol=1e-12)
        if q==1:
            # Find S. St_max
            S_dp180=St_limits(J_dp180,q,S1,S2,r)[1]
            # Find angles
            theta1_dp180,theta2_dp180,deltaphi_dp0,dummy = parametric_angles(0,J_dp180,xi,q,S1,S2,r)
        else:
            # Find S. Minimum of effective potential
            dummy,dummy,S_dp180,dummy= xi_allowed(J_dp180,q,S1,S2,r,more=True)
            # Find angles
            theta1_dp180,theta2_dp180,deltaphi_dp180,dummy = parametric_angles(S_dp180,J_dp180,xi,q,S1,S2,r)

    if False: # Sanity check.
        print "DeltaPhi=0?", deltaphi_dp0
        print "DeltaPhi=pi?", deltaphi_dp180

    if more: # return everything you got
        return J_dp0, S_dp0, theta1_dp0, theta2_dp0, J_dp180, S_dp180, theta1_dp180, theta2_dp180
    else: # return the angles only
        return theta1_dp0, theta2_dp0, theta1_dp180, theta2_dp180


def J_allowed(xi,q,S1,S2,r):

    '''
    Find allowed values of J for fixed xi, i.e the spin-orbit resonances. See
    `precession.resonant_finder`.

    **Call:**

        J_low,J_up=precession.J_allowed(xi,q,S1,S2,r)

    **Parameters:**

    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `r`: binary separation.

    **Returns:**

    - `J_low`: minimum allowed value of J, given the input parameters.
    - `J_up`: maximum allowed value of J, given the input parameters.
    '''

    J_dp0, dummy, dummy, dummy, J_dp180, dummy, dummy, dummy = resonant_finder(xi,q,S1,S2,r, more=True)
    return min(J_dp0,J_dp180),max(J_dp0,J_dp180)


def thetas_inf(xi,kappa_inf,q,S1,S2):

    '''
    Find the asymptotic (constant) values of theta1 and theta2 given xi and
    kappa_inf.

    **Call:**

        theta1_inf,theta2_inf=precession.thetas_inf(xi,kappa_inf,q,S1,S2)

    **Parameters:**

    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `kappa_inf`: asymptotic value of kappa at large separations.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.

    **Returns:**

    - `theta1_inf`: asymptotic value of theta1 at large separations.
    - `theta2_inf`: asymptotic value of theta2 at large separations.
    '''

    if q==1:
        assert False, "[thetas_inf] Error: I'm sorry, can't run for q=1. The angles theta1 and theta2 are not constant at large separations."
    else:
        ct1=(-xi + kappa_inf*(1.+q**(-1)))/(S1*(q**(-1)-q))
        ct2=(xi - kappa_inf*(1.+q))/(S2*(q**(-1)-q))
        return np.arccos(ct1),np.arccos(ct2)


def from_the_angles_inf(theta1_inf,theta2_inf,q,S1,S2):

    '''
    Find xi and kappa_inf given the asymptotic (constant) values of theta1 and
    theta2.

    **Call:**

        xi,kappa_inf=precession.from_the_angles_inf(theta1_inf,theta2_inf,q,S1,S2)

    **Parameters:**

    - `theta1_inf`: asymptotic value of theta1 at large separations.
    - `theta2_inf`: asymptotic value of theta2 at large separations.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.

    **Returns:**

    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `kappa_inf`: asymptotic value of kappa at large separations.
    '''

    if q==1:
        assert False, "[from_the_angles_inf] Error: I'm sorry, can't run for q=1. The angles theta1 and theta2 are not constant at large separations."
    else:
        xi= ((1.+q)*S1*np.cos(theta1_inf)+(1.+q**-1)*S2*np.cos(theta2_inf))*M**-2
        kappa_inf= (S1*np.cos(theta1_inf)+S2*np.cos(theta2_inf))*M**-2
        return xi,kappa_inf


def aligned_configurations(q,S1,S2,r):

    '''
    Values of xi and J corresponding to the four (anti)aligned configuration:
    up-up (spins of both primary and secondary BH aligned with L); up-up (spins
    of both primary and secondary BH antialigned with L); up-down (spin of the
    primary BH aligned with L; spin of the secondary BH antialigned with L);
    down-up (spin of the primary BH aligned with L; spin of the secondary BH
    antialigned with L).

    **Call:**

        xiupup,xidowndown,xiupdown,xidownup,Jupup,Jdowndown,Jupdown,Jdownup=precession.aligned_configurations(q,S1,S2,r)

    **Parameters:**

    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
     - `r`: binary separation.

    **Returns:**

    - `xiupup`: xi of the up-up configuration.
    - `xidowndown`: xi of the down-down configuration.
    - `xiupdown`: xi of the up-down configuration.
    - `xidownup`: xi of the down-up configuration.
    - `Jupup`: J of the up-up configuration.
    - `Jdowndown`: J of the down-down configuration.
    - `Jupdown`: J of the up-down configuration.
    - `Jdownup`: J of the down-up configuration.
    '''



    L=(q/(1.+q)**2)*(r*M**3)**.5

    xiupup=(1.+q)*S1+(1+q**-1)*S2
    xidowndown=-(1.+q)*S1-(1+q**-1)*S2
    xiupdown=(1.+q)*S1-(1+q**-1)*S2
    xidownup=-(1.+q)*S1+(1+q**-1)*S2

    Jupup=L+S1+S2
    Jdowndown=np.abs(L-S1-S2)
    Jupdown=np.abs(L+S1-S2)
    Jdownup=np.abs(L-S1+S2)

    return xiupup,xidowndown,xiupdown,xidownup,Jupup,Jdowndown,Jupdown,Jdownup


def updown(q,S1,S2):

    '''
    Instability range for up-down aligned binaries. Binaries with the primary
    (secondary) spin aligned (antialigned) with the angular momentum are
    unstable between the two separations returned. Hack the code to compute also
    the function *switch*, for a sanity check on the property of the second
    threshold (whether that's on `precession.xi_plus` or on
    `precession.xi_minus`). All up-down binaries are stable in the equal-mass
    case: if q=1 returns Nones.

    **Call:**

        r_udp,r_udm=precession.updown(q,S1,S2)

    **Parameters:**

    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.

    **Returns:**

    - `r_udp`: upper separation for the up-down instability.
    - `r_udm`: lower separation for the up-down instability.
    '''

    if q==1:
        print "[updown] Warning: up-down is always stable for q=1. Retuning Nones."
        return None, None

    r_udp=((1.+q)**2 *((q*S1)**0.5 + S2**0.5)**2/((1.-q)*q))**2
    r_udm=((1.+q)**2 *((q*S1)**0.5 - S2**0.5)**2/((1.-q)*q))**2

    if False:
        switch=q**0.5+ q**(-0.5) - (S2/S1)**0.5 - (S2/S1)**(-0.5)
        return r_udp,r_udm,switch
    else:
        return r_udp,r_udm




#################################
######### MORPHOLOGIES ##########
#################################


def find_morphology(xi,J,q,S1,S2,r):

    '''
    Compute the precessional morphology in DeltaPhi. Returns:

    - -1 if librating about DeltaPhi=0;
    - 0 if circulating in the whole DeltaPhi range [-pi,pi];
    - +1 if librating about DeltaPhi=pi.

    **Call:**

        morphology=precession.find_morphology(xi,J,q,S1,S2,r)

    **Parameters:**

    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `J`: magnitude of the total angular momentum.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `r`: binary separation.

    **Returns:**

    - morphology: precessional morphology in DeltaPhi: -1 if librating about DeltaPhi=0; 0 if circulating; +1 if librating about DeltaPhi=pi.
    '''

    if q==1: # If q=1, the limits must be specified in cos(varphi)
        Sb_min=-1
        Sb_max=1
    else:
        Sb_min,Sb_max=Sb_limits(xi,J,q,S1,S2,r)

    dummy,dummy,deltaphi_Sbmin,dummy = parametric_angles(Sb_min,J,xi,q,S1,S2,r)
    dummy,dummy,deltaphi_Sbmax,dummy = parametric_angles(Sb_max,J,xi,q,S1,S2,r)

    # Both the initial and the final point in a precession cycle are <pi/2. This is a libration about DeltaPhi=0
    if deltaphi_Sbmin<np.pi/2. and deltaphi_Sbmax<np.pi/2.:
        return -1.
    # Both the initial and the final point in a precession cycle are >pi/2. This is a libration about DeltaPhi=180
    elif deltaphi_Sbmin>np.pi/2. and deltaphi_Sbmax>np.pi/2.:
        return 1.
    # The precession orbit crosses both DeltaPhi=0 and DeltaPhi=180. This is  circulation
    else:
        return 0.


def region_selection(varphi,S,J,q,S1,S2,r):

    '''
    Get the morphology in the varphi plane. See `precession.find_morphology`.

    **Call:**

        morphology=precession.region_selection(varphi,S,J,q,S1,S2,r)

    **Parameters:**

    - `varphi`: angle describing the rotation of S1 and S2 about S, in a frame aligned with J.
    - `S`: magnitude of the total spin.
    - `J`: magnitude of the total angular momentum.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `r`: binary separation.

    **Returns:**

    - morphology: precessional morphology in DeltaPhi: -1 if librating about DeltaPhi=0; 0 if circulating; +1 if librating about DeltaPhi=pi.
    '''

    xi=xi_contour(varphi,S,J,q,S1,S2,r)
    return find_morphology(xi,J,q,S1,S2,r)


def phase_checker(q,S1,S2,r,verbose=False):

    '''
    Computes the number of different morphologies you MAY have for a given
    geometrical configuration (i.e. given the lengths of the vectors L, S1 and
    S2). These are just geometrical constraints: the actual number of allowed
    morphologies depends on J, as returned by `precession.phase_xi`, but it
    can't be out of what returned by this function. This function is basically a
    sanity check for `precession.phase_xi`.

    **Call:**

        phases_vals=precession.phase_checker(q,S1,S2,r,verbose=False)

    **Parameters:**

    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `r`: binary separation.
    - `verbose`: if True print additional information.

    **Returns:**

    - phases_vals: number of coexisting morphologies: 1 if only the DeltaPhi~pi phase is present; 2 if two DeltaPhi~pi phases and a circulating phase are present; 3 if a librating DeltaPhi~0, a circulating, and a DeltaPhi~pi phase al all present (array).
    '''

    L=(q/(1.+q)**2)*(r*M**3)**.5

    if L > S1 + S2:
        if verbose:
            print "L > S1 + S2 : 3"
        phases=[3]

    elif (S1 + S2 > L and L > max(S1, S2)):
        if verbose:
            print "S1 + S2 > L > max(S1, S2) : 3,1"
        phases=[3,1]

    elif (max(S1,S2) > L and  L>np.abs(S1-S2)):
        if verbose:
            print "max(S1,S2) > L > |S1-S2| : 3,2,1"
        phases=[3,2,1]

    elif np.abs(S1-S2) > L:
        if verbose:
            print "|S1 - S2| > L : 3,2"
        phases=[3,2]

    else:
        assert False, "[phase_checker] Error. You should never be here!"

    return phases


def phase_xi(J,q,S1,S2,r):

    '''
    Return an integer number, phases, specifying the number of precessional
    morphologies that can coexist for a given value of J. Returns:

    - 1 if only the DeltaPhi~pi phase is present;
    - 2 if two DeltaPhi~pi phases
    and a circulating phase are present;
    - 3 if a librating DeltaPhi~0, a
    circulating, and a DeltaPhi~pi phase al all present.

    The latter is *standard* case studied in [our first
    PRL](http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.114.081103).
    Additionally, return the values of xi that, for given J, separate the
    binaries with different morphologies. If there are no transitions (i.e.
    phase=1), the transition values of xi are returned as Nones. If transitions
    cannot be found for numerical reasons, assume they coincides with the
    extrema of xi (see `precession.xi_allowed`). The output of this function can
    be tested with `precession.phase_checker`.

    **Call:**

        phase,xi_transit_low,xi_transit_up=precession.phase_xi(J,q,S1,S2,r)

    **Parameters:**

    - `J`: magnitude of the total angular momentum.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `r`: binary separation.

    **Returns:**

    - phase: number of coexisting morphologies: 1 if only the DeltaPhi~pi phase is present; 2 if two DeltaPhi~pi phases and a circulating phase are present; 3 if a librating DeltaPhi~0, a circulating, and a DeltaPhi~pi phase al all present.
    - xi_transit_low: value of xi marking the transition between the low and the middle phase
    - xi_transit_up: value of xi marking the transition between the upper and the middle phase
    '''

    xi_min,xi_max=xi_allowed(J,q,S1,S2,r)
    L=(q/(1.+q)**2)*(r*M**3)**.5

    # The following should be equivalent to apply find_morphology at xi_min and xi_max but it turns out to be more stable to numerical noise because it doesn't rely on Sb_limits
    St_min,St_max=St_limits(J,q,S1,S2,r)
    morph_ximin=find_morphology(xi_plus(St_max,J,q,S1,S2,r),J,q,S1,S2,r)
    morph_ximax=find_morphology(xi_plus(St_min,J,q,S1,S2,r),J,q,S1,S2,r)


    # The morphology at xi_max must be librating about pi. The morphology at xi_min can't be circulating.  Check if, because of degeneracies and numerical issues, a different morphology is be detected...
    if morph_ximax!=1 or morph_ximin==0:
        print "[phase_xi] I think this should never ever happen. morph_ximax=",morph_ximax," morph_ximin=",morph_ximin
        if morph_ximin==-1:
            phase=3.
            xi_transit_up=xi_plus(St_min,J,q,S1,S2,r)+1e-9
            try:
                xi_transit_low=sp.optimize.brentq(lambda xi: (find_morphology(xi,J,q,S1,S2,r) +0.5), xi_min,xi_transit_up-1e-5, xtol=1e-5)
            except:
                xi_transit_low=xi_min

            return phase,xi_transit_low,xi_transit_up

    if morph_ximin==-1:

        phase=3. # This is a three-phase case: Deltaphi~pi close to xi_max, Deltaphi~0 at xi_min and a circulating phase in between

        #Find transition Librating 180 - Circulating. Shift the find_morphology output, such that the zero is between the two phases
        try:
            xi_transit_up=sp.optimize.brentq(lambda xi: (find_morphology(xi,J,q,S1,S2,r) -0.5), xi_min,xi_max, xtol=1e-5)
        except:
            xi_transit_up=xi_max

        #Find transition Librating 0 - Circulating. Shift the find_morphology output, such that the zero is between the two phases
        try:
            xi_transit_low=sp.optimize.brentq(lambda xi: (find_morphology(xi,J,q,S1,S2,r) +0.5), xi_min,xi_max, xtol=1e-5)
        except:
            xi_transit_low=xi_min

        #xi_transit_low,xi_transit_up=sorted([xi_transit_0,xi_transit_180])

    elif morph_ximin==1:  # This is either a two-phase or a single-phase case.
        # Here we need to bracket the interval to find two roots. Two possible bracketing points are checked: the values of S for which
            # [first try] cos(theta1)=1 and cos(theta2)= -1
            # [second try] cos(theta1)=-1 and cos(theta2)= 1
        # Either one of the two choice typically gives the correct results for all the cases we tried; we cannot exclude the presence of pathological sets of parameters where both choices fail.

        for xi_bracket in [ -1.*(1.+q)*S1+(1.+1./q)*S2 , (1.+q)*S1-(1.+1./q)*S2 ]:
            if xi_bracket>xi_max or xi_bracket<xi_min:
                phase=1. # Either the bracketing is wrong or this is a single-phase case
                xi_transit_low=xi_min
                xi_transit_up=xi_max
            elif find_morphology(xi_bracket,J,q,S1,S2,r)==0:
                phase=2. # You found a good bracketing point. This must be a two-phase case

                #Find the first transition Librating 180 - Circulating at lower xi. Shift the find_morphology output, such that the zero is between the two phase
                try:
                    xi_transit_low=sp.optimize.brentq(lambda xi: (find_morphology(xi,J,q,S1,S2,r) -0.5), xi_min,xi_bracket, xtol=1e-5)
                except:
                    xi_transit_low=xi_min

                #Find the other transition Librating 180 - Circulating at larger xi. Shift the find_morphology output, such that the zero is between the two phase
                try:
                    xi_transit_up=sp.optimize.brentq(lambda xi:  (find_morphology(xi,J,q,S1,S2,r) -0.5), xi_bracket,xi_max, xtol=1e-5)
                except:
                    xi_transit_up=xi_max

                break # One bracketing point is enough. Get out if you found the solution

            else:
                phase=1.  # Either the bracketing is wrong or this is a single-phase case
                xi_transit_low=xi_min
                xi_transit_up=xi_max

    if phase not in phase_checker(q,S1,S2,r):
        print "[phase_xi] Warning: detected phases not allowed by geometry!"

    return phase, xi_transit_low,xi_transit_up


def Jframe_projection(xi,S,J,q,S1,S2,r):

    '''
    Project the three momenta on the reference frame aligned with the total
    angular momentum J. The z axis points in the J direction, and the x axis
    lies in the plane spanned by J and L. The y axis complete an orthonormal
    triad. Note that this is not an inertial frame (not even on the precession
    time) because it precesses together with L.

    **Call:**

        Jvec,Lvec,S1vec,S2vec,Svec=precession.Jframe_projection(xi,S,J,q,S1,S2,r)

    **Parameters:**

    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `S`: magnitude of the total spin.
    - `J`: magnitude of the total angular momentum.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `r`: binary separation.

    **Returns:**

    - `Jvec`: components of J in a reference frame (3 values for x,y,z).
    - `Lvec`: components of L in a reference frame (3 values for x,y,z).
    - `S1vec`: components of S1 in a reference frame (3 values for x,y,z).
    - `S2vec`: components of S2 in a reference frame (3 values for x,y,z).
    - `Svec`: components of S in a reference frame (3 values for x,y,z).
    '''

    global flags_q1
    L=(q/(1.+q)**2)*(r*M**3)**.5

    if q==1:
        if flags_q1[2]==False:
            print "[Jframe_projection] Warning q=1: input here is cos(varphi), not S."
            flags_q1[2]=True
        varphi=np.arccos(S)
        S=np.sqrt(J**2-L**2-xi*L*M**2)
    else:
        varphi=get_varphi(xi,S,J,q,S1,S2,r,sign=1)

    Jx = 0.
    Jy = 0.
    Jz = J # Definition!

    Lx = ( (((L+S)**2-J**2)**.5) * ((J**2-(L-S)**2)**.5) ) / (2.*J)
    Ly = 0. # Definition!
    Lz = (J**2+L**2-S**2) / (2.*J)

    S1x = (1./(4.*J*S**2))* \
          ( -1.*(S**2+S1**2-S2**2) * (((L+S)**2-J**2)**.5) * ((J**2-(L-S)**2)**.5) \
          + (J**2-L**2+S**2) * (((S1+S2)**2-S**2)**.5) * ((S**2-(S1-S2)**2)**.5) * np.cos(varphi) )
    S1y = (1./(2.*S))* \
          (((S1+S2)**2-S**2)**.5) * ((S**2-(S1-S2)**2)**.5)  * np.sin(varphi)
    S1z = (1./(4.*J*S**2))* \
          ( (S**2+S1**2-S2**2) * (J**2-L**2+S**2) \
          + (((L+S)**2-J**2)**.5) * ((J**2-(L-S)**2)**.5) *(((S1+S2)**2-S**2)**.5) * ((S**2-(S1-S2)**2)**.5) *np.cos(varphi) )

    S2x = (-1./(4.*J*S**2))* \
          ( (S**2+S2**2-S1**2) * (((L+S)**2-J**2)**.5) * ((J**2-(L-S)**2)**.5) \
          + (J**2-L**2+S**2) * (((S1+S2)**2-S**2)**.5) * ((S**2-(S1-S2)**2)**.5) * np.cos(varphi) )
    S2y = (-1./(2.*S))* \
          (((S1+S2)**2-S**2)**.5) * ((S**2-(S1-S2)**2)**.5)  * np.sin(varphi)
    S2z = (1./(4.*J*S**2))* \
          ( (S**2+S2**2-S1**2) * (J**2-L**2+S**2) \
          - (((L+S)**2-J**2)**.5) * ((J**2-(L-S)**2)**.5) *(((S1+S2)**2-S**2)**.5) * ((S**2-(S1-S2)**2)**.5) *np.cos(varphi) )

    Jvec=np.array([Jx,Jy,Jz])
    Lvec=np.array([Lx,Ly,Lz])
    S1vec=np.array([S1x,S1y,S1z])
    S2vec=np.array([S2x,S2y,S2z])
    Svec= S1vec+S2vec

    if False: #Sanity check. These sets of numbers should really be the same
        print "[Jframe_projection] Check varphi", np.cos(varphi), np.sin(varphi)
        print "[Jframe_projection] Check J norm", np.linalg.norm(Jvec), J
        print "[Jframe_projection] Check norm", np.linalg.norm(Lvec), L
        print "[Jframe_projection] Check norm", np.linalg.norm(S1vec), S1
        print "[Jframe_projection] Check norm", np.linalg.norm(S2vec), S2
        print "[Jframe_projection] Check norm", np.linalg.norm(Svec), S

    return Jvec,Lvec,S1vec,S2vec,Svec




#################################
### TIME-DEPENDENT PRECESSION ###
#################################


def Omegaz(S,xi,J,q,S1,S2,r):

    '''
    Compute the (azimuthal) precessional frequency of the orbital angular
    momentum L about the total angular momentum J.

    **Call:**

        Omega=precession.Omegaz(S,xi,J,q,S1,S2,r)

    **Parameters:**

    - `S`: magnitude of the total spin.
    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `J`: magnitude of the total angular momentum.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `r`: binary separation.

    **Returns:**

    - `Omega`: precessional frequency of L about J.
    '''

    L=(q/(1.+q)**2)*(r*M**3)**.5
    eta=q/(1.+q)**2
    t1=(J/2.)*((eta**2*M**3/L**2)**3)
    t2=(3./(2.*eta))*(1.-(eta*M**2*xi/L))
    t3=(3.*(1.+q)/(2.*q*(J**2-(L-S)**2)*((L+S)**2-J**2)))*(1.-(eta*M**2*xi/L))
    t4=4.*(1.-q)*L**2*(S1**2-S2**2)
    t5=(1.+q)*(J**2-L**2-S**2)*(J**2-L**2-S**2-4.*eta*M**2*L*xi)

    return t1*(1.+t2-t3*(t4-t5))


def dSdt(S,xi,J,q,S1,S2,r,sign=1.):

    '''
    Compute the derivative of S with respect to t (on the precessional time
    only, i.e. assuming J is constant). Uses the spin-precession equations, but
    not the radiation reaction equation. The additional sign lets you specifiy
    the sign of the angle deltaphi: for consistency with what presented in our
    papers, use sign=1 if you are in the second half of the precession cycle
    (deltaphi is in [0,pi]) and sign=-1 if you are in the first half of the
    precession cycle (deltaphi is in [-pi,0]). If q=1, this function computes
    d(cos(varphi))/dt.

    **Call:**

        dSdt=precession.dSdt(S,xi,J,q,S1,S2,r,sign=1.)

    **Parameters:**

    - `S`: magnitude of the total spin.
    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `J`: magnitude of the total angular momentum.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `r`: binary separation.
    - `sign`: if 1 return angle in [0,pi], if -1 return angle in [-pi,0].

    **Returns:**

    - `dSdt`: precessional-cycle speed.
    '''

    global flags_q1

    L=(q/(1.+q)**2)*(r*M**3)**.5
    eta=q/(1.+q)**2

    if q==1:
        if flags_q1[3]==False:
            print "[dSdt] Warning q=1: input here is cos(varphi), not S; now computing d(cos(varphi))/dt "
            flags_q1[3]=True

        cosvarphi = S # The input variable is actually cos(varphi)
        S = np.sqrt(J**2-L**2-xi*L*M**2)
        S_min,S_max=St_limits(J,q,S1,S2,r)

        if np.abs(S-S_min)<1e-8 or np.abs(S-S_max)<1e-8:
            print "[dSdt] Warning: you are at resonance, varphi is ill-defined here."
            return 0.

        # Compute d(cos(varphi))/dt
        t6=((S1+S2)**2-S**2)
        t7=(S**2-(S1-S2)**2)
        B=(4*S**2-xi**2*M**4)*t6*t7
        #B=max(0.0,B)
        t1= (12*S**2*S1*S2)/(np.sqrt(B))
        t2= (eta**2*M**3)**3/L**6
        t3= 1-(eta*M**2*xi)/L
        ct1= (1/(4*S1*S**2))*(xi*M**2*(S**2+S1**2-S2**2)+np.sqrt(B)*cosvarphi)
        ct2= (1/(4*S2*S**2))*(xi*M**2*(S**2+S2**2-S1**2)-np.sqrt(B)*cosvarphi)
        ct12=(S**2-S1**2-S2**2)/(2.*S1*S2)
        t4=(np.abs(1.-ct1**2-ct2**2-ct12**2 +2.*ct1*ct2*ct12))**.5
        der=sign*t1*t2*t3*t4

    else:

        # Compute dS/dt
        t1= (-3. * (1.-q**2) *S1*S2*eta**6*M**9)/(2.*q*S*L**5)
        t2= 1.-((eta*M**2*xi)/L)
        #It's faster if you don't call [parametric_angles] here. Equivalent to
            #theta1,theta2,deltaphi,theta12 = parametric_angles(S,J,xi,q,S1,S2,r)
            #der=sign*t1*t2*np.sin(theta1)*np.sin(theta2)*np.sin(deltaphi)
        ct1= ( ((J**2-L**2-S**2)/L) - (2.*q*M**2*xi)/(1.+q) )/(2.*(1.-q)*S1)
        ct2= ( ((J**2-L**2-S**2)*(-q/L)) + (2.*q*M**2*xi)/(1.+q) )/(2.*(1.-q)*S2)
        ct12=(S**2-S1**2-S2**2)/(2.*S1*S2)
        t3=(np.abs(1.-ct1**2-ct2**2-ct12**2 +2.*ct1*ct2*ct12))**.5 # I know abs is dirty, but does the job
        der=sign*t1*t2*t3

    return der


def dtdS(S,xi,J,q,S1,S2,r,sign=1.):

    '''
    Auxiliary function dt/dS=(dS/dt)^-1. See `precession.dSdt`.

    **Call:**

        dtdS=precession.dtdS(S,xi,J,q,S1,S2,r,sign=1.)

    **Parameters:**

    - `S`: magnitude of the total spin.
    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `J`: magnitude of the total angular momentum.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `r`: binary separation.
    - `sign`: if 1 return angle in [0,pi], if -1 return angle in [-pi,0].

    **Returns:**

    - `dtdS`: inverse of the precessional-cycle speed.
    '''

    return 1./(dSdt(S,xi,J,q,S1,S2,r,sign))


def t_of_S( S_initial,S_final ,Sb_min,Sb_max ,xi,J,q,S1,S2,r, t_initial=0, sign=1. ):

    '''
    Integrate `precession.dSdt` to find t (time) as a function of S (magnitude
    of the total spin). Since dS/dt depends on S and not on t, finding t(S) only
    requires a numnerical integration; S(t) is provided in `precession.t_of_S`.
    Sb_min and Sb_max are passed to this function (and not computed within it)
    for computational efficiency. This function can only integrate over half
    precession period (i.e. from Sb_min to Sb_max at most). If you want t(S)
    over more precession periods you should stich different solutions together,
    consistently with the argument sign (in particular, flip sign every half
    period).

    **Call:**

        t=precession.t_of_S(S_initial,S_final,Sb_min,Sb_max,xi,J,q,S1,S2,r,t_initial=0,sign=1.)

    **Parameters:**

    - `S_initial`: lower edge of the integration domain.
    - `S_final`: upper edge of the integration domain.
    - `Sb_min`: minimum value of S from geometrical constraints. This is S- in our papers.
    - `Sb_max`: maximum value of S from geometrical constraints. This is S+ in our papers.
    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `J`: magnitude of the total angular momentum.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `r`: binary separation.
    - `t_initial`: intial integration time.
    - `sign`: if 1 return angle in [0,pi], if -1 return angle in [-pi,0].

    **Returns:**

    - `t`: time (in total mass units).
    '''

    global flags_q1
    if q==1:
        if flags_q1[4]==False:
            print "[t_of_S] Warning q=1: input here is cos(varphi) not S; now computing t( cos(varphi) )"
            flags_q1[4]=True

        L=(q/(1.+q)**2)*(r*M**3)**.5
        S = np.sqrt(J**2-L**2-xi*L*M**2)
        S_min,S_max=St_limits(J,q,S1,S2,r)
        if np.abs(S-S_min)<1e-8 or np.abs(S-S_max)<1e-8:
            print "[t_of_S] Warning: you are at resonance, varphi is ill defined here."
            return 0.
        elif min(S_initial,S_final) < -1 or max(S_initial,S_final) > 1:
            assert False, "[t_of_S] Error. You're trying to integrate over more than one (half)period"
        else:
            res=sp.integrate.quad(dtdS, S_initial, S_final, args=(xi,J,q,S1,S2,r,sign),full_output=1)
            return t_initial + res[0]

    if np.abs(Sb_min-Sb_max)<1e-8: # This happens when [Sb_limits] fails in bracketing of the solutions. In practice, this is a resonant binary.
        return 0.
    elif min(S_initial,S_final) < Sb_min or max(S_initial,S_final) > Sb_max:
        assert False, "[t_of_S] Error. You're trying to integrate over more than one (half)period"
    else:
        res=sp.integrate.quad(dtdS, S_initial, S_final, args=(xi,J,q,S1,S2,r,sign),full_output=1)
        return t_initial + res[0]


def S_of_t(t, Sb_min,Sb_max,xi,J,q,S1,S2,r):

    '''
    Integrate `precession.dSdt` to find S (time) as a function of t (magnitude
    of the total spin). In practice, this is done by inverting
    `precession.t_of_S`. Sb_min and Sb_max are passed to this function (and not
    computed within it) for computational efficiency. This function can only
    integrate over half precession period (i.e. from 0 to tau/2 at most). If you
    want S(t) over more precession periods you should stich different solutions
    together, consistently with the argument sign (in particular, flip sign
    every half period).

    **Call:**

        S=precession.S_of_t(t,Sb_min,Sb_max,xi,J,q,S1,S2,r,t_initial=0,sign=1.)

    **Parameters:**

    - `t`: time (in total mass units).
    - `S_final`: upper edge of the integration domain.
    - `Sb_min`: minimum value of S from geometrical constraints. This is S- in our papers.
    - `Sb_max`: maximum value of S from geometrical constraints. This is S+ in our papers.
    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `J`: magnitude of the total angular momentum.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `r`: binary separation.

    **Returns:**

    - `S`: magnitude of the total spin.
    '''
    tau=precession_period(xi,J,q,S1,S2,r)

    global flags_q1
    if q==1:
        if flags_q1[5]==False:
            print "[S_of_t] Warning q=1: output here is cos(varphi) not S; now computing cos(varphi)(t) )"
            flags_q1[5]=True

        L=(q/(1.+q)**2)*(r*M**3)**.5
        S = np.sqrt(J**2-L**2-xi*L*M**2)
        S_min,S_max=St_limits(J,q,S1,S2,r)

        if np.abs(S-S_min)<1e-8 or np.abs(S-S_max)<1e-8:
            print "[S_of_t] Warning: you are at resonance, varphi is ill defined here."
            return 0.
        elif t < 0 or t > tau/2.:
            assert False, "[S_of_t] Error. You're trying to integrate over more than one (half)period"
        else:
            S= sp.optimize.brentq(lambda S:np.abs( sp.integrate.quad(dtdS, 0, S, args=(xi,J,q,S1,S2,r))[0]) - t , Sb_min, Sb_max)
        return S

    if np.abs(Sb_min-Sb_max)<1e-8: # This happens when [Sb_limits] fails in bracketing of the solutions. In practice, this is a resonant binary.
        return np.average(Sb_min,Sb_min)
    elif t < 0 or t > tau/2.:
        assert False, "[S_of_t] Error. You're trying to integrate over more than one (half)period"
    else:
        S= sp.optimize.brentq(lambda S:np.abs( sp.integrate.quad(dtdS, Sb_min, S, args=(xi,J,q,S1,S2,r))[0]) - t , Sb_min, Sb_max)
        return S


def precession_period(xi,J,q,S1,S2,r):

    '''
    Find the period of S, i.e. the precessional timescale. This is
    `precession.t_of_S` integrated from Sb_min to Sb_max times 2.

    **Call:**

        tau=precession.precession_period(xi,J,q,S1,S2,r)

    **Parameters:**

    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `J`: magnitude of the total angular momentum.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `r`: binary separation.

    **Returns:**

    - `tau`: precessional period (in total mass units).
    '''

    if q==1: # If q=1, the limits must be specified in cos(varphi)
       Sb_min=-1
       Sb_max=1
    else:
        Sb_min,Sb_max=Sb_limits(xi,J,q,S1,S2,r)

    halfperiod = t_of_S(Sb_min,Sb_max,Sb_min,Sb_max,xi,J,q,S1,S2,r)
    #abs because here you don't care about the <sign> issue in dS/dt here
    return np.abs(2*halfperiod)


def OmegazdtdS(S,xi,J,q,S1,S2,r,sign=1.):

    '''
    Auxiliary function Omega_z * |dt/dS|. See `precession.Omegaz` and `precession.dSdt`.

    **Call:**

        OmegadtdS=precession.OmegazdtdS(S,xi,J,q,S1,S2,r,sign=1.):

    **Parameters:**

    - `S`: magnitude of the total spin.
    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `J`: magnitude of the total angular momentum.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `r`: binary separation.
    - `sign`: if 1 return angle in [0,pi], if -1 return angle in [-pi,0].

    **Returns:**

    - `OmegadtdS`: Omega_z * dt/dS.
    '''

    global flags_q1
    if q==1:
        if flags_q1[6]==False:
            print "[OmegazdtdS] Warning q=1: input here is cos(varphi), not S; now computing Omegaz * dt / d(cos(varphi))"
            flags_q1[6]=True
        cosvarphi=S # The input variable is actually cos(varphi)
        L=(q/(1.+q)**2)*(r*M**3)**.5
        S=np.sqrt(J**2-L**2-xi*L*M**2)
        return Omegaz(S,xi,J,q,S1,S2,r)/np.abs(dSdt(cosvarphi,xi,J,q,S1,S2,r,sign))

    else:
        return Omegaz(S,xi,J,q,S1,S2,r)/np.abs(dSdt(S,xi,J,q,S1,S2,r,sign))


def alpha_of_S( S_initial,S_final ,Sb_min,Sb_max ,xi,J,q,S1,S2,r, alpha_initial=0, sign=1.):

    '''
    Integrate `precession.Omegaz' to find the precession angle spanned by L
    about J, phiL, as a function of S. Sb_min and Sb_max are passed to this
    function (and not computed in it) to speed things up. This function can only
    integrate over half precession period (i.e. from Sb_min to Sb_max at most).
    If you want phiL(S) over more precession periods you should stich different
    solutions together, consistently with the argument sign (in particular, flip
    sign every half period).

    **Call:**

        phiL=precession.alpha_of_S(S_initial,S_final,Sb_min,Sb_max,xi,J,q,S1,S2,r,alpha_initial=0,sign=1.):

    **Parameters:**

    - `S_initial`: lower edge of the integration domain.
    - `S_final`: upper edge of the integration domain.
    - `Sb_min`: minimum value of S from geometrical constraints. This is S- in our papers.
    - `Sb_max`: maximum value of S from geometrical constraints. This is S+ in our papers.
    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `J`: magnitude of the total angular momentum.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `r`: binary separation.
    - `alpha_initial`: initial integration angle.
    - `sign`: if 1 return angle in [0,pi], if -1 return angle in [-pi,0].

    **Returns:**

    - `phiL`: azimuthal angle spanned by L about J.
    '''

    global flags_q1
    if q==1:
        if flags_q1[7]==False:
            print "[alpha_of_S] Warning q=1: input here is cos(varphi), not S; now computing alpha(cosvarphi)"
            flags_q1[7]=True

        L=(q/(1.+q)**2)*(r*M**3)**.5
        S = np.sqrt(J**2-L**2-xi*L*M**2)
        S_min,S_max=St_limits(J,q,S1,S2,r)

        if np.abs(S-S_min)<1e-8 or np.abs(S-S_max)<1e-8:
            print "[alpha_of_S] Warning: you are at resonance, varphi is ill defined here."
            return 0.
        elif min(S_initial,S_final) < -1 or max(S_initial,S_final) > 1:
            assert False, "[alpha_of_S] Error. You're trying to integrate over more than one (half)period"
        else:
            # If q=1, S is constant and therefore Omegaz is also constant. It can be taken out of the integral.
            deltat=t_of_S(S_initial,S_final,Sb_min,Sb_max,xi,J,q,S1,S2,r)
            return alpha_initial + Omegaz(S,xi,J,q,S1,S2,r) *deltat

    if np.abs(Sb_min-Sb_max)<1e-8: # This typically happen when [Sb_limits] fails in bracketing of the solutions. In practice, this is a resonant binary.
        return 0.
    elif min(S_initial,S_final) < Sb_min or max(S_initial,S_final) > Sb_max:
        assert False, "[alpha_of_S] Error. You're trying to integrate over more than one (half)period"
    else:
        # Actual integration
        res=sp.integrate.quad(OmegazdtdS, S_initial, S_final, args=(xi,J,q,S1,S2,r,sign),full_output=1)
        return alpha_initial + res[0]


def alphaz(xi,J,q,S1,S2,r):

    '''
    Angle spanned by L about J in a single precession cycle. This is
    `precession.alpha_of_S` integrated from Sb_min to Sb_max times 2.

    **Call:**

        alpha=precession.alphaz(xi,J,q,S1,S2,r)

    **Parameters:**

    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `J`: magnitude of the total angular momentum.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `r`: binary separation.

    **Returns:**

    - `alpha`: azimuthal angle spanned by L about J in an entire precession cycle.
    '''

    if q==1: # If q=1, the limits must be specified in cos(varphi)
        Sb_min=-1
        Sb_max=1
        return 2*alpha_of_S(Sb_min,Sb_max,Sb_min,Sb_max,xi,J,q,S1,S2,r)

    else:
        Sb_min,Sb_max=Sb_limits(xi,J,q,S1,S2,r)
        if np.abs(Sb_min-Sb_max)<1e-8: # This typically happen when [Sb_limits] fails in bracketing of the solutions. In practice, this is a resonant binary.
            return 0.
        else:
            res=sp.integrate.quad(OmegazdtdS, Sb_min, Sb_max, args=(xi,J,q,S1,S2,r), full_output=1)
        return 2*res[0]


def samplingS(xi,J,q,S1,S2,r):

    '''
    Select a value of S weighted with |dt/dS|. Sampling implemented using the
    cumulative distribution:

    1. select a random number epsilon in [0,1];
    2. find the value of S at which the cumulative probability distribution is
    equal to epsilon.

    The cumulative-distribution method is particualry suitable because the
    probability distribution function |dt/dS| diverges at the extrema Sb_min and
    Sb_max (and is troubling to apply a hit-or-miss approach).

    **Call:**

        S=precession.samplingS(xi,J,q,S1,S2,r)

    **Parameters:**

    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `J`: magnitude of the total angular momentum.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `r`: binary separation.

    **Returns:**

    - `S`: magnitude of the total spin.
    '''

    global flags_q1
    if q==1:
        if flags_q1[8]==False:
            print "[samplingS] Warning q=1: sampling is cos(varphi), not S"
            flags_q1[8]=True

        # If q=1, the limits must be specified in cos(varphi)
        tol=1e-10 # Don't go too close to the actual limits
        Sb_min=-1.+tol
        Sb_max=1.-tol

    else:
        Sb_min,Sb_max=Sb_limits(xi,J,q,S1,S2,r)

    if np.abs(Sb_min-Sb_max)<1e-8: # This typically happen when [Sb_limits] fails in bracketing of the solutions. In practice, this is a resonant binary.
        S_sol=(Sb_min+Sb_max)/2.
    else:
        halfperiod=t_of_S(Sb_min,Sb_max,Sb_min,Sb_max,xi,J,q,S1,S2,r)

        eps= random.uniform(0,1)

        S_sol= sp.optimize.brentq(lambda S: np.abs(t_of_S(Sb_min,S,Sb_min,Sb_max,xi,J,q,S1,S2,r) / halfperiod) - eps, Sb_min, Sb_max) # The brentq algorithm works very well with a monotonic function like the cumulative distribution

    return S_sol




#################################
# PRECESSION-AVERAGED INSPIRAL ##
#################################

def St_limits_comp(kappa,q,S1,S2,u):

    '''
    Auxiliary function, see `precession.St_limits`.

    **Call:**

        St_min,St_max=precession.St_limits_comp(kappa,q,S1,S2,u)

    **Parameters:**

    - `kappa`: rescaling of the total angular momentum to compactify the inspiral domain.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `u`: rescaling of the orbital angular momentum to compactify the inspiral domain.

    **Returns:**

    - `St_min`: minimum value of S from geometrical constraints. This is S_min in our papers.
    - `St_max`: maximum value of S from geometrical constraints. This is S_max in our papers.
    '''

    if u==0:
        St_min=max(np.abs(S1-S2),np.abs(kappa))
        St_max=S1+S2
    else:
        St_min=max(np.abs(S1-S2),np.abs( ((1.+4*kappa*u)**0.5-1)/(2.*u)))
        St_max=min(S1+S2,np.abs( ((1.+4*kappa*u)**0.5+1)/(2.*u)))

    return float(St_min),float(St_max)


def xi_plus_comp(S,kappa,q,S1,S2,u):

    '''
    Auxiliary function, see `precession.xi_plus`.

    **Call:**

        xi=precession.xi_plus_comp(S,kappa,q,S1,S2,u)

    **Parameters:**

    - `S`: magnitude of the total spin.
    - `kappa`: rescaling of the total angular momentum to compactify the inspiral domain.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `u`: rescaling of the orbital angular momentum to compactify the inspiral domain.

    **Returns:**

    - `xi`: projection of the effective spin along the orbital angular momentum.
    '''

    t1= kappa-u*S**2
    t2=S**2*(1.+q)**2-(S1**2-S2**2)*(1.-q**2)
    if S in St_limits_comp(kappa,q,S1,S2,u): #if you're on the limits, the second bit must be zero
        t3=t4=t5=0
    else:
        t3=(1.-q**2)*(S**2 - kappa**2-u**2*S**4+2.*u*kappa*S**2)**0.5
        t4=((S1+S2)**2-S**2)**.5
        t5=(S**2-(S1-S2)**2)**.5
    t6= 2.*q*M**2*S**2
    return (t1*t2 + t3*t4*t5)/t6


def xi_minus_comp(S,kappa,q,S1,S2,u):

    '''
    Auxiliary function, see `precession.xi_minus`.

    **Call:**

        xi=precession.xi_minus_comp(S,kappa,q,S1,S2,u)

    **Parameters:**

    - `S`: magnitude of the total spin.
    - `kappa`: rescaling of the total angular momentum to compactify the inspiral domain.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `u`: rescaling of the orbital angular momentum to compactify the inspiral domain.

    **Returns:**

    - `xi`: projection of the effective spin along the orbital angular momentum.
    '''

    t1= kappa-u*S**2
    t2=S**2*(1.+q)**2-(S1**2-S2**2)*(1.-q**2)
    if S in St_limits_comp(kappa,q,S1,S2,u): #if you're on the limits, the second bit must be zero
        t3=t4=t5=0
    else:
        t3=(1.-q**2)*(S**2 - kappa**2-u**2*S**4+2.*u*kappa*S**2)**0.5
        t4=((S1+S2)**2-S**2)**.5
        t5=(S**2-(S1-S2)**2)**.5
    t6= 2.*q*M**2*S**2
    return (t1*t2 - t3*t4*t5)/t6


def Sb_limits_comp(xi,kappa,q,S1,S2,u):

    '''
    Auxiliary function, see `precession.Sb_limits`.

    **Call:**

        Sb_min,Sb_max=precession.Sb_limits_comp(xi,kappa,q,S1,S2,u)

    **Parameters:**

    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `kappa`: rescaling of the total angular momentum to compactify the inspiral domain.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `u`: rescaling of the orbital angular momentum to compactify the inspiral domain.

    **Returns:**

    - `Sb_min`: minimum value of S from geometrical constraints. This is S- in our papers.
    - `Sb_max`: maximum value of S from geometrical constraints. This is S+ in our papers.
    '''

    global flags_q1
    if q==1:
        if u==0:
            if flags_q1[9]==False:
                print "[Sb_limits_comp] Warning q=1,u=0: input for kappa means S"
                flags_q1[9]=True
            Sb_both=kappa
        else:
           Sb_both=np.sqrt(kappa/u-0.5*xi*M**2/u)
        return Sb_both,Sb_both

    St_min,St_max=St_limits_comp(kappa,q,S1,S2,u)
    xi_low=xi_minus_comp(St_min,kappa,q,S1,S2,u)
    xi_up=xi_minus_comp(St_max,kappa,q,S1,S2,u)

    #Debug option: print the parameter-space region of the initial guess
    whereareyou=False

    # Both roots on xi_plus. Split the interval first
    if xi > xi_low and xi > xi_up:

        if whereareyou:
            print "[Sb_limits_comp] Both roots on xi_plus"

        resmax= sp.optimize.fminbound(lambda S: -1.*xi_plus_comp(S,kappa,q,S1,S2,u), St_min, St_max,full_output=1)
        S_up=resmax[0]
        xi_max=-1.*resmax[1]
        if xi_max<xi: #Braket failed!
            print "[Sb_limits_comp] Braket failed on xi_plus at u="+str(u)+". Assume Sb_min=Sb_max"
            #print  xi_plus(S_up,J,q,S1,S2,r), xi
            Sb_min=S_up
            Sb_max=S_up
        else: #Braket succeeded!

            Sb_min= sp.optimize.brentq(lambda S: xi_plus_comp(S,kappa,q,S1,S2,u)-xi, St_min, S_up)
            Sb_max= sp.optimize.brentq(lambda S: xi_plus_comp(S,kappa,q,S1,S2,u)-xi, S_up, St_max)

    # Both roots on xi_minus. Split the interval first
    elif xi < xi_low and xi < xi_up:

        if whereareyou:
            print "[Sb_limits_comp] Both roots on xi_minus"

        resmin= sp.optimize.fminbound(lambda S: xi_minus_comp(S,kappa,q,S1,S2,u), St_min, St_max,full_output=1)
        S_low=resmin[0]
        xi_min=resmin[1]

        if xi_min>xi: #Braket failed!
            print "[Sb_limits_comp] Braket failed on xi_minus at u="+str(u)+". Assume Sb_min=Sb_max"
            Sb_min=S_low
            Sb_max=S_low
        else: #Braket succeeded!
            Sb_min= sp.optimize.brentq(lambda S: xi_minus_comp(S,kappa,q,S1,S2,u)-xi, St_min, S_low)
            Sb_max= sp.optimize.brentq(lambda S: xi_minus_comp(S,kappa,q,S1,S2,u)-xi, S_low, St_max)

    # One root on xi_plus and the other one on xi_plus. No additional maximization is neeeded
    elif xi >= xi_low and xi <= xi_up:

        if whereareyou:
            print "[Sb_limits_comp] Sb_min on xi_plus, Sb_max on xi_minus"

        Sb_min= sp.optimize.brentq(lambda S: xi_plus_comp(S,kappa,q,S1,S2,u)-xi, St_min, St_max)
        Sb_max= sp.optimize.brentq(lambda S: xi_minus_comp(S,kappa,q,S1,S2,u)-xi, St_min, St_max)
    elif xi <= xi_low and xi >= xi_up:

        if whereareyou:
            print "[Sb_limits_comp] Sb_min on xi_minus, Sb_max on xi_plus"

        Sb_min= sp.optimize.brentq(lambda S: xi_minus_comp(S,kappa,q,S1,S2,u)-xi, St_min, St_max)
        Sb_max= sp.optimize.brentq(lambda S: xi_plus_comp(S,kappa,q,S1,S2,u)-xi, St_min, St_max)

    else:
        print "[Sb_limits_comp] Erorr in case selection"
        print "xi=", xi
        print "xi(stmin)=", xi_low
        print "xi(stmax)=", xi_up
        print "Stmin=", St_min
        print "Stmax", St_max
        print "kappa=", kappa
        print "u=", u
        assert False, "[Sb_limits_comp] Erorr in case selection"

    btol=1e-8 # Never go to close to the actual limits, because everything blows up there
    Sb_min+=btol
    Sb_max-=btol

    if whereareyou:
        print "[Sb_limits_comp] Results:", Sb_min,Sb_max


    if Sb_min>Sb_max: # This may happen (numerically) if they're too close to each other. Assume they're the same.
        return np.mean([Sb_min,Sb_max]), np.mean([Sb_min,Sb_max])
    else:
        return Sb_min, Sb_max


def S3sines_comp(S,xi,kappa,q,S1,S2,u):

    '''
    Auxiliary function, see `precession.dkappadu`.

    **Call:**

        denominator=precession.S3sines_comp(S,xi,kappa,q,S1,S2,u)

    **Parameters:**

    - `S`: magnitude of the total spin.
    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `kappa`: rescaling of the total angular momentum to compactify the inspiral domain.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `u`: rescaling of the orbital angular momentum to compactify the inspiral domain.

    **Returns:**

    - `denominator`: denominator in integrand `precession.dkappapu`.
    '''

    ct1= ( kappa - u*S**2  - q*M**2*xi/(1.+q) )/((1.-q)*S1)
    ct2= q*( -kappa + u*S**2  + M**2*xi/(1.+q) )/((1.-q)*S2)
    ct12=(S**2-S1**2-S2**2)/(2.*S1*S2)
    t3=max( (np.abs(1.-ct1**2-ct2**2-ct12**2 +2.*ct1*ct2*ct12))**.5, 1e-20) # I know abs is dirty, but does the job
    if t3==0: # prevent occasional crash
        t3=1e-20
    return S**3/t3


def Ssines_comp(S,xi,kappa,q,S1,S2,u):

    '''
    Auxiliary function, see `precession.dkappadu`.

    **Call:**

        numerator=precession.Ssines_comp(S,xi,kappa,q,S1,S2,u)

    **Parameters:**

    - `S`: magnitude of the total spin.
    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `kappa`: rescaling of the total angular momentum to compactify the inspiral domain.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `u`: rescaling of the orbital angular momentum to compactify the inspiral domain.

    **Returns:**

    - `numerator`: numerator in integrand `precession.dkappapu`.
    '''

    ct1= ( kappa - u*S**2  - q*M**2*xi/(1.+q) )/((1.-q)*S1)
    ct2= q*( -kappa + u*S**2  + M**2*xi/(1.+q) )/((1.-q)*S2)
    ct12=(S**2-S1**2-S2**2)/(2.*S1*S2)
    t3=max( (np.abs(1.-ct1**2-ct2**2-ct12**2 +2.*ct1*ct2*ct12))**.5, 1e-20) # I know abs is dirty, but does the job
    return S/t3


def dkappadu(kappa,u,xi,q,S1,S2):

    '''
    Inspiral ODE to perform precession-averaged inspiral: dkappa/du = S^2_pre.
    We use variables kappa and u (rather than J and L, see `precession.dJdL`)
    because this formulation naturally allows for integration from infinitely
    large separations, i.e. u=0. This function is only the actual equation, not
    the ODE solver.

    **Call:**

        dkappadu=precession.dkappadu(kappa,u,xi,q,S1,S2)

    **Parameters:**

    - `kappa`: rescaling of the total angular momentum to compactify the inspiral domain.
    - `u`: rescaling of the orbital angular momentum to compactify the inspiral domain.
    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.

    **Returns:**

    - `dkappadu`: precession-averaged derivative of kappa with respect to u.
    '''

    dkappadu_debug=False #Debug option
    if dkappadu_debug:
        print "[dkappadu] ODE int: u="+str(u)+"\t\tkappa="+str(float(kappa))

    Sb_min,Sb_max = Sb_limits_comp(xi,kappa,q,S1,S2,u)

    if np.abs(Sb_min-Sb_max)<1e-8:
        if dkappadu_debug:
            print "[dkappadu] Warning. Applyting analytical approximation. u=",u
        return (np.mean([Sb_min,Sb_max]))**2
    else:
        up=sp.integrate.quad(S3sines_comp, Sb_min, Sb_max, args=(xi,kappa,q,S1,S2,u), full_output=1)
        down=sp.integrate.quad(Ssines_comp , Sb_min, Sb_max, args=(xi,kappa,q,S1,S2,u), full_output=1)
        return up[0]/down[0]


def dJdr(J,r,xi,q,S1,S2):

    '''
    Inspiral ODE describing the evolution of the magnitude of the total angular
    momentum vs. the separation r. This function is NOT used by the ODE solvers
    (see `precession.dkappadu`).

    **Call:**

        dJdr=precession.dJdr(J,r,xi,q,S1,S2)

    **Parameters:**

    - `J`: magnitude of the total angular momentum.
    - `r`: binary separation.
    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.

    **Returns:**

    - `dJdr`: precession-averaged derivative of J with respect to r.
    '''

    L=(q/(1.+q)**2)*(r*M**3)**.5
    kappa=(J**2-L**2)/(2.*L)
    u=1./(2.*L)
    S2pre=dkappadu(kappa,u,xi,q,S1,S2)
    dJdL=(1./(2.*L*J))*(J**2+L**2-S2pre)
    dLdr=L/(2.*r)

    return dJdL*dLdr


def dJdL(J,r,xi,q,S1,S2):

    '''
    Inspiral ODE describing the evolution of the magnitude of the total angular
    momentum vs. the separation r. This function is NOT used by the ODE solvers
    (see `precession.dkappadu`).

    **Call:**

        dJdL=precession.dJdL(J,r,xi,q,S1,S2)

    **Parameters:**

    - `J`: magnitude of the total angular momentum.
    - `r`: binary separation.
    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.

    **Returns:**

    - `dJdL`: precession-averaged derivative of J with respect to L.
    '''

    L=(q/(1.+q)**2)*(r*M**3)**.5
    kappa=(J**2-L**2)/(2.*L)
    u=1./(2.*L)
    S2pre=dkappadu(kappa,u,xi,q,S1,S2)
    dJdL=(1./(2.*L*J))*(J**2+L**2-S2pre)
    return dJdL


def Jofr(xi,J_initial,r_vals,q,S1,S2):

    '''
    Single integration of the dJ/dL equation to perfom precession-averaged
    inspiral. Input/output are provided in J and r, but the internal integrator
    uses kappa and u (see `precession.dkappadu`). Integration is performed using
    scipy's `odeint`.

    This function integrates to/from FINITE separations only.

    It takes the desired output separations r_vals, and the intial condition for
    the total angular momentum J_initial. The latter must be consistent with the
    initial separation (i.e. r_vals[0]) and the value of xi; an error is raised
    in case of inconsistencies. It doesn't matter if you integrate from large to
    small separations of the other way round, as long as J_initial is consistent
    with r_vals[0]. It returns a vector with the values of J at each input
    separation, the first item being just the initial condition.

    We recommend to use this function through the wrapper `precession.evolve_J`
    provided.

    **Call:**

        J_vals=precession.Jofr(xi,J_initial,r_vals,q,S1,S2)

    **Parameters:**

    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `J_initial`: initial condition for numerical integration.
    - `r_vals`: binary separation (array).
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.

    **Returns:**

    - `J_vals`: magnitude of the total angular momentum (array).
    '''

    L_vals=[ (q/(1.+q)**2)*(r*M**3)**.5 for r in r_vals]
    kappa_initial= (J_initial**2 - (L_vals[0])**2) / (2.*L_vals[0])
    u_vals=[ 1./(2.*L) for L in L_vals]

    # Analytical solution for q=1. S is constant on the radiation-reaction time
    if q==1:
        L_initial=L_vals[0]
        Ssq=J_initial**2-L_initial**2-xi*L_initial*M**2
        J_vals=[np.sqrt(Ssq+((q/(1.+q)**2)*(r*M**3)**.5)**2+xi*M**2*((q/(1.+q)**2)*(r*M**3)**.5)) for r in r_vals]

    # Numerical integration
    else:
        #sing = [ M*((1.+q)**2*(S1+S2)/(q*M**2))**2 ,  M*((1.+q)**2*(S1-S2)/(q*M**2))**2 ] # Expected singularities. Not needed
        # Increase h0 to prevent occasional slowing down of the integration
        res =integrate.odeint(dkappadu, kappa_initial, u_vals, args=(xi,q,S1,S2), mxstep=50000, full_output=0, printmessg=0)#,h0=0.001)#,tcrit=sing)
        kappa_vals=[x[0] for x in res]
        J_vals= [ (k*2.*L + L**2)**0.5 for k,L in zip(kappa_vals,L_vals)]
    return J_vals


def Jofr_checkpoint(xi,J_initial,r_vals,q,S1,S2):

    '''
    Auxiliary function, see `precession.evolve_J`.

    **Call:**

        savename=precession.Jofr_checkpoint(xi,J_initial,r_vals,q,S1,S2)

    **Parameters:**

    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `J_initial`: initial condition for numerical integration.
    - `r_vals`: binary separation (array).
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.

    **Returns:**

    - `savename`: checkpoint filename.
    '''

    os.system("mkdir -p "+storedir)
    savename= storedir+"/evJ_"+'_'.join([str(x) for x in (xi,J_initial,max(r_vals),min(r_vals),len(r_vals),q,S1,S2)])+".dat"

    if not os.path.isfile(savename):
        print "[evolve_J] Transferring binary. Output:", savename
        outfilesave = open(savename,"w",0)

        J_vals= Jofr(xi,J_initial,r_vals,q,S1,S2)

        for J_f,r_f in zip(J_vals,r_vals):
            outfilesave.write(str(r_f)+" "+str(J_f)+"\n")
        outfilesave.close()

    #else:
    #    print "[evolve_J] Skipping. Output:", savename

    return savename


def evolve_J(xi_vals,J_vals,r_vals,q_vals,S1_vals,S2_vals):

    '''
    Wrapper of `precession.Jofr` to enable parallelization through the python
    `parmap` module; the number of available cores can be specified using the
    integer global variable `precession.CPUs` (all available cores will be used
    by default). Evolve a sequence of binaries with the different q, S1, S2, xi
    and initial values of J and save outputs at the SAME r_vals. Output is a 2D
    array, where e.g. J_vals[0] is the first binary (1D array at all output
    separations) and J_vals[0][0] is the first binary at the first output
    separation (this is a scalar). We strongly reccommend using this function,
    even for a single binary.

    Checkpointing is implemented: results are stored in `precession.storedir`.

    **Call:**

        Jf_vals=precession.evolve_J(xi_vals,Ji_vals,r_vals,q_vals,S1_vals,S2_vals)

    **Parameters:**

    - `xi_vals`: projection of the effective spin along the orbital angular momentum (array).
    - `Ji_vals`: initial condition for numerical integration (array).
    - `r_vals`: binary separation (array).
    - `q_vals`: binary mass ratio. Must be q<=1 (array).
    - `S1_vals`: spin magnitude of the primary BH (array).
    - `S2_vals`: spin magnitude of the secondary BH (array).

    **Returns:**

    - `Jf_vals`: magnitude of the total angular momentum (2D array).
    '''

    global CPUs
    single_flag=False

    try: # Convert float to array if you're evolving just one binary
        len(q_vals)
    except:
        xi_vals=[xi_vals]
        J_vals=[J_vals]
        q_vals=[q_vals]
        S1_vals=[S1_vals]
        S2_vals=[S2_vals]
        single_flag=True
    try: # Set default
        CPUs
    except:
        CPUs=0
        print "[evolve_J] Default parallel computation"
    # Parallelization.
    if CPUs==0: # Run on all cpus on the current machine! (default option)
        filelist=parmap.starmap(Jofr_checkpoint, zip(xi_vals,J_vals, [r_vals for i in range(len(q_vals))],q_vals,S1_vals,S2_vals), pm_parallel=True)
    elif CPUs==1: # 1 cpus done by explicitely switching parallelization off
        filelist=parmap.starmap(Jofr_checkpoint, zip(xi_vals,J_vals, [r_vals for i in range(len(q_vals))],q_vals,S1_vals,S2_vals),pm_parallel=False)
    else: # Run on a given number of CPUs
        filelist=parmap.starmap(Jofr_checkpoint, zip(xi_vals,J_vals, [r_vals for i in range(len(q_vals))],q_vals,S1_vals,S2_vals),pm_processes=CPUs)

    J_fvals=[]
    for index, file in enumerate(filelist):
        print "[evolve_J] Reading:", index, file
        dummy,J_f= np.loadtxt(file,unpack=True)

        J_fvals.append(J_f)

    if single_flag==True:
        return J_fvals[0]
    else:
        return J_fvals


def evolve_angles_single(theta1_i,theta2_i,deltaphi_i,r_vals,q,S1,S2):

    '''
    Auxiliary function, see `precession.evolve_angles`.

    **Call:**

        savename=precession.evolve_angles(theta1_i,theta2_i,deltaphi_i,r_vals,q,S1,S2)

    **Parameters:**

    - `theta1_i`: initial condition for theta1.
    - `theta2_i`: initial condition for theta2
    - `deltaphi_i`: initial condition for deltaphi.
    - `r_vals`: binary separation (array).
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.

    **Returns:**

    - `savename`: checkpoint filename.
    '''


    os.system("mkdir -p "+storedir)
    savename= storedir+"/eva_"+'_'.join([str(x) for x in (theta1_i,theta2_i,deltaphi_i,max(r_vals),min(r_vals),len(r_vals),q,S1,S2)])+".dat"

    if not os.path.isfile(savename):
        print "[evolve_angles] Transferring binary. Output:", savename
        outfilesave = open(savename,"w",0)

        # Step 1. Get xi and J for each intial angle. Forget S
        xi,J_i,dummy= from_the_angles(theta1_i,theta2_i,deltaphi_i,q,S1,S2,r_vals[0])

        # Step 2. Evolve binaires with the dJ/dr ODE.
        J_vals= Jofr(xi,J_i,r_vals,q,S1,S2)

        for J_f,r_f in zip(J_vals,r_vals):
            # Step 3. Select S at the final separation with weight dt/dS
            S_f=samplingS(xi,J_f,q,S1,S2,r_f)
            # Step 4. Back to theta1, theta2, deltaphi
            theta1_f,theta2_f,deltaphi_f,dummy= parametric_angles(S_f,J_f,xi,q,S1,S2,r_f)
            deltaphi_f*=random.choice([-1., 1.])
            # Step 5. Store data
            #outfilesave.write(str(r_f)+" "+str(xi)+" "+str(J_f)+" "+str(S_f)+" "+str(theta1_f)+" "+str(theta2_f)+" "+str(deltaphi_f)+"\n")
            outfilesave.write(str(r_f)+" "+str(theta1_f)+" "+str(theta2_f)+" "+str(deltaphi_f)+"\n")
        outfilesave.close()

    #else:
    #    print "[evolve_angles] Skipping. Output:", savename

    return savename


def evolve_angles(theta1_vals,theta2_vals,deltaphi_vals,r_vals,q_vals,S1_vals,S2_vals):

    '''
    Binary evolution from the angles theta1, theta2 and deltaphi as initial data
    (to/from FINITE separations only). This is our so-called *transfer
    function*. The transfer procedure is implemented as follows:

    1. Convert theta1,theta2, deltaphi into J, xi and S.
    2. Forget S and evolve J.
    3. Resample S at the final separation according to dt/dS.
    4. Covert J, xi and S back to theta1, theta2 and deltaphi; assign a random
    sign to deltaphi.

    Parallelization through the python `parmap` module is implemented; the
    number of available cores can be specified using the integer global variable
    `precession.CPUs` (all available cores will be used by default). Evolve a
    sequence of binaries with different values of q, S1,S2, theta1, theta2,
    deltaphi (assumed to be specified at r_vals[0]) and save outputs at SAME
    separations r_vals. Outputs are 2D arrays, where e.g theta1_fvals[0] is the
    first binary (1D array at all output separations) and theta1_fvals[0][0] is
    the first binary at the first output separation (this is a scalar).

    Checkpointing is implemented: results are stored in `precession.storedir`.

    **Call:**

        theta1f_vals,theta2f_vals,deltaphif_vals=precession.evolve_angles(theta1i_vals,theta2i_vals,deltaphii_vals,r_vals,q_vals,S1_vals,S2_vals)

    **Parameters:**

    - `theta1i_vals`: initial condition for theta1 (array).
    - `theta2i_vals`: initial condition for theta2 (array).
    - `deltaphii_vals`: initial condition for deltaphi (array).
    - `r_vals`: binary separation (array).
    - `q_vals`: binary mass ratio. Must be q<=1 (array).
    - `S1_vals`: spin magnitude of the primary BH (array).
    - `S2_vals`: spin magnitude of the secondary BH (array).

    **Returns:**

    - `theta1f_vals`: solutions for theta1 (2D array).
    - `theta2f_vals`: solutions for theta2 (2D array).
    - `deltaphif_vals`: solutions for deltaphi (2D array).
    '''

    global CPUs
    single_flag=False

    try: # Convert float to array if you're evolving just one binary
        len(q_vals)
    except:
        single_flag=True
        theta1_vals=[theta1_vals]
        theta2_vals=[theta2_vals]
        deltaphi_vals=[deltaphi_vals]
        q_vals=[q_vals]
        S1_vals=[S1_vals]
        S2_vals=[S2_vals]
    try: # Set default
        CPUs
    except:
        CPUs=0
        print "[evolve_angles] Default parallel computation"

    loopflag=True
    while loopflag: # Restart is some of the cores crashed. This happend if you run too many binaries on too many different machines. Nevermind, trash the file and do it again.
        loopflag=False

        #Parallelization
        if CPUs==0: #Run on all cpus on the current machine! (default option)
            filelist=parmap.starmap(evolve_angles_single, zip(theta1_vals,theta2_vals,deltaphi_vals,[r_vals for i in range(len(q_vals))],q_vals,S1_vals,S2_vals),pm_parallel=True)
        elif CPUs==1: #1 cpus done by explicitely removing parallelization
            filelist=parmap.starmap(evolve_angles_single, zip(theta1_vals,theta2_vals,deltaphi_vals,[r_vals for i in range(len(q_vals))],q_vals,S1_vals,S2_vals),pm_parallel=False)
        else: # Run on a given number of CPUs
            filelist=parmap.starmap(evolve_angles_single, zip(theta1_vals,theta2_vals,deltaphi_vals,[r_vals for i in range(len(q_vals))],q_vals,S1_vals,S2_vals),pm_processes=CPUs)

        theta1_fvals=[]
        theta2_fvals=[]
        deltaphi_fvals=[]
        for index, file in enumerate(filelist):
            print "[evolve_angles] Reading:", index, file
            numlines=sum(1 for line in open(file))
            if numlines!=0 and numlines!=len(r_vals): # Restart if core(s) crashed
                print "[evolve_angles] Error on file", file,". Jobs are being restarted!"
                os.system("rm "+file)
                loopflag=True

            else:
                dummy,theta1_f,theta2_f,deltaphi_f= np.loadtxt(file,unpack=True)
                theta1_fvals.append(theta1_f)
                theta2_fvals.append(theta2_f)
                deltaphi_fvals.append(deltaphi_f)

    if single_flag==True:
        return theta1_fvals[0], theta2_fvals[0], deltaphi_fvals[0]
    else:
        return theta1_fvals, theta2_fvals, deltaphi_fvals


def Jofr_infinity(xi,kappa_inf,r_vals,q,S1,S2):

    '''
    Single integration of the dJ/dL equation to perfom precession-averaged
    inspiral. Input/output are provided in J and r, but the internal integrator
    uses kappa and u (see `precession.dkappadu`). Integration is performed using
    scipy's `odeint`.

    This function integrates FROM INFINITE separation (u=0) only.

    The latter must be consistent with `precession.kappainf_lim`; an error is
    raised in case of inconsistencies. It assume that the array r_vals is sorted
    in reversed order, i.e. that you are integrating from large to small
    separations. It returns a vector with the values of J at each input
    separation. The initial condition is NOT returned by this function (unlike
    the `precession.Jofr` for integrations to/from finite separations). If q=1,
    kappa_inf is degenerate with xi: the required initial condition is assumed
    to be S (which is constant).

    We recommend to use this function through the wrapper
    `precession.evolve_J_infinity` provided.

    **Call:**

        J_vals=precession.Jofr_infinity(xi,kappa_inf,r_vals,q,S1,S2)

    **Parameters:**

    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `kappa_inf`: asymptotic value of kappa at large separations.
    - `r_vals`: binary separation (array).
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.

    **Returns:**

    - `J_vals`: magnitude of the total angular momentum (array).
    '''

    # ASSUMES r_vals is sorted in reversed order!!!
    L_vals=[ (q/(1.+q)**2)*(r*M**3)**.5 for r in r_vals]

    global flags_q1
    if q==1:
        if flags_q1[10]==False:
            print "[Jofr_infinity] Warning q=1: required intial condition is S, not kappa_inf."
            flags_q1[10]=True # Suppress future warnings
        S=kappa_inf
        J_vals=[np.sqrt(L**2+S**2+xi*L*M**2) for L in L_vals]
    else:
        u_vals=[ 1./(2.*L) for L in L_vals]
        u_vals.insert(0, 0.) # Add initial condition, r=inifinty u=0

        # Numerical integration from u=0
        # Increase h0 to prevent occasional slowing down of the integration
        res =integrate.odeint(dkappadu, kappa_inf, u_vals, args=(xi,q,S1,S2), mxstep=50000, full_output=0, printmessg=0,h0=2e-4)

        kappa_vals=[x[0] for x in res][1:] # Remove initial condition (not present in r_vals...)
        J_vals= [ (k*2.*L + L**2)**0.5 for k,L in zip(kappa_vals,L_vals)]

    return J_vals


def Jofr_infinity_checkpoint(xi,kappa_inf,r_vals,q,S1,S2):

    '''
    Auxiliary function, see `precession.evolve_J_infinity`.

    **Call:**

        savename=precession.Jofr_infinity_checkpoint(xi,kappa_inf,r_vals,q,S1,S2)

    **Parameters:**

    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `kappa_inf`: asymptotic value of kappa at large separations.
    - `r_vals`: binary separation (array).
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.

    **Returns:**

    - `savename`: checkpoint filename.
    '''

    os.system("mkdir -p "+storedir)
    savename= storedir+"/evJinf_"+'_'.join([str(x) for x in (xi,kappa_inf,max(r_vals),min(r_vals),len(r_vals),q,S1,S2)])+".dat"

    if not os.path.isfile(savename):
        print "[evolve_J_infinity] Transferring binary. Output:", savename
        outfilesave = open(savename,"w",0)

        J_vals= Jofr_infinity(xi,kappa_inf,r_vals,q,S1,S2)

        for J_f,r_f in zip(J_vals,r_vals):
            outfilesave.write(str(r_f)+" "+str(J_f)+"\n")
        outfilesave.close()

    #else:
    #    print "[evolve_J_infinity] Skipping. Output:", savename

    return savename


def evolve_J_infinity(xi_vals,kappainf_vals,r_vals,q_vals,S1_vals,S2_vals):

    '''
    Wrapper of `precession.Jofr_infinity` to enable parallelization through the
    python `parmap` module; the number of available cores can be specified using
    the integer global variable `precession.CPUs` (all available cores will be
    used by default). Evolve a sequence of binaries with the different q, S1,
    S2, xi and initial values of J and save outputs at the SAME separations
    r_vals. Output is a 2D array, where e.g. J_vals[0] is the first binary (1D
    array at all output separations) and J_vals[0][0] is the first binary at the
    first output separation (this is a scalar). We strongly reccommend using
    this function, even for a single binary.

    Checkpointing is implemented: results are stored in `precession.storedir`.

    **Call:**

        Jf_vals=precession.evolve_J_infinity(xi_vals,kappainf_vals,r_vals,q_vals,S1_vals,S2_vals)

    **Parameters:**

    - `xi_vals`: projection of the effective spin along the orbital angular momentum (array).
    - `kappainf_vals`: asymptotic value of kappa at large separations (array).
    - `r_vals`: binary separation (array).
    - `q_vals`: binary mass ratio. Must be q<=1 (array).
    - `S1_vals`: spin magnitude of the primary BH (array).
    - `S2_vals`: spin magnitude of the secondary BH (array).

    **Returns:**

    - `Jf_vals`: magnitude of the total angular momentum (2D array).
    '''

    global CPUs

    single_flag=False
    try: #Convert float to array if you're evolving just one binary
        len(q_vals)
    except:
        xi_vals=[xi_vals]
        kappainf_vals=[kappainf_vals]
        q_vals=[q_vals]
        S1_vals=[S1_vals]
        S2_vals=[S2_vals]
        single_flag=True
    try: # Set default
        CPUs
    except:
        CPUs=0
        print "[evolve_J_infinity] Default parallel computation"
    # Parallelization
    if CPUs==0: # Run on all cpus on the current machine! (default option)
        filelist=parmap.starmap(Jofr_infinity_checkpoint, zip(xi_vals,kappainf_vals,[r_vals for i in range(len(q_vals))],q_vals,S1_vals,S2_vals),pm_parallel=True)
    elif CPUs==1: # 1 cpus done by explicitely removing parallelization
        filelist=parmap.starmap(Jofr_infinity_checkpoint, zip(xi_vals,kappainf_vals,[r_vals for i in range(len(q_vals))],q_vals,S1_vals,S2_vals),pm_parallel=False)
    else: # Run on a given number of CPUs
        filelist=parmap.starmap(Jofr_infinity_checkpoint, zip(xi_vals,kappainf_vals,[r_vals for i in range(len(q_vals))],q_vals,S1_vals,S2_vals),pm_processes=CPUs)

    J_fvals=[]
    for index, file in enumerate(filelist):
        print "[evolve_J_infinity] Reading:", index, file
        dummy,J_f= np.loadtxt(file,unpack=True)

        J_fvals.append(J_f)

    if single_flag==True:
        return J_fvals[0]
    else:
        return J_fvals


def kappa_backwards(xi,J,r,q,S1,S2):

    '''
    Single integration of the dJ/dL equation to perfom precession-averaged
    inspiral. Input/output are provided in J and r, but the internal integrator
    uses kappa and u (see `precession.dkappadu`). Integration is performed using
    scipy's `odeint`.

    This function integrates from some finite separation TO INFINITE separation
    (u=0) only.

    The initial binary is specified at the input separation r through J and xi
    (S not needed). The binary is evolved backwards to r=infinity (u=0) and the
    asymptotic value kappa_inf is returned. If q=1, kappa_inf is degenerate with
    xi: the constant value of S is returned instead.

    We recommend to use this function through the wrapper
    `precession.evolve_J_backwards` provided.

    **Call:**

        kappa_inf=precession.kappa_backwards(xi,J,r,q,S1,S2)

    **Parameters:**

    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `J`: magnitude of the total angular momentum.
    - `r`: binary separation.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.

    **Returns:**

    - `kappa_inf`: asymptotic value of kappa at large separations.
    '''

    L=(q/(1.+q)**2)*(r*M**3)**.5

    global flags_q1
    if q==1:
        if flags_q1[11]==False:
            print "[kappa_backwards] Warning q=1: sensible output is S, not kappa_inf."
            flags_q1[11]=True # Suppress future warnings
        S=np.sqrt(J**2-L**2-xi*L*M**2)
        return S

    else:
        u=1./(2.*L)
        kappa= (J**2 - L**2) / (2.*L)
        u_vals=[u,0.]# Add final condition, r=inifinty u=0
        # Numerical integration to u=0
        res =integrate.odeint(dkappadu, kappa, u_vals, args=(xi,q,S1,S2), mxstep=50000, full_output=0, printmessg=0)#,tcrit=sing)
        kappa_inf=[x[0] for x in res][-1]
        return kappa_inf


def kappa_backwards_checkpoint(xi,J,r,q,S1,S2):

    '''
    Auxiliary function, see `precession.evolve_J_backwards`.

    **Call:**

        savename=precession.kappa_backwards_checkpoint(xi,J,r,q,S1,S2)

    **Parameters:**

    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `J`: magnitude of the total angular momentum.
    - `r`: binary separation.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.

    **Returns:**

    - `savename`: checkpoint filename.
    '''

    os.system("mkdir -p "+storedir)
    savename= storedir+"/evback"+'_'.join([str(x) for x in (xi,J,r,q,S1,S2)])+".dat"

    if not os.path.isfile(savename):
        print "[evolve_J_backwards] Transferring binary. Output:", savename
        outfilesave = open(savename,"w",0)

        kappa_inf=kappa_backwards(xi,J,r,q,S1,S2)
        outfilesave.write(str(kappa_inf))
        outfilesave.close()

    #else:
    #    print "[evolve_J_infinity] Skipping. Output:", savename

    return savename


def evolve_J_backwards(xi_vals,J_vals,r,q_vals,S1_vals,S2_vals):

    '''
    Wrapper of `precession.kappa_backwards` to enable parallelization through
    the python `parmap` module; the number of available cores can be specified
    using the integer global variable `precession.CPUs` (all available cores
    will be used by default). Evolve a sequence of binaries with the different
    q, S1,S2, xi and kappa_inf from the SAME separation r.

    Checkpointing is implemented: results are stored in `precession.storedir`.

    **Call:**

        kappainf_vals=precession.evolve_J_backwards(xi_vals,J_vals,r,q_vals,S1_vals,S2_vals)

    **Parameters:**

    - `xi_vals`: projection of the effective spin along the orbital angular momentum (array).
    - `J`: magnitude of the total angular momentum (array).
    - `r`: binary separation.
    - `q_vals`: binary mass ratio. Must be q<=1 (array).
    - `S1_vals`: spin magnitude of the primary BH (array).
    - `S2_vals`: spin magnitude of the secondary BH (array).

    **Returns:**

    - `kappainf_vals`: asymptotic value of kappa at large separations (array).
    '''

    global CPUs

    flag=False
    try: #Convert float to array, if you're evolving just one binary
        len(q_vals)
    except:
        xi_vals=[xi_vals]
        J_vals=[J_vals]
        q_vals=[q_vals]
        S1_vals=[S1_vals]
        S2_vals=[S2_vals]
        flag=True
    try:
        CPUs
    except:
        CPUs=0
        print "[evolve_J_backwards] Default parallel computation"
    #Parallelization... python is cool indeed
    if CPUs==0: #Run on all cpus on the current machine! (default option)
        filelist=parmap.starmap(kappa_backwards_checkpoint, zip(xi_vals,J_vals,[r for i in range(len(q_vals))],q_vals,S1_vals,S2_vals),pm_parallel=True)
    elif CPUs==1: #1 cpus done by explicitely removing parallelization
        filelist=parmap.starmap(kappa_backwards_checkpoint, zip(xi_vals,J_vals,[r for i in range(len(q_vals))],q_vals,S1_vals,S2_vals),pm_parallel=False)
    else: # Run on a given number of CPUs
        filelist=parmap.starmap(kappa_backwards_checkpoint, zip(xi_vals,J_vals,[r for i in range(len(q_vals))],q_vals,S1_vals,S2_vals),pm_processes=CPUs)

    kappainf_vals=[]
    for index, file in enumerate(filelist):
        print "[evolve_J_backwards] Reading:", index, file
        kappa_inf= np.loadtxt(file,unpack=True)
        kappainf_vals.append(kappa_inf)

    if flag==True:
        return kappainf_vals[0]
    else:
        return kappainf_vals




#################################
#### ORBIT-AVERAGED INSPIRAL ####
#################################


def orbav_eqs(allvars,v,q,S1,S2,eta,m1,m2,chi1,chi2,time=False):

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

        allders=precession.orbav_eqs(allvars,v,q,S1,S2,eta,m1,m2,chi1,chi2,time=False)

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

    # Read variables in
    Lhx=allvars[0]
    Lhy=allvars[1]
    Lhz=allvars[2]
    S1hx=allvars[3]
    S1hy=allvars[4]
    S1hz=allvars[5]
    S2hx=allvars[6]
    S2hy=allvars[7]
    S2hz=allvars[8]
    if time:
        t=allvars[9]

    # Useful variables
    ct1=(Lhx*S1hx+Lhy*S1hy+Lhz*S1hz)
    ct2=(Lhx*S2hx+Lhy*S2hy+Lhz*S2hz)
    ct12=(S1hx*S2hx+S1hy*S2hy+S1hz*S2hz)

    # Spin precession for S1
    Omega1x= eta*v**5*(2.+3.*q/2.)*Lhx/M  \
            + v**6*(S2*S2hx-3.*S2*ct2*Lhx-3.*q*S1*ct1*Lhx)/(2.*M**3)
    Omega1y= eta*v**5*(2.+3.*q/2.)*Lhy/M  \
            + v**6*(S2*S2hy-3.*S2*ct2*Lhy-3.*q*S1*ct1*Lhy)/(2.*M**3)
    Omega1z= eta*v**5*(2.+3.*q/2.)*Lhz/M  \
            + v**6*(S2*S2hz-3.*S2*ct2*Lhz-3.*q*S1*ct1*Lhz)/(2.*M**3)

    dS1hxdt= Omega1y*S1hz - Omega1z*S1hy
    dS1hydt= Omega1z*S1hx - Omega1x*S1hz
    dS1hzdt= Omega1x*S1hy - Omega1y*S1hx

    # Spin precession for S2
    Omega2x= eta*v**5*(2.+3./(2.*q))*Lhx/M  \
            + v**6*(S1*S1hx-3.*S1*ct1*Lhx-3.*S2*ct2*Lhx/q)/(2.*M**3)
    Omega2y= eta*v**5*(2.+3./(2.*q))*Lhy/M  \
            + v**6*(S1*S1hy-3.*S1*ct1*Lhy-3.*S2*ct2*Lhy/q)/(2.*M**3)
    Omega2z= eta*v**5*(2.+3./(2.*q))*Lhz/M  \
            + v**6*(S1*S1hz-3.*S1*ct1*Lhz-3.*S2*ct2*Lhz/q)/(2.*M**3)

    dS2hxdt= Omega2y*S2hz - Omega2z*S2hy
    dS2hydt= Omega2z*S2hx - Omega2x*S2hz
    dS2hzdt= Omega2x*S2hy - Omega2y*S2hx

    # Conservation of angular momentum
    dLhxdt= -1.*v*(S1*dS1hxdt+S2*dS2hxdt)/(eta*M**2)
    dLhydt= -1.*v*(S1*dS1hydt+S2*dS2hydt)/(eta*M**2)
    dLhzdt= -1.*v*(S1*dS1hzdt+S2*dS2hzdt)/(eta*M**2)

    # Radiation reaction
    quadrupole_formula=False
    if quadrupole_formula:
        dvdt= (32.*eta*v**9/(5.*M))
    else:
        dvdt= (32.*eta*v**9/(5.*M))* ( 1.                               \
            - v**2* (743.+924.*eta)/336.                                \
            + v**3* (4.*np.pi                                           \
                     - chi1*ct1*(113.*m1**2/(12.*M**2) + 25.*eta/4. )   \
                     - chi2*ct2*(113.*m2**2/(12.*M**2) + 25.*eta/4. ))  \
            + v**4* (34103./18144. + 13661.*eta/2016. + 59.*eta**2/18.  \
                     + eta*chi1*chi2* (721.*ct1*ct2 - 247.*ct12) /48.   \
                     + ((m1*chi1/M)**2 * (719.*ct1**2-233.))/96.        \
                     + ((m2*chi2/M)**2 * (719.*ct2**2-233.))/96.)       \
            - v**5* np.pi*(4159.+15876.*eta)/672.                       \
            + v**6* (16447322263./139708800. + 16.*np.pi**2/3.          \
                     -1712.*(0.5772156649+np.log(4.*v))/105.            \
                     +(451.*np.pi**2/48. - 56198689./217728.)*eta       \
                     +541.*eta**2/896. - 5605*eta**3/2592.)             \
            + v**7* np.pi*( -4415./4032. + 358675.*eta/6048.            \
                     + 91495.*eta**2/1512.)                             \
            )

    # Integrate in v, not in time
    dtdv=1./dvdt
    dLhxdv=dLhxdt*dtdv
    dLhydv=dLhydt*dtdv
    dLhzdv=dLhzdt*dtdv
    dS1hxdv=dS1hxdt*dtdv
    dS1hydv=dS1hydt*dtdv
    dS1hzdv=dS1hzdt*dtdv
    dS2hxdv=dS2hxdt*dtdv
    dS2hydv=dS2hydt*dtdv
    dS2hzdv=dS2hzdt*dtdv

    if time:
        return dLhxdv, dLhydv, dLhzdv, dS1hxdv, dS1hydv, dS1hzdv, dS2hxdv, dS2hydv, dS2hzdv , dtdv
    else:
        return dLhxdv, dLhydv, dLhzdv, dS1hxdv, dS1hydv, dS1hzdv, dS2hxdv, dS2hydv, dS2hzdv


def orbav_integrator(J,xi,S,r_vals,q,S1,S2,time=False):

    '''
    Single orbit-averaged integration. Integrate the system of ODEs specified in
    `precession.orbav_eqs`. The initial configuration (at r_vals[0]) is
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

    # Get initial condition in a cartesian frame. Use the frame aligned to J at the initial separation
    global flags_q1
    if q==1:
        if flags_q1[12]==False:
            print "[orbav_integrator] Warning q=1: input here is cos(varphi), not S."
            flags_q1[12]=True

    L_vals=[(q/(1.+q)**2)*(comp*M**3)**.5 for comp in r_vals]
    v_vals=[(M/comp)**0.5 for comp in r_vals]
    Jvec,Lvec,S1vec,S2vec,dummy=Jframe_projection(xi,S,J,q,S1,S2,r_vals[0])

    Lh_initial=[comp/L_vals[0] for comp in Lvec]
    S1h_initial=[comp/S1 for comp in S1vec]
    S2h_initial=[comp/S2 for comp in S2vec]

    if time:
        t_initial=0
        allvars_initial=list(Lh_initial)+list(S1h_initial)+list(S2h_initial)+list([t_initial])
    else:
        allvars_initial=list(Lh_initial)+list(S1h_initial)+list(S2h_initial)

    #Compute these numbers only once
    eta=q/(1.+q)**2
    m1=M/(1.+q)
    m2=q*M/(1.+q)
    chi1=S1/m1**2
    chi2=S2/m2**2

    # Actual integration
    res =integrate.odeint(orbav_eqs, allvars_initial, v_vals, args=(q,S1,S2,eta,m1,m2,chi1,chi2,time), mxstep=5000000, full_output=0, printmessg=0,rtol=1e-12,atol=1e-12)#,tcrit=sing)

    # Unzip output
    traxres=zip(*res)
    Lhx_fvals=traxres[0]
    Lhy_fvals=traxres[1]
    Lhz_fvals=traxres[2]
    S1hx_fvals=traxres[3]
    S1hy_fvals=traxres[4]
    S1hz_fvals=traxres[5]
    S2hx_fvals=traxres[6]
    S2hy_fvals=traxres[7]
    S2hz_fvals=traxres[8]
    if time:
        t_fvals=traxres[9]

    if time:
        return Lhx_fvals,Lhy_fvals,Lhz_fvals,S1hx_fvals,S1hy_fvals,S1hz_fvals,S2hx_fvals,S2hy_fvals,S2hz_fvals,t_fvals

    else:
        return Lhx_fvals,Lhy_fvals,Lhz_fvals,S1hx_fvals,S1hy_fvals,S1hz_fvals,S2hx_fvals,S2hy_fvals,S2hz_fvals


def orbit_averaged_single(J,xi,S,r_vals,q,S1,S2):

    '''
    Auxiliary function, see `precession.orbit_averaged`.

    **Call:**

        savename=precession.orbit_averaged_single(J,xi,S,r_vals,q,S1,S2)

    **Parameters:**

    - `J`: magnitude of the total angular momentum.
    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `S`: magnitude of the total spin.
    - `r_vals`: binary separation (array).
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.

    **Returns:**

    - `savename`: checkpoint filename.
    '''

    global flags_q1
    if q==1:
        if flags_q1[13]==False:
            print "[orbit_averaged] Warning q=1: Input/output for S is actually cos(varphi)"
            flags_q1[13]=True

    os.system("mkdir -p "+storedir)
    savename= storedir+"/orbav_"+'_'.join([str(x) for x in (J,xi,S,max(r_vals),min(r_vals),len(r_vals),q,S1,S2)])+".dat"

    if not os.path.isfile(savename):
        print "[orbit_averaged] Transferring binary. Output:", savename
        outfilesave = open(savename,"w",0)
        Lhx_fvals,Lhy_fvals,Lhz_fvals,S1hx_fvals,S1hy_fvals,S1hz_fvals,S2hx_fvals,S2hy_fvals,S2hz_fvals = orbav_integrator(J,xi,S,r_vals,q,S1,S2)

        for r_f,Lhx,Lhy,Lhz,S1hx,S1hy,S1hz,S2hx,S2hy,S2hz in zip(r_vals,Lhx_fvals,Lhy_fvals,Lhz_fvals,S1hx_fvals,S1hy_fvals,S1hz_fvals,S2hx_fvals,S2hy_fvals,S2hz_fvals):

            L_f=(q/(1.+q)**2)*(r_f*M**3)**.5
            J_f= ((L_f*Lhx+S1*S1hx+S2*S2hx)**2 + (L_f*Lhy+S1*S1hy+S2*S2hy)**2 + (L_f*Lhz+S1*S1hz+S2*S2hz)**2 )**0.5
            xi_f= ((1.+q)*S1*(Lhx*S1hx+Lhy*S1hy+Lhz*S1hz)+(1.+q**-1)*S2*(Lhx*S2hx+Lhy*S2hy+Lhz*S2hz))*M**-2
            S_f= ((S1*S1hx+S2*S2hx)**2 + (S1*S1hy+S2*S2hy)**2 + (S1*S1hz+S2*S2hz)**2 )**0.5

            if q==1:
                A1=np.sqrt(J_f**2-(L_f-S_f)**2)
                A2=np.sqrt((L_f+S_f)**2-J_f**2)
                A3=np.sqrt(S_f**2-(S1-S2)**2)
                A4=np.sqrt((S1+S2)**2-S_f**2)
                cosvarphi = (4*J_f*S_f**2*S1hz*S1-(S_f**2+S1**2-S2**2)*(J_f**2-L_f**2+S_f**2))/(A1*A2*A3*A4)
                S_f=cosvarphi

            outfilesave.write(str(r_f)+" "+str(J_f)+" "+str(xi_f)+" "+str(S_f)+"\n")
        outfilesave.close()

    #else:
    #    print "[evolve_J_infinity] Skipping. Output:", savename

    return savename


def orbit_averaged(J_vals,xi_vals,S_vals,r_vals,q_vals,S1_vals,S2_vals):

    '''
    Wrapper of `precession.orbav_integrator` to enable parallelization through
    the python parmap module; the number of available cores can be specified
    using the integer global variable `precession.CPUs` (all available cores
    will be used by default). Input/outputs are given in terms of J, xi and S.
    Evolve a sequence of binaries with the different q, S1,S2, xi and initial
    values of J and S; save outputs at the SAME separations r_vals. The initial
    configuration must be compatible with r_vals[0]. Output is a 2D array, where
    e.g. J_vals[0] is the first binary (1D array at all output separations) and
    J_vals[0][0] is the first binary at the first output separation (this is a
    scalar).

    Checkpointing is implemented: results are stored in `precession.storedir`.

    **Call:**

        Jf_vals,xif_vals,Sf_vals=precession.orbit_averaged(J_vals,xi_vals,S_vals,r_vals,q,S1,S2)

    **Parameters:**

    - `Ji_vals`: initial condition for J (array).
    - `xii_vals`: initial condition for xi (array).
    - `Si_vals`: initial condition for S (array).
    - `r_vals`: binary separation (array).
    - `q_vals`: binary mass ratio. Must be q<=1 (array).
    - `S1_vals`: spin magnitude of the primary BH (array).
    - `S2_vals`: spin magnitude of the secondary BH (array).

    **Returns:**

    - `Jf_vals`: solutions for J (2D array).
    - `xif_vals`: solutions for xi (2D array).
    - `Sf_vals`: solutions for S (2D array).
    '''

    global CPUs

    single_flag=False
    try: #Convert float to array if you're evolving just one binary
        len(q_vals)
    except:
        J_vals=[J_vals]
        xi_vals=[xi_vals]
        S_vals=[S_vals]
        q_vals=[q_vals]
        S1_vals=[S1_vals]
        S2_vals=[S2_vals]
        single_flag=True
    try: # Set default
        CPUs
    except:
        CPUs=0
        print "[orbit_averaged] Default parallel computation"

    # Parallelization
    if CPUs==0: # Run on all cpus on the current machine! (default option)
        filelist=parmap.starmap(orbit_averaged_single, zip(J_vals,xi_vals,S_vals,[r_vals for i in range(len(q_vals))],q_vals,S1_vals,S2_vals),pm_parallel=True)
    elif CPUs==1: # 1 cpus done by explicitely removing parallelization
        filelist=parmap.starmap(orbit_averaged_single, zip(J_vals,xi_vals,S_vals,[r_vals for i in range(len(q_vals))],q_vals,S1_vals,S2_vals),pm_parallel=False)
    else: # Run on a given number of CPUs
        filelist=parmap.starmap(orbit_averaged_single, zip(J_vals,xi_vals,S_vals,[r_vals for i in range(len(q_vals))],q_vals,S1_vals,S2_vals),pm_processes=CPUs)

    J_fvals=[]
    S_fvals=[]
    xi_vals=[]
    for index, file in enumerate(filelist):
        print "[orbit_averaged] Reading:", index, file
        dummy,J_f,xi_f,S_f= np.loadtxt(file,unpack=True)
        J_fvals.append(J_f)
        xi_vals.append(xi_f)
        S_fvals.append(S_f)

    if single_flag==True:
        return J_fvals[0], xi_vals[0], S_fvals[0]
    else:
        return J_fvals, xi_vals, S_fvals


def orbit_angles_single(theta1_i,theta2_i,deltaphi_i,r_vals,q,S1,S2):

    '''
    Auxiliary function, see `precession.orbit_angles`.

    **Call:**

        savename=precession.orbit_angles_single(theta1_i,theta2_i,deltaphi_i,r_vals,q,S1,S2)

    **Parameters:**

    - `theta1_i`: initial condition for theta1.
    - `theta2_i`: initial condition for theta2
    - `deltaphi_i`: initial condition for deltaphi.
    - `r_vals`: binary separation (array).
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.

    **Returns:**

    - `savename`: checkpoint filename.
    '''

    os.system("mkdir -p "+storedir)
    savename= storedir+"/orbang_"+'_'.join([str(x) for x in (theta1_i,theta2_i,deltaphi_i,max(r_vals),min(r_vals),len(r_vals),q,S1,S2)])+".dat"

    if not os.path.isfile(savename):
        print "[orbit_angles] Transferring binary. Output:", savename
        outfilesave = open(savename,"w",0)

        # Step 1. Get xi and J for each intial angle. Keep S now
        xi_i,J_i,S_i= from_the_angles(theta1_i,theta2_i,deltaphi_i,q,S1,S2,r_vals[0])
        # Note that S_i is actually cos(varphi_i) when q=1.

        # Step 2. Evolve ODE system
        Lhx_fvals,Lhy_fvals,Lhz_fvals,S1hx_fvals,S1hy_fvals,S1hz_fvals,S2hx_fvals,S2hy_fvals,S2hz_fvals = orbav_integrator(J_i,xi_i,S_i,r_vals,q,S1,S2)

        for r_f,Lhx,Lhy,Lhz,S1hx,S1hy,S1hz,S2hx,S2hy,S2hz in zip(r_vals,Lhx_fvals,Lhy_fvals,Lhz_fvals,S1hx_fvals,S1hy_fvals,S1hz_fvals,S2hx_fvals,S2hy_fvals,S2hz_fvals):

            L_f=(q/(1.+q)**2)*(r_f*M**3)**.5
            S_f= ((S1*S1hx+S2*S2hx)**2 + (S1*S1hy+S2*S2hy)**2 + (S1*S1hz+S2*S2hz)**2 )**0.5
            J_f= ((L_f*Lhx+S1*S1hx+S2*S2hx)**2 + (L_f*Lhy+S1*S1hy+S2*S2hy)**2 + (L_f*Lhz+S1*S1hz+S2*S2hz)**2 )**0.5
            xi_f= ((1.+q)*S1*(Lhx*S1hx+Lhy*S1hy+Lhz*S1hz)+(1.+q**-1)*S2*(Lhx*S2hx+Lhy*S2hy+Lhz*S2hz))*M**-2
            if q==1:
                A1=np.sqrt(J_f**2-(L_f-S_f)**2)
                A2=np.sqrt((L_f+S_f)**2-J_f**2)
                A3=np.sqrt(S_f**2-(S1-S2)**2)
                A4=np.sqrt((S1+S2)**2-S_f**2)
                cosvarphi = (4*J_f*S_f**2*S1hz*S1-(S_f**2+S1**2-S2**2)*(J_f**2-L_f**2+S_f**2))/(A1*A2*A3*A4)
                S_f=cosvarphi

            # Step 3. Back to theta1, theta2, deltaphi
            theta1_f,theta2_f,deltaphi_f,dummy= parametric_angles(S_f,J_f,xi_f,q,S1,S2,r_f)

            # Step 4. Track the precessional phase to set the sign of DeltaPhi. In symbols, the sign of DeltaPhi must be the sign of
            #L dot [ ( S1 - (S1 dot L) dot L ) cross ( S2 - (S2 dot L) dot L ) ]
            S1px=(S1hx-theta1_f*Lhx)
            S1py=(S1hy-theta1_f*Lhy)
            S1pz=(S1hz-theta1_f*Lhz)
            S2px=(S2hx-theta2_f*Lhx)
            S2py=(S2hy-theta2_f*Lhy)
            S2pz=(S2hz-theta2_f*Lhz)
            proj=Lhx*(S1py*S2pz-S1pz*S2py) + Lhy*(S1pz*S2px-S1px*S2pz) + Lhz*(S1px*S2py-S1py*S2px)
            deltaphi_f*=math.copysign(1., proj)

            # Step 4. Store data
            outfilesave.write(str(r_f)+" "+str(theta1_f)+" "+str(theta2_f)+" "+str(deltaphi_f)+"\n")
        outfilesave.close()

    #else:
    #    print "[evolve_angles] Skipping. Output:", savename

    return savename


def orbit_angles(theta1_vals,theta2_vals,deltaphi_vals,r_vals,q_vals,S1_vals,S2_vals):

    '''
    Wrapper of `precession.orbav_integrator` to enable parallelization through
    the python parmap module; the number of available cores can be specified
    using the integer global variable `precession.CPUs` (all available cores
    will be used by default). Input/outputs are given in terms of the angles
    theta1, theta2 and deltaphi. Evolve a sequence of binaries with the
    different q, S1, S2 and initial values for the angles; save outputs at SAME
    separations r_vals. Output is a 2D array, where e.g. theta1_vals[0] is the
    first binary (1D array at all output separations) and theta1_vals[0][0] is
    the first binary at the first output separation (this is a scalar).

    Checkpointing is implemented: results are stored in `precession.storedir`.

    **Call:**

        theta1f_vals,theta2f_vals,deltaphif_vals=precession.orbit_angles(theta1i_vals,theta2i_vals,deltaphii_vals,r_vals,q_vals,S1_vals,S2_vals)

    **Parameters:**

    - `theta1i_vals`: initial condition for theta1 (array).
    - `theta2i_vals`: initial condition for theta2 (array).
    - `deltaphii_vals`: initial condition for deltaphi (array).
    - `r_vals`: binary separation (array).
    - `q_vals`: binary mass ratio. Must be q<=1 (array).
    - `S1_vals`: spin magnitude of the primary BH (array).
    - `S2_vals`: spin magnitude of the secondary BH (array).

    **Returns:**

    - `theta1f_vals`: solutions for theta1 (2D array).
    - `theta2f_vals`: solutions for theta2 (2D array).
    - `deltaphif_vals`: solutions for deltaphi (2D array).
    '''

    global CPUs
    flag=False

    try: #Convert float to array, if you're evolving just one binary
        len(theta1_vals)
        len(theta1_vals)
        len(deltaphi_vals)
    except:
        flag=True
        theta1_vals=[theta1_vals]
        theta2_vals=[theta2_vals]
        deltaphi_vals=[deltaphi_vals]
        q_vals=[q_vals]
        S1_vals=[S1_vals]
        S2_vals=[S2_vals]
    try:
        CPUs
    except:
        CPUs=0
        print "[orbit_angles] Default parallel computation"

    loopflag=True
    while loopflag: # Restart is some of the cores crashed. This happend if you run too many things on too many different machines. Nevermind, trash the file and do it again.
        loopflag=False

        #Parallelization... python is cool indeed
        if CPUs==0: #Run on all cpus on the current machine! (default option)
            filelist=parmap.starmap(orbit_angles_single, zip(theta1_vals,theta2_vals,deltaphi_vals,[r_vals for i in range(len(q_vals))],q_vals,S1_vals,S2_vals),pm_parallel=True)
        elif CPUs==1: #1 cpus done by explicitely removing parallelization
            filelist=parmap.starmap(orbit_angles_single, zip(theta1_vals,theta2_vals,deltaphi_vals,[r_vals for i in range(len(q_vals))],q_vals,S1_vals,S2_vals),pm_parallel=False)
        else: # Run on a given number of CPUs
            filelist=parmap.starmap(orbit_angles_single, zip(theta1_vals,theta2_vals,deltaphi_vals,[r_vals for i in range(len(q_vals))],q_vals,S1_vals,S2_vals),pm_processes=CPUs)

        theta1_fvals=[]
        theta2_fvals=[]
        deltaphi_fvals=[]
        for index, file in enumerate(filelist):
            print "[orbit_angles] Reading:", index, file
            numlines=sum(1 for line in open(file))
            if numlines!=0 and numlines!=len(r_vals): # Restar if core(s) crashed
                print "[orbit_angles] Error on file", file,". Jobs are being restarting!!!"
                os.system("rm "+file)
                loopflag=True

            else:
                dummy,theta1_f,theta2_f,deltaphi_f= np.loadtxt(file,unpack=True)
                theta1_fvals.append(theta1_f)
                theta2_fvals.append(theta2_f)
                deltaphi_fvals.append(deltaphi_f)
    if flag==True:
        return theta1_fvals[0], theta2_fvals[0], deltaphi_fvals[0]
    else:
        return theta1_fvals, theta2_fvals, deltaphi_fvals


def orbit_vectors_single(Lxi,Lyi,Lzi,S1xi,S1yi,S1zi,S2xi,S2yi,S2zi,r_vals,q,time=False):

    '''
    Auxiliary function, see `precession.orbit_vector`.

    **Call:**

        savename=precession.orbit_vectors_single(Lxi,Lyi,Lzi,S1xi,S1yi,S1zi,S2xi,S2yi,S2zi,r_vals,q,time=False)

    **Parameters:**

    - `Lxi`: x component of the vector L, initial.
    - `Lyi`: y component of the vector L, initial.
    - `Lzi`: z component of the vector L, initial.
    - `S1xi`: x component of the vector S1, initial.
    - `S1yi`: y component of the vector S1, initial.
    - `S1zi`: z component of the vector S1, initial.
    - `S2xi`: x component of the vector S2, initial.
    - `S2yi`: y component of the vector S2, initial.
    - `S2zi`: z component of the vector S2, initial.
    - `r_vals`: binary separation (array).
    - `q`: binary mass ratio. Must be q<=1.
    - `time`: if `True` also integrate t(r).

    **Returns:**

    - `savename`: checkpoint filename.
    '''

    os.system("mkdir -p "+storedir)
    savename= storedir+"/orbvec_"+'_'.join([str(x) for x in (Lxi,Lyi,Lzi,S1xi,S1yi,S1zi,S2xi,S2yi,S2zi,max(r_vals),min(r_vals),len(r_vals),q)])+".dat"

    if not os.path.isfile(savename):
        print "[orbit_vectors] Transferring binary. Output:", savename
        outfilesave = open(savename,"w",0)

        v_vals=[(M/comp)**0.5 for comp in r_vals]
        Li=(Lxi**2 + Lyi**2 + Lzi**2)**0.5
        if np.abs(Li - (q/(1.+q)**2)*(r_vals[0]*M**3)**.5) > 1e-8:
            assert False, "[orbit_vector] Initial condition for L not compatible with r_vals[0]"

        S1=(S1xi**2 + S1yi**2 + S1zi**2)**0.5
        S2=(S2xi**2 + S2yi**2 + S2zi**2)**0.5

        Lhxi=Lxi/Li
        Lhyi=Lyi/Li
        Lhzi=Lzi/Li
        S1hxi=S1xi/S1
        S1hyi=S1yi/S1
        S1hzi=S1zi/S1
        S2hxi=S2xi/S2
        S2hyi=S2yi/S2
        S2hzi=S2zi/S2

        if time:
            t_initial=0
            allvars_initial=[Lhxi,Lhyi,Lhzi,S1hxi,S1hyi,S1hzi,S2hxi,S2hyi,S2hzi,t_initial]
        else:
            allvars_initial=[Lhxi,Lhyi,Lhzi,S1hxi,S1hyi,S1hzi,S2hxi,S2hyi,S2hzi]

        #Compute these numbers only once
        eta=q/(1.+q)**2
        m1=M/(1.+q)
        m2=q*M/(1.+q)
        chi1=S1/m1**2
        chi2=S2/m2**2

        # Actual integration
        res =integrate.odeint(orbav_eqs, allvars_initial, v_vals, args=(q,S1,S2,eta,m1,m2,chi1,chi2,time), mxstep=5000000, full_output=0, printmessg=0,rtol=1e-12,atol=1e-12)#,tcrit=sing)

        # Unzip output
        traxres=zip(*res)
        Lhx_fvals=traxres[0]
        Lhy_fvals=traxres[1]
        Lhz_fvals=traxres[2]
        S1hx_fvals=traxres[3]
        S1hy_fvals=traxres[4]
        S1hz_fvals=traxres[5]
        S2hx_fvals=traxres[6]
        S2hy_fvals=traxres[7]
        S2hz_fvals=traxres[8]
        if time:
            t_fvals=traxres[9]

        if time:
            for r_f,Lhx,Lhy,Lhz,S1hx,S1hy,S1hz,S2hx,S2hy,S2hz,t_f in zip(r_vals,Lhx_fvals,Lhy_fvals,Lhz_fvals,S1hx_fvals,S1hy_fvals,S1hz_fvals,S2hx_fvals,S2hy_fvals,S2hz_fvals,t_fvals):
                L_f=(q/(1.+q)**2)*(r_f*M**3)**.5
                outfilesave.write(str(r_f)+" "+str(Lhx*L_f)+" "+str(Lhy*L_f)+" "+str(Lhz*L_f)+" "+str(S1hx*S1)+" "+str(S1hy*S1)+" "+str(S1hz*S1)+" "+str(S2hx*S2)+" "+str(S2hy*S2)+" "+str(S2hz*S2)+" "+str(t_f)+"\n")
        else:
            for r_f,Lhx,Lhy,Lhz,S1hx,S1hy,S1hz,S2hx,S2hy,S2hz in zip(r_vals,Lhx_fvals,Lhy_fvals,Lhz_fvals,S1hx_fvals,S1hy_fvals,S1hz_fvals,S2hx_fvals,S2hy_fvals,S2hz_fvals):
                L_f=(q/(1.+q)**2)*(r_f*M**3)**.5
                outfilesave.write(str(r_f)+" "+str(Lhx*L_f)+" "+str(Lhy*L_f)+" "+str(Lhz*L_f)+" "+str(S1hx*S1)+" "+str(S1hy*S1)+" "+str(S1hz*S1)+" "+str(S2hx*S2)+" "+str(S2hy*S2)+" "+str(S2hz*S2)+"\n")
        outfilesave.close()

    #else:
    #    print "[orbit_vectors] Skipping. Output:", savename

    return savename


def orbit_vectors(Lxi_vals,Lyi_vals,Lzi_vals,S1xi_vals,S1yi_vals,S1zi_vals,S2xi_vals,S2yi_vals,S2zi_vals,r_vals,q_vals,time=False):

    '''
    Wrapper of the orbit-averaged PN integrator to enable parallelization
    through the python parmap module; the number of available cores can be
    specified using the integer global variable `precession.CPUs` (all available
    cores will be used by default). Inputs and outputs are given in terms of the
    components of L, S1 and S2 in some inertial frame (cf. e.g.
    `precession.Jframe_projection`). Vectors, not unit vectros!, are returned.
    Evolve a sequence of binaries with the different initial configurations;
    save outputs at SAME separations r_vals. The initial configuration must be
    compatible with r_vals[0]. Output is a 2D array, where e.g. Lx_fvals[0] is
    the first binary (1D array at all output separations) and Lx_fvals[0][0] is
    the first binary at the first output separation (this is a scalar).

    Checkpointing is implemented: results are stored in `precession.storedir`.

    **Call:**

        Lx_fvals,Ly_fvals,Lz_fvals,S1x_fvals,S1y_fvals,S1z_fvals,S2x_fvals,S2y_fvals,S2z_fvals=precession.orbit_vectors(Lxi_vals,Lyi_vals,Lzi_vals,S1xi_vals,S1yi_vals,S1zi_vals,S2xi_vals,S2yi_vals,S2zi_vals,r_vals,q_vals,time=False)

    **Parameters:**

    - `Lxi_vals`: x component of the vector L, initial (array).
    - `Lyi_vals`: y component of the vector L, initial (array).
    - `Lzi_vals`: z component of the vector L, initial (array).
    - `S1xi_vals`: x component of the vector S1, initial (array).
    - `S1yi_vals`: y component of the vector S1, initial (array).
    - `S1zi_vals`: z component of the vector S1, initial (array).
    - `S2xi_vals`: x component of the vector S2, initial (array).
    - `S2yi_vals`: y component of the vector S2, initial (array).
    - `S2zi_vals`: z component of the vector S2, initial (array).
    - `r_vals`: binary separation (array).
    - `q_vals`: binary mass ratio. Must be q<=1 (array).
    - `time`: if `True` also integrate t(r).

    **Returns:**

    - `Lx_fvals`: x component of the vector L, final (2D array).
    - `Ly_fvals`: y component of the vector L, final (2D array).
    - `Lz_fvals`: z component of the vector L, final (2D array).
    - `S1x_fvals`: x component of the vector S1, final (2D array).
    - `S1y_fvals`: y component of the vector S1, final (2D array).
    - `S1z_fvals`: z component of the vector S1, final (2D array).
    - `S2x_fvals`: x component of the vector S2, final (2D array).
    - `S2y_fvals`: y component of the vector S2, final (2D array).
    - `S2z_fvals`: z component of the vector S2, final (2D array).
    '''

    global CPUs

    single_flag=False
    try: #Convert float to array if you're evolving just one binary
        len(q_vals)
    except:
        Lxi_vals=[Lxi_vals]
        Lyi_vals=[Lyi_vals]
        Lzi_vals=[Lzi_vals]
        S1xi_vals=[S1xi_vals]
        S1yi_vals=[S1yi_vals]
        S1zi_vals=[S1zi_vals]
        S2xi_vals=[S2xi_vals]
        S2yi_vals=[S2yi_vals]
        S2zi_vals=[S2zi_vals]
        q_vals=[q_vals]
        single_flag=True
    try: # Set default
        CPUs
    except:
        CPUs=0
        print "[orbit_vectors] Default parallel computation"

    # Parallelization
    if CPUs==0: # Run on all cpus on the current machine! (default option)
        filelist=parmap.starmap(orbit_vectors_single, zip(Lxi_vals,Lyi_vals,Lzi_vals,S1xi_vals,S1yi_vals,S1zi_vals,S2xi_vals,S2yi_vals,S2zi_vals,[r_vals for i in range(len(q_vals))],q_vals),time,pm_parallel=True)
    elif CPUs==1: # 1 cpus done by explicitely removing parallelization
        filelist=parmap.starmap(orbit_vectors_single, zip(Lxi_vals,Lyi_vals,Lzi_vals,S1xi_vals,S1yi_vals,S1zi_vals,S2xi_vals,S2yi_vals,S2zi_vals,[r_vals for i in range(len(q_vals))],q_vals),time,pm_parallel=False)
    else: # Run on a given number of CPUs
        filelist=parmap.starmap(orbit_vectors_single, zip(Lxi_vals,Lyi_vals,Lzi_vals,S1xi_vals,S1yi_vals,S1zi_vals,S2xi_vals,S2yi_vals,S2zi_vals,[r_vals for i in range(len(q_vals))],q_vals),time,pm_processes=CPUs)

    Lx_fvals=[]
    Ly_fvals=[]
    Lz_fvals=[]
    S1x_fvals=[]
    S1y_fvals=[]
    S1z_fvals=[]
    S2x_fvals=[]
    S2y_fvals=[]
    S2z_fvals=[]
    if time:
        t_fvals=[]
    for index, file in enumerate(filelist):
        print "[orbit_vectors] Reading:", index, file
        if time:
            dummy,Lx,Ly,Lz,S1x,S1y,S1z,S2x,S2y,S2z,tf= np.loadtxt(file,unpack=True)
        else:
            dummy,Lx,Ly,Lz,S1x,S1y,S1z,S2x,S2y,S2z= np.loadtxt(file,unpack=True)
        Lx_fvals.append(Lx)
        Ly_fvals.append(Ly)
        Lz_fvals.append(Lz)
        S1x_fvals.append(S1x)
        S1y_fvals.append(S1y)
        S1z_fvals.append(S1z)
        S2x_fvals.append(S2x)
        S2y_fvals.append(S2y)
        S2z_fvals.append(S2z)
        if time:
            t_fvals.append(tf)

    if single_flag==True:
        if time:
            return Lx_fvals[0], Ly_fvals[0], Lz_fvals[0], S1x_fvals[0], S1y_fvals[0], S1z_fvals[0], S2x_fvals[0], S2y_fvals[0], S2z_fvals[0],t_fvals[0]
        else:
            return Lx_fvals[0], Ly_fvals[0], Lz_fvals[0], S1x_fvals[0], S1y_fvals[0], S1z_fvals[0], S2x_fvals[0], S2y_fvals[0], S2z_fvals[0]
    else:
        if time:
            return Lx_fvals, Ly_fvals, Lz_fvals, S1x_fvals, S1y_fvals, S1z_fvals, S2x_fvals, S2y_fvals, S2z_fvals, t_fvals
        else:
            return Lx_fvals, Ly_fvals, Lz_fvals, S1x_fvals, S1y_fvals, S1z_fvals, S2x_fvals, S2y_fvals, S2z_fvals


def hybrid_single(xi,kappa_inf,r_vals,q,S1,S2,r_t):

    '''
    Auxiliary function, see `hybrid`.

    **Call:**

        savename=precession.hybrid_single(xi,kappa_inf,r_vals,q,S1,S2,r_t)

    **Parameters:**

    - `xi`: projection of the effective spin along the orbital angular momentum.
    - `kappa_inf`: asymtotic value of kappa at large separations.
    - `r_vals`: binary separation (array).
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `r_t`: transition radius between orbit- and precession-averaged approach.

    **Returns:**

    - `savename`: checkpoint filename.
    '''
    global flags_q1
    if q==1:
        if flags_q1[14]==False:
            print "[hybrid] Warning q=1: required intial condition is S, not kappa_inf."
            flags_q1[14]=True # Suppress future warnings

    os.system("mkdir -p "+storedir)
    savename= storedir+"/hybrid_"+'_'.join([str(x) for x in (xi,kappa_inf,max(r_vals),min(r_vals),len(r_vals),q,S1,S2,r_t)])+".dat"

    if not os.path.isfile(savename):
        print "[hybrid] Transferring binary. Output:", savename
        outfilesave = open(savename,"w",0)

        # Split the output separations: precession-average before r_t and orbit-average after it
        r_vals_pa=[r for r in r_vals if r>r_t]
        r_vals_oa=[r for r in r_vals if r<=r_t] # Keep r_t (if present) in the orbit-average part

        if not [r for r in r_vals_oa if r!=r_t]: # If there's nothing but r_t in the orbit-averaged part
            assert False, "[hybrid] No output required below r_t. You don't need a hybrid integration, use evolve_J_infinity instead"

        # Add the threshold at the end of the precession-average part and at the beginning of the orbit-average part
        r_vals_pa.append(r_t)
        r_vals_oa.insert(0,r_t)

        # Evolve from r=infinity to r=r_t using precession-averaged integration
        J_vals_pa=Jofr_infinity(xi,kappa_inf,r_vals_pa,q,S1,S2)

        # Store the angles theta1, theta2 and deltaphi (need S resampling at each output separation)
        # Don't use the latest values in the arrays, because you added one value at the end earlier on
        for J_f,r_f in zip(J_vals_pa[:-1],r_vals_pa[:-1]):

            S_f=samplingS(xi,J_f,q,S1,S2,r_f)
            theta1_f,theta2_f,deltaphi_f,dummy = parametric_angles(S_f,J_f,xi,q,S1,S2,r_f)
            deltaphi_f*=random.choice([-1., 1.])
            outfilesave.write(str(r_f)+" "+str(theta1_f)+" "+str(theta2_f)+" "+str(deltaphi_f)+"\n")

        # Last S resampling at r=r_t
        S_t=samplingS(xi,J_vals_pa[-1],q,S1,S2,r_t)

        # Evolve from r_t to min(r_vals) using orbit-average integration
        Lhx_vals_oa,Lhy_vals_oa,Lhz_vals_oa,S1hx_vals_oa,S1hy_vals_oa,S1hz_vals_oa,S2hx_vals_oa,S2hy_vals_oa,S2hz_vals_oa = orbav_integrator(J_vals_pa[-1],xi,S_t,r_vals_oa,q,S1,S2)

        # Store the angles theta1, theta2 and deltaphi (S resampling not needed)
        # Don't use the first values in the arrays, because you added one value on top earlier on
        for r_f,Lhx,Lhy,Lhz,S1hx,S1hy,S1hz,S2hx,S2hy,S2hz in zip(r_vals_oa[1:],Lhx_vals_oa[1:],Lhy_vals_oa[1:],Lhz_vals_oa[1:],S1hx_vals_oa[1:],S1hy_vals_oa[1:],S1hz_vals_oa[1:],S2hx_vals_oa[1:],S2hy_vals_oa[1:],S2hz_vals_oa[1:]):
            L_f=(q/(1.+q)**2)*(r_f*M**3)**.5
            S_f=((S1*S1hx+S2*S2hx)**2 + (S1*S1hy+S2*S2hy)**2 + (S1*S1hz+S2*S2hz)**2 )**0.5
            J_f=((L_f*Lhx+S1*S1hx+S2*S2hx)**2 + (L_f*Lhy+S1*S1hy+S2*S2hy)**2 + (L_f*Lhz+S1*S1hz+S2*S2hz)**2 )**0.5
            xi_f=((1.+q)*S1*(Lhx*S1hx+Lhy*S1hy+Lhz*S1hz)+(1.+q**-1)*S2*(Lhx*S2hx+Lhy*S2hy+Lhz*S2hz))*M**-2
            if q==1: # You need to compute varphi, not S
                A1=np.sqrt(J_f**2-(L_f-S_f)**2)
                A2=np.sqrt((L_f+S_f)**2-J_f**2)
                A3=np.sqrt(S_f**2-(S1-S2)**2)
                A4=np.sqrt((S1+S2)**2-S_f**2)
                cosvarphi=(4*J_f*S_f**2*S1hz*S1-(S_f**2+S1**2-S2**2)*(J_f**2-L_f**2+S_f**2))/(A1*A2*A3*A4)
                S_f=cosvarphi

            theta1_f,theta2_f,deltaphi_f,dummy= parametric_angles(S_f,J_f,xi_f,q,S1,S2,r_f)
            # Track the precessional phase to set the sign of DeltaPhi. In symbols, the sign of DeltaPhi must be the sign of
            #L dot [ ( S1 - (S1 dot L) dot L ) cross ( S2 - (S2 dot L) dot L ) ]
            S1px=(S1hx-theta1_f*Lhx)
            S1py=(S1hy-theta1_f*Lhy)
            S1pz=(S1hz-theta1_f*Lhz)
            S2px=(S2hx-theta2_f*Lhx)
            S2py=(S2hy-theta2_f*Lhy)
            S2pz=(S2hz-theta2_f*Lhz)
            proj=Lhx*(S1py*S2pz-S1pz*S2py) + Lhy*(S1pz*S2px-S1px*S2pz) + Lhz*(S1px*S2py-S1py*S2px)
            deltaphi_f*=math.copysign(1., proj)

            outfilesave.write(str(r_f)+" "+str(theta1_f)+" "+str(theta2_f)+" "+str(deltaphi_f)+"\n")
        outfilesave.close()

    return savename


def hybrid(xi_vals,kappainf_vals,r_vals,q_vals,S1_vals,S2_vals,r_t):

    '''
    Hybrid inspiral. Evolve a binary FROM INIFINITELY large separations (as
    specified by kappa_inf and xi) till the threshold r_t using the
    precession-averaged approach, and then from r_t to the end of the inspiral
    using an orbit-averaged integration to track the precessional phase.

    Parallelization is implemented through the python parmap module; the number
    of available cores can be specified using the integer global variable
    `precession.CPUs` (all available cores will be used by default). Evolve a
    sequence of binaries with the different q, S1,S2, xi and kappa_inf. Save
    outputs at SAME separations r_vals; r_t must also be the same for all
    binaries

    The initial condition is NOT returned by this function. Outputs are given in
    terms of the angles theta1, theta2 and deltaphi as 2D arrays, where e.g
    theta1_fvals[0] is the first binary (1D array at all output separations) and
    theta1_fvals[0][0] is the first binary at the first output separation (this
    is a scalar).

    **Call:**
        theta1f_vals,theta2f_vals,deltaphif_vals=precession.hybrid(xi_vals,kappainf_vals,r_vals,q_vals,S1_vals,S2_vals,r_t)

    **Parameters:**

    - `xi_vals`: projection of the effective spin along the orbital angular momentum (array).
    - `kappainf_vals`: asymtotic value of kappa at large separations (array).
    - `r_vals`: binary separation (array).
    - `q_vals`: binary mass ratio. Must be q<=1 (array).
    - `S1_vals`: spin magnitude of the primary BH (array).
    - `S2_vals`: spin magnitude of the secondary BH (array).
    - `r_t`: transition radius between orbit- and precession-averaged approach.

    **Returns:**

    - `theta1f_vals`: solutions for theta1 (2D array).
    - `theta2f_vals`: solutions for theta2 (2D array).
    - `deltaphif_vals`: solutions for deltaphi (2D array).
    '''

    global CPUs

    single_flag=False

    try: #Convert float to array if you're evolving just one binary
        len(q_vals)
    except:
        single_flag=True
        xi_vals=[xi_vals]
        kappainf_vals=[kappainf_vals]
        q_vals=[q_vals]
        S1_vals=[S1_vals]
        S2_vals=[S2_vals]
    try: # Set defaults
        CPUs
    except:
        CPUs=0
        print "[hybrid] Default parallel computation"

    loopflag=True
    while loopflag: # Restart is some of the cores crashed. This happend if you run too many things on too many different machines. Nevermind, trash the file and do it again.
        loopflag=False

        #Parallelization
        if CPUs==0: #Run on all cpus on the current machine! (default option)
            filelist=parmap.starmap(hybrid_single, zip(xi_vals,kappainf_vals,[r_vals for i in range(len(q_vals))],q_vals,S1_vals,S2_vals,[r_t for i in range(len(q_vals))]),pm_parallel=True)
        elif CPUs==1: #1 cpus done by explicitely removing parallelization
            filelist=parmap.starmap(hybrid_single, zip(xi_vals,kappainf_vals,[r_vals for i in range(len(q_vals))],q_vals,S1_vals,S2_vals,[r_t for i in range(len(q_vals))]),pm_parallel=False)
        else: # Run on a given number of CPUs
            filelist=parmap.starmap(hybrid_single, zip(xi_vals,kappainf_vals,[r_vals for i in range(len(q_vals))],q_vals,S1_vals,S2_vals,[r_t for i in range(len(q_vals))]),pm_processes=CPUs)

        theta1_fvals=[]
        theta2_fvals=[]
        deltaphi_fvals=[]
        for index, file in enumerate(filelist):
            print "[hybrid] Reading:", index, file
            numlines=sum(1 for line in open(file))
            if numlines!=0 and numlines!=len(r_vals): # Restart if core(s) crashed
                print "[hybrid] Error on file", file,". Jobs are being restarted!"
                os.system("rm "+file)
                loopflag=True

            else:
                dummy,theta1_f,theta2_f,deltaphi_f= np.loadtxt(file,unpack=True)
                theta1_fvals.append(theta1_f)
                theta2_fvals.append(theta2_f)
                deltaphi_fvals.append(deltaphi_f)
    if single_flag==True:
        return theta1_fvals[0], theta2_fvals[0], deltaphi_fvals[0]
    else:
        return theta1_fvals, theta2_fvals, deltaphi_fvals



#################################
########## BH REMNANT ###########
#################################


def finalmass(theta1,theta2,deltaPhi,q,S1,S2):

    '''
    Estimate the final mass of the BH renmant following a BH merger. We
    implement the fitting formula to numerical relativity simulations by
    Barausse Morozova Rezzolla 2012.  See also Gerosa and Sesana 2015. This
    formula has to be applied *close to merger*, where numerical relativity
    simulations are available. You should do a PN evolution to transfer binaries
    at r~10M.

    **Call:**

        Mfin=precession.finalmass(theta1,theta2,deltaPhi,q,S1,S2)

    **Parameters:**

    - `theta1`: angle between the spin of the primary and the orbital angular momentum.
    - `theta2`: angle between the spin of the secondary and the orbital angular momentum.
    - `deltaphi`: angle between the projection of the two spins on the orbital plane.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.

    **Returns:**

    - `Mfin`: mass of the BH remnant, in units of the (pre-merger) binary total mass
    '''

    chi1=S1/(M/(1.+q))**2   # Dimensionless spin
    chi2=S2/(q*M/(1.+q))**2 # Dimensionless spin
    eta=q*pow(1.+q,-2.)     # Symmetric mass ratio

    # Spins here are defined in a frame with L along z and S1 in xz
    hatL=np.array([0,0,1])
    hatS1=np.array([np.sin(theta1),0,np.cos(theta1)])
    hatS2 = np.array([np.sin(theta2)*np.cos(deltaPhi),np.sin(theta2)*np.sin(deltaPhi),np.cos(theta2)])
    #Useful spin combinations.
    Delta= (q*chi2*hatS2-chi1*hatS1)/(1.+q)
    Delta_par=np.dot(Delta,hatL)
    Delta_perp=np.linalg.norm(np.cross(Delta,hatL))
    chit= (q*q*chi2*hatS2+chi1*hatS1)/pow(1.+q,2.)
    chit_par=np.dot(chit,hatL)
    chit_perp=np.linalg.norm(np.cross(chit,hatL))

    #Final mass. Barausse Morozova Rezzolla 2012
    p0=0.04827
    p1=0.01707
    Z1=1.+ pow(1.-pow(chit_par,2.),1./3.)* (pow(1.+chit_par,1./3.)+pow(1.-chit_par,1./3.))
    Z2=pow(3.*pow(chit_par,2.)+pow(Z1,2.),1./2.)
    risco=3.+Z2-math.copysign(1.,chit_par)*pow((3.-Z1)*(3.+Z1+2.*Z2),1./2.)
    Eisco=pow(1.-2./(3.*risco),1./2)
    #Radiated energy, in unit of the initial total mass of the binary
    Erad= eta*(1.-Eisco)+4.*pow(eta,2.)*(4.*p0+16.*p1*chit_par*(chit_par+1.)+Eisco-1.)
    Mfin=M*(1.- Erad) # Final mass

    return Mfin


def finalspin(theta1,theta2,deltaPhi,q,S1,S2):

    '''
    Estimate the final mass of the BH renmant following a BH merger. We
    implement the fitting formula to numerical relativity simulations by
    Barausse Rezzolla 2009.  See also Gerosa and Sesana 2015. We return the
    dimensionless spin, which is the spin in units of the (pre-merger) binary
    total mass, not the spin in units of the actual BH remnant. This can be
    obtained combing this function with `precession.finalmass`. Maximally
    spinning BHs are returned if/whenever the fitting formula returns
    dimensionless spins greater than 1. This formula has to be applied *close to
    merger*, where numerical relativity simulations are available. You should do
    a PN evolution to transfer binaries at r~10M.

    **Call:**

        chifin=precession.finalspin(theta1,theta2,deltaPhi,q,S1,S2)

    **Parameters:**

    - `theta1`: angle between the spin of the primary and the orbital angular momentum.
    - `theta2`: angle between the spin of the secondary and the orbital angular momentum.
    - `deltaphi`: angle between the projection of the two spins on the orbital plane.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.

    **Returns:**

    - `chifin`: dimensionless spin of the BH remnant
    '''

    chi1=S1/(M/(1.+q))**2   # Dimensionless spin
    chi2=S2/(q*M/(1.+q))**2 # Dimensionless spin
    eta=q*pow(1.+q,-2.)     # Symmetric mass ratio

    # Spins here are defined in a frame with L along z and S1 in xz
    hatL=np.array([0,0,1])
    hatS1=np.array([np.sin(theta1),0,np.cos(theta1)])
    hatS2 = np.array([np.sin(theta2)*np.cos(deltaPhi),np.sin(theta2)*np.sin(deltaPhi),np.cos(theta2)])
    #Useful spin combinations.
    Delta= (q*chi2*hatS2-chi1*hatS1)/(1.+q)
    Delta_par=np.dot(Delta,hatL)
    Delta_perp=np.linalg.norm(np.cross(Delta,hatL))
    chit= (q*q*chi2*hatS2+chi1*hatS1)/pow(1.+q,2.)
    chit_par=np.dot(chit,hatL)
    chit_perp=np.linalg.norm(np.cross(chit,hatL))

    #Final spin. Barausse Rezzolla 2009
    t0=-2.8904
    t2=-3.51712
    t3=2.5763
    s4=-0.1229
    s5=0.4537
    smalll = 2.*pow(3.,1./2.) + t2*eta+t3*pow(eta,2.) + s4*np.dot(chit,chit)*pow(1.+q,4.)*pow(1+q*q,-2.) + (s5*eta+t0+2.)*chit_par*pow(1.+q,2)*pow(1.+q*q,-1)
    chifin=np.linalg.norm( chit+hatL*smalll*q/pow(1.+q,2.) )
    if chifin>1.: #Check on the final spin, as suggested by Emanuele
        print "[finalspin] Warning: got chi>1, force chi=1"
        chifin==1.

    return chifin


def finalkick(theta1,theta2,deltaPhi,q,S1,S2,maxkick=False,kms=False,more=False):

    '''
    Estimate the final kick of the BH remnant following a BH merger. We
    implement the fitting formula to numerical relativity simulations developed
    by the Rochester group. The larger contribution comes from the component of
    the kick parallel to L. Flags let you switch on and off the various
    contributions (all on by default): superkicks (Gonzalez et al. 2007a;
    Campanelli et al. 2007), hang-up kicks (Lousto & Zlochower 2011),
    cross-kicks (Lousto & Zlochower 2013). The orbital-plane kick components are
    implemented as described in Kesden et al. 2010a. See also Gerosa and Sesana
    2015.

    The final kick depends on the orbital phase at merger Theta. By default,
    this is assumed to be randonly distributed in [0,2pi]. The maximum kick is
    realized for Theta=0 and can be computed with the optional argument
    maxkick=True. This formula has to be applied *close to merger*, where
    numerical relativity simulations are available. You should do a PN evolution
    to transfer binaries at r~10M.

    The final kick is returned in geometrical units (i.e. vkick/c) by default,
    and converted to km/s if kms=True.

    **Call:**

        vkick=precession.finalkick(theta1,theta2,deltaphi,q,S1,S2,maxkick=False,kms=False,more=False)

    **Parameters:**

    - `theta1`: angle between the spin of the primary and the orbital angular momentum.
    - `theta2`: angle between the spin of the secondary and the orbital angular momentum.
    - `deltaphi`: angle between the projection of the two spins on the orbital plane.
    - `q`: binary mass ratio. Must be q<=1.
    - `S1`: spin magnitude of the primary BH.
    - `S2`: spin magnitude of the secondary BH.
    - `maxkick`: if `True` maximizes over the orbital phase at merger.
    - `kms`: if `True` convert result to km/s.
    - `more`: if `True` returns additional quantities.

    **Returns:**

    - `vkick`: kick of the BH remnant
    - `vm`: (optional) mass-asymmetry term
    - `vperp`: (optional) spin-asymmetry term perpendicular to L
    - `v_e1`: (optional) component of the orbital-plane kick
    - `v_e2`: (optional) component of the orbital-plane kick
    - `vpar`: (optional) spin-asymmetry term along L

    '''

    chi1=S1/(M/(1.+q))**2   # Dimensionless spin
    chi2=S2/(q*M/(1.+q))**2 # Dimensionless spin
    eta=q*pow(1.+q,-2.)     # Symmetric mass ratio

    # Spins here are defined in a frame with L along z and S1 in xz
    hatL=np.array([0,0,1])
    hatS1=np.array([np.sin(theta1),0,np.cos(theta1)])
    hatS2 = np.array([np.sin(theta2)*np.cos(deltaPhi),np.sin(theta2)*np.sin(deltaPhi),np.cos(theta2)])
    #Useful spin combinations.
    Delta= -(q*chi2*hatS2-chi1*hatS1)/(1.+q) # Minus sign added in v1.0.2. Typo in the paper.
    Delta_par=np.dot(Delta,hatL)
    Delta_perp=np.linalg.norm(np.cross(Delta,hatL))
    chit= (q*q*chi2*hatS2+chi1*hatS1)/pow(1.+q,2.)
    chit_par=np.dot(chit,hatL)
    chit_perp=np.linalg.norm(np.cross(chit,hatL))

    #Kick. Coefficients are quoted in km/s

    # vm and vperp are like in Kesden at 2010a, vpar is modified from Lousto Zlochower 2013
    zeta=np.radians(145.)
    A=1.2e4
    B=-0.93
    H=6.9e3

    # Switch on/off the various (super)kick contribution. Default are all on
    superkick=True
    hangupkick=True
    crosskick=True

    if superkick==True:
        V11=3677.76
    else:
        V11=0.
    if hangupkick==True:
        VA=2481.21
        VB=1792.45
        VC=1506.52
    else:
        VA=0.
        VB=0.
        VC=0.
    if crosskick==True:
        C2=1140.
        C3=2481.
    else:
        C2=0.
        C3=0.

    if maxkick==True:
        bigTheta=0
    else:
        bigTheta=np.random.uniform(0., 2.*np.pi)

    vm=A*eta*eta*(1.+B*eta)*(1.-q)/(1.+q)
    vperp=H*eta*eta*Delta_par
    vpar=16.*eta*eta* (Delta_perp*(V11+2.*VA*chit_par+4.*VB*pow(chit_par,2.)+8.*VC*pow(chit_par,3.)) + chit_perp*Delta_par*(2.*C2+4.*C3*chit_par)) * np.cos(bigTheta)
    vkick=np.linalg.norm([vm+vperp*np.cos(zeta),vperp*np.sin(zeta),vpar])

    if vkick>5000:
        print "[finalkick] Warning; I got v_kick>5000km/s. This shouldn't be possibile"

    if not kms: # divide by the speed of light in km/s
        c_kms=299792.458
        vkick=vkick/299792.458
        vm=vm/299792.458
        vperp=vperp/299792.458
        vpar=vpar/299792.458

    if more:
        return vkick, vm, vperp, vm+vperp*np.cos(zeta), vperp*np.sin(zeta), vpar
    else:
        return vkick




#################################
########## UTILITIES ############
#################################


def ftor(f,M_msun):

    '''
    Conversion between binary separation r (in mass unit) and emitted GW
    frequency f (in Hertz). We use the Newtonian expression: f^2 = G M / (pi^2
    r^3) in cgs units. Mass units: r--> GMr/c^2

    **Call:**

        r=precession.ftor(f,M_msun)

    **Parameters:**

    - `f`: emitted GW frequency in Hertz.
    - `M_msun`: binary total mass in solar masses.

    **Returns:**

    - `r`: binary separation.
    '''



    M_cgs=M_msun*(2e33)
    c_cgs=2.99e10
    G_cgs=6.67e-8
    r=pow(pow(c_cgs,3.),2./3.)*pow(math.pi*f*G_cgs*M_cgs,-2./3.)
    return r


def rtof(r,M_msun):

    '''
    Conversion between emitted GW frequency f (in Hertz) and binary separation r
    (in mass unit). We use the Newtonian expression: f^2 = G M / (pi^2 r^3) in
    cgs units. Mass units: r--> GMr/c^2

    **Call:**

        f=precession.rtof(r,M_msun)

    **Parameters:**

    - `r`: binary separation.
    - `M_msun`: binary total mass in solar masses.

    **Returns:**

    - `f`: emitted GW frequency in Hertz.
    '''

    M_cgs=M_msun*(2e33)
    c_cgs=2.99e10
    G_cgs=6.67e-8
    f=pow(c_cgs,3.)/(math.pi*G_cgs*M_cgs*pow(r,3./2.))
    return f


def cutoff(detector,M_msun):

    '''
    Return the GW frequency and binary separation (in total-mass units) when
    binary enter the sensitivity window of a typical ground-based LIGO-like
    detector or a LISA-like space mission.

    **Call:**

        r,f=precession.cutoff(detector,M_msun)

    **Parameters:**

    - `detector`: specify either *space* or *ground*.
    - `M_msun`: binary total mass in solar masses.

    **Returns:**

    - `r`: binary separation.
    - `f`: emitted GW frequency in Hertz.
    '''

    if detector=="ground":
        fcut=10 # Hz
    elif detector=="space":
        fcut= 1e-5 # Hz
    else:
        assert False, "[cutoff] Please select 'space' or 'ground'. Otherwise run ftor with the chosen frequency"
    rcut=ftor(fcut,M_msun)
    return rcut, fcut
